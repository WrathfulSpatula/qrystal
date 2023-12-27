use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

pub struct ParallelFor {
    p_stride: u64,
    dispatch_threshold: u64,
    num_cores: u32,
}

impl ParallelFor {
    pub fn new() -> Self {
        let p_stride = match std::env::var("QRACK_PSTRIDEPOW") {
            Ok(val) => (1 << val.parse::<usize>().unwrap()),
            Err(_) => (1 << PSTRIDEPOW),
        };
        let num_cores = num_cpus::get();
        let p_stride_pow = (p_stride as f64).log2() as usize;
        let min_stride_pow = if num_cores > 1 {
            (num_cores - 1).next_power_of_two().trailing_zeros() as usize
        } else {
            0
        };
        let dispatch_threshold = if p_stride_pow > min_stride_pow {
            p_stride_pow - min_stride_pow
        } else {
            0
        };
        ParallelFor {
            p_stride,
            num_cores,
            dispatch_threshold,
        }
    }

    pub fn set_concurrency_level(&mut self, num: u32) {
        if self.num_cores == num {
            return;
        }
        self.num_cores = num;
        let p_stride_pow = self.p_stride.log2();
        let min_stride_pow = if self.num_cores > 1 {
            (self.num_cores - 1).log2()
        } else {
            0
        };
        self.dispatch_threshold = if p_stride_pow > min_stride_pow {
            p_stride_pow - min_stride_pow
        } else {
            0
        };
    }

    pub fn get_concurrency_level(&self) -> u32 {
        self.num_cores
    }

    pub fn get_stride(&self) -> u64 {
        self.p_stride
    }

    pub fn get_preferred_concurrency_power(&self) -> u64 {
        self.dispatch_threshold
    }

    fn par_for_inc<F, G>(&self, begin: usize, item_count: usize, inc: G, fn: F)
    where
        F: Fn(usize, usize) + Send + Sync,
        G: Fn(usize) -> usize + Send + Sync,
    {
        let stride = self.p_stride;
        let threads = if item_count / stride > self.num_cores {
            self.num_cores
        } else {
            item_count / stride
        };
        if threads <= 1 {
            for j in begin..(begin + item_count) {
                fn(inc(j), 0);
            }
            return;
        }
        let idx = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for cpu in 0..threads {
            let idx = Arc::clone(&idx);
            let begin = begin;
            let item_count = item_count;
            let stride = stride;
            let inc = inc;
            let handle = thread::spawn(move || {
                loop {
                    let i = idx.fetch_add(1, Ordering::SeqCst);
                    let l = i * stride;
                    if l >= item_count {
                        break;
                    }
                    let max_j = if l + stride < item_count {
                        stride
                    } else {
                        item_count - l
                    };
                    for j in 0..max_j {
                        let k = j + l;
                        fn(inc(begin + k), cpu);
                    }
                }
            });
            handles.push(handle);
        }
        for handle in handles {
            handle.join().unwrap();
        }
    }

    pub fn par_for<F>(&self, begin: u64, end: u64, fn: F)
    where
        F: Fn(u64) + Sync,
    {
        self.par_for_inc(begin, end - begin, |i| i, fn);
    }

    pub fn par_for_skip<F>(
        &self,
        begin: usize,
        end: usize,
        skip_mask: usize,
        mask_width: usize,
        fn: F,
    ) where
        F: Fn(usize) + Send + Sync,
    {
        if (skip_mask << mask_width) >= end {
            self.par_for(begin, skip_mask, fn);
            return;
        }
        let low_mask = skip_mask - 1;
        let high_mask = !low_mask;
        let inc_fn = if low_mask == 0 {
            |i| i << mask_width
        } else {
            |i| (i & low_mask) | ((i & high_mask) << mask_width)
        };
        self.par_for_inc(begin, (end - begin) >> mask_width, inc_fn, fn);
    }

    pub fn par_for_mask<F>(
        &self,
        begin: usize,
        end: usize,
        mask_array: &[usize],
        fn: F,
    ) where
        F: Fn(usize) + Send + Sync,
    {
        let mask_len = mask_array.len();
        let mut masks = vec![[0usize; 2]; mask_len];
        let mut only_low = true;
        for i in 0..mask_len {
            masks[i][0] = mask_array[i] - 1;
            masks[i][1] = !(masks[i][0] + mask_array[i]);
            if mask_array[mask_len - i - 1] != (end >> (i + 1)) {
                only_low = false;
            }
        }
        if only_low {
            self.par_for(begin, end >> mask_len, fn);
        } else {
            self.par_for_inc(begin, (end - begin) >> mask_len, |i| {
                let mut i = i;
                for m in 0..mask_len {
                    i = ((i << 1) & masks[m][1]) | (i & masks[m][0]);
                }
                i
            }, fn);
        }
    }

    pub fn par_for_set<F>(&self, sparse_set: HashSet<u64>, fn: F)
    where
        F: Fn(u64) + Sync,
    {
        let sparse_set: Vec<usize> = sparse_set.iter().cloned().collect();
        self.par_for_inc(0, sparse_set.len(), |i| sparse_set[i], fn);
    }

    pub fn par_for_sparse_compose<F>(
        &self,
        low_set: Vec<u64>,
        high_set: Vec<u64>,
        high_start: u64,
        fn: F,
    )
    where
        F: Fn(u64) + Sync,
    {
        let low_size = low_set.len();
        self.par_for_inc(0, low_size * high_set.len(), |i| {
            let low_perm = i % low_size;
            let high_perm = (i - low_perm) / low_size;
            let perm = low_set[low_perm] | (high_set[high_perm] << high_start);
            perm
        }, fn);
    }

    pub fn par_norm<F>(
        &self,
        item_count: usize,
        state_array: &StateVectorPtr,
        norm_thresh: f64,
    ) -> f64 {
        if norm_thresh <= 0.0 {
            return self.par_norm_exact(item_count, state_array);
        }
        let stride = self.p_stride;
        let threads = if item_count / stride > self.num_cores {
            self.num_cores
        } else {
            item_count / stride
        };
        if threads <= 1 {
            let mut nrm_sqr = 0.0;
            let norm_thresh = norm_thresh as f64;
            for j in 0..item_count {
                let nrm = norm(state_array.read(j));
                if nrm >= norm_thresh {
                    nrm_sqr += nrm;
                }
            }
            return nrm_sqr;
        }
        let idx = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..threads {
            let idx = Arc::clone(&idx);
            let item_count = item_count;
            let state_array = state_array.clone();
            let stride = stride;
            let norm_thresh = norm_thresh as f64;
            let handle = thread::spawn(move || {
                let mut sqr_norm = 0.0;
                loop {
                    let i = idx.fetch_add(1, Ordering::SeqCst);
                    let l = i * stride;
                    if l >= item_count {
                        break;
                    }
                    let max_j = if l + stride < item_count {
                        stride
                    } else {
                        item_count - l
                    };
                    for j in 0..max_j {
                        let k = i * stride + j;
                        let nrm = norm(state_array.read(k));
                        if nrm >= norm_thresh {
                            sqr_norm += nrm;
                        }
                    }
                }
                sqr_norm
            });
            handles.push(handle);
        }
        let mut nrm_sqr = 0.0;
        for handle in handles {
            nrm_sqr += handle.join().unwrap();
        }
        nrm_sqr
    }

    pub fn par_norm_exact<F>(&self, item_count: usize, state_array: &StateVectorPtr) -> f64 {
        let stride = self.p_stride;
        let threads = if item_count / stride > self.num_cores {
            self.num_cores
        } else {
            item_count / stride
        };
        if threads <= 1 {
            let mut nrm_sqr = 0.0;
            for j in 0..item_count {
                nrm_sqr += norm(state_array.read(j));
            }
            return nrm_sqr;
        }
        let idx = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..threads {
            let idx = Arc::clone(&idx);
            let item_count = item_count;
            let stride = stride;
            let state_array = state_array.clone();
            let handle = thread::spawn(move || {
                let mut sqr_norm = 0.0;
                loop {
                    let i = idx.fetch_add(1, Ordering::SeqCst);
                    let l = i * stride;
                    if l >= item_count {
                        break;
                    }
                    let max_j = if l + stride < item_count {
                        stride
                    } else {
                        item_count - l
                    };
                    for j in 0..max_j {
                        sqr_norm += norm(state_array.read(i * stride + j));
                    }
                }
                sqr_norm
            });
            handles.push(handle);
        }
        let mut nrm_sqr = 0.0;
        for handle in handles {
            nrm_sqr += handle.join().unwrap();
        }
        nrm_sqr
    }
}
