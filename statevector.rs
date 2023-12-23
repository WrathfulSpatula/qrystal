use std::collections::HashMap;
use std::sync::Mutex;

type bitCapIntOcl = u64;
type complex = (f64, f64);
type complex2 = (complex, complex);
type real1 = f64;
const ZERO_CMPLX: complex = (0.0, 0.0);
const REAL1_EPSILON: f64 = 0.000001;
const QRACK_ALIGN_SIZE: usize = 16;

trait ParallelFor {
    fn par_for<F>(&self, start: bitCapIntOcl, end: bitCapIntOcl, f: F)
    where
        F: Fn(bitCapIntOcl, usize) + Sync + Send;
}

struct StateVectorArray {
    capacity: bitCapIntOcl,
    is_read_locked: bool,
    amplitudes: Vec<complex>,
}

impl StateVectorArray {
    fn new(cap: bitCapIntOcl) -> Self {
        Self {
            capacity: cap,
            is_read_locked: true,
            amplitudes: vec![(0.0, 0.0); cap as usize],
        }
    }

    fn read(&self, i: bitCapIntOcl) -> complex {
        self.amplitudes[i as usize]
    }

    fn write(&mut self, i: bitCapIntOcl, c: complex) {
        self.amplitudes[i as usize] = c;
    }

    fn clear(&mut self) {
        self.amplitudes = vec![(0.0, 0.0); self.capacity as usize];
    }

    fn copy_in(&mut self, copy_in: Option<&[complex]>) {
        match copy_in {
            Some(arr) => self.amplitudes.copy_from_slice(arr),
            None => self.clear(),
        }
    }

    fn copy_out(&self, copy_out: &mut [complex]) {
        copy_out.copy_from_slice(&self.amplitudes);
    }

    fn copy(&mut self, to_copy: &Self) {
        self.amplitudes.copy_from_slice(&to_copy.amplitudes);
    }

    fn shuffle(&mut self, svp: &mut StateVectorArray) {
        let half_cap = self.capacity / 2;
        self.amplitudes[half_cap as usize..].swap_with_slice(&mut svp.amplitudes);
    }

    fn get_probs(&self, out_array: &mut [real1]) {
        for (i, c) in self.amplitudes.iter().enumerate() {
            out_array[i] = c.0 * c.0 + c.1 * c.1;
        }
    }

    fn is_sparse(&self) -> bool {
        false
    }
}

struct StateVectorSparse {
    capacity: bitCapIntOcl,
    is_read_locked: bool,
    amplitudes: Mutex<HashMap<bitCapIntOcl, complex>>,
}

impl StateVectorSparse {
    fn new(cap: bitCapIntOcl) -> Self {
        Self {
            capacity: cap,
            is_read_locked: true,
            amplitudes: Mutex::new(HashMap::new()),
        }
    }

    fn read(&self, i: bitCapIntOcl) -> complex {
        let amplitudes = self.amplitudes.lock().unwrap();
        *amplitudes.get(&i).unwrap_or(&(0.0, 0.0))
    }

    fn write(&self, i: bitCapIntOcl, c: complex) {
        let mut amplitudes = self.amplitudes.lock().unwrap();
        if c.0.abs() > REAL1_EPSILON || c.1.abs() > REAL1_EPSILON {
            amplitudes.insert(i, c);
        } else {
            amplitudes.remove(&i);
        }
    }

    fn write2(&self, i1: bitCapIntOcl, c1: complex, i2: bitCapIntOcl, c2: complex) {
        let mut amplitudes = self.amplitudes.lock().unwrap();
        if c1.0.abs() > REAL1_EPSILON || c1.1.abs() > REAL1_EPSILON {
            amplitudes.insert(i1, c1);
        } else {
            amplitudes.remove(&i1);
        }
        if c2.0.abs() > REAL1_EPSILON || c2.1.abs() > REAL1_EPSILON {
            amplitudes.insert(i2, c2);
        } else {
            amplitudes.remove(&i2);
        }
    }

    fn clear(&self) {
        let mut amplitudes = self.amplitudes.lock().unwrap();
        amplitudes.clear();
    }

    fn copy_in(&self, copy_in: Option<&[complex]>) {
        let mut amplitudes = self.amplitudes.lock().unwrap();
        match copy_in {
            Some(arr) => {
                for (i, c) in arr.iter().enumerate() {
                    if c.0.abs() > REAL1_EPSILON || c.1.abs() > REAL1_EPSILON {
                        amplitudes.insert(i as bitCapIntOcl, *c);
                    } else {
                        amplitudes.remove(&(i as bitCapIntOcl));
                    }
                }
            }
            None => {
                amplitudes.clear();
            }
        }
    }

    fn copy_out(&self, copy_out: &mut [complex]) {
        let amplitudes = self.amplitudes.lock().unwrap();
        for (i, c) in amplitudes.iter() {
            copy_out[*i as usize] = *c;
        }
    }

    fn copy(&self, to_copy: &Self) {
        let mut amplitudes = self.amplitudes.lock().unwrap();
        let to_copy_amplitudes = to_copy.amplitudes.lock().unwrap();
        *amplitudes = to_copy_amplitudes.clone();
    }

    fn shuffle(&self, svp: &mut StateVectorSparse) {
        let half_cap = self.capacity / 2;
        let mut amplitudes = self.amplitudes.lock().unwrap();
        let mut svp_amplitudes = svp.amplitudes.lock().unwrap();
        for i in 0..half_cap {
            let amp = amplitudes.remove(&i).unwrap_or((0.0, 0.0));
            svp_amplitudes.insert(i, amp);
        }
    }

    fn get_probs(&self, out_array: &mut [real1]) {
        let amplitudes = self.amplitudes.lock().unwrap();
        for (i, c) in amplitudes.iter() {
            out_array[*i as usize] = c.0 * c.0 + c.1 * c.1;
        }
    }

    fn is_sparse(&self) -> bool {
        let amplitudes = self.amplitudes.lock().unwrap();
        amplitudes.len() < self.capacity as usize / 2
    }
}

trait StateVector {
    fn read(&self, i: bitCapIntOcl) -> complex;
    fn write(&self, i: bitCapIntOcl, c: complex);
    fn clear(&self);
    fn copy_in(&self, copy_in: Option<&[complex]>);
    fn copy_out(&self, copy_out: &mut [complex]);
    fn copy(&self, to_copy: &Self);
    fn shuffle(&self, svp: &mut StateVectorSparse);
    fn get_probs(&self, out_array: &mut [real1]);
    fn is_sparse(&self) -> bool;
}

impl StateVector for StateVectorArray {
    fn read(&self, i: bitCapIntOcl) -> complex {
        StateVectorArray::read(self, i)
    }

    fn write(&self, i: bitCapIntOcl, c: complex) {
        StateVectorArray::write(self, i, c);
    }

    fn clear(&self) {
        StateVectorArray::clear(self);
    }

    fn copy_in(&self, copy_in: Option<&[complex]>) {
        StateVectorArray::copy_in(self, copy_in);
    }

    fn copy_out(&self, copy_out: &mut [complex]) {
        StateVectorArray::copy_out(self, copy_out);
    }

    fn copy(&self, to_copy: &Self) {
        StateVectorArray::copy(self, to_copy);
    }

    fn shuffle(&self, svp: &mut StateVectorSparse) {
        StateVectorArray::shuffle(self, svp);
    }

    fn get_probs(&self, out_array: &mut [real1]) {
        StateVectorArray::get_probs(self, out_array);
    }

    fn is_sparse(&self) -> bool {
        StateVectorArray::is_sparse(self)
    }
}

impl StateVector for StateVectorSparse {
    fn read(&self, i: bitCapIntOcl) -> complex {
        StateVectorSparse::read(self, i)
    }

    fn write(&self, i: bitCapIntOcl, c: complex) {
        StateVectorSparse::write(self, i, c);
    }

    fn clear(&self) {
        StateVectorSparse::clear(self);
    }

    fn copy_in(&self, copy_in: Option<&[complex]>) {
        StateVectorSparse::copy_in(self, copy_in);
    }

    fn copy_out(&self, copy_out: &mut [complex]) {
        StateVectorSparse::copy_out(self, copy_out);
    }

    fn copy(&self, to_copy: &Self) {
        StateVectorSparse::copy(self, to_copy);
    }

    fn shuffle(&self, svp: &mut StateVectorSparse) {
        StateVectorSparse::shuffle(self, svp);
    }

    fn get_probs(&self, out_array: &mut [real1]) {
        StateVectorSparse::get_probs(self, out_array);
    }

    fn is_sparse(&self) -> bool {
        StateVectorSparse::is_sparse(self)
    }
}


