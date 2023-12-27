use std::cmp::Ordering;
use std::f64::EPSILON;
use std::mem;
use std::ptr;
use std::sync::Arc;

mod QEngine;
mod QInterface;
mod QAlu;
mod QParity;

pub struct QBdt {
    bdt_stride: usize,
    dev_id: i64,
    root: Arc<dyn QBdtNodeInterface>,
    bdt_max_q_power: usize,
    device_ids: Vec<i64>,
    engines: Vec<QInterfaceEngine>,
    shards: Vec<Arc<MpsShard>>,
}

impl QBdt {
    pub fn new(
        eng: Vec<QInterfaceEngine>,
        q_bit_count: usize,
        init_state: usize,
        rgp: Option<qrack_rand_gen_ptr>,
        phase_fac: Complex,
        do_norm: bool,
        random_global_phase: bool,
        use_host_mem: bool,
        device_id: i64,
        use_hardware_rng: bool,
        use_sparse_state_vec: bool,
        norm_thresh: f64,
        ignored: Vec<i64>,
        qubit_threshold: usize,
        separation_thresh: f64,
    ) -> Self {
        Self {
            bdt_stride: 0,
            dev_id: device_id,
            root: Arc::new(QBdtNode::new()),
            bdt_max_q_power: 0,
            device_ids: vec![],
            engines: vec![],
            shards: vec![],
        }
    }

    fn dump_buffers(&mut self) {
        for shard in &mut self.shards {
            *shard = Arc::new(MpsShard::default());
        }
    }

    fn flush_buffer(&mut self, t: usize) {
        let shard = self.shards[t].clone();
        if let Some(shard) = shard {
            self.shards[t] = Arc::new(MpsShard::default());
            self.apply_single(shard.gate, t);
        }
    }

    fn flush_buffers(&mut self) {
        for i in 0..self.shards.len() {
            self.flush_buffer(i);
        }
    }

    fn flush_if_blocked(&mut self, target: usize, controls: &[usize]) {
        self.flush_if_blocked(controls);
        self.flush_buffer(target);
    }

    fn flush_if_blocked(&mut self, controls: &[usize]) {
        for &control in controls {
            let shard = self.shards[control].clone();
            if let Some(shard) = shard {
                self.shards[control] = Arc::new(MpsShard::default());
                self.apply_single(shard.gate, control);
            }
        }
    }

    fn flush_non_phase_buffers(&mut self) {
        for i in 0..self.shards.len() {
            let shard = self.shards[i].clone();
            if let Some(shard) = shard {
                self.shards[i] = Arc::new(MpsShard::default());
                self.apply_single(shard.gate, i);
            }
        }
    }

    fn make_q_engine(&mut self, qb_count: usize, perm: usize) -> QEnginePtr {
        unimplemented!()
    }

    fn get_traversal<F>(&mut self, get_lambda: F)
    where
        F: Fn(usize, Complex),
    {
        self.flush_buffers();
        for i in 0..self.bdt_max_q_power {
            let mut leaf = self.root.clone();
            let mut scale = leaf.get_scale();
            for j in 0..self.qubit_count {
                if norm(leaf.get_scale()) <= _qrack_qbdt_sep_thresh {
                    break;
                }
                leaf = leaf.branches[select_bit(i, j)].clone();
                scale *= leaf.get_scale();
            }
            get_lambda(i, scale);
        }
    }

    fn set_traversal<F>(&mut self, set_lambda: F)
    where
        F: Fn(usize, Arc<dyn QBdtNodeInterface>),
    {
        self.dump_buffers();
        self.root = Arc::new(QBdtNode::new());
        self.root.branch(self.qubit_count);
        for i in 0..self.bdt_max_q_power {
            let mut prev_leaf = self.root.clone();
            let mut leaf = self.root.clone();
            for j in 0..self.qubit_count {
                prev_leaf = leaf.clone();
                leaf = leaf.branches[select_bit(i, j)].clone();
            }
            set_lambda(i, leaf);
        }
        self.root.pop_state_vector(self.qubit_count);
        self.root.prune(self.qubit_count);
    }

    fn execute_as_state_vector<F>(&mut self, operation: F)
    where
        F: Fn(&mut dyn QInterface),
    {
        let mut q_reg = self.make_q_engine(self.qubit_count);
        self.get_quantum_state(&mut q_reg);
        operation(&mut q_reg);
        self.set_quantum_state(&mut q_reg);
    }

    fn bit_cap_int_as_state_vector<F>(&mut self, operation: F) -> usize
    where
        F: Fn(&mut dyn QInterface) -> usize,
    {
        let mut q_reg = self.make_q_engine(self.qubit_count);
        self.get_quantum_state(&mut q_reg);
        let to_ret = operation(&mut q_reg);
        self.set_quantum_state(&mut q_reg);
        to_ret
    }

    fn par_for_qbdt(&mut self, end: usize, max_qubit: usize, fn: BdtFunc, branch: bool) {
        unimplemented!()
    }

    fn _par_for(&mut self, end: usize, fn: ParallelFuncBdt) {
        unimplemented!()
    }

    fn decompose_dispose(&mut self, start: usize, length: usize, dest: &mut QBdt) {
        unimplemented!()
    }

    fn apply_controlled_single(&mut self, mtrx: &[Complex], controls: &[usize], target: usize, is_anti: bool) {
        unimplemented!()
    }

    fn select_bit(perm: usize, bit: usize) -> usize {
        (perm >> bit) & 1
    }

    fn remove_power(perm: usize, power: usize) -> usize {
        (perm & power) | ((perm >> 1) & !power)
    }

    fn apply_single(&mut self, mtrx: &[Complex], target: usize) {
        unimplemented!()
    }

    fn init(&mut self) {
        unimplemented!()
    }
}

trait QBdtNodeInterface {
    fn get_scale(&self) -> Complex;
    fn branch(&mut self, qubit_count: usize);
    fn pop_state_vector(&mut self, qubit_count: usize);
    fn prune(&mut self, qubit_count: usize);
}

struct QBdtNode {
    scale: Complex,
    branches: Vec<Arc<dyn QBdtNodeInterface>>,
}

impl QBdtNode {
    fn new() -> Self {
        Self {
            scale: Complex::default(),
            branches: vec![],
        }
    }
}

impl QBdtNodeInterface for QBdtNode {
    fn get_scale(&self) -> Complex {
        self.scale
    }

    fn branch(&mut self, qubit_count: usize) {
        self.branches = vec![Arc::new(QBdtNode::new()); qubit_count];
    }

    fn pop_state_vector(&mut self, qubit_count: usize) {
        unimplemented!()
    }

    fn prune(&mut self, qubit_count: usize) {
        unimplemented!()
    }
}

