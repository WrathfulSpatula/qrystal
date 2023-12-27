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

    fn select_bit(perm: usize, bit: usize) -> usize {
        (perm >> bit) & 1
    }

    fn remove_power(perm: usize, power: usize) -> usize {
        (perm & power) | ((perm >> 1) & !power)
    }

    fn new(
        eng: Vec<QInterfaceEngine>,
        q_bit_count: usize,
        init_state: usize,
        rgp: qrack_rand_gen_ptr,
        phase_fac: Complex,
        do_norm: bool,
        random_global_phase: bool,
        use_host_mem: bool,
        device_id: i64,
        use_hardware_rng: bool,
        use_sparse_state_vec: bool,
        norm_thresh: f64,
        dev_ids: Vec<i64>,
        qubit_threshold: usize,
        sep_thresh: f64,
    ) -> Self {
        QBdt {
            qubit_count: q_bit_count,
            root: None,
            shards: vec![None; q_bit_count],
        }
    }

    fn init(&mut self) {
        let bdt_stride = (self.get_stride() + 1) >> 1;
        if bdt_stride == 0 {
            bdt_stride = 1;
        }
        let mut engine_level = 0;
        if self.engines.is_empty() {
            self.engines.push(QINTERFACE_OPTIMAL_BASE);
        }
        let mut root_engine = self.engines[0];
        while engine_level < self.engines.len()
            && root_engine != QINTERFACE_CPU
            && root_engine != QINTERFACE_OPENCL
            && root_engine != QINTERFACE_HYBRID
        {
            engine_level += 1;
            root_engine = self.engines[engine_level];
        }
    }

    fn make_q_engine(&self, qb_count: usize, perm: usize) -> QEnginePtr {
        CreateQuantumInterface(
            self.engines.clone(),
            qb_count,
            perm,
            self.rand_generator,
            ONE_CMPLX,
            self.do_normalize,
            false,
            false,
            self.dev_id,
            self.hardware_rand_generator != NULL,
            false,
            self.amplitude_floor as f32,
            self.device_ids.clone(),
        )
    }

    fn par_for_qbdt(
        &self,
        end: usize,
        max_qubit: usize,
        fn: BdtFunc,
        branch: bool,
    ) {
        if branch {
            self.root.as_ref().unwrap().lock().unwrap().branch(max_qubit);
        }
        let stride = self.bdt_stride;
        let under_threads = (end / stride).min(1);
        let nm_crs = self.get_concurrency_level() / (under_threads + 1);
        let threads = (end / stride).min(nm_crs);
        if threads <= 1 {
            for j in 0..end {
                fn(j);
            }
            if branch {
                self.root.as_ref().unwrap().lock().unwrap().prune(max_qubit);
            }
            return;
        }
        let my_mutex = Arc::new(Mutex::new(()));
        let idx = Arc::new(Mutex::new(0));
        let mut futures = Vec::new();
        for _ in 0..threads {
            let my_mutex = my_mutex.clone();
            let idx = idx.clone();
            futures.push(thread::spawn(move || {
                loop {
                    let i;
                    {
                        let mut idx = idx.lock().unwrap();
                        i = *idx;
                        *idx += 1;
                    }
                    let l = i * stride;
                    if l >= end {
                        break;
                    }
                    let max_j = (l + stride).min(end) - l;
                    for j in 0..max_j {
                        let k = j + l;
                        fn(k);
                    }
                }
            }));
        }
        for future in futures {
            future.join().unwrap();
        }
    }

    fn _par_for(&self, end: usize, fn: ParallelFuncBdt) {
        let stride = self.bdt_stride;
        let nm_crs = self.get_concurrency_level();
        let threads = (end / stride).min(nm_crs);
        if threads <= 1 {
            for j in 0..end {
                fn(j, 0);
            }
            return;
        }
        let my_mutex = Arc::new(Mutex::new(()));
        let idx = Arc::new(Mutex::new(0));
        let mut futures = Vec::new();
        for _ in 0..threads {
            let my_mutex = my_mutex.clone();
            let idx = idx.clone();
            futures.push(thread::spawn(move || {
                loop {
                    let i;
                    {
                        let mut idx = idx.lock().unwrap();
                        i = *idx;
                        *idx += 1;
                    }
                    let l = i * stride;
                    if l >= end {
                        break;
                    }
                    let max_j = (l + stride).min(end) - l;
                    for j in 0..max_j {
                        fn(j + l, 0);
                    }
                }
            }));
        }
        for future in futures {
            future.join().unwrap();
        }
    }

    fn count_branches(&self) -> usize {
        let max_qubit_index = self.qubit_count - 1;
        let mut nodes = HashSet::new();
        nodes.insert(self.root.as_ref().unwrap().lock().unwrap());
        self.par_for_qbdt(
            self.max_q_power,
            max_qubit_index,
            |i| {
                let mut leaf = self.root.as_ref().unwrap().lock().unwrap();
                for j in 0..max_qubit_index {
                    if leaf.scale.norm() <= _qrack_qbdt_sep_thresh {
                        return (2usize.pow((max_qubit_index - j) as u32) - 1) as usize;
                    }
                    leaf = leaf.branches[SelectBit(i, max_qubit_index - (j + 1))].as_ref().unwrap().lock().unwrap();
                    nodes.insert(leaf);
                }
                0
            },
            false,
        );
        nodes.len()
    }

    fn apply_single(&mut self, mtrx: &[Complex], target: usize) {
        if target >= self.qubit_count {
            panic!("QBdt::ApplySingle target parameter must be within allocated qubit bounds!");
        }
        if mtrx[1].norm() == 0.0
            && mtrx[2].norm() == 0.0
            && (mtrx[0] - mtrx[3]).norm() == 0.0
            && (self.rand_global_phase || (ONE_CMPLX - mtrx[0]).norm() == 0.0)
        {
            return;
        }
        let q_power = 2usize.pow(target as u32);
        self.par_for_qbdt(q_power, target, |i| {
            let mut leaf = self.root.as_ref().unwrap().lock().unwrap();
            for j in 0..target {
                if leaf.scale.norm() <= _qrack_qbdt_sep_thresh {
                    return;
                }
                leaf = leaf.branches[SelectBit(i, j)].as_ref().unwrap().lock().unwrap();
            }
            if leaf.scale.norm() <= _qrack_qbdt_sep_thresh {
                return;
            }
            leaf.apply_2x2(mtrx, self.qubit_count - target);
        }, false);
    }

    fn apply_controlled_single(&self, mtrx: &[f64], controls: &[usize], target: usize, is_anti: bool) {
        if target >= self.qubit_count {
            panic!("QBdt::ApplyControlledSingle target parameter must be within allocated qubit bounds!");
        }
        self.throw_if_qb_id_array_is_bad(controls);
        let is_phase = mtrx[1] == 0.0 && mtrx[2] == 0.0 && (is_anti && mtrx[3] == 0.0 || !is_anti && mtrx[0] == 0.0);
        if is_phase && mtrx[0] == 0.0 && mtrx[3] == 0.0 {
            return;
        }
        let mut controls = controls.to_vec();
        controls.sort();
        if target < *controls.last().unwrap() {
            let mut target = target;
            let mut controls_last = *controls.last().unwrap();
            if !is_phase {
                std::mem::swap(&mut target, &mut controls_last);
                self.swap(target, controls_last);
                self.apply_controlled_single(mtrx, &controls, target, is_anti);
                self.swap(target, controls_last);
                return;
            }
        }
        let q_power = 2usize.pow(target as u32);
        let mut control_mask = 0;
        for &control in controls {
            control_mask |= 2usize.pow(target as u32 - (control + 1) as u32);
        }
        let control_perm = if is_anti { 0 } else { control_mask };
        let mtrx_col1 = (mtrx[0], mtrx[2]);
        let mtrx_col2 = (mtrx[1], mtrx[3]);
        let mtrx_col1_shuff = mtrx_col_shuff(mtrx_col1);
        let mtrx_col2_shuff = mtrx_col_shuff(mtrx_col2);
        (0..q_power).into_par_iter().for_each(|i| {
            if (i & control_mask) != control_perm {
                return control_mask - 1;
            }
            let mut leaf = &self.root;
            for j in 0..target {
                if leaf.scale <= _qrack_qbdt_sep_thresh {
                    return 2usize.pow(target - j) - 1;
                }
                leaf = &leaf.branches[select_bit(i, target - (j + 1))];
            }
            let _lock = leaf.mtx.lock().unwrap();
            if leaf.scale <= _qrack_qbdt_sep_thresh {
                return 0;
            }
            leaf.apply_2x2(mtrx_col1, mtrx_col2, mtrx_col1_shuff, mtrx_col2_shuff, self.qubit_count - target);
            return 0;
        });
    }
}

impl QInterface for QBdt {
    fn set_permutation(&mut self, init_state: usize, phase_fac: Complex) {
        self.dump_buffers();
        if self.qubit_count == 0 {
            return;
        }
        if phase_fac == CMPLX_DEFAULT_ARG {
            if self.rand_global_phase {
                let angle = self.rand() * 2.0 * PI;
                phase_fac = Complex::new(angle.cos(), angle.sin());
            } else {
                phase_fac = ONE_CMPLX;
            }
        }
        self.root = Some(Arc::new(Mutex::new(QBdtNodeInterface::new(phase_fac))));
        let mut leaf = self.root.as_ref().unwrap().lock().unwrap();
        for qubit in 0..self.qubit_count {
            let bit = SelectBit(init_state, qubit);
            leaf.branches[bit] = Some(Arc::new(Mutex::new(QBdtNodeInterface::new(ONE_CMPLX))));
            leaf.branches[bit ^ 1] = Some(Arc::new(Mutex::new(QBdtNodeInterface::new(ZERO_CMPLX))));
            leaf = leaf.branches[bit].as_ref().unwrap().lock().unwrap();
        }
    }

    fn clone(&self) -> Self {
        let mut c = QBdt {
            engines: self.engines.clone(),
            qubit_count: 0,
            rand_generator: self.rand_generator,
            do_normalize: self.do_normalize,
            rand_global_phase: self.rand_global_phase,
            hardware_rand_generator: self.hardware_rand_generator,
            amplitude_floor: self.amplitude_floor,
            root: None,
            shards: Vec::new(),
        };
        c.root = match &self.root {
            Some(root) => Some(Arc::new(Mutex::new(root.lock().unwrap().shallow_clone()))),
            None => None,
        };
        c.shards.resize(self.shards.len(), None);
        c.set_qubit_count(self.qubit_count);
        for i in 0..self.shards.len() {
            if let Some(shard) = &self.shards[i] {
                c.shards[i] = Some(shard.clone());
            }
        }
        c
    }

    fn sum_sqr_diff(&self, to_compare: &QBdt) -> f64 {
        if self as *const QBdt == to_compare as *const QBdt {
            return 0.0;
        }
        if self.qubit_count != to_compare.qubit_count {
            return 1.0;
        }
        self.flush_buffers();
        to_compare.flush_buffers();
        let num_cores = self.get_concurrency_level();
        let projection_buff = vec![Complex::new(0.0, 0.0); num_cores];
        let l_phase_arg = self.first_nonzero_phase();
        let r_phase_arg = to_compare.first_nonzero_phase();
        self.root.as_ref().unwrap().lock().unwrap().scale *= Complex::new(
            (r_phase_arg - l_phase_arg).cos(),
            (r_phase_arg - l_phase_arg).sin(),
        );
        self._par_for(self.max_q_power, |i, _| {
            let mut leaf = self.root.as_ref().unwrap().lock().unwrap();
            let mut scale = leaf.scale;
            for j in 0..self.qubit_count {
                if leaf.scale.norm() <= _qrack_qbdt_sep_thresh {
                    return;
                }
                leaf = leaf.branches[SelectBit(i, j)].as_ref().unwrap().lock().unwrap();
                scale *= leaf.scale;
            }
            projection_buff[0] += scale.conj() * to_compare.get_amplitude(i);
        });
        let mut projection = Complex::new(0.0, 0.0);
        for i in 0..num_cores {
            projection += projection_buff[i];
        }
        1.0 - clamp_prob(projection.norm())
    }

    fn get_amplitude(&self, perm: usize) -> Complex {
        if perm >= self.max_q_power {
            panic!("QBdt::GetAmplitude argument out-of-bounds!");
        }
        self.flush_buffers();
        let mut leaf = self.root.as_ref().unwrap().lock().unwrap();
        let mut scale = leaf.scale;
        for j in 0..self.qubit_count {
            if leaf.scale.norm() <= _qrack_qbdt_sep_thresh {
                break;
            }
            leaf = leaf.branches[SelectBit(perm, j)].as_ref().unwrap().lock().unwrap();
            scale *= leaf.scale;
        }
        scale
    }

    fn compose(&mut self, to_copy: &QBdt, start: usize) -> usize {
        if start > self.qubit_count {
            panic!("QBdt::Compose start index is out-of-bounds!");
        }
        if to_copy.qubit_count == 0 {
            return start;
        }
        self.root.as_ref().unwrap().lock().unwrap().insert_at_depth(
            to_copy.root.as_ref().unwrap().lock().unwrap().shallow_clone(),
            start,
            to_copy.qubit_count,
        );
        self.shards.splice(
            start..start,
            to_copy.shards.iter().cloned(),
        );
        for i in 0..to_copy.qubit_count {
            if let Some(shard) = &to_copy.shards[i] {
                self.shards[start + i] = Some(shard.clone());
            }
        }
        self.set_qubit_count(self.qubit_count + to_copy.qubit_count);
        start
    }

    fn decompose(&self, start: usize, length: usize) -> QBdtPtr {
        let mut dest = QBdt::new(
            self.engines.clone(),
            length,
            0,
            self.rand_generator,
            ONE_CMPLX,
            self.do_normalize,
            self.rand_global_phase,
            false,
            -1,
            self.hardware_rand_generator != NULL,
            false,
            self.amplitude_floor,
        );
        self.decompose_dispose(start, length, Some(&mut dest));
        dest
    }

    fn decompose_dispose(&mut self, start: usize, length: usize, dest: Option<&mut QBdt>) {
        if is_bad_bit_range(start, length, self.qubit_count) {
            panic!("QBdt::DecomposeDispose range is out-of-bounds!");
        }
        if length == 0 {
            return;
        }
        if let Some(dest) = dest {
            dest.root = Some(Arc::new(Mutex::new(
                self.root.as_ref().unwrap().lock().unwrap().remove_separable_at_depth(start, length).shallow_clone(),
            )));
            dest.shards.splice(
                0..0,
                self.shards[start..start + length].iter().cloned(),
            );
        } else {
            self.root.as_ref().unwrap().lock().unwrap().remove_separable_at_depth(start, length);
        }
        self.shards.splice(start..start + length, vec![None; length]);
        self.set_qubit_count(self.qubit_count - length);
        self.root.as_ref().unwrap().lock().unwrap().prune(self.qubit_count);
    }

    fn allocate(&mut self, start: usize, length: usize) -> usize {
        if length == 0 {
            return start;
        }
        let mut n_qubits = QBdt::new(
            self.engines.clone(),
            length,
            0,
            self.rand_generator,
            ONE_CMPLX,
            self.do_normalize,
            self.rand_global_phase,
            false,
            -1,
            self.hardware_rand_generator != NULL,
            false,
            self.amplitude_floor,
        );
        n_qubits.root.as_ref().unwrap().lock().unwrap().insert_at_depth(
            self.root.as_ref().unwrap().lock().unwrap().shallow_clone(),
            length,
            self.qubit_count,
        );
        self.root = n_qubits.root;
        self.shards.splice(
            start..start,
            n_qubits.shards.iter().cloned(),
        );
        self.set_qubit_count(self.qubit_count + length);
        self.ror(length, 0, start + length);
        start
    }

    fn prob(&mut self, qubit: usize) -> f64 {
        if qubit >= self.qubit_count {
            panic!("QBdt::Prob qubit index parameter must be within allocated qubit bounds!");
        }
        let shard = self.shards[qubit];
        if let Some(shard) = shard {
            if !shard.is_phase() {
                self.shards[qubit] = None;
                self.apply_single(shard.gate, qubit);
            }
        }
        let q_power = 2usize.pow(qubit as u32);
        let num_cores = self.get_concurrency_level();
        let mut qi_probs = HashMap::new();
        let mut one_chance_buff = vec![0.0; num_cores];
        self._par_for(q_power, |i, _| {
            let mut leaf = self.root.as_ref().unwrap().lock().unwrap();
            let mut scale = leaf.scale;
            for j in 0..qubit {
                if leaf.scale.norm() <= _qrack_qbdt_sep_thresh {
                    return;
                }
                leaf = leaf.branches[SelectBit(i, j)].as_ref().unwrap().lock().unwrap();
                scale *= leaf.scale;
            }
            if leaf.scale.norm() <= _qrack_qbdt_sep_thresh {
                return;
            }
            one_chance_buff[0] += (scale * leaf.branches[1].as_ref().unwrap().lock().unwrap().scale.conj()).norm();
        });
        let mut one_chance = 0.0;
        for i in 0..num_cores {
            one_chance += one_chance_buff[i];
        }
        clamp_prob(one_chance)
    }

    fn prob_all(&self, perm: usize) -> f64 {
        self.flush_buffers();
        let mut leaf = self.root.as_ref().unwrap().lock().unwrap();
        let mut scale = leaf.scale;
        for j in 0..self.qubit_count {
            if leaf.scale.norm() <= _qrack_qbdt_sep_thresh {
                break;
            }
            leaf = leaf.branches[SelectBit(perm, j)].as_ref().unwrap().lock().unwrap();
            scale *= leaf.scale;
        }
        clamp_prob(scale.norm())
    }

    fn force_m(&mut self, qubit: usize, result: bool, do_force: bool, do_apply: bool) -> bool {
        if qubit >= self.qubit_count {
            panic!("QBdt::Prob qubit index parameter must be within allocated qubit bounds!");
        }
        let one_chance = self.prob(qubit);
        let result = if one_chance >= 1.0 {
            true
        } else if one_chance <= 0.0 {
            false
        } else if !do_force {
            self.rand() <= one_chance
        } else {
            result
        };
        if !do_apply {
            return result;
        }
        self.shards[qubit] = None;
        let q_power = 2usize.pow(qubit as u32);
        self.root.as_ref().unwrap().lock().unwrap().scale = self.get_nonunitary_phase();
        self._par_for(q_power, |i, _| {
            let mut leaf = self.root.as_ref().unwrap().lock().unwrap();
            for j in 0..qubit {
                if leaf.scale.norm() <= _qrack_qbdt_sep_thresh {
                    return;
                }
                leaf.branch(j);
                leaf = leaf.branches[SelectBit(i, j)].as_ref().unwrap().lock().unwrap();
            }
            if leaf.scale.norm() <= _qrack_qbdt_sep_thresh {
                return;
            }
            leaf.branch(qubit);
            let b0 = leaf.branches[0].as_ref().unwrap();
            let b1 = leaf.branches[1].as_ref().unwrap();
            if result {
                if b1.lock().unwrap().scale.norm() <= _qrack_qbdt_sep_thresh {
                    b1.lock().unwrap().set_zero();
                    return;
                }
                b0.lock().unwrap().set_zero();
                b1.lock().unwrap().scale /= b1.lock().unwrap().scale.norm();
            } else {
                if b0.lock().unwrap().scale.norm() <= _qrack_qbdt_sep_thresh {
                    b0.lock().unwrap().set_zero();
                    return;
                }
                b0.lock().unwrap().scale /= b0.lock().unwrap().scale.norm();
                b1.lock().unwrap().set_zero();
            }
        });
        self.root.as_ref().unwrap().lock().unwrap().prune(qubit);
        result
    }

    fn m_all(&mut self) -> usize {
        let mut result = 0;
        let mut leaf = self.root.as_ref().unwrap().lock().unwrap();
        for i in 0..self.qubit_count {
            let shard = self.shards[i];
            if let Some(shard) = shard {
                if !shard.is_phase() {
                    self.apply_single(shard.gate, i);
                }
            }
            self.shards[i] = None;
        }
        for i in 0..self.qubit_count {
            let one_chance = clamp_prob(leaf.branches[1].as_ref().unwrap().lock().unwrap().scale.norm());
            let bit_result = if one_chance >= 1.0 {
                true
            } else if one_chance <= 0.0 {
                false
            } else {
                self.rand() <= one_chance
            };
            leaf.branch(i);
            if bit_result {
                leaf.branches[0].as_ref().unwrap().lock().unwrap().set_zero();
                leaf.branches[1].as_ref().unwrap().lock().unwrap().scale = ONE_CMPLX;
                leaf = leaf.branches[1].as_ref().unwrap().lock().unwrap();
                result |= 1 << i;
            } else {
                leaf.branches[0].as_ref().unwrap().lock().unwrap().scale = ONE_CMPLX;
                leaf.branches[1].as_ref().unwrap().lock().unwrap().set_zero();
                leaf = leaf.branches[0].as_ref().unwrap().lock().unwrap();
            }
        }
        result
    }

    fn mtrx(&mut self, mtrx: &[f64], target: usize) {
        let shard = &mut self.shards[target];
        if let Some(shard) = shard {
            shard.compose(mtrx);
        } else {
            *shard = Some(Box::new(MpsShard::new(mtrx)));
        }
    }

    fn mc_mtrx(&mut self, controls: &[usize], mtrx: &[f64], target: usize) {
        if controls.is_empty() {
            self.mtrx(mtrx, target);
        } else if mtrx[1] == 0.0 && mtrx[2] == 0.0 {
            self.mc_phase(controls, mtrx[0], mtrx[3], target);
        } else if mtrx[0] == 0.0 && mtrx[3] == 0.0 {
            self.mc_invert(controls, mtrx[1], mtrx[2], target);
        } else {
            self.flush_non_phase_buffers();
            self.flush_if_blocked(target, controls);
            self.apply_controlled_single(mtrx, controls, target, false);
        }
    }

    fn mac_mtrx(&mut self, controls: &[usize], mtrx: &[f64], target: usize) {
        if controls.is_empty() {
            self.mtrx(mtrx, target);
        } else if mtrx[1] == 0.0 && mtrx[2] == 0.0 {
            self.mac_phase(controls, mtrx[0], mtrx[3], target);
        } else if mtrx[0] == 0.0 && mtrx[3] == 0.0 {
            self.mac_invert(controls, mtrx[1], mtrx[2], target);
        } else {
            self.flush_non_phase_buffers();
            self.flush_if_blocked(target, controls);
            self.apply_controlled_single(mtrx, controls, target, true);
        }
    }

    fn mc_phase(&mut self, controls: &[usize], top_left: f64, bottom_right: f64, target: usize) {
        if controls.is_empty() {
            self.phase(top_left, bottom_right, target);
            return;
        }
        let mtrx = [top_left, 0.0, 0.0, bottom_right];
        if top_left != 1.0 {
            self.flush_non_phase_buffers();
            self.apply_controlled_single(&mtrx, controls, target, false);
            return;
        }
        if bottom_right == 1.0 {
            return;
        }
        let mut l_controls = controls.to_vec();
        l_controls.push(target);
        l_controls.sort();
        let target = *l_controls.last().unwrap();
        l_controls.pop();
        self.flush_non_phase_buffers();
        self.apply_controlled_single(&mtrx, &l_controls, target, false);
    }

    fn mc_invert(&mut self, controls: &[usize], top_right: f64, bottom_left: f64, target: usize) {
        if controls.is_empty() {
            self.invert(top_right, bottom_left, target);
            return;
        }
        let mtrx = [0.0, top_right, bottom_left, 0.0];
        if top_right != 1.0 || bottom_left != 1.0 {
            self.flush_non_phase_buffers();
            self.flush_if_blocked(target, controls);
            self.apply_controlled_single(&mtrx, controls, target, false);
            return;
        }
        let mut l_controls = controls.to_vec();
        l_controls.sort();
        if *l_controls.last().unwrap() < target {
            self.flush_non_phase_buffers();
            self.flush_if_blocked(target, &l_controls);
            self.apply_controlled_single(&mtrx, &l_controls, target, false);
            return;
        }
        self.h(target);
        self.mc_phase(&l_controls, 1.0, -1.0, target);
        self.h(target);
    }

    fn f_sim(&mut self, theta: f64, phi: f64, qubit1: usize, qubit2: usize) {
        if qubit1 == qubit2 {
            return;
        }
        let controls = vec![qubit1];
        let sin_theta = f64::sin(theta);
        if sin_theta * sin_theta <= FP_NORM_EPSILON {
            self.mc_phase(&controls, 1.0, f64::exp(complex(0.0, phi)), qubit2);
            return;
        }
        let exp_i_phi = f64::exp(complex(0.0, phi));
        let sin_theta_diff_neg = 1.0 + sin_theta;
        if sin_theta_diff_neg * sin_theta_diff_neg <= FP_NORM_EPSILON {
            self.i_swap(qubit1, qubit2);
            self.mc_phase(&controls, 1.0, exp_i_phi, qubit2);
            return;
        }
        let sin_theta_diff_pos = 1.0 - sin_theta;
        if sin_theta_diff_pos * sin_theta_diff_pos <= FP_NORM_EPSILON {
            self.ii_swap(qubit1, qubit2);
            self.mc_phase(&controls, 1.0, exp_i_phi, qubit2);
            return;
        }
        self.execute_as_state_vector(|eng| {
            eng.f_sim(theta, phi, qubit1, qubit2);
        });
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

