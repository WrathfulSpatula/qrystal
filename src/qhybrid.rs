use std::sync::Arc;
use std::vec::Vec;

pub struct QHybrid {
    is_gpu: bool,
    is_pager: bool,
    use_rdrand: bool,
    is_sparse: bool,
    gpu_threshold_qubits: u64,
    pager_threshold_qubits: u64,
    separability_threshold: f64,
    dev_id: i64,
    engine: Arc<dyn QEngine>,
    phase_factor: Complex,
    device_ids: Vec<i64>,
}

impl QHybrid {
    pub fn new(
        q_bit_count: u64,
        init_state: u64,
        rgp: Option<qrack_rand_gen_ptr>,
        phase_fac: Complex,
        do_norm: bool,
        random_global_phase: bool,
        use_host_mem: bool,
        device_id: i64,
        use_hardware_rng: bool,
        use_sparse_state_vec: bool,
        norm_thresh: f64,
        dev_list: Vec<i64>,
        qubit_threshold: u64,
        ignored2: f32,
    ) -> Self {
        let engine = Self::make_engine(true);
        Self {
            is_gpu: true,
            is_pager: false,
            use_rdrand: false,
            is_sparse: false,
            gpu_threshold_qubits: 0,
            pager_threshold_qubits: 0,
            separability_threshold: 0.0,
            dev_id: -1,
            engine,
            phase_factor: Complex::default(),
            device_ids: Vec::new(),
        }
    }

    pub fn set_qubit_count(&mut self, qb: u64) {
        let is_higher = qb > self.qubit_count;
        if is_higher {
            self.switch_modes(qb >= self.gpu_threshold_qubits, qb > self.pager_threshold_qubits);
        }
        self.qubit_count = qb;
        if !is_higher {
            self.switch_modes(qb >= self.gpu_threshold_qubits, qb > self.pager_threshold_qubits);
        }
        if self.engine.is_zero_amplitude() {
            self.engine.set_qubit_count(qb);
        }
    }

    pub fn make_engine(&self, is_opencl: bool) -> Arc<dyn QEngine> {
        unimplemented!()
    }

    pub fn is_opencl(&self) -> bool {
        self.is_gpu
    }

    pub fn set_concurrency(&mut self, thread_count: u32) {
        self.engine.set_concurrency(thread_count);
    }

    pub fn switch_gpu_mode(&mut self, use_gpu: bool) {
        let n_engine = if !self.is_gpu && use_gpu {
            self.make_engine(true)
        } else if self.is_gpu && !use_gpu {
            self.make_engine(false)
        } else {
            None
        };
        if let Some(n_engine) = n_engine {
            n_engine.copy_state_vec(&self.engine);
            self.engine = n_engine;
        }
        self.is_gpu = use_gpu;
    }

    pub fn switch_pager_mode(&mut self, use_pager: bool) {
        if !self.is_pager && use_pager {
            let engines = if self.is_gpu {
                vec![QRACK_GPU_ENGINE, QINTERFACE_CPU]
            } else {
                vec![QINTERFACE_CPU]
            };
            self.engine = Arc::new(QPager::new(
                self.engine.clone(),
                engines,
                self.qubit_count,
                ZERO_BCI,
                rand_generator,
                phase_factor,
                do_normalize,
                rand_global_phase,
                use_host_ram,
                dev_id,
                use_rdrand,
                is_sparse,
                amplitude_floor,
                device_ids,
                0U,
                separability_threshold,
            ));
        } else if self.is_pager && !use_pager {
            self.engine = self.engine.release_engine();
        }
        self.is_pager = use_pager;
    }

    pub fn switch_modes(&mut self, use_gpu: bool, use_pager: bool) {
        if !use_pager {
            self.switch_pager_mode(false);
        }
        self.switch_gpu_mode(use_gpu);
        if use_pager {
            self.switch_pager_mode(true);
        }
    }

    pub fn get_running_norm(&self) -> f64 {
        self.engine.get_running_norm()
    }

    pub fn zero_amplitudes(&mut self) {
        self.engine.zero_amplitudes();
    }

    pub fn is_zero_amplitude(&self) -> bool {
        self.engine.is_zero_amplitude()
    }

    pub fn copy_state_vec(&mut self, src: Arc<dyn QEngine>) {
        self.switch_modes(src.is_gpu(), src.is_pager());
        self.engine.copy_state_vec(src.engine());
    }

    pub fn get_amplitude_page(&self, page_ptr: *mut Complex, offset: u64, length: u64) {
        self.engine.get_amplitude_page(page_ptr, offset, length);
    }

    pub fn set_amplitude_page(&mut self, page_ptr: *const Complex, offset: u64, length: u64) {
        self.engine.set_amplitude_page(page_ptr, offset, length);
    }

    pub fn set_amplitude_page_qhybrid(&mut self, page_engine_ptr: Arc<QHybrid>, src_offset: u64, dst_offset: u64, length: u64) {
        page_engine_ptr.switch_modes(self.is_gpu, self.is_pager);
        self.engine.set_amplitude_page(page_engine_ptr.engine, src_offset, dst_offset, length);
    }

    pub fn set_amplitude_page_qengine(&mut self, page_engine_ptr: Arc<dyn QEngine>, src_offset: u64, dst_offset: u64, length: u64) {
        self.set_amplitude_page_qhybrid(page_engine_ptr, src_offset, dst_offset, length);
    }

    pub fn shuffle_buffers(&mut self, o_engine: Arc<dyn QEngine>) {
        let o_engine = o_engine.downcast::<QHybrid>().unwrap();
        o_engine.switch_modes(self.is_gpu, self.is_pager);
        self.engine.shuffle_buffers(o_engine.engine);
    }

    pub fn clone_empty(&self) -> Arc<dyn QEngine> {
        self.engine.clone_empty()
    }

    pub fn queue_set_do_normalize(&mut self, do_norm: bool) {
        self.engine.queue_set_do_normalize(do_norm);
    }

    pub fn queue_set_running_norm(&mut self, running_nrm: f64) {
        self.engine.queue_set_running_norm(running_nrm);
    }

    pub fn apply_m(&mut self, reg_mask: u64, result: u64, nrm: Complex) {
        self.engine.apply_m(reg_mask, result, nrm);
    }

    pub fn prob_reg(&self, start: u64, length: u64, permutation: u64) -> f64 {
        self.engine.prob_reg(start, length, permutation)
    }

    pub fn compose(&mut self, to_copy: Arc<QHybrid>) -> u64 {
        self.set_qubit_count(self.qubit_count + to_copy.qubit_count);
        to_copy.switch_modes(self.is_gpu, self.is_pager);
        self.engine.compose(to_copy.engine)
    }

    pub fn compose_qinterface(&mut self, to_copy: Arc<dyn QInterface>) -> u64 {
        self.compose(to_copy.downcast::<QHybrid>().unwrap())
    }

    pub fn compose_start(&mut self, to_copy: Arc<QHybrid>, start: u64) -> u64 {
        self.set_qubit_count(self.qubit_count + to_copy.qubit_count);
        to_copy.switch_modes(self.is_gpu, self.is_pager);
        self.engine.compose_start(to_copy.engine, start)
    }

    pub fn compose_start_qinterface(&mut self, to_copy: Arc<dyn QInterface>, start: u64) -> u64 {
        self.compose_start(to_copy.downcast::<QHybrid>().unwrap(), start)
    }

    pub fn compose_no_clone(&mut self, to_copy: Arc<QHybrid>) -> u64 {
        self.set_qubit_count(self.qubit_count + to_copy.qubit_count);
        to_copy.switch_modes(self.is_gpu, self.is_pager);
        self.engine.compose_no_clone(to_copy.engine)
    }

    pub fn compose_no_clone_qinterface(&mut self, to_copy: Arc<dyn QInterface>) -> u64 {
        self.compose_no_clone(to_copy.downcast::<QHybrid>().unwrap())
    }

    pub fn decompose(&mut self, start: u64, dest: Arc<dyn QInterface>) {
        self.decompose(start, dest.downcast::<QHybrid>().unwrap())
    }

    pub fn try_decompose(&mut self, start: u64, dest: Arc<dyn QInterface>, error_tol: f64) -> bool {
        self.try_decompose(start, dest.downcast::<QHybrid>().unwrap(), error_tol)
    }

    pub fn decompose_qhybrid(&mut self, start: u64, dest: Arc<QHybrid>) {
        dest.switch_modes(self.is_gpu, self.is_pager);
        self.engine.decompose(start, dest.engine);
        self.set_qubit_count(self.qubit_count - dest.get_qubit_count());
    }

    pub fn dispose(&mut self, start: u64, length: u64) {
        self.engine.dispose(start, length);
        self.set_qubit_count(self.qubit_count - length);
    }

    pub fn dispose_perm(&mut self, start: u64, length: u64, disposed_perm: u64) {
        self.engine.dispose_perm(start, length, disposed_perm);
        self.set_qubit_count(self.qubit_count - length);
    }

    pub fn uniform_parity_rz(&mut self, mask: u64, angle: f64) {
        self.engine.uniform_parity_rz(mask, angle);
    }

    pub fn cuniform_parity_rz(&mut self, controls: Vec<u64>, mask: u64, angle: f64) {
        self.engine.cuniform_parity_rz(controls, mask, angle);
    }

    pub fn cswap(&mut self, controls: Vec<u64>, qubit1: u64, qubit2: u64) {
        self.engine.cswap(controls, qubit1, qubit2);
    }

    pub fn anti_cswap(&mut self, controls: Vec<u64>, qubit1: u64, qubit2: u64) {
        self.engine.anti_cswap(controls, qubit1, qubit2);
    }

    pub fn csqrt_swap(&mut self, controls: Vec<u64>, qubit1: u64, qubit2: u64) {
        self.engine.csqrt_swap(controls, qubit1, qubit2);
    }

    pub fn anti_csqrt_swap(&mut self, controls: Vec<u64>, qubit1: u64, qubit2: u64) {
        self.engine.anti_csqrt_swap(controls, qubit1, qubit2);
    }

    pub fn cisqrt_swap(&mut self, controls: Vec<u64>, qubit1: u64, qubit2: u64) {
        self.engine.cisqrt_swap(controls, qubit1, qubit2);
    }

    pub fn anti_cisqrt_swap(&mut self, controls: Vec<u64>, qubit1: u64, qubit2: u64) {
        self.engine.anti_cisqrt_swap(controls, qubit1, qubit2);
    }

    pub fn force_m(&mut self, qubit: u64, result: bool, do_force: bool, do_apply: bool) -> bool {
        self.engine.force_m(qubit, result, do_force, do_apply)
    }

    pub fn inc(&mut self, to_add: u64, start: u64, length: u64) {
        self.engine.inc(to_add, start, length);
    }

    pub fn cinc(&mut self, to_add: u64, in_out_start: u64, length: u64, controls: Vec<u64>) {
        self.engine.cinc(to_add, in_out_start, length, controls);
    }

    pub fn incc(&mut self, to_add: u64, start: u64, length: u64, carry_index: u64) {
        self.engine.incc(to_add, start, length, carry_index);
    }

    pub fn incs(&mut self, to_add: u64, start: u64, length: u64, overflow_index: u64) {
        self.engine.incs(to_add, start, length, overflow_index);
    }

    pub fn incsc(&mut self, to_add: u64, start: u64, length: u64, overflow_index: u64, carry_index: u64) {
        self.engine.incsc(to_add, start, length, overflow_index, carry_index);
    }

    pub fn incsc_no_overflow(&mut self, to_add: u64, start: u64, length: u64, carry_index: u64) {
        self.engine.incsc_no_overflow(to_add, start, length, carry_index);
    }

    pub fn dec(&mut self, to_sub: u64, start: u64, length: u64, carry_index: u64) {
        self.engine.dec(to_sub, start, length, carry_index);
    }

    pub fn decsc(&mut self, to_sub: u64, start: u64, length: u64, overflow_index: u64, carry_index: u64) {
        self.engine.decsc(to_sub, start, length, overflow_index, carry_index);
    }

    pub fn decsc_no_overflow(&mut self, to_sub: u64, start: u64, length: u64, carry_index: u64) {
        self.engine.decsc_no_overflow(to_sub, start, length, carry_index);
    }

    pub fn mul(&mut self, to_mul: u64, in_out_start: u64, carry_start: u64, length: u64) {
        self.engine.mul(to_mul, in_out_start, carry_start, length);
    }

    pub fn div(&mut self, to_div: u64, in_out_start: u64, carry_start: u64, length: u64) {
        self.engine.div(to_div, in_out_start, carry_start, length);
    }

    pub fn mul_mod_n_out(&mut self, to_mul: u64, mod_n: u64, in_start: u64, out_start: u64, length: u64) {
        self.engine.mul_mod_n_out(to_mul, mod_n, in_start, out_start, length);
    }

    pub fn imul_mod_n_out(&mut self, to_mul: u64, mod_n: u64, in_start: u64, out_start: u64, length: u64) {
        self.engine.imul_mod_n_out(to_mul, mod_n, in_start, out_start, length);
    }

    pub fn pow_mod_n_out(&mut self, base: u64, mod_n: u64, in_start: u64, out_start: u64, length: u64) {
        self.engine.pow_mod_n_out(base, mod_n, in_start, out_start, length);
    }

    pub fn cmul(&mut self, to_mul: u64, in_out_start: u64, carry_start: u64, length: u64, controls: Vec<u64>) {
        self.engine.cmul(to_mul, in_out_start, carry_start, length, controls);
    }

    pub fn cdiv(&mut self, to_div: u64, in_out_start: u64, carry_start: u64, length: u64, controls: Vec<u64>) {
        self.engine.cdiv(to_div, in_out_start, carry_start, length, controls);
    }

    pub fn cmul_mod_n_out(&mut self, to_mul: u64, mod_n: u64, in_start: u64, out_start: u64, length: u64, controls: Vec<u64>) {
        self.engine.cmul_mod_n_out(to_mul, mod_n, in_start, out_start, length, controls);
    }

    pub fn cimul_mod_n_out(&mut self, to_mul: u64, mod_n: u64, in_start: u64, out_start: u64, length: u64, controls: Vec<u64>) {
        self.engine.cimul_mod_n_out(to_mul, mod_n, in_start, out_start, length, controls);
    }

    pub fn cpow_mod_n_out(&mut self, base: u64, mod_n: u64, in_start: u64, out_start: u64, length: u64, controls: Vec<u64>) {
        self.engine.cpow_mod_n_out(base, mod_n, in_start, out_start, length, controls);
    }

    pub fn indexed_lda(&mut self, index_start: u64, index_length: u64, value_start: u64, value_length: u64, values: *const u8, reset_value: bool) -> u64 {
        self.engine.indexed_lda(index_start, index_length, value_start, value_length, values, reset_value)
    }

    pub fn indexed_adc(&mut self, index_start: u64, index_length: u64, value_start: u64, value_length: u64, carry_index: u64, values: *const u8) -> u64 {
        self.engine.indexed_adc(index_start, index_length, value_start, value_length, carry_index, values)
    }

    pub fn indexed_sbc(&mut self, index_start: u64, index_length: u64, value_start: u64, value_length: u64, carry_index: u64, values: *const u8) -> u64 {
        self.engine.indexed_sbc(index_start, index_length, value_start, value_length, carry_index, values)
    }

    pub fn hash(&mut self, start: u64, length: u64, values: *const u8) {
        self.engine.hash(start, length, values);
    }

    pub fn swap(&mut self, qubit_index1: u64, qubit_index2: u64) {
        self.engine.swap(qubit_index1, qubit_index2);
    }

    pub fn iswap(&mut self, qubit_index1: u64, qubit_index2: u64) {
        self.engine.iswap(qubit_index1, qubit_index2);
    }

    pub fn iiswap(&mut self, qubit_index1: u64, qubit_index2: u64) {
        self.engine.iiswap(qubit_index1, qubit_index2);
    }

    pub fn sqrt_swap(&mut self, qubit_index1: u64, qubit_index2: u64) {
        self.engine.sqrt_swap(qubit_index1, qubit_index2);
    }

    pub fn isqrt_swap(&mut self, qubit_index1: u64, qubit_index2: u64) {
        self.engine.isqrt_swap(qubit_index1, qubit_index2);
    }

    pub fn fsim(&mut self, theta: f64, phi: f64, qubit_index1: u64, qubit_index2: u64) {
        self.engine.fsim(theta, phi, qubit_index1, qubit_index2);
    }

    pub fn prob(&self, qubit_index: u64) -> f64 {
        self.engine.prob(qubit_index)
    }

    pub fn ctrl_or_anti_prob(&self, control_state: bool, control: u64, target: u64) -> f64 {
        self.engine.ctrl_or_anti_prob(control_state, control, target)
    }

    pub fn prob_all(&self, full_register: u64) -> f64 {
        self.engine.prob_all(full_register)
    }

    pub fn prob_mask(&self, mask: u64, permutation: u64) -> f64 {
        self.engine.prob_mask(mask, permutation)
    }

    pub fn prob_parity(&self, mask: u64) -> f64 {
        self.engine.prob_parity(mask)
    }

    pub fn force_m_parity(&mut self, mask: u64, result: bool, do_force: bool) -> bool {
        self.engine.force_m_parity(mask, result, do_force)
    }

    pub fn sum_sqr_diff(&self, to_compare: Arc<QInterface>) -> f64 {
        self.sum_sqr_diff(to_compare.downcast::<QHybrid>().unwrap())
    }

    pub fn sum_sqr_diff_qhybrid(&self, to_compare: Arc<QHybrid>) -> f64 {
        to_compare.switch_modes(self.is_gpu, self.is_pager);
        self.engine.sum_sqr_diff(to_compare.engine)
    }

    pub fn update_running_norm(&mut self, norm_thresh: f64) {
        self.engine.update_running_norm(norm_thresh);
    }

    pub fn normalize_state(&mut self, nrm: f64, norm_thresh: f64, phase_arg: f64) {
        self.engine.normalize_state(nrm, norm_thresh, phase_arg);
    }

    pub fn expectation_bits_all(&self, bits: Vec<u64>, offset: u64) -> f64 {
        self.engine.expectation_bits_all(bits, offset)
    }

    pub fn finish(&mut self) {
        self.engine.finish();
    }

    pub fn is_finished(&self) -> bool {
        self.engine.is_finished()
    }

    pub fn dump(&self) {
        self.engine.dump();
    }

    pub fn clone(&self) -> Arc<dyn QInterface> {
        let c = Arc::new(QHybrid::new(
            self.qubit_count,
            ZERO_BCI,
            rand_generator,
            phase_factor,
            do_normalize,
            rand_global_phase,
            use_host_ram,
            dev_id,
            use_rdrand,
            is_sparse,
            amplitude_floor,
            device_ids,
            gpu_threshold_qubits,
            separability_threshold,
        ));
        c.running_norm = running_norm;
        c.set_concurrency(self.get_concurrency_level());
        c.engine.copy_state_vec(&self.engine);
        c
    }

    pub fn set_device(&mut self, d_id: i64) {
        self.dev_id = d_id;
        self.engine.set_device(d_id);
    }

    pub fn get_device(&self) -> i64 {
        self.dev_id
    }

    pub fn get_max_size(&self) -> u64 {
        self.engine.get_max_size()
    }

    fn get_expectation(&self, value_start: u64, value_length: u64) -> f64 {
        self.engine.get_expectation(value_start, value_length)
    }

    fn apply_2x2(&mut self, offset1: u64, offset2: u64, mtrx: *const Complex, bit_count: u64, q_powers_sorted: *const u64, do_calc_norm: bool, norm_thresh: f64) {
        self.engine.apply_2x2(offset1, offset2, mtrx, bit_count, q_powers_sorted, do_calc_norm, norm_thresh);
    }

    fn apply_controlled_2x2(&mut self, controls: Vec<u64>, target: u64, mtrx: *const Complex) {
        self.engine.apply_controlled_2x2(controls, target, mtrx);
    }

    fn apply_anti_controlled_2x2(&mut self, controls: Vec<u64>, target: u64, mtrx: *const Complex) {
        self.engine.apply_anti_controlled_2x2(controls, target, mtrx);
    }
}

