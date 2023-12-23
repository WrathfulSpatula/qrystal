use std::sync::Arc;
use std::cell::RefCell;

pub struct QPager {
    use_gpu_threshold: bool,
    is_sparse: bool,
    use_t_gadget: bool,
    max_page_setting: i64,
    max_page_qubits: i64,
    threshold_qubits_per_page: i64,
    base_qubits_per_page: i64,
    max_qubits: i64,
    dev_id: i64,
    root_engine: QInterfaceEngine,
    base_page_max_q_power: i64,
    base_page_count: i64,
    phase_factor: Complex,
    devices_host_pointer: Vec<bool>,
    device_ids: Vec<i64>,
    engines: Vec<QInterfaceEngine>,
    q_pages: Vec<QEnginePtr>,
}

pub type QPagerPtr = Arc<RefCell<QPager>>;

impl QPager {
    pub fn new(eng: Vec<QInterfaceEngine>, q_bit_count: i64, init_state: i64, rgp: qrack_rand_gen_ptr, phase_fac: Complex, do_norm: bool, ignored: bool, use_host_mem: bool, device_id: i64, use_hardware_rng: bool, use_sparse_state_vec: bool, norm_thresh: f64, dev_list: Vec<i64>, qubit_threshold: i64, separation_thresh: f64) -> QPagerPtr {
        let qubit_count = q_bit_count;
        let is_sparse = use_sparse_state_vec;
        let use_t_gadget = true;
        let max_page_setting = -1;
        let max_page_qubits = -1;
        let threshold_qubits_per_page = qubit_threshold;
        let dev_id = device_id;
        let phase_factor = phase_fac;
        let device_ids = dev_list;
        let engines = eng;

        let mut q_pages = Vec::new();
        let max_qubits = qubit_count;
        let base_qubits_per_page = if qubit_count < threshold_qubits_per_page { qubit_count } else { threshold_qubits_per_page };
        let base_page_count = pow2_ocl(qubit_count - base_qubits_per_page);
        let base_page_max_q_power = pow2_ocl(base_qubits_per_page);

        let q_pages = vec![MakeEngine(qubit_count, 0)];

        let qpager = QPager {
            use_gpu_threshold,
            is_sparse,
            use_t_gadget,
            max_page_setting,
            max_page_qubits,
            threshold_qubits_per_page,
            base_qubits_per_page,
            max_qubits,
            dev_id,
            root_engine,
            base_page_max_q_power,
            base_page_count,
            phase_factor,
            devices_host_pointer,
            device_ids,
            engines,
            q_pages,
        };

        Arc::new(RefCell::new(qpager))
    }

    fn make_engine(length: i64, page_id: i64) -> QEnginePtr {
        // TODO: Implement this function
    }

    fn set_qubit_count(&mut self, qb: i64) {
        self.qubit_count = qb;
        self.base_qubits_per_page = if self.qubit_count < self.threshold_qubits_per_page { self.qubit_count } else { self.threshold_qubits_per_page };
        self.base_page_count = pow2_ocl(self.qubit_count - self.base_qubits_per_page);
        self.base_page_max_q_power = pow2_ocl(self.base_qubits_per_page);
    }

    fn page_max_q_power(&self) -> i64 {
        let to_ret;
        bi_div_mod_small(self.max_q_power, self.q_pages.len(), &to_ret, None);
        to_ret as i64
    }

    fn paged_qubit_count(&self) -> i64 {
        log2_ocl(self.q_pages.len()) as i64
    }

    fn qubits_per_page(&self) -> i64 {
        log2_ocl(self.page_max_q_power()) as i64
    }

    fn get_page_device(&self, page: i64) -> i64 {
        self.device_ids[page % self.device_ids.len()]
    }

    fn get_page_host_pointer(&self, page: i64) -> bool {
        self.devices_host_pointer[page % self.devices_host_pointer.len()]
    }

    fn combine_engines(&mut self, threshold_bits: i64) {
        // TODO: Implement this function
    }

    fn separate_engines(&mut self, threshold_bits: i64, no_base_floor: bool) {
        // TODO: Implement this function
    }

    fn single_bit_gate<Qubit1Fn>(&mut self, target: i64, fn: Qubit1Fn, is_sqi_ctrl: bool, is_anti: bool) {
        // TODO: Implement this function
    }

    fn meta_controlled<Qubit1Fn>(&mut self, control_perm: i64, controls: Vec<i64>, target: i64, fn: Qubit1Fn, mtrx: &Complex, is_sqi_ctrl: bool, is_intra_ctrled: bool) {
        // TODO: Implement this function
    }

    fn semi_meta_controlled<Qubit1Fn>(&mut self, control_perm: i64, controls: Vec<i64>, target: i64, fn: Qubit1Fn) {
        // TODO: Implement this function
    }

    fn meta_swap(&mut self, qubit1: i64, qubit2: i64, is_i_phase_fac: bool, is_inverse: bool) {
        // TODO: Implement this function
    }

    fn combine_and_op<F>(&mut self, fn: F, bits: Vec<i64>) {
        // TODO: Implement this function
    }

    fn combine_and_op_controlled<F>(&mut self, fn: F, bits: Vec<i64>, controls: &Vec<i64>) {
        // TODO: Implement this function
    }

    fn apply_single_either(&mut self, is_invert: bool, top: Complex, bottom: Complex, target: i64) {
        // TODO: Implement this function
    }

    fn apply_either_controlled_single_bit(&mut self, control_perm: i64, controls: &Vec<i64>, target: i64, mtrx: &Complex) {
        // TODO: Implement this function
    }

    fn either_i_swap(&mut self, qubit1: i64, qubit2: i64, is_inverse: bool) {
        // TODO: Implement this function
    }

    fn init(&mut self) {
        // TODO: Implement this function
    }

    fn get_set_amplitude_page(&mut self, page_ptr: &mut Complex, c_page_ptr: &Complex, offset: i64, length: i64) {
        // TODO: Implement this function
    }

    fn set_concurrency(&mut self, threads_per_engine: u32) {
        // TODO: Implement this function
    }

    fn set_t_injection(&mut self, use_gadget: bool) {
        self.use_t_gadget = use_gadget;
        for i in 0..self.q_pages.len() {
            self.q_pages[i].set_t_injection(use_gadget);
        }
    }

    fn get_t_injection(&self) -> bool {
        self.use_t_gadget
    }

    fn is_open_cl(&self) -> bool {
        self.q_pages[0].is_open_cl()
    }

    fn release_engine(&mut self) -> QEnginePtr {
        self.combine_engines();
        self.q_pages[0].clone()
    }

    fn lock_engine(&mut self, eng: QEnginePtr) {
        self.q_pages.resize(1);
        self.q_pages[0] = eng;
        eng.set_device(self.device_ids[0]);
    }

    fn zero_amplitudes(&mut self) {
        for i in 0..self.q_pages.len() {
            self.q_pages[i].zero_amplitudes();
        }
    }

    fn copy_state_vec(&mut self, src: QEnginePtr) {
        self.copy_state_vec(src.clone());
    }

    fn copy_state_vec(&mut self, src: QPagerPtr) {
        let qpp = self.qubits_per_page();
        src.borrow_mut().combine_engines(qpp);
        src.borrow_mut().separate_engines(qpp, true);
        for i in 0..self.q_pages.len() {
            self.q_pages[i].copy_state_vec(src.borrow().q_pages[i].clone());
        }
    }

    fn is_zero_amplitude(&self) -> bool {
        for i in 0..self.q_pages.len() {
            if !self.q_pages[i].is_zero_amplitude() {
                return false;
            }
        }
        true
    }

    fn get_amplitude_page(&mut self, page_ptr: &mut Complex, offset: i64, length: i64) {
        self.get_set_amplitude_page(page_ptr, None, offset, length);
    }

    fn set_amplitude_page(&mut self, page_ptr: &Complex, offset: i64, length: i64) {
        self.get_set_amplitude_page(None, page_ptr, offset, length);
    }

    fn set_amplitude_page(&mut self, page_engine_ptr: QEnginePtr, src_offset: i64, dst_offset: i64, length: i64) {
        self.set_amplitude_page(page_engine_ptr.clone(), src_offset, dst_offset, length);
    }

    fn set_amplitude_page(&mut self, page_engine_ptr: QPagerPtr, src_offset: i64, dst_offset: i64, length: i64) {
        self.combine_engines();
        page_engine_ptr.borrow_mut().combine_engines();
        self.q_pages[0].set_amplitude_page(page_engine_ptr.borrow().q_pages[0].clone(), src_offset, dst_offset, length);
    }

    fn shuffle_buffers(&mut self, engine: QEnginePtr) {
        self.shuffle_buffers(engine.clone());
    }

    fn shuffle_buffers(&mut self, engine: QPagerPtr) {
        let qpp = self.qubits_per_page();
        let tcqpp = engine.borrow().qubits_per_page();
        engine.borrow_mut().separate_engines(qpp, true);
        self.separate_engines(tcqpp, true);
        if self.q_pages.len() == 1 {
            self.q_pages[0].shuffle_buffers(engine.borrow().q_pages[0].clone());
            return;
        }
        let offset = self.q_pages.len() >> 1;
        for i in 0..offset {
            self.q_pages[offset + i].swap(&mut engine.borrow_mut().q_pages[i]);
        }
    }

    fn clone_empty(&self) -> QEnginePtr {
        // TODO: Implement this function
    }

    fn queue_set_do_normalize(&mut self, do_norm: bool) {
        self.finish();
        self.do_normalize = do_norm;
    }

    fn queue_set_running_norm(&mut self, running_nrm: f64) {
        self.finish();
        self.running_norm = running_nrm;
    }

    fn prob_reg(&mut self, start: i64, length: i64, permutation: i64) -> f64 {
        self.combine_engines();
        self.q_pages[0].prob_reg(start, length, permutation)
    }

    fn apply_m(&mut self, reg_mask: i64, result: i64, nrm: Complex) {
        self.combine_engines();
        self.q_pages[0].apply_m(reg_mask, result, nrm);
    }

    fn get_expectation(&mut self, value_start: i64, value_length: i64) -> f64 {
        self.combine_engines();
        self.q_pages[0].get_expectation(value_start, value_length)
    }

    fn apply_2x2(&mut self, offset1: i64, offset2: i64, mtrx: &Complex, bit_count: i64, q_powers_sorted: &i64, do_calc_norm: bool, norm_thresh: f64) {
        self.combine_engines();
        self.q_pages[0].apply_2x2(offset1, offset2, mtrx, bit_count, q_powers_sorted, do_calc_norm, norm_thresh);
    }

    fn get_running_norm(&self) -> f64 {
        let mut to_ret = 0.0;
        for i in 0..self.q_pages.len() {
            to_ret += self.q_pages[i].get_running_norm();
        }
        to_ret
    }

    fn first_nonzero_phase(&self) -> f64 {
        for i in 0..self.q_pages.len() {
            if !self.q_pages[i].is_zero_amplitude() {
                return self.q_pages[i].first_nonzero_phase();
            }
        }
        0.0
    }

    fn set_quantum_state(&mut self, input_state: &Complex) {
        // TODO: Implement this function
    }

    fn get_quantum_state(&mut self, output_state: &mut Complex) {
        // TODO: Implement this function
    }

    fn get_probs(&mut self, output_probs: &mut f64) {
        // TODO: Implement this function
    }

    fn get_amplitude(&mut self, perm: i64) -> Complex {
        let mut p = 0;
        let mut a = 0;
        bi_div_mod(perm, self.page_max_q_power(), &mut p, &mut a);
        self.q_pages[p as usize].get_amplitude(a)
    }

    fn set_amplitude(&mut self, perm: i64, amp: Complex) {
        let mut p = 0;
        let mut a = 0;
        bi_div_mod(perm, self.page_max_q_power(), &mut p, &mut a);
        self.q_pages[p as usize].set_amplitude(a, amp);
    }

    fn prob_all(&mut self, perm: i64) -> f64 {
        let mut p = 0;
        let mut a = 0;
        bi_div_mod(perm, self.page_max_q_power(), &mut p, &mut a);
        self.q_pages[p as usize].prob_all(a)
    }

    fn set_permutation(&mut self, perm: i64, phase_fac: Complex) {
        // TODO: Implement this function
    }

    fn compose(&mut self, to_copy: QPagerPtr) -> i64 {
        self.compose_either(to_copy, false)
    }

    fn compose(&mut self, to_copy: QInterfacePtr) -> i64 {
        self.compose(to_copy.clone())
    }

    fn compose_no_clone(&mut self, to_copy: QPagerPtr) -> i64 {
        self.compose_either(to_copy, true)
    }

    fn compose_no_clone(&mut self, to_copy: QInterfacePtr) -> i64 {
        self.compose_no_clone(to_copy.clone())
    }

    fn compose_either(&mut self, to_copy: QPagerPtr, will_destroy: bool) -> i64 {
        // TODO: Implement this function
    }

    fn decompose(&mut self, start: i64, dest: QPagerPtr) {
        self.decompose(start, dest.clone())
    }

    fn decompose(&mut self, start: i64, dest: QInterfacePtr) {
        self.decompose(start, dest.clone())
    }

    fn decompose(&mut self, start: i64, length: i64) -> QInterfacePtr {
        // TODO: Implement this function
    }

    fn dispose(&mut self, start: i64, length: i64) {
        self.dispose(start, length, 0)
    }

    fn dispose(&mut self, start: i64, length: i64, disposed_perm: i64) {
        // TODO: Implement this function
    }

    fn allocate(&mut self, start: i64, length: i64) -> i64 {
        // TODO: Implement this function
    }

    fn mtrx(&mut self, mtrx: &Complex, target: i64) {
        // TODO: Implement this function
    }

    fn phase(&mut self, top_left: Complex, bottom_right: Complex, qubit_index: i64) {
        self.apply_single_either(false, top_left, bottom_right, qubit_index);
    }

    fn invert(&mut self, top_right: Complex, bottom_left: Complex, qubit_index: i64) {
        self.apply_single_either(true, top_right, bottom_left, qubit_index);
    }

    fn mcmtrx(&mut self, controls: &Vec<i64>, mtrx: &Complex, target: i64) {
        let p = pow2(controls.len());
        bi_decrement(&mut p, 1);
        self.apply_either_controlled_single_bit(p, controls, target, mtrx);
    }

    fn macmtrx(&mut self, controls: &Vec<i64>, mtrx: &Complex, target: i64) {
        self.apply_either_controlled_single_bit(0, controls, target, mtrx);
    }

    fn uniform_parity_rz(&mut self, mask: i64, angle: f64) {
        // TODO: Implement this function
    }

    fn c_uniform_parity_rz(&mut self, controls: &Vec<i64>, mask: i64, angle: f64) {
        // TODO: Implement this function
    }

    fn x_mask(&mut self, mask: i64) {
        // TODO: Implement this function
    }

    fn z_mask(&mut self, mask: i64) {
        self.phase_parity(std::f64::consts::PI, mask);
    }

    fn phase_parity(&mut self, radians: f64, mask: i64) {
        // TODO: Implement this function
    }

    fn force_m(&mut self, qubit: i64, result: bool, do_force: bool, do_apply: bool) -> bool {
        // TODO: Implement this function
    }

    fn force_m_reg(&mut self, start: i64, length: i64, result: i64, do_force: bool, do_apply: bool) -> i64 {
        self.force_m_reg(start, length, result, do_force, do_apply)
    }

    fn incdecsc(&mut self, to_add: i64, start: i64, length: i64, overflow_index: i64, carry_index: i64) {
        self.incdecsc(to_add, start, length, carry_index);
    }

    fn incdecsc(&mut self, to_add: i64, start: i64, length: i64, carry_index: i64) {
        // TODO: Implement this function
    }

    fn incbcd(&mut self, to_add: i64, start: i64, length: i64) {
        // TODO: Implement this function
    }

    fn incdec_bcdc(&mut self, to_add: i64, start: i64, length: i64, carry_index: i64) {
        // TODO: Implement this function
    }

    fn mul(&mut self, to_mul: i64, in_out_start: i64, carry_start: i64, length: i64) {
        // TODO: Implement this function
    }

    fn div(&mut self, to_div: i64, in_out_start: i64, carry_start: i64, length: i64) {
        // TODO: Implement this function
    }

    fn mul_mod_n_out(&mut self, to_mul: i64, mod_n: i64, in_start: i64, out_start: i64, length: i64) {
        // TODO: Implement this function
    }

    fn imul_mod_n_out(&mut self, to_mul: i64, mod_n: i64, in_start: i64, out_start: i64, length: i64) {
        // TODO: Implement this function
    }

    fn pow_mod_n_out(&mut self, base: i64, mod_n: i64, in_start: i64, out_start: i64, length: i64) {
        // TODO: Implement this function
    }

    fn cmul(&mut self, to_mul: i64, in_out_start: i64, carry_start: i64, length: i64, controls: &Vec<i64>) {
        // TODO: Implement this function
    }

    fn cdiv(&mut self, to_div: i64, in_out_start: i64, carry_start: i64, length: i64, controls: &Vec<i64>) {
        // TODO: Implement this function
    }

    fn cmul_mod_n_out(&mut self, to_mul: i64, mod_n: i64, in_start: i64, out_start: i64, length: i64, controls: &Vec<i64>) {
        // TODO: Implement this function
    }

    fn cimul_mod_n_out(&mut self, to_mul: i64, mod_n: i64, in_start: i64, out_start: i64, length: i64, controls: &Vec<i64>) {
        // TODO: Implement this function
    }

    fn cpow_mod_n_out(&mut self, base: i64, mod_n: i64, in_start: i64, out_start: i64, length: i64, controls: &Vec<i64>) {
        // TODO: Implement this function
    }

    fn indexed_lda(&mut self, index_start: i64, index_length: i64, value_start: i64, value_length: i64, values: &Vec<u8>, reset_value: bool) -> i64 {
        // TODO: Implement this function
    }

    fn indexed_adc(&mut self, index_start: i64, index_length: i64, value_start: i64, value_length: i64, carry_index: i64, values: &Vec<u8>) -> i64 {
        // TODO: Implement this function
    }

    fn indexed_sbc(&mut self, index_start: i64, index_length: i64, value_start: i64, value_length: i64, carry_index: i64, values: &Vec<u8>) -> i64 {
        // TODO: Implement this function
    }

    fn hash(&mut self, start: i64, length: i64, values: &Vec<u8>) {
        // TODO: Implement this function
    }

    fn c_phase_flip_if_less(&mut self, greater_perm: i64, start: i64, length: i64, flag_index: i64) {
        // TODO: Implement this function
    }

    fn phase_flip_if_less(&mut self, greater_perm: i64, start: i64, length: i64) {
        // TODO: Implement this function
    }

    fn swap(&mut self, qubit_index1: i64, qubit_index2: i64) {
        // TODO: Implement this function
    }

    fn i_swap(&mut self, qubit1: i64, qubit2: i64) {
        self.either_i_swap(qubit1, qubit2, false);
    }

    fn ii_swap(&mut self, qubit1: i64, qubit2: i64) {
        self.either_i_swap(qubit1, qubit2, true);
    }

    fn f_sim(&mut self, theta: f64, phi: f64, qubit_index1: i64, qubit_index2: i64) {
        // TODO: Implement this function
    }

    fn prob(&mut self, qubit_index: i64) -> f64 {
        // TODO: Implement this function
    }

    fn prob_mask(&mut self, mask: i64, permutation: i64) -> f64 {
        // TODO: Implement this function
    }

    fn prob_parity(&mut self, mask: i64) -> f64 {
        if bi_compare_0(mask) == 0 {
            return 0.0;
        }
        self.combine_engines();
        self.q_pages[0].prob_parity(mask)
    }

    fn force_m_parity(&mut self, mask: i64, result: bool, do_force: bool) -> bool {
        if bi_compare_0(mask) == 0 {
            return 0.0;
        }
        self.combine_engines();
        self.q_pages[0].force_m_parity(mask, result, do_force)
    }

    fn expectation_bits_all(&mut self, bits: &Vec<i64>, offset: i64) -> f64 {
        // TODO: Implement this function
    }

    fn update_running_norm(&mut self, norm_thresh: f64) {
        // TODO: Implement this function
    }

    fn normalize_state(&mut self, nrm: f64, norm_thresh: f64, phase_arg: f64) {
        // TODO: Implement this function
    }

    fn finish(&mut self) {
        for i in 0..self.q_pages.len() {
            self.q_pages[i].finish();
        }
    }

    fn is_finished(&self) -> bool {
        for i in 0..self.q_pages.len() {
            if !self.q_pages[i].is_finished() {
                return false;
            }
        }
        true
    }

    fn dump(&self) {
        for i in 0..self.q_pages.len() {
            self.q_pages[i].dump();
        }
    }

    fn clone(&self) -> QInterfacePtr {
        // TODO: Implement this function
    }

    fn set_device(&mut self, d_id: i64) {
        self.device_ids.clear();
        self.device_ids.push(d_id);
        for i in 0..self.q_pages.len() {
            self.q_pages[i].set_device(d_id);
        }
        #[cfg(ENABLE_OPENCL)]
        if self.root_engine != QINTERFACE_CPU {
            self.max_page_qubits = log2_ocl(OCLEngine::Instance().GetDeviceContextPtr(dev_id).GetMaxAlloc() / std::mem::size_of::<Complex>());
            self.max_page_qubits = if self.max_page_setting < self.max_page_qubits { self.max_page_setting } else { 1 };
        }
        if !self.use_gpu_threshold {
            return;
        }
        self.threshold_qubits_per_page = self.max_page_qubits;
    }

    fn get_device(&self) -> i64 {
        self.q_pages[0].get_device()
    }

    fn get_max_size(&self) -> i64 {
        self.q_pages[0].get_max_size()
    }

    fn sum_sqr_diff(&self, to_compare: QPagerPtr) -> f64 {
        self.sum_sqr_diff(to_compare)
    }

    fn sum_sqr_diff(&self, to_compare: QPagerPtr) -> f64 {
        // TODO: Implement this function
    }
}


