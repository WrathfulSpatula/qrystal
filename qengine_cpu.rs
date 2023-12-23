use std::sync::Arc;
use std::collections::HashMap;

pub struct QEngineCPU {
    is_sparse: bool,
    max_qubits: usize,
    state_vec: Option<Arc<StateVector>>,
    dispatch_queue: Option<DispatchQueue>,
}

impl QEngineCPU {
    pub fn new(
        q_bit_count: usize,
        init_state: usize,
        rgp: Option<qrack_rand_gen_ptr>,
        phase_fac: complex,
        do_norm: bool,
        random_global_phase: bool,
        ignored: bool,
        ignored2: i64,
        use_hardware_rng: bool,
        use_sparse_state_vec: bool,
        norm_thresh: real1_f,
        ignored3: Vec<i64>,
        ignored4: usize,
        ignored5: real1_f,
    ) -> Self {
        Self {
            is_sparse: use_sparse_state_vec,
            max_qubits: q_bit_count,
            state_vec: None,
            dispatch_queue: None,
        }
    }

    pub fn finish(&self) {
        if let Some(dispatch_queue) = &self.dispatch_queue {
            dispatch_queue.finish();
        }
    }

    pub fn is_finished(&self) -> bool {
        if let Some(dispatch_queue) = &self.dispatch_queue {
            dispatch_queue.is_finished()
        } else {
            true
        }
    }

    pub fn dump(&self) {
        if let Some(dispatch_queue) = &self.dispatch_queue {
            dispatch_queue.dump();
        }
    }

    pub fn set_device(&self, d_id: i64) {}

    pub fn first_nonzero_phase(&self) -> complex {
        if let Some(state_vec) = &self.state_vec {
            state_vec.first_nonzero_phase()
        } else {
            ZERO_R1_F
        }
    }

    pub fn zero_amplitudes(&mut self) {
        self.dump();
        self.state_vec = None;
        self.running_norm = ZERO_R1;
    }

    pub fn free_state_vec(&mut self, sv: Option<*mut complex>) {
        self.state_vec = None;
    }

    pub fn is_zero_amplitude(&self) -> bool {
        self.state_vec.is_none()
    }

    pub fn get_amplitude_page(&self, page_ptr: *mut complex, offset: usize, length: usize) {
        if let Some(state_vec) = &self.state_vec {
            state_vec.get_amplitude_page(page_ptr, offset, length);
        }
    }

    pub fn set_amplitude_page(&mut self, page_ptr: *const complex, offset: usize, length: usize) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.set_amplitude_page(page_ptr, offset, length);
        }
    }

    pub fn set_amplitude_page_from_engine(
        &mut self,
        page_engine_ptr: QEnginePtr,
        src_offset: usize,
        dst_offset: usize,
        length: usize,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            if let Some(page_engine) = page_engine_ptr.downcast::<QEngineCPU>() {
                state_vec.set_amplitude_page_from_engine(
                    &page_engine.state_vec,
                    src_offset,
                    dst_offset,
                    length,
                );
            }
        }
    }

    pub fn shuffle_buffers(&mut self, engine: QEnginePtr) {
        if let Some(state_vec) = &mut self.state_vec {
            if let Some(engine) = engine.downcast::<QEngineCPU>() {
                state_vec.shuffle_buffers(&engine.state_vec);
            }
        }
    }

    pub fn copy_state_vec(&mut self, src: QEnginePtr) {
        if let Some(state_vec) = &mut self.state_vec {
            if let Some(src) = src.downcast::<QEngineCPU>() {
                state_vec.copy_state_vec(&src.state_vec);
            }
        }
    }

    pub fn clone_empty(&self) -> QEnginePtr {
        QEngineCPUPtr::new(self.max_qubits)
    }

    pub fn queue_set_do_normalize(&self, do_norm: bool) {
        if let Some(dispatch_queue) = &self.dispatch_queue {
            dispatch_queue.dispatch(|| {
                self.do_normalize = do_norm;
            });
        }
    }

    pub fn queue_set_running_norm(&self, running_nrm: real1_f) {
        if let Some(dispatch_queue) = &self.dispatch_queue {
            dispatch_queue.dispatch(|| {
                self.running_norm = running_nrm;
            });
        }
    }

    pub fn set_quantum_state(&mut self, input_state: *const complex) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.set_quantum_state(input_state);
        }
    }

    pub fn get_quantum_state(&self, output_state: *mut complex) {
        if let Some(state_vec) = &self.state_vec {
            state_vec.get_quantum_state(output_state);
        }
    }

    pub fn get_probs(&self, output_probs: *mut real1) {
        if let Some(state_vec) = &self.state_vec {
            state_vec.get_probs(output_probs);
        }
    }

    pub fn get_amplitude(&self, perm: usize) -> complex {
        if let Some(state_vec) = &self.state_vec {
            state_vec.get_amplitude(perm)
        } else {
            ZERO_C
        }
    }

    pub fn set_amplitude(&mut self, perm: usize, amp: complex) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.set_amplitude(perm, amp);
        }
    }

    pub fn compose(&mut self, to_copy: QEngineCPUPtr) -> usize {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.compose(&to_copy.state_vec)
        } else {
            0
        }
    }

    pub fn decompose(&mut self, start: usize, dest: QInterfacePtr) {
        if let Some(state_vec) = &mut self.state_vec {
            if let Some(dest) = dest.downcast::<QEngineCPU>() {
                state_vec.decompose(start, &dest.state_vec);
            }
        }
    }

    pub fn dispose(&mut self, start: usize, length: usize) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.dispose(start, length);
        }
    }

    pub fn dispose_with_perm(&mut self, start: usize, length: usize, disposed_perm: usize) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.dispose_with_perm(start, length, disposed_perm);
        }
    }

    pub fn allocate(&mut self, start: usize, length: usize) -> usize {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.allocate(start, length)
        } else {
            0
        }
    }

    pub fn x_mask(&mut self, mask: usize) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.x_mask(mask);
        }
    }

    pub fn phase_parity(&mut self, radians: real1_f, mask: usize) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.phase_parity(radians, mask);
        }
    }

    pub fn rol(&mut self, shift: usize, start: usize, length: usize) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.rol(shift, start, length);
        }
    }

    #[cfg(feature = "alu")]
    pub fn inc(&mut self, to_add: usize, start: usize, length: usize) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.inc(to_add, start, length);
        }
    }

    #[cfg(feature = "alu")]
    pub fn cinc(
        &mut self,
        to_add: usize,
        in_out_start: usize,
        length: usize,
        controls: Vec<usize>,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.cinc(to_add, in_out_start, length, &controls);
        }
    }

    #[cfg(feature = "alu")]
    pub fn incs(&mut self, to_add: usize, start: usize, length: usize, overflow_index: usize) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.incs(to_add, start, length, overflow_index);
        }
    }

    #[cfg(feature = "alu")]
    pub fn incbcd(&mut self, to_add: usize, start: usize, length: usize) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.incbcd(to_add, start, length);
        }
    }

    #[cfg(feature = "alu")]
    pub fn mul(&mut self, to_mul: usize, in_out_start: usize, carry_start: usize, length: usize) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.mul(to_mul, in_out_start, carry_start, length);
        }
    }

    #[cfg(feature = "alu")]
    pub fn div(&mut self, to_div: usize, in_out_start: usize, carry_start: usize, length: usize) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.div(to_div, in_out_start, carry_start, length);
        }
    }

    #[cfg(feature = "alu")]
    pub fn mul_mod_n_out(
        &mut self,
        to_mul: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.mul_mod_n_out(to_mul, mod_n, in_start, out_start, length);
        }
    }

    #[cfg(feature = "alu")]
    pub fn imul_mod_n_out(
        &mut self,
        to_mul: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.imul_mod_n_out(to_mul, mod_n, in_start, out_start, length);
        }
    }

    #[cfg(feature = "alu")]
    pub fn pow_mod_n_out(
        &mut self,
        base: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.pow_mod_n_out(base, mod_n, in_start, out_start, length);
        }
    }

    #[cfg(feature = "alu")]
    pub fn cmul(
        &mut self,
        to_mul: usize,
        in_out_start: usize,
        carry_start: usize,
        length: usize,
        controls: Vec<usize>,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.cmul(to_mul, in_out_start, carry_start, length, &controls);
        }
    }

    #[cfg(feature = "alu")]
    pub fn cdiv(
        &mut self,
        to_div: usize,
        in_out_start: usize,
        carry_start: usize,
        length: usize,
        controls: Vec<usize>,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.cdiv(to_div, in_out_start, carry_start, length, &controls);
        }
    }

    #[cfg(feature = "alu")]
    pub fn cmul_mod_n_out(
        &mut self,
        to_mul: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
        controls: Vec<usize>,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.cmul_mod_n_out(
                to_mul,
                mod_n,
                in_start,
                out_start,
                length,
                &controls,
            );
        }
    }

    #[cfg(feature = "alu")]
    pub fn cimul_mod_n_out(
        &mut self,
        to_mul: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
        controls: Vec<usize>,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.cimul_mod_n_out(
                to_mul,
                mod_n,
                in_start,
                out_start,
                length,
                &controls,
            );
        }
    }

    #[cfg(feature = "alu")]
    pub fn cpow_mod_n_out(
        &mut self,
        base: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
        controls: Vec<usize>,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.cpow_mod_n_out(
                base,
                mod_n,
                in_start,
                out_start,
                length,
                &controls,
            );
        }
    }

    #[cfg(feature = "alu")]
    pub fn full_add(
        &mut self,
        input_bit1: usize,
        input_bit2: usize,
        carry_in_sum_out: usize,
        carry_out: usize,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.full_add(
                input_bit1,
                input_bit2,
                carry_in_sum_out,
                carry_out,
            );
        }
    }

    #[cfg(feature = "alu")]
    pub fn ifull_add(
        &mut self,
        input_bit1: usize,
        input_bit2: usize,
        carry_in_sum_out: usize,
        carry_out: usize,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.ifull_add(
                input_bit1,
                input_bit2,
                carry_in_sum_out,
                carry_out,
            );
        }
    }

    #[cfg(feature = "alu")]
    pub fn indexed_lda(
        &mut self,
        index_start: usize,
        index_length: usize,
        value_start: usize,
        value_length: usize,
        values: &[u8],
        reset_value: bool,
    ) -> usize {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.indexed_lda(
                index_start,
                index_length,
                value_start,
                value_length,
                values,
                reset_value,
            )
        } else {
            0
        }
    }

    #[cfg(feature = "alu")]
    pub fn indexed_adc(
        &mut self,
        index_start: usize,
        index_length: usize,
        value_start: usize,
        value_length: usize,
        carry_index: usize,
        values: &[u8],
    ) -> usize {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.indexed_adc(
                index_start,
                index_length,
                value_start,
                value_length,
                carry_index,
                values,
            )
        } else {
            0
        }
    }

    #[cfg(feature = "alu")]
    pub fn indexed_sbc(
        &mut self,
        index_start: usize,
        index_length: usize,
        value_start: usize,
        value_length: usize,
        carry_index: usize,
        values: &[u8],
    ) -> usize {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.indexed_sbc(
                index_start,
                index_length,
                value_start,
                value_length,
                carry_index,
                values,
            )
        } else {
            0
        }
    }

    #[cfg(feature = "alu")]
    pub fn hash(&mut self, start: usize, length: usize, values: &[u8]) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.hash(start, length, values);
        }
    }

    #[cfg(feature = "alu")]
    pub fn c_phase_flip_if_less(
        &mut self,
        greater_perm: usize,
        start: usize,
        length: usize,
        flag_index: usize,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.c_phase_flip_if_less(
                greater_perm,
                start,
                length,
                flag_index,
            );
        }
    }

    #[cfg(feature = "alu")]
    pub fn phase_flip_if_less(
        &mut self,
        greater_perm: usize,
        start: usize,
        length: usize,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.phase_flip_if_less(
                greater_perm,
                start,
                length,
            );
        }
    }

    pub fn set_permutation(&mut self, perm: usize, phase_fac: complex) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.set_permutation(perm, phase_fac);
        }
    }

    pub fn uniformly_controlled_single_bit(
        &mut self,
        controls: &[usize],
        qubit_index: usize,
        mtrxs: &[complex],
        mtrx_skip_powers: &[usize],
        mtrx_skip_value_mask: usize,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.uniformly_controlled_single_bit(
                controls,
                qubit_index,
                mtrxs,
                mtrx_skip_powers,
                mtrx_skip_value_mask,
            );
        }
    }

    pub fn uniform_parity_rz(&mut self, mask: usize, angle: real1_f) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.uniform_parity_rz(mask, angle);
        }
    }

    pub fn c_uniform_parity_rz(
        &mut self,
        controls: &[usize],
        mask: usize,
        angle: real1_f,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.c_uniform_parity_rz(controls, mask, angle);
        }
    }

    pub fn prob(&self, qubit_index: usize) -> real1_f {
        if let Some(state_vec) = &self.state_vec {
            state_vec.prob(qubit_index)
        } else {
            0.0
        }
    }

    pub fn ctrl_or_anti_prob(
        &self,
        control_state: bool,
        control: usize,
        target: usize,
    ) -> real1_f {
        if let Some(state_vec) = &self.state_vec {
            state_vec.ctrl_or_anti_prob(control_state, control, target)
        } else {
            0.0
        }
    }

    pub fn prob_reg(&self, start: usize, length: usize, permutation: usize) -> real1_f {
        if let Some(state_vec) = &self.state_vec {
            state_vec.prob_reg(start, length, permutation)
        } else {
            0.0
        }
    }

    pub fn prob_mask(&self, mask: usize, permutation: usize) -> real1_f {
        if let Some(state_vec) = &self.state_vec {
            state_vec.prob_mask(mask, permutation)
        } else {
            0.0
        }
    }

    pub fn prob_parity(&self, mask: usize) -> real1_f {
        if let Some(state_vec) = &self.state_vec {
            state_vec.prob_parity(mask)
        } else {
            0.0
        }
    }

    pub fn m_all(&self) -> usize {
        if let Some(state_vec) = &self.state_vec {
            state_vec.m_all()
        } else {
            0
        }
    }

    pub fn force_m_parity(&mut self, mask: usize, result: bool, do_force: bool) -> bool {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.force_m_parity(mask, result, do_force)
        } else {
            false
        }
    }

    pub fn normalize_state(
        &mut self,
        nrm: real1_f,
        norm_thresh: real1_f,
        phase_arg: real1_f,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.normalize_state(nrm, norm_thresh, phase_arg);
        }
    }

    pub fn sum_sqr_diff(&self, to_compare: QInterfacePtr) -> real1_f {
        if let Some(state_vec) = &self.state_vec {
            if let Some(to_compare) = to_compare.downcast::<QEngineCPU>() {
                state_vec.sum_sqr_diff(&to_compare.state_vec)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    pub fn clone(&self) -> QInterfacePtr {
        if let Some(state_vec) = &self.state_vec {
            QEngineCPUPtr::new(state_vec.clone())
        } else {
            QEngineCPUPtr::new(None)
        }
    }

    fn get_expectation(&self, value_start: usize, value_length: usize) -> real1_f {
        if let Some(state_vec) = &self.state_vec {
            state_vec.get_expectation(value_start, value_length)
        } else {
            0.0
        }
    }

    fn alloc_state_vec(&self, elem_count: usize) -> StateVectorPtr {
        StateVectorPtr::new(elem_count)
    }

    fn reset_state_vec(&mut self, sv: StateVectorPtr) {
        self.state_vec = Some(sv);
    }

    fn dispatch(&self, work_item_count: usize, fn: DispatchFn) {
        if let Some(dispatch_queue) = &self.dispatch_queue {
            if work_item_count >= pow2(GetPreferredConcurrencyPower())
                && work_item_count < self.get_stride()
            {
                dispatch_queue.dispatch(fn);
            } else {
                self.finish();
                fn();
            }
        } else {
            fn();
        }
    }

    fn decompose_dispose(&mut self, start: usize, length: usize, dest: QEngineCPUPtr) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.decompose_dispose(start, length, &dest.state_vec);
        }
    }

    fn apply_2x2(
        &mut self,
        offset1: usize,
        offset2: usize,
        mtrx: &[complex],
        bit_count: usize,
        q_powers_sorted: &[usize],
        do_calc_norm: bool,
        norm_thresh: real1_f,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.apply_2x2(
                offset1,
                offset2,
                mtrx,
                bit_count,
                q_powers_sorted,
                do_calc_norm,
                norm_thresh,
            );
        }
    }

    fn update_running_norm(&mut self, norm_thresh: real1_f) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.update_running_norm(norm_thresh);
        }
    }

    fn apply_m(&mut self, mask: usize, result: usize, nrm: complex) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.apply_m(mask, result, nrm);
        }
    }

    #[cfg(feature = "alu")]
    fn incdecc(
        &mut self,
        to_mod: usize,
        in_out_start: usize,
        length: usize,
        carry_index: usize,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.incdecc(to_mod, in_out_start, length, carry_index);
        }
    }

    #[cfg(feature = "alu")]
    fn incdecsc(
        &mut self,
        to_mod: usize,
        in_out_start: usize,
        length: usize,
        carry_index: usize,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.incdecsc(to_mod, in_out_start, length, carry_index);
        }
    }

    #[cfg(feature = "alu")]
    fn incdecsc_with_overflow(
        &mut self,
        to_mod: usize,
        in_out_start: usize,
        length: usize,
        overflow_index: usize,
        carry_index: usize,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.incdecsc_with_overflow(
                to_mod,
                in_out_start,
                length,
                overflow_index,
                carry_index,
            );
        }
    }

    #[cfg(feature = "alu")]
    fn incdec_bcd(&mut self, to_mod: usize, in_out_start: usize, length: usize, carry_index: usize) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.incdec_bcd(to_mod, in_out_start, length, carry_index);
        }
    }

    #[cfg(feature = "alu")]
    fn mod_n_out(
        &mut self,
        kernel_fn: MFn,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
        inverse: bool,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.mod_n_out(
                kernel_fn,
                mod_n,
                in_start,
                out_start,
                length,
                inverse,
            );
        }
    }

    #[cfg(feature = "alu")]
    fn c_mod_n_out(
        &mut self,
        kernel_fn: MFn,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
        controls: &[usize],
        inverse: bool,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.c_mod_n_out(
                kernel_fn,
                mod_n,
                in_start,
                out_start,
                length,
                controls,
                inverse,
            );
        }
    }
}

pub type QEngineCPUPtr = Arc<QEngineCPU>;

pub struct StateVector {
    // implementation details
}

impl StateVector {
    pub fn new(elem_count: usize) -> Self {
        Self {
            // implementation details
        }
    }

    pub fn first_nonzero_phase(&self) -> complex {
        // implementation details
    }

    pub fn get_amplitude_page(&self, page_ptr: *mut complex, offset: usize, length: usize) {
        // implementation details
    }

    pub fn set_amplitude_page(&mut self, page_ptr: *const complex, offset: usize, length: usize) {
        // implementation details
    }

    pub fn set_amplitude_page_from_engine(
        &mut self,
        page_engine: &Option<Arc<StateVector>>,
        src_offset: usize,
        dst_offset: usize,
        length: usize,
    ) {
        // implementation details
    }

    pub fn shuffle_buffers(&mut self, engine: &Option<Arc<StateVector>>) {
        // implementation details
    }

    pub fn copy_state_vec(&mut self, src: &Option<Arc<StateVector>>) {
        // implementation details
    }

    pub fn compose(&mut self, to_copy: &Option<Arc<StateVector>>) -> usize {
        // implementation details
    }

    pub fn decompose(&mut self, start: usize, dest: &Option<Arc<StateVector>>) {
        // implementation details
    }

    pub fn dispose(&mut self, start: usize, length: usize) {
        // implementation details
    }

    pub fn dispose_with_perm(&mut self, start: usize, length: usize, disposed_perm: usize) {
        // implementation details
    }

    pub fn allocate(&mut self, start: usize, length: usize) -> usize {
        // implementation details
    }

    pub fn x_mask(&mut self, mask: usize) {
        // implementation details
    }

    pub fn phase_parity(&mut self, radians: real1_f, mask: usize) {
        // implementation details
    }

    pub fn rol(&mut self, shift: usize, start: usize, length: usize) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn inc(&mut self, to_add: usize, start: usize, length: usize) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn cinc(&mut self, to_add: usize, in_out_start: usize, length: usize, controls: &[usize]) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn incs(&mut self, to_add: usize, start: usize, length: usize, overflow_index: usize) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn incbcd(&mut self, to_add: usize, start: usize, length: usize) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn mul(
        &mut self,
        to_mul: usize,
        in_out_start: usize,
        carry_start: usize,
        length: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn div(
        &mut self,
        to_div: usize,
        in_out_start: usize,
        carry_start: usize,
        length: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn mul_mod_n_out(
        &mut self,
        to_mul: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn imul_mod_n_out(
        &mut self,
        to_mul: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn pow_mod_n_out(
        &mut self,
        base: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn cmul(
        &mut self,
        to_mul: usize,
        in_out_start: usize,
        carry_start: usize,
        length: usize,
        controls: &[usize],
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn cdiv(
        &mut self,
        to_div: usize,
        in_out_start: usize,
        carry_start: usize,
        length: usize,
        controls: &[usize],
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn cmul_mod_n_out(
        &mut self,
        to_mul: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
        controls: &[usize],
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn cimul_mod_n_out(
        &mut self,
        to_mul: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
        controls: &[usize],
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn cpow_mod_n_out(
        &mut self,
        base: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
        controls: &[usize],
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn full_add(
        &mut self,
        input_bit1: usize,
        input_bit2: usize,
        carry_in_sum_out: usize,
        carry_out: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn ifull_add(
        &mut self,
        input_bit1: usize,
        input_bit2: usize,
        carry_in_sum_out: usize,
        carry_out: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn indexed_lda(
        &mut self,
        index_start: usize,
        index_length: usize,
        value_start: usize,
        value_length: usize,
        values: &[u8],
        reset_value: bool,
    ) -> usize {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn indexed_adc(
        &mut self,
        index_start: usize,
        index_length: usize,
        value_start: usize,
        value_length: usize,
        carry_index: usize,
        values: &[u8],
    ) -> usize {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn indexed_sbc(
        &mut self,
        index_start: usize,
        index_length: usize,
        value_start: usize,
        value_length: usize,
        carry_index: usize,
        values: &[u8],
    ) -> usize {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn hash(&mut self, start: usize, length: usize, values: &[u8]) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn c_phase_flip_if_less(
        &mut self,
        greater_perm: usize,
        start: usize,
        length: usize,
        flag_index: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn phase_flip_if_less(
        &mut self,
        greater_perm: usize,
        start: usize,
        length: usize,
    ) {
        // implementation details
    }

    pub fn set_permutation(&mut self, perm: usize, phase_fac: complex) {
        // implementation details
    }

    pub fn uniformly_controlled_single_bit(
        &mut self,
        controls: &[usize],
        qubit_index: usize,
        mtrxs: &[complex],
        mtrx_skip_powers: &[usize],
        mtrx_skip_value_mask: usize,
    ) {
        // implementation details
    }

    pub fn uniform_parity_rz(&mut self, mask: usize, angle: real1_f) {
        // implementation details
    }

    pub fn c_uniform_parity_rz(
        &mut self,
        controls: &[usize],
        mask: usize,
        angle: real1_f,
    ) {
        // implementation details
    }

    pub fn prob(&self, qubit_index: usize) -> real1_f {
        // implementation details
    }

    pub fn ctrl_or_anti_prob(
        &self,
        control_state: bool,
        control: usize,
        target: usize,
    ) -> real1_f {
        // implementation details
    }

    pub fn prob_reg(&self, start: usize, length: usize, permutation: usize) -> real1_f {
        // implementation details
    }

    pub fn prob_mask(&self, mask: usize, permutation: usize) -> real1_f {
        // implementation details
    }

    pub fn prob_parity(&self, mask: usize) -> real1_f {
        // implementation details
    }

    pub fn m_all(&self) -> usize {
        // implementation details
    }

    pub fn force_m_parity(&mut self, mask: usize, result: bool, do_force: bool) -> bool {
        // implementation details
    }

    pub fn normalize_state(
        &mut self,
        nrm: real1_f,
        norm_thresh: real1_f,
        phase_arg: real1_f,
    ) {
        // implementation details
    }

    pub fn sum_sqr_diff(&self, to_compare: &Option<Arc<StateVector>>) -> real1_f {
        // implementation details
    }

    pub fn clone(&self) -> Option<Arc<StateVector>> {
        // implementation details
    }
}

pub type StateVectorPtr = Arc<StateVector>;

pub struct DispatchQueue {
    // implementation details
}

impl DispatchQueue {
    pub fn new() -> Self {
        Self {
            // implementation details
        }
    }

    pub fn finish(&self) {
        // implementation details
    }

    pub fn is_finished(&self) -> bool {
        // implementation details
    }

    pub fn dump(&self) {
        // implementation details
    }

    pub fn dispatch(&self, fn: DispatchFn) {
        // implementation details
    }
}

pub type DispatchFn = Box<dyn Fn() + Send>;

pub struct QInterface {
    // implementation details
}

impl QInterface {
    pub fn new() -> Self {
        Self {
            // implementation details
        }
    }

    pub fn first_nonzero_phase(&self) -> complex {
        // implementation details
    }

    pub fn get_amplitude_page(&self, page_ptr: *mut complex, offset: usize, length: usize) {
        // implementation details
    }

    pub fn set_amplitude_page(&mut self, page_ptr: *const complex, offset: usize, length: usize) {
        // implementation details
    }

    pub fn set_amplitude_page_from_engine(
        &mut self,
        page_engine: &Option<Arc<StateVector>>,
        src_offset: usize,
        dst_offset: usize,
        length: usize,
    ) {
        // implementation details
    }

    pub fn shuffle_buffers(&mut self, engine: &Option<Arc<StateVector>>) {
        // implementation details
    }

    pub fn copy_state_vec(&mut self, src: &Option<Arc<StateVector>>) {
        // implementation details
    }

    pub fn compose(&mut self, to_copy: &Option<Arc<StateVector>>) -> usize {
        // implementation details
    }

    pub fn decompose(&mut self, start: usize, dest: &Option<Arc<StateVector>>) {
        // implementation details
    }

    pub fn dispose(&mut self, start: usize, length: usize) {
        // implementation details
    }

    pub fn dispose_with_perm(&mut self, start: usize, length: usize, disposed_perm: usize) {
        // implementation details
    }

    pub fn allocate(&mut self, start: usize, length: usize) -> usize {
        // implementation details
    }

    pub fn x_mask(&mut self, mask: usize) {
        // implementation details
    }

    pub fn phase_parity(&mut self, radians: real1_f, mask: usize) {
        // implementation details
    }

    pub fn rol(&mut self, shift: usize, start: usize, length: usize) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn inc(&mut self, to_add: usize, start: usize, length: usize) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn cinc(&mut self, to_add: usize, in_out_start: usize, length: usize, controls: &[usize]) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn incs(&mut self, to_add: usize, start: usize, length: usize, overflow_index: usize) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn incbcd(&mut self, to_add: usize, start: usize, length: usize) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn mul(
        &mut self,
        to_mul: usize,
        in_out_start: usize,
        carry_start: usize,
        length: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn div(
        &mut self,
        to_div: usize,
        in_out_start: usize,
        carry_start: usize,
        length: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn mul_mod_n_out(
        &mut self,
        to_mul: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn imul_mod_n_out(
        &mut self,
        to_mul: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn pow_mod_n_out(
        &mut self,
        base: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn cmul(
        &mut self,
        to_mul: usize,
        in_out_start: usize,
        carry_start: usize,
        length: usize,
        controls: &[usize],
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn cdiv(
        &mut self,
        to_div: usize,
        in_out_start: usize,
        carry_start: usize,
        length: usize,
        controls: &[usize],
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn cmul_mod_n_out(
        &mut self,
        to_mul: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
        controls: &[usize],
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn cimul_mod_n_out(
        &mut self,
        to_mul: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
        controls: &[usize],
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn cpow_mod_n_out(
        &mut self,
        base: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
        controls: &[usize],
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn full_add(
        &mut self,
        input_bit1: usize,
        input_bit2: usize,
        carry_in_sum_out: usize,
        carry_out: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn ifull_add(
        &mut self,
        input_bit1: usize,
        input_bit2: usize,
        carry_in_sum_out: usize,
        carry_out: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn indexed_lda(
        &mut self,
        index_start: usize,
        index_length: usize,
        value_start: usize,
        value_length: usize,
        values: &[u8],
        reset_value: bool,
    ) -> usize {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn indexed_adc(
        &mut self,
        index_start: usize,
        index_length: usize,
        value_start: usize,
        value_length: usize,
        carry_index: usize,
        values: &[u8],
    ) -> usize {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn indexed_sbc(
        &mut self,
        index_start: usize,
        index_length: usize,
        value_start: usize,
        value_length: usize,
        carry_index: usize,
        values: &[u8],
    ) -> usize {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn hash(&mut self, start: usize, length: usize, values: &[u8]) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn c_phase_flip_if_less(
        &mut self,
        greater_perm: usize,
        start: usize,
        length: usize,
        flag_index: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn phase_flip_if_less(
        &mut self,
        greater_perm: usize,
        start: usize,
        length: usize,
    ) {
        // implementation details
    }

    pub fn set_permutation(&mut self, perm: usize, phase_fac: complex) {
        // implementation details
    }

    pub fn uniformly_controlled_single_bit(
        &mut self,
        controls: &[usize],
        qubit_index: usize,
        mtrxs: &[complex],
        mtrx_skip_powers: &[usize],
        mtrx_skip_value_mask: usize,
    ) {
        // implementation details
    }

    pub fn uniform_parity_rz(&mut self, mask: usize, angle: real1_f) {
        // implementation details
    }

    pub fn c_uniform_parity_rz(
        &mut self,
        controls: &[usize],
        mask: usize,
        angle: real1_f,
    ) {
        // implementation details
    }

    pub fn prob(&self, qubit_index: usize) -> real1_f {
        // implementation details
    }

    pub fn ctrl_or_anti_prob(
        &self,
        control_state: bool,
        control: usize,
        target: usize,
    ) -> real1_f {
        // implementation details
    }

    pub fn prob_reg(&self, start: usize, length: usize, permutation: usize) -> real1_f {
        // implementation details
    }

    pub fn prob_mask(&self, mask: usize, permutation: usize) -> real1_f {
        // implementation details
    }

    pub fn prob_parity(&self, mask: usize) -> real1_f {
        // implementation details
    }

    pub fn m_all(&self) -> usize {
        // implementation details
    }

    pub fn force_m_parity(&mut self, mask: usize, result: bool, do_force: bool) -> bool {
        // implementation details
    }

    pub fn normalize_state(
        &mut self,
        nrm: real1_f,
        norm_thresh: real1_f,
        phase_arg: real1_f,
    ) {
        // implementation details
    }

    pub fn sum_sqr_diff(&self, to_compare: &Option<Arc<StateVector>>) -> real1_f {
        // implementation details
    }

    pub fn clone(&self) -> Option<Arc<StateVector>> {
        // implementation details
    }
}

pub type QInterfacePtr = Arc<QInterface>;

pub struct QEngine {
    // implementation details
}

impl QEngine {
    pub fn new() -> Self {
        Self {
            // implementation details
        }
    }

    pub fn first_nonzero_phase(&self) -> complex {
        // implementation details
    }

    pub fn get_amplitude_page(&self, page_ptr: *mut complex, offset: usize, length: usize) {
        // implementation details
    }

    pub fn set_amplitude_page(&mut self, page_ptr: *const complex, offset: usize, length: usize) {
        // implementation details
    }

    pub fn set_amplitude_page_from_engine(
        &mut self,
        page_engine: &Option<Arc<StateVector>>,
        src_offset: usize,
        dst_offset: usize,
        length: usize,
    ) {
        // implementation details
    }

    pub fn shuffle_buffers(&mut self, engine: &Option<Arc<StateVector>>) {
        // implementation details
    }

    pub fn copy_state_vec(&mut self, src: &Option<Arc<StateVector>>) {
        // implementation details
    }

    pub fn compose(&mut self, to_copy: &Option<Arc<StateVector>>) -> usize {
        // implementation details
    }

    pub fn decompose(&mut self, start: usize, dest: &Option<Arc<StateVector>>) {
        // implementation details
    }

    pub fn dispose(&mut self, start: usize, length: usize) {
        // implementation details
    }

    pub fn dispose_with_perm(&mut self, start: usize, length: usize, disposed_perm: usize) {
        // implementation details
    }

    pub fn allocate(&mut self, start: usize, length: usize) -> usize {
        // implementation details
    }

    pub fn x_mask(&mut self, mask: usize) {
        // implementation details
    }

    pub fn phase_parity(&mut self, radians: real1_f, mask: usize) {
        // implementation details
    }

    pub fn rol(&mut self, shift: usize, start: usize, length: usize) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn inc(&mut self, to_add: usize, start: usize, length: usize) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn cinc(&mut self, to_add: usize, in_out_start: usize, length: usize, controls: &[usize]) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn incs(&mut self, to_add: usize, start: usize, length: usize, overflow_index: usize) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn incbcd(&mut self, to_add: usize, start: usize, length: usize) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn mul(
        &mut self,
        to_mul: usize,
        in_out_start: usize,
        carry_start: usize,
        length: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn div(
        &mut self,
        to_div: usize,
        in_out_start: usize,
        carry_start: usize,
        length: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn mul_mod_n_out(
        &mut self,
        to_mul: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn imul_mod_n_out(
        &mut self,
        to_mul: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn pow_mod_n_out(
        &mut self,
        base: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn cmul(
        &mut self,
        to_mul: usize,
        in_out_start: usize,
        carry_start: usize,
        length: usize,
        controls: &[usize],
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn cdiv(
        &mut self,
        to_div: usize,
        in_out_start: usize,
        carry_start: usize,
        length: usize,
        controls: &[usize],
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn cmul_mod_n_out(
        &mut self,
        to_mul: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
        controls: &[usize],
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn cimul_mod_n_out(
        &mut self,
        to_mul: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
        controls: &[usize],
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn cpow_mod_n_out(
        &mut self,
        base: usize,
        mod_n: usize,
        in_start: usize,
        out_start: usize,
        length: usize,
        controls: &[usize],
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn full_add(
        &mut self,
        input_bit1: usize,
        input_bit2: usize,
        carry_in_sum_out: usize,
        carry_out: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn ifull_add(
        &mut self,
        input_bit1: usize,
        input_bit2: usize,
        carry_in_sum_out: usize,
        carry_out: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn indexed_lda(
        &mut self,
        index_start: usize,
        index_length: usize,
        value_start: usize,
        value_length: usize,
        values: &[u8],
        reset_value: bool,
    ) -> usize {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn indexed_adc(
        &mut self,
        index_start: usize,
        index_length: usize,
        value_start: usize,
        value_length: usize,
        carry_index: usize,
        values: &[u8],
    ) -> usize {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn indexed_sbc(
        &mut self,
        index_start: usize,
        index_length: usize,
        value_start: usize,
        value_length: usize,
        carry_index: usize,
        values: &[u8],
    ) -> usize {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn hash(&mut self, start: usize, length: usize, values: &[u8]) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn c_phase_flip_if_less(
        &mut self,
        greater_perm: usize,
        start: usize,
        length: usize,
        flag_index: usize,
    ) {
        // implementation details
    }

    #[cfg(feature = "alu")]
    pub fn phase_flip_if_less(
        &mut self,
        greater_perm: usize,
        start: usize,
        length: usize,
    ) {
        // implementation details
    }

    pub fn set_permutation(&mut self, perm: usize, phase_fac: complex) {
        // implementation details
    }

    pub fn uniformly_controlled_single_bit(
        &mut self,
        controls: &[usize],
        qubit_index: usize,
        mtrxs: &[complex],
        mtrx_skip_powers: &[usize],
        mtrx_skip_value_mask: usize,
    ) {
        // implementation details
    }

    pub fn uniform_parity_rz(&mut self, mask: usize, angle: real1_f) {
        // implementation details
    }

    pub fn c_uniform_parity_rz(
        &mut self,
        controls: &[usize],
        mask: usize,
        angle: real1_f,
    ) {
        // implementation details
    }

    pub fn prob(&self, qubit_index: usize) -> real1_f {
        // implementation details
    }

    pub fn ctrl_or_anti_prob(
        &self,
        control_state: bool,
        control: usize,
        target: usize,
    ) -> real1_f {
        // implementation details
    }

    pub fn prob_reg(&self, start: usize, length: usize, permutation: usize) -> real1_f {
        // implementation details
    }

    pub fn prob_mask(&self, mask: usize, permutation: usize) -> real1_f {
        // implementation details
    }

    pub fn prob_parity(&self, mask: usize) -> real1_f {
        // implementation details
    }

    pub fn m_all(&self) -> usize {
        // implementation details
    }

    pub fn force_m_parity(&mut self, mask: usize, result: bool, do_force: bool) -> bool {
        // implementation details
    }

    pub fn normalize_state(
        &mut self,
        nrm: real1_f,
        norm_thresh: real1_f,
        phase_arg: real1_f,
    ) {
        // implementation details
    }

    pub fn sum_sqr_diff(&self, to_compare: &Option<Arc<StateVector>>) -> real1_f {
        // implementation details
    }

    pub fn clone(&self) -> Option<Arc<StateVector>> {
        // implementation details
    }
}

pub type QEnginePtr = Arc<QEngine>;

pub struct QEngineCPU {
    is_sparse: bool,
    max_qubits: usize,
    state_vec: Option<Arc<StateVector>>,
    dispatch_queue: Option<DispatchQueue>,
}

impl QEngineCPU {
    pub fn new(
        q_bit_count: usize,
        init_state: usize,
        rgp: Option<qrack_rand_gen_ptr>,
        phase_fac: complex,
        do_norm: bool,
        random_global_phase: bool,
        ignored: bool,
        ignored2: i64,
        use_hardware_rng: bool,
        use_sparse_state_vec: bool,
        norm_thresh: real1_f,
        ignored3: Vec<i64>,
        ignored4: usize,
        ignored5: real1_f,
    ) -> Self {
        Self {
            is_sparse: use_sparse_state_vec,
            max_qubits: q_bit_count,
            state_vec: None,
            dispatch_queue: None,
        }
    }

    pub fn finish(&self) {
        if let Some(dispatch_queue) = &self.dispatch_queue {
            dispatch_queue.finish();
        }
    }

    pub fn is_finished(&self) -> bool {
        if let Some(dispatch_queue) = &self.dispatch_queue {
            dispatch_queue.is_finished()
        } else {
            true
        }
    }

    pub fn dump(&self) {
        if let Some(dispatch_queue) = &self.dispatch_queue {
            dispatch_queue.dump();
        }
    }

    pub fn set_device(&self, d_id: i64) {}

    pub fn first_nonzero_phase(&self) -> complex {
        if let Some(state_vec) = &self.state_vec {
            state_vec.first_nonzero_phase()
        } else {
            ZERO_R1_F
        }
    }

    pub fn zero_amplitudes(&mut self) {
        self.dump();
        self.state_vec = None;
        self.running_norm = ZERO_R1;
    }

    pub fn free_state_vec(&mut self, sv: Option<*mut complex>) {
        self.state_vec = None;
    }

    pub fn is_zero_amplitude(&self) -> bool {
        self.state_vec.is_none()
    }

    pub fn get_amplitude_page(&self, page_ptr: *mut complex, offset: usize, length: usize) {
        if let Some(state_vec) = &self.state_vec {
            state_vec.get_amplitude_page(page_ptr, offset, length);
        }
    }

    pub fn set_amplitude_page(&mut self, page_ptr: *const complex, offset: usize, length: usize) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.set_amplitude_page(page_ptr, offset, length);
        }
    }

    pub fn set_amplitude_page_from_engine(
        &mut self,
        page_engine_ptr: QEnginePtr,
        src_offset: usize,
        dst_offset: usize,
        length: usize,
    ) {
        if let Some(state_vec) = &mut self.state_vec {
            if let Some(page_engine) = page_engine_ptr.downcast::<QEngineCPU>() {
                state_vec.set_amplitude_page_from_engine(
                    &page_engine.state_vec,
                    src_offset,
                    dst_offset,
                    length,
                );
            }
        }
    }

    pub fn shuffle_buffers(&mut self, engine: QEnginePtr) {
        if let Some(state_vec) = &mut self.state_vec {
            if let Some(engine) = engine.downcast::<QEngineCPU>() {
                state_vec.shuffle_buffers(&engine.state_vec);
            }
        }
    }

    pub fn copy_state_vec(&mut self, src: QEnginePtr) {
        if let Some(state_vec) = &mut self.state_vec {
            if let Some(src) = src.downcast::<QEngineCPU>() {
                state_vec.copy_state_vec(&src.state_vec);
            }
        }
    }

    pub fn clone_empty(&self) -> QEnginePtr {
        QEngineCPUPtr::new(self.max_qubits)
    }

    pub fn queue_set_do_normalize(&self, do_norm: bool) {
        if let Some(dispatch_queue) = &self.dispatch_queue {
            dispatch_queue.dispatch(|| {
                self.do_normalize = do_norm;
            });
        }
    }

    pub fn queue_set_running_norm(&self, running_nrm: real1_f) {
        if let Some(dispatch_queue) = &self.dispatch_queue {
            dispatch_queue.dispatch(|| {
                self.running_norm = running_nrm;
            });
        }
    }

    pub fn set_quantum_state(&mut self, input_state: *const complex) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.set_quantum_state(input_state);
        }
    }

    pub fn get_quantum_state(&self, output_state: *mut complex) {
        if let Some(state_vec) = &self.state_vec {
            state_vec.get_quantum_state(output_state);
        }
    }

    pub fn get_probs(&self, output_probs: *mut real1) {
        if let Some(state_vec) = &self.state_vec {
            state_vec.get_probs(output_probs);
        }
    }

    pub fn get_amplitude(&self, perm: usize) -> complex {
        if let Some(state_vec) = &self.state_vec {
            state_vec.get_amplitude(perm)
        } else {
            ZERO_C
        }
    }

    pub fn set_amplitude(&mut self, perm: usize, amp: complex) {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.set_amplitude(perm, amp);
        }
    }

    pub fn compose(&mut self, to_copy: QEngineCPUPtr) -> usize {
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.compose(&to_copy.state_vec)
        } else {
            0
        }
    }

    pub fn decompose(&mut self, start: usize, dest: QInterfacePtr) {
        if let Some(state_vec) = &mut self.state_vec {
            if let Some(dest) = dest.downcast::<QEngineCPU>() {
                state_vec.decompose(start, &dest.state_vec);
            }
        }
    }

    pub fn dispose(&mut

