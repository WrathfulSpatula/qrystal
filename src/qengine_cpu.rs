use std::sync::Arc;
use std::collections::HashMap;
use std::ptr;
use std::slice;
use std::cmp::Ordering;
use std::f64::consts::PI;
use QEngine

pub struct QEngineCPU {
    is_sparse: bool,
    max_qubits: usize,
    state_vec: Option<Arc<StateVector>>,
    dispatch_queue: Option<DispatchQueue>,
}

impl QEngineCPU {
    fn new(
        qBitCount: i32,
        initState: i64,
        rgp: *mut c_void,
        phaseFac: c_double,
        doNorm: bool,
        randomGlobalPhase: bool,
        useHostMem: bool,
        deviceID: i64,
        useHardwareRNG: bool,
        useSparseStateVec: bool,
        norm_thresh: f64,
        devList: Vec<i64>,
        qubitThreshold: i32,
        sep_thresh: f64,
    ) -> Result<Self, &'static str> {
        let maxQubits = match std::env::var("QRACK_MAX_CPU_QB") {
            Ok(val) => val.parse::<i32>().unwrap(),
            Err(_) => -1,
        };

        if qBitCount > maxQubits {
            return Err(
                "Cannot instantiate a QEngineCPU with greater capacity than environment variable QRACK_MAX_CPU_QB."
            );
        }

        if qBitCount == 0 {
            ZeroAmplitudes();
            return Ok(Self {
                qBitCount,
                rgp,
                doNorm,
                randomGlobalPhase,
                useHostMem,
                deviceID,
                useHardwareRNG,
                useSparseStateVec,
                norm_thresh,
                devList,
                qubitThreshold,
                sep_thresh,
                isSparse: useSparseStateVec,
                maxQubits,
            });
        }

        let stateVec = AllocStateVec(maxQPowerOcl);
        stateVec.clear();

        if phaseFac == CMPLX_DEFAULT_ARG {
            stateVec.write(initState as bitCapIntOcl, GetNonunitaryPhase());
        } else {
            stateVec.write(initState as bitCapIntOcl, phaseFac);
        }

        Ok(Self {
            qBitCount,
            rgp,
            doNorm,
            randomGlobalPhase,
            useHostMem,
            deviceID,
            useHardwareRNG,
            useSparseStateVec,
            norm_thresh,
            devList,
            qubitThreshold,
            sep_thresh,
            isSparse: useSparseStateVec,
            maxQubits,
        })
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

    pub fn shuffle_buffers(&mut self, engine: QEnginePtr) {
        if let Some(state_vec) = &mut self.state_vec {
            if let Some(engine) = engine.downcast::<QEngineCPU>() {
                state_vec.shuffle_buffers(&engine.state_vec);
            }
        }
    }

    fn copy_state_vec(&mut self, src: &QEngineCPU) {
        if self.qubit_count != src.qubit_count {
            panic!("QEngineCPU::CopyStateVec argument size differs from this!");
        }
        if src.is_zero_amplitude() {
            self.zero_amplitudes();
            return;
        }
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.clear();
        }
        if src.is_sparse {
            let mut sv = vec![Complex::new(0.0, 0.0); self.max_q_power];
            src.get_quantum_state(&mut sv);
            self.set_quantum_state(&sv);
        } else {
            let state_vec = self.state_vec.as_mut().unwrap();
            src.get_quantum_state(&mut state_vec.amplitudes);
        }
        self.running_norm = src.running_norm;
    }

    fn get_amplitude(&self, perm: usize) -> Complex {
        if perm >= self.max_q_power {
            panic!("QEngineCPU::GetAmplitude argument out-of-bounds!");
        }
        if let Some(state_vec) = &self.state_vec {
            state_vec[perm]
        } else {
            Complex::new(0.0, 0.0)
        }
    }

    fn set_amplitude(&mut self, perm: usize, amp: Complex) {
        if perm >= self.max_q_power {
            panic!("QEngineCPU::SetAmplitude argument out-of-bounds!");
        }
        if let Some(state_vec) = &mut self.state_vec {
            if self.running_norm != 0.0 {
                self.running_norm += amp.norm() - state_vec[perm].norm();
            }
            state_vec[perm] = amp;
        } else if amp.norm() != 0.0 {
            self.reset_state_vec(vec![Complex::new(0.0, 0.0); self.max_q_power]);
        }
    }

    fn set_permutation(&mut self, perm: usize, phase_fac: Complex) {
        self.dump();
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.clear();
            if phase_fac == Complex::new(0.0, 0.0) {
                let phase = if self.rand_global_phase {
                    let angle = rand() * 2.0 * PI;
                    Complex::new(angle.cos(), angle.sin())
                } else {
                    Complex::new(1.0, 0.0)
                };
                state_vec[perm] = phase;
            } else {
                let nrm = phase_fac.norm();
                state_vec[perm] = phase_fac / nrm;
            }
        } else {
            self.reset_state_vec(vec![Complex::new(0.0, 0.0); self.max_q_power]);
            if let Some(state_vec) = &mut self.state_vec {
                if phase_fac == Complex::new(0.0, 0.0) {
                    let phase = if self.rand_global_phase {
                        let angle = rand() * 2.0 * PI;
                        Complex::new(angle.cos(), angle.sin())
                    } else {
                        Complex::new(1.0, 0.0)
                    };
                    state_vec[perm] = phase;
                } else {
                    let nrm = phase_fac.norm();
                    state_vec[perm] = phase_fac / nrm;
                }
            }
        }
        self.running_norm = 1.0;
    }

    fn set_quantum_state(&mut self, input_state: &[Complex]) {
        self.dump();
        if let Some(state_vec) = &mut self.state_vec {
            state_vec.copy_from_slice(input_state);
        } else {
            self.reset_state_vec(input_state.to_vec());
        }
        self.running_norm = 0.0;
    }

    fn get_quantum_state(&self, output_state: &mut [Complex]) {
        if let Some(state_vec) = &self.state_vec {
            if self.do_normalize {
                self.normalize_state();
            }
            self.finish();
            output_state.copy_from_slice(&state_vec);
        } else {
            for amplitude in output_state.iter_mut() {
                *amplitude = Complex::new(0.0, 0.0);
            }
        }
    }

    fn get_probs(&self, output_probs: &mut [f64]) {
        if let Some(state_vec) = &self.state_vec {
            if self.do_normalize {
                self.normalize_state();
            }
            self.finish();
            for (i, amplitude) in state_vec.iter().enumerate() {
                output_probs[i] = amplitude.norm();
            }
        } else {
            for prob in output_probs.iter_mut() {
                *prob = 0.0;
            }
        }
    }
    
    pub fn x_mask(&self, mask: u64) {
        if mask >= self.max_q_power {
            panic!("QEngineCPU::XMask mask out-of-bounds!");
        }
        self.check_zero_skip();
        if mask == 0 {
            return;
        }
        if mask.is_power_of_two() {
            self.x(mask.trailing_zeros() as usize);
            return;
        }
        if self.state_vec.is_sparse() {
            QInterface::x_mask(mask);
            return;
        }
        self.dispatch(self.max_q_power_ocl, |_, _| {
            let mask_ocl = mask as u32;
            let other_mask = (self.max_q_power_ocl - 1) ^ mask_ocl;
            let fn_par = |lcv: u32, _: u32| {
                let other_res = lcv & other_mask;
                let set_int = lcv & mask_ocl;
                let reset_int = set_int ^ mask_ocl;
                if set_int < reset_int {
                    return;
                }
                let mut set_int = set_int | other_res;
                let mut reset_int = reset_int | other_res;
                let y0 = self.state_vec.read(reset_int);
                self.state_vec.write(reset_int, self.state_vec.read(set_int));
                self.state_vec.write(set_int, y0);
            };
            par_for(0, self.max_q_power_ocl, fn_par);
        });
    }

    pub fn phase_parity(&self, radians: f64, mask: u64) {
        if mask >= self.max_q_power {
            panic!("QEngineCPU::PhaseParity mask out-of-bounds!");
        }
        self.check_zero_skip();
        if mask == 0 {
            return;
        }
        if mask.is_power_of_two() {
            let phase_fac = Complex::from_polar(&1.0, &(radians / 2.0));
            self.phase(Complex::new(1.0, 0.0) / phase_fac, phase_fac, mask.trailing_zeros() as usize);
            return;
        }
        if self.state_vec.is_sparse() {
            QInterface::phase_parity(radians, mask);
            return;
        }
        self.dispatch(self.max_q_power_ocl, |_, _| {
            let parity_start_size = 4 * std::mem::size_of::<u32>();
            let phase_fac = Complex::from_polar(&1.0, &(radians / 2.0));
            let i_phase_fac = Complex::new(1.0, 0.0) / phase_fac;
            let mask_ocl = mask as u32;
            let other_mask = (self.max_q_power_ocl - 1) ^ mask_ocl;
            let fn_par = |lcv: u32, _: u32| {
                let other_res = lcv & other_mask;
                let set_int = lcv & mask_ocl;
                let mut v = set_int;
                let mut parity_size = parity_start_size;
                while parity_size > 0 {
                    v ^= v >> parity_size;
                    parity_size >>= 1;
                }
                v &= 1;
                let set_int = set_int | other_res;
                self.state_vec.write(set_int, if v != 0 { phase_fac } else { i_phase_fac } * self.state_vec.read(set_int));
            };
            par_for(0, self.max_q_power_ocl, fn_par);
        });
    }

    pub fn uniformly_controlled_single_bit(&self, controls: &[usize], qubit_index: usize, mtrxs: &[Complex<f64>], mtrx_skip_powers: &[u64], mtrx_skip_value_mask: u64) {
        self.check_zero_skip();
        if controls.is_empty() {
            self.mtrx(&mtrxs[(mtrx_skip_value_mask * 4) as usize..], qubit_index);
            return;
        }
        if qubit_index >= self.qubit_count {
            panic!("QEngineCPU::UniformlyControlledSingleBit qubitIndex is out-of-bounds!");
        }
        self.throw_if_qb_id_array_is_bad(controls, self.qubit_count, "QEngineCPU::UniformlyControlledSingleBit control is out-of-bounds!");
        let target_power = 1 << qubit_index;
        let q_powers: Vec<u64> = controls.iter().map(|&control| 1 << control).collect();
        let mtrx_skip_powers_ocl: Vec<u32> = mtrx_skip_powers.iter().map(|&i| i as u32).collect();
        let mtrx_skip_value_mask_ocl = mtrx_skip_value_mask as u32;
        let nrm = if self.running_norm > 0.0 { 1.0 / self.running_norm.sqrt() } else { 1.0 };
        let fn_par = |lcv: u32, _: u32| {
            let mut offset = 0;
            for (j, &control) in controls.iter().enumerate() {
                if lcv & q_powers[j] != 0 {
                    offset |= 1 << j;
                }
            }
            let mut i = 0;
            let mut i_high = offset;
            for &p in &mtrx_skip_powers_ocl {
                let i_low = i_high & (p - 1);
                i |= i_low;
                i_high = (i_high ^ i_low) << 1;
            }
            i |= i_high;
            let offset = (i | mtrx_skip_value_mask_ocl) * 4;
            let mut qubit = [Complex::new(0.0, 0.0); 2];
            let y0 = self.state_vec.read(lcv);
            qubit[1] = self.state_vec.read(lcv | target_power);
            qubit[0] = nrm * (mtrxs[0 + offset] * y0 + mtrxs[1 + offset] * qubit[1]);
            qubit[1] = nrm * (mtrxs[2 + offset] * y0 + mtrxs[3 + offset] * qubit[1]);
            self.state_vec.write2(lcv, qubit[0], lcv | target_power, qubit[1]);
        };
        self.finish();
        self.par_for_skip(0, self.max_q_power_ocl, target_power, 1, fn_par);
        if self.do_normalize {
            self.running_norm = 1.0;
        }
    }

    pub fn uniform_parity_rz(&self, mask: u64, angle: f64) {
        if mask >= self.max_q_power {
            panic!("QEngineCPU::UniformParityRZ mask out-of-bounds!");
        }
        self.check_zero_skip();
        self.dispatch(self.max_q_power_ocl, |_, _| {
            let cosine = angle.cos();
            let sine = angle.sin();
            let phase_fac = Complex::new(cosine, sine);
            let phase_fac_adj = Complex::new(cosine, -sine);
            let fn_par = |lcv: u32, _: u32| {
                let perm = lcv & mask as u32;
                let mut c = 0;
                let mut perm = perm;
                while perm != 0 {
                    perm &= perm - 1;
                    c += 1;
                }
                self.state_vec.write(lcv, self.state_vec.read(lcv) * if c & 1 != 0 { phase_fac } else { phase_fac_adj });
            };
            if self.state_vec.is_sparse() {
                self.par_for_set(self.cast_state_vec_sparse().iterable(), fn_par);
            } else {
                self.par_for(0, self.max_q_power_ocl, fn_par);
            }
        });
    }

    pub fn c_uniform_parity_rz(&self, c_controls: &[usize], mask: u64, angle: f64) {
        if c_controls.is_empty() {
            return self.uniform_parity_rz(mask, angle);
        }
        if mask >= self.max_q_power {
            panic!("QEngineCPU::CUniformParityRZ mask out-of-bounds!");
        }
        self.throw_if_qb_id_array_is_bad(c_controls, self.qubit_count, "QEngineCPU::CUniformParityRZ control is out-of-bounds!");
        self.check_zero_skip();
        let mut controls = c_controls.to_vec();
        controls.sort();
        self.dispatch(self.max_q_power_ocl >> c_controls.len(), |_, _| {
            let mut control_mask = 0;
            let mut control_powers: Vec<u32> = Vec::with_capacity(controls.len());
            for &control in &controls {
                control_powers.push(1 << control);
                control_mask |= 1 << control;
            }
            let cosine = angle.cos();
            let sine = angle.sin();
            let phase_fac = Complex::new(cosine, sine);
            let phase_fac_adj = Complex::new(cosine, -sine);
            let fn_par = |lcv: u32, _: u32| {
                let perm = lcv & mask as u32;
                let mut c = 0;
                let mut perm = perm;
                while perm != 0 {
                    perm &= perm - 1;
                    c += 1;
                }
                self.state_vec.write(control_mask | lcv, self.state_vec.read(control_mask | lcv) * if c & 1 != 0 { phase_fac } else { phase_fac_adj });
            };
            self.par_for_mask(0, self.max_q_power_ocl, &control_powers, fn_par);
        });
    }

    pub fn compose(&self, to_copy: &QEngineCPU) -> u32 {
        let result = self.qubit_count;
        if to_copy.qubit_count == 0 {
            return result;
        }
        let n_qubit_count = self.qubit_count + to_copy.qubit_count;
        if n_qubit_count > self.max_qubits {
            panic!("Cannot instantiate a QEngineCPU with greater capacity than environment variable QRACK_MAX_CPU_QB.");
        }
        if self.qubit_count == 0 {
            self.finish();
            self.set_qubit_count(to_copy.qubit_count);
            to_copy.finish();
            self.running_norm = to_copy.running_norm;
            if let Some(state_vec) = &to_copy.state_vec {
                state_vec = self.alloc_state_vec(to_copy.max_q_power_ocl);
                state_vec.copy(to_copy.state_vec);
            }
            return 0;
        }
        if to_copy.qubit_count == 0 {
            return self.qubit_count;
        }
        if self.state_vec.is_none() || to_copy.state_vec.is_none() {
            self.zero_amplitudes();
            self.set_qubit_count(n_qubit_count);
            return result;
        }
        let n_max_q_power = self.pow2_ocl(n_qubit_count);
        let start_mask = self.max_q_power_ocl - 1;
        let end_mask = (to_copy.max_q_power_ocl - 1) << self.qubit_count;
        if self.do_normalize {
            self.normalize_state();
        }
        self.finish();
        let n_state_vec = self.alloc_state_vec(n_max_q_power);
        self.state_vec.is_read_locked = false;
        let fn = |lcv: &bit_cap_int_ocl, cpu: &u32| {
            n_state_vec.write(
                lcv,
                self.state_vec.read(lcv & start_mask) * to_copy.state_vec.read((lcv & end_mask) >> self.qubit_count),
            );
        };
        if to_copy.do_normalize && to_copy.running_norm != ONE_R1 {
            to_copy.normalize_state();
        }
        to_copy.finish();
        if self.state_vec.is_sparse() || to_copy.state_vec.is_sparse() {
            self.par_for_sparse_compose(
                self.cast_state_vec_sparse().iterable(),
                to_copy.cast_state_vec_sparse().iterable(),
                self.qubit_count,
                fn,
            );
        } else {
            self.par_for(0, n_max_q_power, fn);
        }
        self.set_qubit_count(n_qubit_count);
        self.reset_state_vec(n_state_vec);
        return result;
    }

    pub fn compose_with_start(&self, to_copy: &QEngineCPU, start: u32) -> u32 {
        if start > self.qubit_count {
            panic!("QEngineCPU::Compose start index is out-of-bounds!");
        }
        if self.qubit_count == 0 {
            self.compose(to_copy);
            return 0;
        }
        if to_copy.qubit_count == 0 {
            return self.qubit_count;
        }
        let n_qubit_count = self.qubit_count + to_copy.qubit_count;
        if n_qubit_count > self.max_qubits {
            panic!("Cannot instantiate a QEngineCPU with greater capacity than environment variable QRACK_MAX_CPU_QB.");
        }
        if self.state_vec.is_none() || to_copy.state_vec.is_none() {
            self.zero_amplitudes();
            self.set_qubit_count(n_qubit_count);
            return start;
        }
        let o_qubit_count = to_copy.qubit_count;
        let n_max_q_power = self.pow2_ocl(n_qubit_count);
        let start_mask = self.pow2_mask_ocl(start);
        let mid_mask = self.bit_reg_mask_ocl(start, o_qubit_count);
        let end_mask = self.pow2_mask_ocl(self.qubit_count + o_qubit_count) & !(start_mask | mid_mask);
        if self.do_normalize {
            self.normalize_state();
        }
        self.finish();
        if to_copy.do_normalize {
            to_copy.normalize_state();
        }
        to_copy.finish();
        let n_state_vec = self.alloc_state_vec(n_max_q_power);
        self.state_vec.is_read_locked = false;
        self.par_for(0, n_max_q_power, |lcv: &bit_cap_int_ocl, cpu: &u32| {
            n_state_vec.write(
                lcv,
                self.state_vec.read((lcv & start_mask) | ((lcv & end_mask) >> o_qubit_count))
                    * to_copy.state_vec.read((lcv & mid_mask) >> start),
            );
        });
        self.set_qubit_count(n_qubit_count);
        self.reset_state_vec(n_state_vec);
        return start;
    }

    pub fn compose_multiple(&self, to_copy: Vec<QInterfacePtr>) -> HashMap<QInterfacePtr, u32> {
        let to_compose_count = to_copy.len();
        let mut n_qubit_count = self.qubit_count;
        let mut ret = HashMap::new();
        let mut offset = Vec::with_capacity(to_compose_count);
        let mut mask = Vec::with_capacity(to_compose_count);
        let start_mask = self.max_q_power_ocl - 1;
        if self.do_normalize {
            self.normalize_state();
        }
        self.finish();
        for i in 0..to_compose_count {
            let src = to_copy[i].downcast::<Qrack::QEngineCPU>().unwrap();
            if src.do_normalize {
                src.normalize_state();
            }
            src.finish();
            mask.push((src.max_q_power_ocl - 1) << n_qubit_count);
            offset.push(n_qubit_count);
            ret.insert(to_copy[i], n_qubit_count);
            n_qubit_count += src.get_qubit_count();
        }
        let n_max_q_power = self.pow2_ocl(n_qubit_count);
        let n_state_vec = self.alloc_state_vec(n_max_q_power);
        self.state_vec.is_read_locked = false;
        self.par_for(0, n_max_q_power, |lcv: &bit_cap_int_ocl, cpu: &u32| {
            n_state_vec.write(lcv, self.state_vec.read(lcv & start_mask));
            for j in 0..to_compose_count {
                let src = to_copy[j].downcast::<Qrack::QEngineCPU>().unwrap();
                n_state_vec.write(
                    lcv,
                    n_state_vec.read(lcv) * src.state_vec.read((lcv & mask[j]) >> offset[j]),
                );
            }
        });
        self.set_qubit_count(n_qubit_count);
        self.reset_state_vec(n_state_vec);
        return ret;
    }

    pub fn decompose_dispose(&self, start: u32, length: u32, destination: Option<QEngineCPUPtr>) {
        if self.is_bad_bit_range(start, length, self.qubit_count) {
            panic!("QEngineCPU::DecomposeDispose range is out-of-bounds!");
        }
        if length == 0 {
            return;
        }
        let n_length = self.qubit_count - length;
        if self.state_vec.is_none() {
            self.set_qubit_count(n_length);
            if let Some(destination) = destination {
                destination.zero_amplitudes();
            }
            return;
        }
        if n_length == 0 {
            if let Some(destination) = destination {
                destination.state_vec = self.state_vec;
            }
            self.state_vec = None;
            self.set_qubit_count(0);
            return;
        }
        if let Some(destination) = destination {
            if destination.state_vec.is_none() {
                destination.set_permutation(ZERO_BCI);
            }
        }
        let part_power = self.pow2_ocl(length);
        let remainder_power = self.pow2_ocl(n_length);
        let mut remainder_state_prob = vec![0.0; remainder_power];
        let mut remainder_state_angle = vec![0.0; remainder_power];
        let mut part_state_prob: Option<Vec<f64>> = None;
        let mut part_state_angle: Option<Vec<f64>> = None;
        if let Some(destination) = destination {
            part_state_prob = Some(vec![0.0; part_power]);
            part_state_angle = Some(vec![0.0; part_power]);
        }
        if self.do_normalize {
            self.normalize_state();
        }
        self.finish();
        if let Some(destination) = destination {
            self.par_for(0, remainder_power, |lcv: &bit_cap_int_ocl, cpu: &u32| {
                let mut j = lcv & self.pow2_mask_ocl(start);
                j |= (lcv ^ j) << length;
                for k in 0..part_power {
                    let l = j | (k << start);
                    let amp = self.state_vec.read(l);
                    let nrm = amp.norm();
                    remainder_state_prob[lcv] += nrm;
                    if nrm > self.amplitude_floor {
                        part_state_angle[k] = amp.arg();
                    }
                }
            });
            self.par_for(0, part_power, |lcv: &bit_cap_int_ocl, cpu: &u32| {
                let mut j = lcv << start;
                for k in 0..remainder_power {
                    let mut l = k & self.pow2_mask_ocl(start);
                    l |= j | ((k ^ l) << length);
                    let amp = self.state_vec.read(l);
                    let nrm = amp.norm();
                    part_state_prob[lcv] += nrm;
                    if nrm > self.amplitude_floor {
                        remainder_state_angle[k] = amp.arg();
                    }
                }
            });
        } else {
            self.par_for(0, remainder_power, |lcv: &bit_cap_int_ocl, cpu: &u32| {
                let mut j = lcv & self.pow2_mask_ocl(start);
                j |= (lcv ^ j) << length;
                for k in 0..part_power {
                    remainder_state_prob[lcv] += self.state_vec.read(j | (k << start)).norm();
                }
            });
            self.par_for(0, part_power, |lcv: &bit_cap_int_ocl, cpu: &u32| {
                let mut j = lcv << start;
                for k in 0..remainder_power {
                    let mut l = k & self.pow2_mask_ocl(start);
                    l |= j | ((k ^ l) << length);
                    let amp = self.state_vec.read(l);
                    if amp.norm() > self.amplitude_floor {
                        remainder_state_angle[k] = amp.arg();
                    }
                }
            });
        }
        if let Some(destination) = destination {
            destination.dump();
            self.par_for(0, part_power, |lcv: &bit_cap_int_ocl, cpu: &u32| {
                destination.state_vec.write(
                    lcv,
                    (part_state_prob[lcv].sqrt() as f64)
                        * Complex::new(part_state_angle[lcv].cos(), part_state_angle[lcv].sin()),
                );
            });
            part_state_prob = None;
            part_state_angle = None;
        }
        self.set_qubit_count(n_length);
        self.reset_state_vec(self.alloc_state_vec(self.max_q_power_ocl));
        self.par_for(0, remainder_power, |lcv: &bit_cap_int_ocl, cpu: &u32| {
            self.state_vec.write(
                lcv,
                (remainder_state_prob[lcv].sqrt() as f64)
                    * Complex::new(remainder_state_angle[lcv].cos(), remainder_state_angle[lcv].sin()),
            );
        });
    }

    pub fn decompose(&self, start: u32, destination: QInterfacePtr) {
        self.decompose_dispose(start, destination.get_qubit_count(), Some(destination.downcast::<Qrack::QEngineCPU>().unwrap()));
    }

    pub fn dispose(&self, start: u32, length: u32) {
        self.decompose_dispose(start, length, None);
    }

    pub fn dispose_with_perm(&self, start: u32, length: u32, disposed_perm: bit_cap_int) {
        if self.is_bad_bit_range(start, length, self.qubit_count) {
            panic!("QEngineCPU::Dispose range is out-of-bounds!");
        }
        if length == 0 {
            return;
        }
        let n_length = self.qubit_count - length;
        if self.state_vec.is_none() {
            self.set_qubit_count(n_length);
            return;
        }
        let disposed_perm_ocl = disposed_perm as bit_cap_int_ocl;
        let remainder_power = self.pow2_ocl(n_length);
        let skip_mask = self.pow2_ocl(start) - 1;
        let disposed_res = disposed_perm_ocl << start;
        if self.do_normalize {
            self.normalize_state();
        }
        self.finish();
        let n_state_vec = self.alloc_state_vec(remainder_power);
        self.state_vec.is_read_locked = false;
        self.par_for(0, remainder_power, |i_high: &bit_cap_int_ocl, cpu: &u32| {
            let i_low = i_high & skip_mask;
            n_state_vec.write(
                i_high,
                self.state_vec.read(i_low | ((i_high ^ i_low) << length) | disposed_res),
            );
        });
        if n_length == 0 {
            self.set_qubit_count(1);
        } else {
            self.set_qubit_count(n_length);
        }
        self.reset_state_vec(n_state_vec);
    }

    pub fn prob(&self, qubit: usize) -> Result<f64, &'static str> {
        if qubit >= self.qubit_count {
            return Err("QEngineCPU::Prob qubit index parameter must be within allocated qubit bounds!");
        }
        if self.do_normalize {
            self.normalize_state();
        }
        self.finish();
        if self.state_vec.is_none() {
            return Ok(0.0);
        }
        if self.qubit_count == 1 {
            return Ok(self.state_vec.read(1).norm());
        }
        let q_power = 2u64.pow(qubit as u32);
        let num_cores = self.get_concurrency_level();
        let mut one_chance_buff = vec![0.0; num_cores];
        let fn_closure = |lcv: u64, cpu: u32| {
            one_chance_buff[cpu as usize] += self.state_vec.read(lcv | q_power).norm();
        };
        self.state_vec.is_read_locked = false;
        if self.state_vec.is_sparse() {
            self.par_for_set(self.cast_state_vec_sparse().iterable(q_power, q_power, q_power), fn_closure);
        } else {
            self.par_for_skip(0, self.max_q_power >> 1, q_power >> 1, 1, fn_closure);
        }
        self.state_vec.is_read_locked = true;
        let mut one_chance = 0.0;
        for i in 0..num_cores {
            one_chance += one_chance_buff[i];
        }
        Ok(clamp_prob(one_chance))
    }

    pub fn ctrl_or_anti_prob(&self, control_state: bool, control: usize, target: usize) -> Result<f64, &'static str> {
        if self.state_vec.is_none() {
            return Ok(0.0);
        }
        let control_prob = self.prob(control)?;
        let control_prob = if !control_state { 1.0 - control_prob } else { control_prob };
        if control_prob <= FP_NORM_EPSILON {
            return Ok(0.0);
        }
        if (1.0 - control_prob) <= FP_NORM_EPSILON {
            return self.prob(target);
        }
        if target >= self.qubit_count {
            return Err("QEngineCPU::CtrlOrAntiProb target index parameter must be within allocated qubit bounds!");
        }
        let q_control_power = 2u64.pow(control as u32);
        let q_control_mask = if control_state { q_control_power } else { 0 };
        let q_power = 2u64.pow(target as u32);
        let num_cores = self.get_concurrency_level();
        let mut one_chance_buff = vec![0.0; num_cores];
        let fn_closure = |lcv: u64, cpu: u32| {
            if (lcv & q_control_power) == q_control_mask {
                one_chance_buff[cpu as usize] += self.state_vec.read(lcv | q_power).norm();
            }
        };
        self.state_vec.is_read_locked = false;
        if self.state_vec.is_sparse() {
            self.par_for_set(self.cast_state_vec_sparse().iterable(q_power, q_power, q_power), fn_closure);
        } else {
            self.par_for_skip(0, self.max_q_power, q_power, 1, fn_closure);
        }
        self.state_vec.is_read_locked = true;
        let mut one_chance = 0.0;
        for i in 0..num_cores {
            one_chance += one_chance_buff[i];
        }
        one_chance /= control_prob;
        Ok(clamp_prob(one_chance))
    }

    pub fn prob_reg(&self, start: usize, length: usize, permutation: u64) -> Result<f64, &'static str> {
        if self.do_normalize {
            self.normalize_state();
        }
        self.finish();
        if self.state_vec.is_none() {
            return Ok(0.0);
        }
        let num_threads = self.get_concurrency_level();
        let mut probs = vec![0.0; num_threads];
        let perm = permutation << start as u64;
        let fn_closure = |lcv: u64, cpu: u32| {
            probs[cpu as usize] += self.state_vec.read(lcv | perm).norm();
        };
        self.state_vec.is_read_locked = false;
        if self.state_vec.is_sparse() {
            self.par_for_set(self.cast_state_vec_sparse().iterable(0, bit_reg_mask(start, length), perm), fn_closure);
        } else {
            self.par_for_skip(0, self.max_q_power, 2u64.pow(start as u32), length, fn_closure);
        }
        self.state_vec.is_read_locked = true;
        let mut prob = 0.0;
        for thrd in 0..num_threads {
            prob += probs[thrd];
        }
        Ok(clamp_prob(prob))
    }

    pub fn prob_mask(&self, mask: u64, permutation: u64) -> Result<f64, &'static str> {
        if mask >= self.max_q_power {
            return Err("QEngineCPU::ProbMask mask out-of-bounds!");
        }
        if self.do_normalize {
            self.normalize_state();
        }
        self.finish();
        if self.state_vec.is_none() {
            return Ok(0.0);
        }
        let mut v = mask;
        let mut skip_powers_vec = Vec::new();
        while v != 0 {
            let old_v = v;
            v &= v - 1;
            skip_powers_vec.push((v ^ old_v) & old_v);
        }
        let num_threads = self.get_concurrency_level();
        let mut probs = vec![0.0; num_threads];
        let permutation_ocl = permutation;
        self.state_vec.is_read_locked = false;
        self.par_for_mask(0, self.max_q_power, &skip_powers_vec, |lcv: u64, cpu: u32| {
            probs[cpu as usize] += self.state_vec.read(lcv | permutation_ocl).norm();
        });
        self.state_vec.is_read_locked = true;
        let mut prob = 0.0;
        for thrd in 0..num_threads {
            prob += probs[thrd];
        }
        Ok(clamp_prob(prob))
    }

    pub fn prob_parity(&self, mask: u64) -> Result<f64, &'static str> {
        if mask >= self.max_q_power {
            return Err("QEngineCPU::ProbParity mask out-of-bounds!");
        }
        if self.do_normalize {
            self.normalize_state();
        }
        self.finish();
        if self.state_vec.is_none() || mask == 0 {
            return Ok(0.0);
        }
        let num_cores = self.get_concurrency_level();
        let mut odd_chance_buff = vec![0.0; num_cores];
        let mask_ocl = mask;
        let fn_closure = |lcv: u64, cpu: u32| {
            let mut parity = false;
            let mut v = lcv & mask_ocl;
            while v != 0 {
                parity = !parity;
                v = v & (v - 1);
            }
            if parity {
                odd_chance_buff[cpu as usize] += self.state_vec.read(lcv).norm();
            }
        };
        self.state_vec.is_read_locked = false;
        if self.state_vec.is_sparse() {
            self.par_for_set(self.cast_state_vec_sparse().iterable(), fn_closure);
        } else {
            self.par_for(0, self.max_q_power, fn_closure);
        }
        self.state_vec.is_read_locked = true;
        let mut odd_chance = 0.0;
        for i in 0..num_cores {
            odd_chance += odd_chance_buff[i];
        }
        Ok(clamp_prob(odd_chance))
    }

    pub fn m_all(&self) -> u64 {
        let rnd = self.rand();
        let mut tot_prob = 0.0;
        let mut last_nonzero = self.max_q_power;
        last_nonzero -= 1;
        let mut perm = 0;
        while perm < self.max_q_power {
            let part_prob = self.prob_all(perm).unwrap();
            if part_prob > REAL1_EPSILON {
                tot_prob += part_prob;
                if tot_prob > rnd || (1.0 - tot_prob) <= FP_NORM_EPSILON {
                    self.set_permutation(perm);
                    return perm;
                }
                last_nonzero = perm;
            }
            perm += 1;
        }
        self.set_permutation(last_nonzero);
        last_nonzero
    }

    pub fn force_m_parity(&self, mask: u64, result: bool, do_force: bool) -> bool {
        if mask >= self.max_q_power || mask == 0 {
            return false;
        }
        if self.state_vec.is_none() {
            return false;
        }
        if !do_force {
            let result = self.rand() <= self.prob_parity(mask).unwrap();
        }
        let mut odd_chance = 0.0;
        let num_cores = self.get_concurrency_level();
        let mut odd_chance_buff = vec![0.0; num_cores];
        let mask_ocl = mask;
        let fn_closure = |lcv: u64, cpu: u32| {
            let mut parity = false;
            let mut v = lcv & mask_ocl;
            while v != 0 {
                parity = !parity;
                v = v & (v - 1);
            }
            if parity == result {
                odd_chance_buff[cpu as usize] += self.state_vec.read(lcv).norm();
            } else {
                self.state_vec.write(lcv, 0.0);
            }
        };
        self.state_vec.is_read_locked = false;
        if self.state_vec.is_sparse() {
            self.par_for_set(self.cast_state_vec_sparse().iterable(), fn_closure);
        } else {
            self.par_for(0, self.max_q_power, fn_closure);
        }
        self.state_vec.is_read_locked = true;
        for i in 0..num_cores {
            odd_chance += odd_chance_buff[i];
        }
        self.running_norm = odd_chance;
        if !self.do_normalize {
            self.normalize_state();
        }
        result
    }

    pub fn sum_sqr_diff(&self, to_compare: &QEngineCPU) -> f64 {
        if to_compare.is_none() {
            return 1.0;
        }
        if self as *const QEngineCPU == to_compare as *const QEngineCPU {
            return 0.0;
        }
        if self.qubit_count != to_compare.qubit_count {
            return 1.0;
        }
        if self.do_normalize {
            self.normalize_state();
        }
        self.finish();
        if to_compare.do_normalize {
            to_compare.normalize_state();
        }
        to_compare.finish();
        if self.state_vec.is_none() && to_compare.state_vec.is_none() {
            return 0.0;
        }
        if self.state_vec.is_none() {
            to_compare.update_running_norm();
            return to_compare.running_norm;
        }
        if to_compare.state_vec.is_none() {
            self.update_running_norm();
            return self.running_norm;
        }
        self.state_vec.is_read_locked = false;
        to_compare.state_vec.is_read_locked = false;
        let num_cores = self.get_concurrency_level();
        let mut part_inner = vec![0.0; num_cores];
        let fn_closure = |lcv: u64, cpu: u32| {
            part_inner[cpu as usize] += self.state_vec.read(lcv).conj() * to_compare.state_vec.read(lcv);
        };
        self.state_vec.is_read_locked = true;
        to_compare.state_vec.is_read_locked = true;
        let mut tot_inner = 0.0;
        for i in 0..num_cores {
            tot_inner += part_inner[i];
        }
        1.0 - clamp_prob(tot_inner.norm())
    }

    pub fn apply_m(&self, reg_mask: u64, result: u64, nrm: Complex) {
        self.check_zero_skip();
        self.dispatch(self.max_q_power, |lcv, cpu| {
            if (lcv & reg_mask) == result {
                self.state_vec.write(lcv, nrm * self.state_vec.read(lcv));
            } else {
                self.state_vec.write(lcv, 0.0);
            }
        });
    }

    pub fn normalize_state(&self, nrm_f: f64, norm_thresh_f: f64, phase_arg: f64) {
        self.check_zero_skip();
        if self.running_norm == REAL1_DEFAULT_ARG && nrm_f == REAL1_DEFAULT_ARG {
            self.update_running_norm();
        }
        let mut nrm = nrm_f;
        let mut norm_thresh = norm_thresh_f;
        if nrm < 0.0 {
            self.finish();
            nrm = self.running_norm;
        }
        if nrm <= FP_NORM_EPSILON {
            self.zero_amplitudes();
            return;
        }
        if (1.0 - nrm).abs() <= FP_NORM_EPSILON && phase_arg.powi(2) <= FP_NORM_EPSILON {
            return;
        }
        self.finish();
        if norm_thresh < 0.0 {
            norm_thresh = self.amplitude_floor;
        }
        nrm = 1.0 / nrm.sqrt();
        let c_nrm = Complex::from_polar(nrm, phase_arg);
        if norm_thresh <= 0.0 {
            self.par_for(0, self.max_q_power, |lcv, cpu| {
                self.state_vec.write(lcv, c_nrm * self.state_vec.read(lcv));
            });
        } else {
            self.par_for(0, self.max_q_power, |lcv, cpu| {
                let mut amp = self.state_vec.read(lcv);
                if amp.norm() < norm_thresh {
                    amp = 0.0;
                }
                self.state_vec.write(lcv, c_nrm * amp);
            });
        }
        self.running_norm = 1.0;
    }

    pub fn update_running_norm(&self, norm_thresh: f64) {
        self.finish();
        if self.state_vec.is_none() {
            self.running_norm = 0.0;
            return;
        }
        if norm_thresh < 0.0 {
            norm_thresh = self.amplitude_floor;
        }
        self.running_norm = par_norm(self.max_q_power, &self.state_vec, norm_thresh);
        if self.running_norm <= FP_NORM_EPSILON {
            self.zero_amplitudes();
        }
    }

    pub fn alloc_state_vec(&self, elem_count: u64) -> StateVectorPtr {
        if self.is_sparse {
            StateVectorSparse::new(elem_count)
        } else {
            StateVectorArray::new(elem_count)
        }
    }

    #[cfg(feature = "sse")]
    fn apply_2x2(&mut self, offset1: bitCapIntOcl, offset2: bitCapIntOcl, matrix: &[complex], bit_count: bitLenInt, q_pows_sorted: &[bitCapIntOcl], do_calc_norm: bool, nrm_thresh: real1_f) {
        CHECK_ZERO_SKIP!();
        if offset1 >= self.maxQPowerOcl || offset2 >= self.maxQPowerOcl {
            panic!("QEngineCPU::Apply2x2 offset1 and offset2 parameters must be within allocated qubit bounds!");
        }
        for i in 0..bit_count {
            if q_pows_sorted[i] >= self.maxQPowerOcl {
                panic!("QEngineCPU::Apply2x2 parameter qPowsSorted array values must be within allocated qubit bounds!");
            }
            if i > 0 && q_pows_sorted[i - 1] == q_pows_sorted[i] {
                panic!("QEngineCPU::Apply2x2 parameter qPowSorted array values cannot be duplicated (for control and target qubits)!");
            }
        }
        let mtrx_s = vec![complex::default(); 4];
        mtrx_s.copy_from_slice(matrix);
        let q_powers_sorted = q_pows_sorted.to_vec();
        let do_apply_norm = self.do_normalize && bit_count == 1 && self.running_norm > ZERO_R1;
        let mut do_calc_norm = do_calc_norm && (do_apply_norm || self.running_norm <= ZERO_R1);
        let nrm = if do_apply_norm { ONE_R1 / (self.running_norm.sqrt()) } else { ONE_R1 };
        if do_calc_norm {
            self.running_norm = ONE_R1;
        }
        self.dispatch(self.maxQPowerOcl >> bit_count, |_, _| {
            let mtrx = &mtrx_s;
            let norm_thresh = if nrm_thresh < ZERO_R1 { self.amplitude_floor } else { nrm_thresh };
            let num_cores = self.get_concurrency_level();
            let mtrx_col1 = complex2::new(mtrx[0], mtrx[2]);
            let mtrx_col2 = complex2::new(mtrx[1], mtrx[3]);
            let mtrx_col1_shuff = self.mtrx_col_shuff(mtrx_col1);
            let mtrx_col2_shuff = self.mtrx_col_shuff(mtrx_col2);
            let mtrx_phase = if IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2]) {
                complex2::new(mtrx[0], mtrx[3])
            } else {
                complex2::new(mtrx[1], mtrx[2])
            };
            let mut rng_nrm = vec![ZERO_R1; num_cores];
            let fn: ParallelFunc;
            if !do_calc_norm {
                if IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2]) {
                    fn = |lcv, _| {
                        let mut qubit = self.state_vec.read2(lcv + offset1, lcv + offset2);
                        qubit = mtrx_phase * qubit;
                        self.state_vec.write2(lcv + offset1, qubit.c(0), lcv + offset2, qubit.c(1));
                    };
                } else if IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3]) {
                    fn = |lcv, _| {
                        let mut qubit = self.state_vec.read2(lcv + offset2, lcv + offset1);
                        qubit = mtrx_phase * qubit;
                        self.state_vec.write2(lcv + offset1, qubit.c(0), lcv + offset2, qubit.c(1));
                    };
                } else {
                    fn = |lcv, _| {
                        let mut qubit = self.state_vec.read2(lcv + offset1, lcv + offset2);
                        qubit = self.matrix_mul(mtrx_col1, mtrx_col2, mtrx_col1_shuff, mtrx_col2_shuff, qubit);
                        self.state_vec.write2(lcv + offset1, qubit.c(0), lcv + offset2, qubit.c(1));
                    };
                }
            } else if norm_thresh > ZERO_R1 {
                if (ONE_R1 - nrm).abs() > REAL1_EPSILON {
                    if IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2]) {
                        fn = NORM_THRESH_KERNEL(offset1, offset2, nrm * mtrx_phase * qubit);
                    } else if IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3]) {
                        fn = NORM_THRESH_KERNEL(offset2, offset1, nrm * mtrx_phase * qubit);
                    } else {
                        fn = NORM_THRESH_KERNEL(
                            offset1,
                            offset2,
                            self.matrix_mul(nrm, mtrx_col1, mtrx_col2, mtrx_col1_shuff, mtrx_col2_shuff, qubit),
                        );
                    }
                } else {
                    if IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2]) {
                        fn = NORM_THRESH_KERNEL(offset1, offset2, mtrx_phase * qubit);
                    } else if IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3]) {
                        fn = NORM_THRESH_KERNEL(offset2, offset1, nrm * mtrx_phase * qubit);
                    } else {
                        fn = NORM_THRESH_KERNEL(
                            offset1,
                            offset2,
                            self.matrix_mul(mtrx_col1, mtrx_col2, mtrx_col1_shuff, mtrx_col2_shuff, qubit),
                        );
                    }
                }
            } else {
                if (ONE_R1 - nrm).abs() > REAL1_EPSILON {
                    if IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2]) {
                        fn = NORM_CALC_KERNEL(offset1, offset2, nrm * mtrx_phase * qubit);
                    } else if IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3]) {
                        fn = NORM_CALC_KERNEL(offset2, offset1, nrm * mtrx_phase * qubit);
                    } else {
                        fn = NORM_CALC_KERNEL(
                            offset1,
                            offset2,
                            self.matrix_mul(nrm, mtrx_col1, mtrx_col2, mtrx_col1_shuff, mtrx_col2_shuff, qubit),
                        );
                    }
                } else {
                    if IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2]) {
                        fn = NORM_CALC_KERNEL(offset1, offset2, mtrx_phase * qubit);
                    } else if IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3]) {
                        fn = NORM_CALC_KERNEL(offset2, offset1, mtrx_phase * qubit);
                    } else {
                        fn = NORM_CALC_KERNEL(
                            offset1,
                            offset2,
                            self.matrix_mul(mtrx_col1, mtrx_col2, mtrx_col1_shuff, mtrx_col2_shuff, qubit),
                        );
                    }
                }
            }
            if self.state_vec.is_sparse() {
                let set_mask = offset1 ^ offset2;
                let mut filter_mask = 0;
                for i in 0..bit_count {
                    filter_mask |= q_powers_sorted[i];
                }
                filter_mask &= !set_mask;
                let filter_values = filter_mask & offset1 & offset2;
                self.par_for_set(self.cast_state_vec_sparse().iterable(set_mask, filter_mask, filter_values), fn);
            } else {
                self.par_for_mask(0, self.maxQPowerOcl, &q_powers_sorted, fn);
            }
            if do_apply_norm {
                self.running_norm = ONE_R1;
            }
            if !do_calc_norm {
                return;
            }
            let mut r_nrm = ZERO_R1;
            for i in 0..num_cores {
                r_nrm += rng_nrm[i];
            }
            rng_nrm.clear();
            self.running_norm = r_nrm;
            if self.running_norm <= FP_NORM_EPSILON {
                self.zero_amplitudes();
            }
        });
    }

    #[cfg(not(feature = "sse"))]
    fn apply_2x2(&mut self, offset1: bitCapIntOcl, offset2: bitCapIntOcl, matrix: &[complex], bit_count: bitLenInt, q_pows_sorted: &[bitCapIntOcl], do_calc_norm: bool, nrm_thresh: real1_f) {
        CHECK_ZERO_SKIP!();
        if offset1 >= self.maxQPowerOcl || offset2 >= self.maxQPowerOcl {
            panic!("QEngineCPU::Apply2x2 offset1 and offset2 parameters must be within allocated qubit bounds!");
        }
        for i in 0..bit_count {
            if q_pows_sorted[i] >= self.maxQPowerOcl {
                panic!("QEngineCPU::Apply2x2 parameter qPowsSorted array values must be within allocated qubit bounds!");
            }
            if i > 0 && q_pows_sorted[i - 1] == q_pows_sorted[i] {
                panic!("QEngineCPU::Apply2x2 parameter qPowsSorted array values cannot be duplicated (for control and target qubits)!");
            }
        }
        let mtrx_s = Box::new([complex; 4]::from_slice(matrix));
        let q_powers_sorted = q_pows_sorted.to_vec();
        let do_apply_norm = self.do_normalize && bit_count == 1 && self.running_norm > ZERO_R1;
        let mut do_calc_norm = do_calc_norm && (do_apply_norm || self.running_norm <= ZERO_R1);
        let nrm = if do_apply_norm { ONE_R1 / (self.running_norm.sqrt()) } else { ONE_R1 };
        if do_calc_norm {
            self.running_norm = ONE_R1;
        }
        self.dispatch(self.maxQPowerOcl >> bit_count, |_, _| {
            let mtrx = &mtrx_s[..];
            let mtrx0 = mtrx[0];
            let mtrx1 = mtrx[1];
            let mtrx2 = mtrx[2];
            let mtrx3 = mtrx[3];
            let norm_thresh = if nrm_thresh < ZERO_R1 { self.amplitude_floor } else { nrm_thresh };
            let num_cores = self.get_concurrency_level();
            let mut rng_nrm = vec![ZERO_R1; num_cores];
            let fn: ParallelFunc;
            if !do_calc_norm {
                if IS_NORM_0(mtrx1) && IS_NORM_0(mtrx2) {
                    fn = |lcv, _| {
                        self.state_vec.write2(lcv + offset1, mtrx0 * self.state_vec.read(lcv + offset1), lcv + offset2, mtrx3 * self.state_vec.read(lcv + offset2));
                    };
                } else if IS_NORM_0(mtrx0) && IS_NORM_0(mtrx3) {
                    fn = |lcv, _| {
                        self.state_vec.write2(lcv + offset1, mtrx1 * self.state_vec.read(lcv + offset2), lcv + offset2, mtrx2 * self.state_vec.read(lcv + offset1));
                    };
                } else {
                    fn = |lcv, _| {
                        let y0 = self.state_vec.read(lcv + offset1);
                        let y1 = self.state_vec.read(lcv + offset2);
                        self.state_vec.write2(lcv + offset1, (mtrx0 * y0) + (mtrx1 * y1), lcv + offset2, (mtrx2 * y0) + (mtrx3 * y1));
                    };
                }
            } else if norm_thresh > ZERO_R1 {
                if (ONE_R1 - nrm).abs() > REAL1_EPSILON {
                    if IS_NORM_0(mtrx1) && IS_NORM_0(mtrx2) {
                        fn = NORM_THRESH_KERNEL(nrm * (mtrx0 * Y0), nrm * (mtrx3 * qubit[1U]));
                    } else if IS_NORM_0(mtrx0) && IS_NORM_0(mtrx3) {
                        fn = NORM_THRESH_KERNEL(nrm * (mtrx1 * qubit[1U]), nrm * (mtrx2 * Y0));
                    } else {
                        fn = NORM_THRESH_KERNEL(
                            nrm * ((mtrx0 * Y0) + (mtrx1 * qubit[1U])), nrm * ((mtrx2 * Y0) + (mtrx3 * qubit[1U])));
                    }
                } else {
                    if IS_NORM_0(mtrx1) && IS_NORM_0(mtrx2) {
                        fn = NORM_THRESH_KERNEL(mtrx0 * Y0, mtrx3 * qubit[1U]);
                    } else if IS_NORM_0(mtrx0) && IS_NORM_0(mtrx3) {
                        fn = NORM_THRESH_KERNEL(mtrx1 * qubit[1U], mtrx2 * Y0);
                    } else {
                        fn = NORM_THRESH_KERNEL((mtrx0 * Y0) + (mtrx1 * qubit[1U]), (mtrx2 * Y0) + (mtrx3 * qubit[1U]));
                    }
                }
            } else {
                if (ONE_R1 - nrm).abs() > REAL1_EPSILON {
                    if IS_NORM_0(mtrx1) && IS_NORM_0(mtrx2) {
                        fn = NORM_CALC_KERNEL(nrm * (mtrx0 * Y0), nrm * (mtrx3 * qubit[1U]));
                    } else if IS_NORM_0(mtrx0) && IS_NORM_0(mtrx3) {
                        fn = NORM_CALC_KERNEL(nrm * (mtrx1 * qubit[1U]), nrm * (mtrx2 * Y0));
                    } else {
                        fn = NORM_CALC_KERNEL(
                            nrm * ((mtrx0 * Y0) + (mtrx1 * qubit[1U])), nrm * ((mtrx2 * Y0) + (mtrx3 * qubit[1U])));
                    }
                } else {
                    if IS_NORM_0(mtrx1) && IS_NORM_0(mtrx2) {
                        fn = NORM_CALC_KERNEL(mtrx0 * Y0, mtrx3 * qubit[1U]);
                    } else if IS_NORM_0(mtrx0) && IS_NORM_0(mtrx3) {
                        fn = NORM_CALC_KERNEL(mtrx1 * qubit[1U], mtrx2 * Y0);
                    } else {
                        fn = NORM_CALC_KERNEL((mtrx0 * Y0) + (mtrx1 * qubit[1U]), (mtrx2 * Y0) + (mtrx3 * qubit[1U]));
                    }
                }
            }
            if self.state_vec.is_sparse() {
                let set_mask = offset1 ^ offset2;
                let mut filter_mask = 0;
                for i in 0..bit_count {
                    filter_mask |= q_powers_sorted[i];
                }
                filter_mask &= !set_mask;
                let filter_values = filter_mask & offset1 & offset2;
                self.par_for_set(self.cast_state_vec_sparse().iterable(set_mask, filter_mask, filter_values), fn);
            } else {
                self.par_for_mask(0, self.maxQPowerOcl, &q_powers_sorted, fn);
            }
            if do_apply_norm {
                self.running_norm = ONE_R1;
            }
            if !do_calc_norm {
                return;
            }
            let mut r_nrm = ZERO_R1;
            for i in 0..num_cores {
                r_nrm += rng_nrm[i];
            }
            rng_nrm.clear();
            self.running_norm = r_nrm;
            if self.running_norm <= FP_NORM_EPSILON {
                self.zero_amplitudes();
            }
        });
    }
}

pub type QEngineCPUPtr = Arc<QEngineCPU>;
