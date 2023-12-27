use std::rc::Rc;
use std::cell::RefCell;

pub struct QUnit {
    freeze_basis_2_qb: bool,
    use_host_ram: bool,
    is_sparse: bool,
    is_reactive_separate: bool,
    use_t_gadget: bool,
    threshold_qubits: i32,
    separability_threshold: f64,
    log_fidelity: f64,
    dev_id: i64,
    phase_factor: Complex<f64>,
    shards: QEngineShardMap,
    device_ids: Vec<i64>,
    engines: Vec<QInterfaceEngine>,
}

pub type QUnitPtr = Rc<RefCell<QUnit>>;

impl QUnit {
    pub fn new(
        eng: Vec<QInterfaceEngine>,
        q_bit_count: i32,
        init_state: i64,
        rgp: qrack_rand_gen_ptr,
        phase_fac: Complex<f64>,
        do_norm: bool,
        random_global_phase: bool,
        use_host_mem: bool,
        device_id: i64,
        use_hardware_rng: bool,
        use_sparse_state_vec: bool,
        norm_thresh: f32,
        dev_ids: Vec<i64>,
        qubit_threshold: i32,
        separation_thresh: f32,
    ) -> QUnit {
        QUnit {
            freeze_basis_2_qb: false,
            use_host_ram: false,
            is_sparse: false,
            is_reactive_separate: false,
            use_t_gadget: false,
            threshold_qubits: 0,
            separability_threshold: 0.0,
            log_fidelity: 0.0,
            dev_id: -1,
            phase_factor: Complex::new(0.0, 0.0),
            shards: QEngineShardMap::new(),
            device_ids: Vec::new(),
            engines: Vec::new(),
        }
    }

    pub fn set_concurrency(&mut self, threads_per_engine: u32) {
        QInterface::set_concurrency(threads_per_engine);
        self.parallel_unit_apply(
            |unit, unused1, unused2, unused3, threads_per_engine| {
                unit.set_concurrency(threads_per_engine as u32);
                true
            },
            0.0,
            0.0,
            0.0,
            threads_per_engine as i64,
        );
    }

    pub fn set_t_injection(&mut self, use_gadget: bool) {
        self.use_t_gadget = use_gadget;
        self.parallel_unit_apply(
            |unit, unused1, unused2, unused3, use_gadget| {
                unit.set_t_injection(use_gadget != 0);
                true
            },
            0.0,
            0.0,
            0.0,
            if use_gadget { 1 } else { 0 },
        );
    }

    pub fn set_reactive_separate(&mut self, is_agg_sep: bool) {
        self.is_reactive_separate = is_agg_sep;
    }

    pub fn get_reactive_separate(&self) -> bool {
        self.is_reactive_separate
    }

    pub fn get_device(&self) -> i64 {
        self.dev_id
    }

    pub fn prob_rdm(&self, qubit: i32) -> f64 {
        let shard = &self.shards[qubit as usize];
        if shard.unit.is_none() {
            return self.prob(qubit);
        }
        shard.unit.prob_rdm(qubit)
    }

    pub fn c_prob_rdm(&self, control: i32, target: i32) -> f64 {
        self.anti_cnot(control, target);
        let prob = self.prob_rdm(target);
        self.anti_cnot(control, target);
        prob
    }

    pub fn ac_prob_rdm(&self, control: i32, target: i32) -> f64 {
        self.cnot(control, target);
        let prob = self.prob_rdm(target);
        self.cnot(control, target);
        prob
    }

    pub fn set_amplitude(&mut self, perm: i64, amp: Complex<f64>) {
        if perm >= max_q_power {
            panic!("QUnit::SetAmplitude argument out-of-bounds!");
        }
        self.entangle_all();
        self.shards[0].unit.set_amplitude(perm, amp);
    }

    pub fn compose(&mut self, to_copy: QUnitPtr) -> i32 {
        self.compose(to_copy, self.qubit_count)
    }

    pub fn compose(&mut self, to_copy: QInterfacePtr) -> i32 {
        self.compose(to_copy.downcast::<QUnitPtr>().unwrap())
    }

    pub fn compose(&mut self, to_copy: QUnitPtr, start: i32) -> i32 {
        if start > self.qubit_count {
            panic!("QUnit::Compose start index is out-of-bounds!");
        }
        let clone = to_copy.clone();
        self.shards.insert(start, clone.shards);
        self.set_qubit_count(self.qubit_count + to_copy.get_qubit_count());
        start
    }

    pub fn compose(&mut self, to_copy: QInterfacePtr, start: i32) -> i32 {
        self.compose(to_copy.downcast::<QUnitPtr>().unwrap(), start)
    }

    pub fn decompose(&mut self, start: i32, dest: QInterfacePtr) {
        self.decompose(start, dest.downcast::<QUnitPtr>().unwrap())
    }

    pub fn decompose(&mut self, start: i32, dest: QUnitPtr) {
        self.detach(start, dest.get_qubit_count(), dest);
    }

    pub fn decompose(&mut self, start: i32, length: i32) -> QInterfacePtr {
        let dest = QUnit::new(
            self.engines.clone(),
            length,
            ZERO_BCI,
            rand_generator,
            phase_factor,
            do_normalize,
            rand_global_phase,
            use_host_ram,
            dev_id,
            use_rdrand,
            is_sparse,
            amplitude_floor as f32,
            device_ids.clone(),
            threshold_qubits,
            separability_threshold as f32,
        );
        self.decompose(start, dest);
        dest
    }

    pub fn dispose(&mut self, start: i32, length: i32) {
        self.detach(start, length, None);
    }

    pub fn dispose(&mut self, start: i32, length: i32, disposed_perm: i64) {
        self.detach(start, length, None);
    }

    pub fn z_mask(&mut self, mask: i64) {
        self.phase_parity(PI_R1, mask);
    }

    pub fn mc_phase(
        &mut self,
        controls: Vec<i32>,
        top_left: Complex<f64>,
        bottom_right: Complex<f64>,
        target: i32,
    ) {
        let m = 2_i64.pow(controls.len() as u32) - 1;
        self.uc_phase(controls, top_left, bottom_right, target, m);
    }

    pub fn mc_invert(
        &mut self,
        controls: Vec<i32>,
        top_right: Complex<f64>,
        bottom_left: Complex<f64>,
        target: i32,
    ) {
        let m = 2_i64.pow(controls.len() as u32) - 1;
        self.uc_invert(controls, top_right, bottom_left, target, m);
    }

    pub fn mac_phase(
        &mut self,
        controls: Vec<i32>,
        top_left: Complex<f64>,
        bottom_right: Complex<f64>,
        target: i32,
    ) {
        self.uc_phase(controls, top_left, bottom_right, target, 0);
    }

    pub fn mac_invert(
        &mut self,
        controls: Vec<i32>,
        top_right: Complex<f64>,
        bottom_left: Complex<f64>,
        target: i32,
    ) {
        self.uc_invert(controls, top_right, bottom_left, target, 0);
    }

    pub fn mc_mtrx(&mut self, controls: Vec<i32>, mtrx: &[Complex<f64>], target: i32) {
        let m = 2_i64.pow(controls.len() as u32) - 1;
        self.uc_mtrx(controls, mtrx, target, m);
    }

    pub fn mac_mtrx(&mut self, controls: Vec<i32>, mtrx: &[Complex<f64>], target: i32) {
        self.uc_mtrx(controls, mtrx, target, 0);
    }

    pub fn set_reg(&mut self, start: i32, length: i32, value: i64) {
        let mut perm = value;
        for i in 0..length {
            self.set_qubit(start + i, perm & 1);
            perm >>= 1;
        }
    }

    pub fn swap(&mut self, qubit1: i32, qubit2: i32) {
        if qubit1 >= self.qubit_count {
            panic!("QUnit::Swap qubit index must be within allocated qubit bounds!");
        }
        if qubit2 >= self.qubit_count {
            panic!("QUnit::Swap qubit index must be within allocated qubit bounds!");
        }
        if qubit1 == qubit2 {
            return;
        }
        self.shards.swap(qubit1, qubit2);
    }

    pub fn i_swap(&mut self, qubit1: i32, qubit2: i32) {
        self.either_i_swap(qubit1, qubit2, false);
    }

    pub fn ii_swap(&mut self, qubit1: i32, qubit2: i32) {
        self.either_i_swap(qubit1, qubit2, true);
    }

    pub fn prob(&self, qubit: i32) -> f64 {
        if qubit >= self.qubit_count {
            panic!("QUnit::Prob target must be within allocated qubit bounds!");
        }
        self.to_perm_basis_prob(qubit);
        self.prob_base(qubit)
    }

    pub fn prob_all(&self, perm: i64) -> f64 {
        self.clamp_prob(norm(self.get_amplitude_or_prob(perm, true) as f64))
    }

    pub fn prob_all_rdm(&self, round_rz: bool, perm: i64) -> f64 {
        if self.shards[0].unit.is_some() && self.shards[0].unit.get_qubit_count() == self.qubit_count {
            self.order_contiguous(self.shards[0].unit);
            return self.shards[0].unit.prob_all_rdm(round_rz, perm);
        }
        let clone = self.clone();
        let unit = clone.entangle_all(true);
        clone.order_contiguous(unit);
        unit.prob_all_rdm(round_rz, perm)
    }

    pub fn sum_sqr_diff(&self, to_compare: QInterfacePtr) -> f64 {
        self.sum_sqr_diff(to_compare.downcast::<QUnitPtr>().unwrap())
    }

    pub fn expectation_bits_factorized(
        &self,
        bits: Vec<i32>,
        perms: Vec<i64>,
        offset: i64,
    ) -> f64 {
        self.expectation_factorized(false, false, bits, perms, vec![], offset, false)
    }

    pub fn expectation_bits_factorized_rdm(
        &self,
        round_rz: bool,
        bits: Vec<i32>,
        perms: Vec<i64>,
        offset: i64,
    ) -> f64 {
        self.expectation_factorized(true, false, bits, perms, vec![], offset, round_rz)
    }

    pub fn normalize_state(&mut self, nrm: f32, norm_thresh: f32, phase_arg: f64) {
        let nrm = if nrm == REAL1_DEFAULT_ARG { 0.0 } else { nrm };
        let norm_thresh = if norm_thresh == REAL1_DEFAULT_ARG {
            0.0
        } else {
            norm_thresh
        };
        self.normalize_state(nrm, norm_thresh, phase_arg);
    }

    pub fn dump(&mut self) {
        for shard in &mut self.shards {
            shard.unit = None;
        }
    }

    pub fn is_clifford(&self, qubit: i32) -> bool {
        self.shards[qubit as usize].is_clifford()
    }

    pub fn get_unitary_fidelity(&self) -> f64 {
        self.log_fidelity.exp()
    }

    pub fn reset_unitary_fidelity(&mut self) {
        self.log_fidelity = 0.0;
    }

    pub fn set_sdrp(&mut self, sdrp: f32) {
        self.separability_threshold = sdrp as f64;
    }

    fn expectation_factorized(
        &self,
        is_rdm: bool,
        is_float: bool,
        bits: Vec<i32>,
        perms: Vec<i64>,
        weights: Vec<f32>,
        offset: i64,
        round_rz: bool,
    ) -> f64 {
        if (is_float && weights.len() < bits.len())
            || (!is_float && perms.len() < bits.len())
        {
            panic!("QUnit::ExpectationFactorized() must supply at least as many weights as bits!");
        }
        self.throw_if_qb_id_array_is_bad(
            &bits,
            self.qubit_count,
            "QUnit::ExpectationFactorized qubits vector values must be within allocated qubit bounds!",
        );
        if self.shards[0].unit.is_some() && self.shards[0].unit.get_qubit_count() == self.qubit_count {
            self.order_contiguous(self.shards[0].unit);
            if is_float {
                if is_rdm {
                    return self.shards[0].unit.expectation_floats_factorized_rdm(round_rz, &bits, &weights);
                } else {
                    return self.shards[0].unit.expectation_floats_factorized(&bits, &weights);
                }
            } else {
                if is_rdm {
                    return self.shards[0].unit.expectation_bits_factorized_rdm(round_rz, &bits, &perms, offset);
                } else {
                    return self.shards[0].unit.expectation_bits_factorized(&bits, &perms, offset);
                }
            }
        }
        let clone = self.clone();
        let unit = clone.entangle_all(true);
        clone.order_contiguous(unit);
        if is_float {
            if is_rdm {
                return unit.expectation_floats_factorized_rdm(round_rz, &bits, &weights);
            } else {
                return unit.expectation_floats_factorized(&bits, &weights);
            }
        } else {
            if is_rdm {
                return unit.expectation_bits_factorized_rdm(round_rz, &bits, &perms, offset);
            } else {
                return unit.expectation_bits_factorized(&bits, &perms, offset);
            }
        }
    }

    fn entangle_range(&mut self, start: i32, length: i32, is_for_prob: bool) -> QInterfacePtr {
        let to_ret = self.entangle_range(start, length, is_for_prob);
        self.order_contiguous(to_ret);
        to_ret
    }

    fn entangle_all(&mut self, is_for_prob: bool) -> QInterfacePtr {
        let to_ret = self.entangle_range(0, self.qubit_count, is_for_prob);
        self.order_contiguous(to_ret);
        to_ret
    }

    fn entangle_in_current_basis(
        &mut self,
        first: &mut [i32],
        last: &mut [i32],
    ) -> QInterfacePtr {
        let to_ret = self.entangle_in_current_basis(first, last);
        self.order_contiguous(to_ret);
        to_ret
    }

    fn clamp_shard(&mut self, qubit: i32) {
        let shard = &mut self.shards[qubit as usize];
        if !shard.clamp_amps() || shard.unit.is_none() {
            return;
        }
        if shard.amp1.norm() <= FP_NORM_EPSILON {
            self.log_fidelity += (ONE_R1_F - shard.amp1.norm()).log();
            self.separate_bit(false, qubit);
        } else if shard.amp0.norm() <= FP_NORM_EPSILON {
            self.log_fidelity += (ONE_R1_F - shard.amp0.norm()).log();
            self.separate_bit(true, qubit);
        }
    }

    fn transform_x2x2(&self, mtrx_in: &[Complex<f64>], mtrx_out: &mut [Complex<f64>]) {
        mtrx_out[0] = (ONE_R1 / 2.0) * (mtrx_in[0] + mtrx_in[1] + mtrx_in[2] + mtrx_in[3]);
        mtrx_out[1] = (ONE_R1 / 2.0) * (mtrx_in[0] - mtrx_in[1] + mtrx_in[2] - mtrx_in[3]);
        mtrx_out[2] = (ONE_R1 / 2.0) * (mtrx_in[0] + mtrx_in[1] - mtrx_in[2] - mtrx_in[3]);
        mtrx_out[3] = (ONE_R1 / 2.0) * (mtrx_in[0] - mtrx_in[1] - mtrx_in[2] + mtrx_in[3]);
    }

    fn transform_x_invert(
        &self,
        top_right: Complex<f64>,
        bottom_left: Complex<f64>,
        mtrx_out: &mut [Complex<f64>],
    ) {
        mtrx_out[0] = (ONE_R1 / 2.0) * (top_right + bottom_left);
        mtrx_out[1] = (ONE_R1 / 2.0) * (-top_right + bottom_left);
        mtrx_out[2] = -mtrx_out[1];
        mtrx_out[3] = -mtrx_out[0];
    }

    fn transform_y2x2(&self, mtrx_in: &[Complex<f64>], mtrx_out: &mut [Complex<f64>]) {
        mtrx_out[0] = (ONE_R1 / 2.0) * (mtrx_in[0] + I_CMPLX * (mtrx_in[1] - mtrx_in[2]) + mtrx_in[3]);
        mtrx_out[1] = (ONE_R1 / 2.0) * (mtrx_in[0] - I_CMPLX * (mtrx_in[1] + mtrx_in[2]) - mtrx_in[3]);
        mtrx_out[2] = (ONE_R1 / 2.0) * (mtrx_in[0] + I_CMPLX * (mtrx_in[1] + mtrx_in[2]) - mtrx_in[3]);
        mtrx_out[3] = (ONE_R1 / 2.0) * (mtrx_in[0] - I_CMPLX * (mtrx_in[1] - mtrx_in[2]) + mtrx_in[3]);
    }

    fn transform_y_invert(
        &self,
        top_right: Complex<f64>,
        bottom_left: Complex<f64>,
        mtrx_out: &mut [Complex<f64>],
    ) {
        mtrx_out[0] = I_CMPLX * (ONE_R1 / 2.0) * (top_right - bottom_left);
        mtrx_out[1] = I_CMPLX * (ONE_R1 / 2.0) * (-top_right - bottom_left);
        mtrx_out[2] = -mtrx_out[1];
        mtrx_out[3] = -mtrx_out[0];
    }

    fn transform_phase(
        &self,
        top_left: Complex<f64>,
        bottom_right: Complex<f64>,
        mtrx_out: &mut [Complex<f64>],
    ) {
        mtrx_out[0] = (ONE_R1 / 2.0) * (top_left + bottom_right);
        mtrx_out[1] = (ONE_R1 / 2.0) * (top_left - bottom_right);
        mtrx_out[2] = mtrx_out[1];
        mtrx_out[3] = mtrx_out[0];
    }

    fn revert_basis_x(&mut self, i: i32) {
        let shard = &mut self.shards[i as usize];
        if shard.pauli_basis != PauliX {
            return;
        }
        self.convert_z_to_x(i);
    }

    fn revert_basis_y(&mut self, i: i32) {
        let shard = &mut self.shards[i as usize];
        if shard.pauli_basis != PauliY {
            return;
        }
        shard.pauli_basis = PauliX;
        let diag = Complex::new(ONE_R1 / 2.0, ONE_R1 / 2.0);
        let cross = Complex::new(ONE_R1 / 2.0, -ONE_R1 / 2.0);
        let mtrx = [diag, cross, cross, diag];
        if shard.unit.is_some() {
            shard.unit.mtrx(&mtrx, shard.mapped);
        }
        if shard.is_phase_dirty || shard.is_prob_dirty {
            shard.is_prob_dirty = true;
            return;
        }
        let y0 = shard.amp0;
        shard.amp0 = mtrx[0] * y0 + mtrx[1] * shard.amp1;
        shard.amp1 = mtrx[2] * y0 + mtrx[3] * shard.amp1;
        self.clamp_shard(i);
    }

    fn revert_basis_1_qb(&mut self, i: i32) {
        let shard = &mut self.shards[i as usize];
        if shard.pauli_basis == PauliY {
            self.convert_y_to_z(i);
        } else {
            self.revert_basis_x(i);
        }
    }

    fn revert_basis_to_x_1_qb(&mut self, i: i32) {
        let shard = &mut self.shards[i as usize];
        if shard.pauli_basis == PauliZ {
            self.convert_z_to_x(i);
        } else if shard.pauli_basis == PauliY {
            self.revert_basis_y(i);
        }
    }

    fn revert_basis_to_y_1_qb(&mut self, i: i32) {
        let shard = &mut self.shards[i as usize];
        if shard.pauli_basis == PauliZ {
            self.convert_z_to_y(i);
        } else if shard.pauli_basis == PauliX {
            self.convert_x_to_y(i);
        }
    }

    fn convert_z_to_x(&mut self, i: i32) {
        let shard = &mut self.shards[i as usize];
        shard.pauli_basis = PauliX;
        let mtrx = [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        if shard.unit.is_some() {
            shard.unit.mtrx(&mtrx, shard.mapped);
        }
        if shard.is_phase_dirty || shard.is_prob_dirty {
            shard.is_prob_dirty = true;
            return;
        }
        let z0 = shard.amp0;
        shard.amp0 = mtrx[0] * z0 + mtrx[1] * shard.amp1;
        shard.amp1 = mtrx[2] * z0 + mtrx[3] * shard.amp1;
        self.clamp_shard(i);
    }

    fn convert_x_to_y(&mut self, i: i32) {
        let shard = &mut self.shards[i as usize];
        shard.pauli_basis = PauliY;
        let mtrx = [Complex::new(0.0, 0.0), Complex::new(0.0, -1.0), Complex::new(0.0, 1.0), Complex::new(0.0, 0.0)];
        if shard.unit.is_some() {
            shard.unit.mtrx(&mtrx, shard.mapped);
        }
        if shard.is_phase_dirty || shard.is_prob_dirty {
            shard.is_prob_dirty = true;
            return;
        }
        let x0 = shard.amp0;
        shard.amp0 = mtrx[0] * x0 + mtrx[1] * shard.amp1;
        shard.amp1 = mtrx[2] * x0 + mtrx[3] * shard.amp1;
        self.clamp_shard(i);
    }

    fn convert_y_to_z(&mut self, i: i32) {
        let shard = &mut self.shards[i as usize];
        shard.pauli_basis = PauliZ;
        let mtrx = [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0), Complex::new(0.0, 0.0)];
        if shard.unit.is_some() {
            shard.unit.mtrx(&mtrx, shard.mapped);
        }
        if shard.is_phase_dirty || shard.is_prob_dirty {
            shard.is_prob_dirty = true;
            return;
        }
        let y0 = shard.amp0;
        shard.amp0 = mtrx[0] * y0 + mtrx[1] * shard.amp1;
        shard.amp1 = mtrx[2] * y0 + mtrx[3] * shard.amp1;
        self.clamp_shard(i);
    }

    fn dirty_shard_range(&mut self, start: i32, length: i32) {
        for i in 0..length {
            self.shards[start + i].make_dirty();
        }
    }

    fn dirty_shard_range_phase(&mut self, start: i32, length: i32) {
        for i in 0..length {
            self.shards[start + i].is_phase_dirty = true;
        }
    }

    fn dirty_shard_index_vector(&mut self, bit_indices: Vec<i32>) {
        for i in 0..bit_indices.len() {
            self.shards[bit_indices[i] as usize].make_dirty();
        }
    }

    fn end_emulation(&mut self, target: i32) {
        let shard = &mut self.shards[target as usize];
        if shard.unit.is_some() {
            return;
        }
        if shard.amp1.norm() <= FP_NORM_EPSILON {
            shard.unit = Some(self.make_engine(1, ZERO_BCI));
        } else if shard.amp0.norm() <= FP_NORM_EPSILON {
            shard.unit = Some(self.make_engine(1, ONE_BCI));
        } else {
            let bit_state = [shard.amp0, shard.amp1];
            shard.unit = Some(self.make_engine(1, ZERO_BCI));
            shard.unit.set_quantum_state(&bit_state);
        }
    }

    fn find_shard_index(&mut self, shard: QEngineShardPtr) -> i32 {
        shard.found = true;
        for i in 0..self.shards.len() {
            if self.shards[i].found {
                shard.found = false;
                return i as i32;
            }
        }
        shard.found = false;
        self.shards.len() as i32
    }
}



