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

fn is_amp_0(c: Complex) -> bool {
    (2.0 * c.norm()) <= separability_threshold
}

fn queued_phase(shard: &QEngineShard) -> bool {
    shard.target_of_shards.len() != 0 || shard.controls_shards.len() != 0 ||
    shard.anti_target_of_shards.len() != 0 || shard.anti_controls_shards.len() != 0
}

fn shard_state(shard: &Shard) -> bool {
    (2 * norm(shard.amp0)) < ONE_R1
}

fn cached_x(shard: &Shard) -> bool {
    (shard.pauliBasis == PauliX) && !dirty(shard) && !queued_phase(shard)
}

fn cached_x_or_y(shard: &Shard) -> bool {
    (shard.pauliBasis != PauliZ) && !dirty(shard) && !queued_phase(shard)
}

fn cached_z(shard: &Shard) -> bool {
    (shard.pauliBasis == PauliZ) && !dirty(shard) && !queued_phase(shard)
}

fn cached_zero(q: usize) -> bool {
    cached_z(&shards[q]) && !(shards[q].unit && shards[q].unit.is_clifford() && shards[q].unit.get_t_injection()) &&
        (prob_base(q) <= FP_NORM_EPSILON)
}

fn cached_one(q: usize) -> bool {
    cached_z(&shards[q]) && !(shards[q].unit && shards[q].unit.is_clifford() && shards[q].unit.get_t_injection()) &&
        ((ONE_R1_F - prob_base(q)) <= FP_NORM_EPSILON)
}

fn cached_plus(q: usize) -> bool {
    cached_x(&shards[q]) && !(shards[q].unit && shards[q].unit.is_clifford() && shards[q].unit.get_t_injection()) &&
        (prob_base(q) <= FP_NORM_EPSILON)
}

fn unsafe_cached_zero_or_one(shard: &Shard) -> bool {
    !shard.is_prob_dirty && (shard.pauliBasis == PauliZ) && (IS_NORM_0(shard.amp0) || IS_NORM_0(shard.amp1))
}

fn unsafe_cached_x(shard: &Shard) -> bool {
    !shard.is_prob_dirty && (shard.pauliBasis == PauliX) && (IS_NORM_0(shard.amp0) || IS_NORM_0(shard.amp1))
}

fn unsafe_cached_one(shard: &Shard) -> bool {
    !shard.is_prob_dirty && (shard.pauliBasis == PauliZ) && IS_NORM_0(shard.amp0)
}

fn unsafe_cached_zero(shard: &Shard) -> bool {
    !shard.is_prob_dirty && (shard.pauliBasis == PauliZ) && IS_NORM_0(shard.amp1)
}

impl QUnit {
    pub fn new(
        eng: Vec<QInterfaceEngine>,
        qBitCount: bitLenInt,
        initState: bitCapInt,
        rgp: qrack_rand_gen_ptr,
        phaseFac: complex,
        doNorm: bool,
        randomGlobalPhase: bool,
        useHostMem: bool,
        deviceID: int64_t,
        useHardwareRNG: bool,
        useSparseStateVec: bool,
        norm_thresh: real1_f,
        devList: Vec<int64_t>,
        qubitThreshold: bitLenInt,
        sep_thresh: real1_f,
    ) -> Self {
        let mut engines = eng;
        if engines.is_empty() {
            engines.push(QINTERFACE_STABILIZER_HYBRID);
        }
        #if ENABLE_ENV_VARS
        if std::env::var("QRACK_QUNIT_SEPARABILITY_THRESHOLD").is_ok() {
            separabilityThreshold = std::env::var("QRACK_QUNIT_SEPARABILITY_THRESHOLD")
                .unwrap()
                .parse::<real1_f>()
                .unwrap();
        }
        #endif
        if qBitCount != 0 {
            SetPermutation(initState);
        }
        QUnit {
            engines,
            freezeBasis2Qb: false,
            useHostRam: useHostMem,
            isSparse: useSparseStateVec,
            isReactiveSeparate: true,
            useTGadget: true,
            thresholdQubits: qubitThreshold,
            separabilityThreshold: sep_thresh,
            logFidelity: 0.0,
            devID: deviceID,
            phaseFactor: phaseFac,
            deviceIDs: devList,
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

    fn make_engine(&self, length: i32, perm: i64) -> QInterfacePtr {
        let to_ret = create_quantum_interface(
            self.engines,
            length,
            perm,
            self.rand_generator,
            self.phase_factor,
            self.do_normalize,
            self.rand_global_phase,
            self.use_host_ram,
            self.dev_id,
            self.use_rdrand,
            self.is_sparse,
            self.amplitude_floor as real1_f,
            self.device_ids,
            self.threshold_qubits,
            self.separability_threshold,
        );
        to_ret.set_concurrency(self.get_concurrency_level());
        to_ret.set_t_injection(self.use_t_gadget);
        to_ret
    }

    pub fn set_permutation(&self, perm: i64, phase_fac: Complex) {
        self.dump();
        self.log_fidelity = 0.0;
        self.shards = QEngineShardMap();
        for i in 0..self.qubit_count {
            self.shards.push(QEngineShard(perm >> i & 1 != 0, self.get_nonunitary_phase()));
        }
    }

    pub fn set_quantum_state(&mut self, input_state: &[Complex]) {
        self.dump();
        if self.qubit_count == 1 {
            let shard = &mut self.shards[0];
            shard.unit = None;
            shard.mapped = 0;
            shard.is_prob_dirty = false;
            shard.is_phase_dirty = false;
            shard.amp0 = input_state[0];
            shard.amp1 = input_state[1];
            shard.pauli_basis = PauliZ;
            if is_amp_0(shard.amp0 - shard.amp1) {
                self.log_fidelity += (shard.amp0 - shard.amp1).norm().log();
                shard.pauli_basis = PauliX;
                shard.amp0 = shard.amp0 / shard.amp0.norm();
                shard.amp1 = Complex::new(0.0, 0.0);
            } else if is_amp_0(shard.amp0 + shard.amp1) {
                self.log_fidelity += (shard.amp0 + shard.amp1).norm().log();
                shard.pauli_basis = PauliX;
                shard.amp1 = shard.amp0 / shard.amp0.norm();
                shard.amp0 = Complex::new(0.0, 0.0);
            } else if is_amp_0((I_CMPLX * input_state[0]) - input_state[1]) {
                self.log_fidelity += ((I_CMPLX * input_state[0]) - input_state[1]).norm().log();
                shard.pauli_basis = PauliY;
                shard.amp0 = shard.amp0 / shard.amp0.norm();
                shard.amp1 = Complex::new(0.0, 0.0);
            } else if is_amp_0((I_CMPLX * input_state[0]) + input_state[1]) {
                self.log_fidelity += ((I_CMPLX * input_state[0]) + input_state[1]).norm().log();
                shard.pauli_basis = PauliY;
                shard.amp1 = shard.amp0 / shard.amp0.norm();
                shard.amp0 = Complex::new(0.0, 0.0);
            }
            return;
        }
        let unit = make_engine(self.qubit_count, ZERO_BCI);
        unit.set_quantum_state(input_state);
        for idx in 0..self.qubit_count {
            self.shards[idx] = QEngineShard::new(unit.clone(), idx);
        }
    }

    pub fn get_quantum_state(&self, output_state: &mut [Complex]) {
        if self.qubit_count == 1 {
            self.revert_basis_1_qb(0);
            if let None = self.shards[0].unit {
                output_state[0] = self.shards[0].amp0;
                output_state[1] = self.shards[0].amp1;
                return;
            }
        }
        let this_copy_shared: QUnitPtr;
        let this_copy: &QUnit;
        if self.shards[0].get_qubit_count() == self.qubit_count {
            self.to_perm_basis_all();
            self.order_contiguous(self.shards[0].unit.as_ref().unwrap());
            this_copy = self;
        } else {
            this_copy_shared = self.clone().into();
            this_copy_shared.entangle_all();
            this_copy = this_copy_shared.as_ref();
        }
        this_copy.shards[0].unit.as_ref().unwrap().get_quantum_state(output_state);
    }

    pub fn get_probs(&self, output_probs: &mut [Real]) {
        if self.qubit_count == 1 {
            self.revert_basis_1_qb(0);
            if let None = self.shards[0].unit {
                output_probs[0] = self.shards[0].amp0.norm();
                output_probs[1] = self.shards[0].amp1.norm();
                return;
            }
        }
        let this_copy_shared: QUnitPtr;
        let this_copy: &QUnit;
        if self.shards[0].get_qubit_count() == self.qubit_count {
            self.to_perm_basis_prob();
            self.order_contiguous(self.shards[0].unit.as_ref().unwrap());
            this_copy = self;
        } else {
            this_copy_shared = self.clone().into();
            this_copy_shared.entangle_all(true);
            this_copy = this_copy_shared.as_ref();
        }
        this_copy.shards[0].unit.as_ref().unwrap().get_probs(output_probs);
    }

    pub fn get_amplitude(&self, perm: bit_cap_int) -> Complex {
        self.get_amplitude_or_prob(perm, false)
    }

    fn get_amplitude_or_prob(&self, perm: bit_cap_int, is_prob: bool) -> Complex {
        if perm >= self.max_q_power {
            panic!("QUnit::GetAmplitudeOrProb argument out-of-bounds!");
        }
        if is_prob {
            self.to_perm_basis_prob();
        } else {
            self.to_perm_basis_all();
        }
        let mut result = Complex::new(1.0, 0.0);
        let mut perms = HashMap::new();
        for i in 0..self.qubit_count {
            let shard = &self.shards[i];
            if shard.unit.is_none() {
                result *= if bi_and_1(perm >> i) { shard.amp1 } else { shard.amp0 };
                continue;
            }
            if !perms.contains_key(&shard.unit) {
                perms.insert(shard.unit.clone(), 0);
            }
            if bi_and_1(perm >> i) {
                bi_or_ip(perms.get_mut(&shard.unit).unwrap(), pow2(shard.mapped));
            }
        }
        for (qi, perm) in perms {
            result *= qi.get_amplitude(perm);
            if is_amp_0(result) {
                break;
            }
        }
        result
    }

    fn detach(&self, start: u32, length: u32, dest: Option<&mut QUnitPtr>) {
        if self.is_bad_bit_range(start, length, qubit_count) {
            panic!("QUnit::Detach range is out-of-bounds!");
        }
        for i in 0..length {
            self.revert_basis_2_qb(start + i);
        }

        let mut subunits: HashMap<&QInterfacePtr, u32> = HashMap::new();
        for i in 0..length {
            let shard = &mut shards[start + i];
            if let Some(unit) = &shard.unit {
                *subunits.entry(unit).or_insert(0) += 1;
            } else if let Some(dest) = dest {
                dest.shards[i] = *shard;
            }
        }

        if length > 1 {
            for subunit in subunits.keys() {
                self.order_contiguous(*subunit);
            }
        }

        let mut decomposed_units: HashMap<&QInterfacePtr, u32> = HashMap::new();
        for i in 0..length {
            let shard = &mut self.shards[start + i];
            let unit = shard.unit;
            if let Some(unit) = unit {
                if !decomposed_units.contains_key(&unit) {
                    decomposed_units.insert(unit, start + i);
                    let sub_len = subunits[&unit];
                    let orig_len = unit.get_qubit_count();
                    if sub_len != orig_len {
                        if let Some(dest) = dest {
                            let n_unit = make_engine(sub_len, ZERO_BCI);
                            shard.unit.unwrap().decompose(shard.mapped, n_unit);
                            shard.unit = Some(n_unit);
                        } else {
                            shard.unit.unwrap().dispose(shard.mapped, sub_len);
                        }
                        if sub_len == 1 && dest.is_some() {
                            let mut amps = [Complex::default(); 2];
                            shard.unit.unwrap().get_quantum_state(&mut amps);
                            shard.amp0 = amps[0];
                            shard.amp1 = amps[1];
                            shard.is_prob_dirty = false;
                            shard.is_phase_dirty = false;
                            shard.unit = None;
                            shard.mapped = 0;
                            shard.clamp_amps();
                        }
                        if sub_len == orig_len - 1 {
                            let mapped = self.shards[decomposed_units[&unit]].mapped;
                            let mapped = if mapped == 0 { sub_len } else { 0 };
                            for shard in &mut self.shards {
                                if shard.unit == unit && shard.mapped == mapped {
                                    let mut amps = [Complex::default(); 2];
                                    shard.unit.unwrap().get_quantum_state(&mut amps);
                                    shard.amp0 = amps[0];
                                    shard.amp1 = amps[1];
                                    shard.is_prob_dirty = false;
                                    shard.is_phase_dirty = false;
                                    shard.unit = None;
                                    shard.mapped = 0;
                                    shard.clamp_amps();
                                    break;
                                }
                            }
                        }
                    }
                } else {
                    shard.unit = shards[decomposed_units[&unit]].unit;
                }
            }
            if let Some(dest) = dest {
                dest.shards[i] = *shard;
            }
        }

        for shard in &mut self.shards {
            if let Some(subunit) = subunits.get(&shard.unit) {
                if shard.mapped >= shards[decomposed_units[&shard.unit]].mapped + subunit {
                    shard.mapped -= subunit;
                }
            }
        }
        self.shards.drain(start as usize..(start + length) as usize);
        self.set_qubit_count(qubit_count - length);
    }

    fn entangle_in_current_basis(&self, first: &mut VecDeque<u32>, last: &mut VecDeque<u32>) -> QInterfacePtr {
        for bit in first.iter() {
            end_emulation(*bit);
        }
        let mut units: Vec<QInterfacePtr> = Vec::with_capacity(last.len());
        let mut found: HashMap<&QInterfacePtr, bool> = HashMap::new();

        for bit in last.iter() {
            if !found.contains_key(&(self.shards)[*bit].unit) {
                found.insert(&(self.shards)[*bit].unit, true);
                units.push(self.shards[*bit].unit);
            }
        }

        while units.len() > 1 {
            if units.len() & 1 == 1 {
                let consumed = units[1].clone();
                let offset = unit1.compose_no_clone(consumed);
                units.remove(1);
                for shard in &mut shards {
                    if shard.unit == consumed {
                        shard.mapped += offset;
                        shard.unit = unit1.clone();
                    }
                }
            }
            let mut n_units: Vec<QInterfacePtr> = Vec::new();
            let mut offsets: HashMap<&QInterfacePtr, u32> = HashMap::new();
            let mut offset_partners: HashMap<&QInterfacePtr, QInterfacePtr> = HashMap::new();
            for i in (0..units.len()).step_by(2) {
                let retained = units[i].clone();
                let consumed = units[i + 1].clone();
                n_units.push(retained.clone());
                offsets.insert(&consumed, retained.compose_no_clone(consumed));
                offset_partners.insert(&consumed, retained);
            }

            for shard in &mut self.shards {
                if let Some(offset) = offsets.get(&shard.unit) {
                    shard.mapped += *offset;
                    shard.unit = offset_partners[&shard.unit].clone();
                }
            }
            units = n_units;
        }

        for bit in first.iter() {
            *bit = self.shards[*bit].mapped;
        }
        unit1
    }

    pub fn allocate(&self, start: u32, length: u32) -> u32 {
        if length == 0 {
            return start;
        }
        let n_qubits = QUnitPtr::new(
            engines,
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
            amplitude_floor as real1_f,
            device_ids,
            threshold_qubits,
            separability_threshold,
        );
        n_qubits.set_reactive_separate(is_reactive_separate);
        n_qubits.set_t_injection(use_t_gadget);
        self.compose(n_qubits, start)
    }

    fn entangle(&self, bits: &mut Vec<u32>) -> QInterfacePtr {
        bits.sort();
        let mut ebits: VecDeque<&mut u32> = bits.iter_mut().collect();
        self.entangle_in_current_basis(&mut ebits, &mut ebits)
    }

    fn entangle_range(&self, start: u32, length: u32, is_for_prob: bool) -> QInterfacePtr {
        if is_for_prob {
            self.to_perm_basis_prob(start, length);
        } else {
            self.to_perm_basis(start, length);
        }
        if length == 1 {
            self.end_emulation(start);
            return shards[start].unit.clone();
        }
        let mut bits: Vec<u32> = (start..start + length).collect();
        let mut ebits: VecDeque<&mut u32> = bits.iter_mut().collect();
        let to_ret = entangle_in_current_basis(&mut ebits, &mut ebits);
        self.order_contiguous(&to_ret);
        to_ret
    }

    fn entangle_range(&self, start1: u32, length1: u32, start2: u32, length2: u32) -> QInterfacePtr {
        self.to_perm_basis(start1, length1);
        self.to_perm_basis(start2, length2);
        let mut bits: Vec<u32> = Vec::with_capacity((length1 + length2) as usize);
        let mut ebits: VecDeque<&mut u32> = VecDeque::with_capacity((length1 + length2) as usize);
        if start2 < start1 {
            std::mem::swap(&mut start1, &mut start2);
            std::mem::swap(&mut length1, &mut length2);
        }
        for i in 0..length1 {
            bits.push(i + start1);
            ebits.push_back(&mut bits[i as usize]);
        }
        for i in 0..length2 {
            bits.push(i + start2);
            ebits.push_back(&mut bits[(i + length1) as usize]);
        }
        let to_ret = entangle_in_current_basis(&mut ebits, &mut ebits);
        self.order_contiguous(&to_ret);
        to_ret
    }

    fn entangle_range(&self, start1: u32, length1: u32, start2: u32, length2: u32, start3: u32, length3: u32) -> QInterfacePtr {
        self.to_perm_basis(start1, length1);
        self.to_perm_basis(start2, length2);
        self.to_perm_basis(start3, length3);
        let mut bits: Vec<u32> = Vec::with_capacity((length1 + length2 + length3) as usize);
        let mut ebits: VecDeque<&mut u32> = VecDeque::with_capacity((length1 + length2 + length3) as usize);
        if start2 < start1 {
            std::mem::swap(&mut start1, &mut start2);
            std::mem::swap(&mut length1, &mut length2);
        }
        if start3 < start1 {
            std::mem::swap(&mut start1, &mut start3);
            std::mem::swap(&mut length1, &mut length3);
        }
        if start3 < start2 {
            std::mem::swap(&mut start2, &mut start3);
            std::mem::swap(&mut length2, &mut length3);
        }
        for i in 0..length1 {
            bits.push(i + start1);
            ebits.push_back(&mut bits[i as usize]);
        }
        for i in 0..length2 {
            bits.push(i + start2);
            ebits.push_back(&mut bits[(i + length1) as usize]);
        }
        for i in 0..length3 {
            bits.push(i + start3);
            ebits.push_back(&mut bits[(i + length1 + length2) as usize]);
        }
        let to_ret = entangle_in_current_basis(&mut ebits, &mut ebits);
        self.order_contiguous(&to_ret);
        to_ret
    }

    fn try_separate_clifford(&mut self, qubit: bitLenInt) -> bool {
        let shard = &mut self.shards[qubit];
        if !shard.unit.try_separate(shard.mapped) {
            return false;
        }
        
        let sep_unit = shard.unit.decompose(shard.mapped, 1);
        let is_pair = shard.unit.get_qubit_count() == 1;
        let mut o_qubit = 0;
        for i in 0..self.shards.len() {
            if shard.unit == self.shards[i].unit && shard.mapped != self.shards[i].mapped {
                o_qubit = i;
                if shard.mapped < self.shards[i].mapped {
                    self.shards[i].mapped -= 1;
                }
            }
        }
        shard.mapped = 0;
        shard.unit = sep_unit;
        self.prob_base(qubit);
        if is_pair {
            self.prob_base(o_qubit);
        }
        true
    }

    fn try_separate(&mut self, qubits: &[bitLenInt], error_tol: real1_f) -> bool {
        if qubits.len() == 1 {
            let qubit = qubits[0];
            let shard = &mut self.shards[qubit];
            if shard.get_qubit_count() == 1 {
                if shard.unit.is_some() {
                    self.prob_base(qubit);
                }
                return true;
            }
            if self.blocked_separate(shard) {
                return false;
            }
            let mapped = shard.mapped;
            let o_unit = shard.unit;
            let n_unit = self.make_engine(1, ZERO_BCI);
            if o_unit.try_decompose(mapped, n_unit, error_tol) {
                for i in 0..self.shards.len() {
                    if self.shards[i].unit == o_unit && self.shards[i].mapped > mapped {
                        self.shards[i].mapped -= 1;
                    }
                }
                shard.unit = n_unit;
                shard.mapped = 0;
                shard.make_dirty();
                self.prob_base(qubit);
                if o_unit.get_qubit_count() == 1 {
                    return true;
                }
                for i in 0..self.shards.len() {
                    if shard.unit == o_unit {
                        self.prob_base(i);
                        break;
                    }
                }
                return true;
            }
            return false;
        }
        let mut q = qubits.to_vec();
        q.sort();
        
        for i in 0..q.len() {
            self.swap(i, q[i]);
        }
        let dest = QUnit::new(engines, q.len(), ZERO_BCI, rand_generator, ONE_CMPLX, do_normalize, rand_global_phase, use_host_ram);
        let to_ret = self.try_decompose(0, &dest, error_tol);
        if to_ret {
            if q.len() == 1 {
                dest.prob_base(0);
            }
            self.compose(&dest, 0);
        }
        for i in 0..q.len() {
            self.swap(i, q[i]);
        }
        to_ret
    }

    fn try_separate(&mut self, qubit: bitLenInt) -> bool {
        if qubit >= self.shards.len() {
            panic!("QUnit::TrySeparate target parameter must be within allocated qubit bounds!");
        }
        let shard = &mut self.shards[qubit];
        if shard.get_qubit_count() == 1 {
            if shard.unit.is_some() {
                self.prob_base(qubit);
            }
            return true;
        }
        if shard.unit.is_clifford() {
            return self.try_separate_clifford(qubit);
        }
        let mut prob;
        let mut x = ZERO_R1_F;
        let mut y = ZERO_R1_F;
        let mut z = ZERO_R1_F;
        for i in 0..3 {
            prob = ONE_R1_F - 2 * self.prob_base(qubit);
            if shard.unit.is_none() {
                return true;
            }
            if shard.pauli_basis == PauliZ {
                z = prob;
            } else if shard.pauli_basis == PauliX {
                x = prob;
            } else {
                y = prob;
            }
            if i >= 2 {
                continue;
            }
            if shard.pauli_basis == PauliZ {
                self.convert_z_to_x(qubit);
            } else if shard.pauli_basis == PauliX {
                self.convert_x_to_y(qubit);
            } else {
                self.convert_y_to_z(qubit);
            }
        }
        let one_min_r = 1.0 - f64::sqrt((x * x + y * y + z * z) as f64);
        if one_min_r > separability_threshold {
            return false;
        }
        
        if shard.pauli_basis == PauliX {
            self.revert_basis_1_qb(qubit);
        } else if shard.pauli_basis == PauliY {
            std::mem::swap(&mut x, &mut z);
            std::mem::swap(&mut y, &mut z);
        }
        let inclination = f64::atan2(f64::sqrt(x * x + y * y), z);
        let azimuth = f64::atan2(y, x);
        shard.unit.iai(shard.mapped, azimuth, inclination);
        prob = 2 * shard.unit.prob(shard.mapped);
        if prob > separability_threshold {
            shard.unit.ai(shard.mapped, azimuth, inclination);
            return false;
        }
        self.separate_bit(false, qubit);
        self.shard_ai(qubit, azimuth, inclination);
        self.log_fidelity += f64::log(clamp_prob(1.0 - one_min_r / 2));
        true
    }

    fn try_separate(&mut self, qubit1: bitLenInt, qubit2: bitLenInt) -> bool {
        if qubit1 >= self.shards.len() {
            panic!("QUnit::TrySeparate target parameter must be within allocated qubit bounds!");
        }
        if qubit2 >= self.shards.len() {
            panic!("QUnit::TrySeparate target parameter must be within allocated qubit bounds!");
        }
        let shard1 = &mut self.shards[qubit1];
        let shard2 = &mut self.shards[qubit2];
        if self.freeze_basis_2_qb || shard1.unit.is_none() || shard2.unit.is_none() || shard1.unit != shard2.unit {
            
            let is_shard1_sep = self.try_separate(qubit1);
            let is_shard2_sep = self.try_separate(qubit2);
            return is_shard1_sep && is_shard2_sep;
        }
        let unit = shard1.unit;
        let mapped1 = shard1.mapped;
        let mapped2 = shard2.mapped;
        
        if unit.is_clifford() && !unit.try_separate(mapped1, mapped2) {
            return false;
        }
        if self.queued_phase(shard1) || self.queued_phase(shard2) {
            
            let is_shard1_sep = self.try_separate(qubit1);
            let is_shard2_sep = self.try_separate(qubit2);
            return is_shard1_sep && is_shard2_sep;
        }
        self.revert_basis_1_qb(qubit1);
        self.revert_basis_1_qb(qubit2);
        
        let mtrx = [complex(SQRT1_2_R1, ZERO_R1), complex(ZERO_R1, -SQRT1_2_R1), complex(SQRT1_2_R1, ZERO_R1),
            complex(ZERO_R1, SQRT1_2_R1)];
        let controls = vec![mapped1];
        let z = ONE_R1_F - 2 * unit.cprob(mapped1, mapped2);
        unit.ch(shard1.mapped, shard2.mapped);
        let x = ONE_R1_F - 2 * unit.cprob(mapped1, mapped2);
        unit.cs(shard1.mapped, shard2.mapped);
        let y = ONE_R1_F - 2 * unit.cprob(mapped1, mapped2);
        unit.mcmtrx(&controls, &mtrx, mapped2);
        let inclination = f64::atan2(f64::sqrt(x * x + y * y), z);
        let azimuth = f64::atan2(y, x);
        unit.ciai(mapped1, mapped2, azimuth, inclination);
        let z = ONE_R1_F - 2 * unit.acprob(mapped1, mapped2);
        unit.anti_ch(shard1.mapped, shard2.mapped);
        let x = ONE_R1_F - 2 * unit.acprob(mapped1, mapped2);
        unit.anti_cs(shard1.mapped, shard2.mapped);
        let y = ONE_R1_F - 2 * unit.acprob(mapped1, mapped2);
        unit.macmtrx(&controls, &mtrx, mapped2);
        let inclination_anti = f64::atan2(f64::sqrt(x * x + y * y), z);
        let azimuth_anti = f64::atan2(y, z);
        unit.anti_ciai(mapped1, mapped2, azimuth_anti, inclination_anti);
        shard1.make_dirty();
        shard2.make_dirty();
        let is_shard1_sep = self.try_separate(qubit1);
        let is_shard2_sep = self.try_separate(qubit2);
        self.anti_cai(qubit1, qubit2, azimuth_anti, inclination_anti);
        self.cai(qubit1, qubit2, azimuth, inclination);
        is_shard1_sep && is_shard2_sep
    }

    fn order_contiguous(&self, unit: &QInterfacePtr) {
        if unit.is_none() || (unit.get_qubit_count() == 1) {
            return;
        }
        
        let mut bits = Vec::new();
        let mut j = 0;
        for i in 0..self.qubit_count {
            if self.shards[i].unit == *unit {
                bits.push(QSortEntry {
                    mapped: self.shards[i].mapped,
                    bit: i,
                });
                j += 1;
            }
        }
        self.sort_unit(unit, &mut bits, 0, bits.len() - 1);
    }
    
    fn sort_unit(&self, unit: &QInterfacePtr, bits: &mut [QSortEntry], low: usize, high: usize) {
        let mut i = low;
        let mut j = high;
        if i == (j - 1) {
            if bits[j] < bits[i] {
                unit.swap(bits[i].mapped, bits[j].mapped);
                self.shards[bits[i].bit].mapped.swap(&mut self.shards[bits[j].bit].mapped);
                bits.swap(i, j);
            }
            return;
        }
        let pivot = bits[(low + high) / 2];
        while i <= j {
            while bits[i] < pivot {
                i += 1;
            }
            while bits[j] > pivot {
                j -= 1;
            }
            if i < j {
                unit.swap(bits[i].mapped, bits[j].mapped);
                self.shards[bits[i].bit].mapped.swap(&mut self.shards[bits[j].bit].mapped);
                bits.swap(i, j);
                i += 1;
                j -= 1;
            } else if i == j {
                i += 1;
                j -= 1;
            }
        }
        if low < j {
            self.sort_unit(unit, bits, low, j);
        }
        if i < high {
            self.sort_unit(unit, bits, i, high);
        }
    }
    
    fn check_bits_permutation(&self, start: usize, length: usize) -> bool {
        self.to_perm_basis_prob(start, length);
        for i in 0..length {
            let shard = &self.shards[start + i];
            if !self.unsafe_cached_zero_or_one(shard) {
                return false;
            }
        }
        true
    }
    
    fn get_cached_permutation(&self, start: usize, length: usize) -> bitCapInt {
        let mut res = ZERO_BCI;
        for i in 0..length {
            if self.shard_state(&self.shards[start + i]) {
                bi_or_ip(&mut res, pow2(i));
            }
        }
        res
    }
    
    fn get_cached_permutation_bit_array(&self, bit_array: &[usize]) -> bitCapInt {
        let mut res = ZERO_BCI;
        for i in 0..bit_array.len() {
            if self.shard_state(&self.shards[bit_array[i]]) {
                bi_or_ip(&mut res, pow2(i));
            }
        }
        res
    }
    
    fn check_bits_plus(&self, qubit_index: usize, length: usize) -> bool {
        let mut is_h_basis = true;
        for i in 0..length {
            if !self.cached_plus(qubit_index + i) {
                is_h_basis = false;
                break;
            }
        }
        is_h_basis
    }

    pub fn phase_parity(&mut self, radians: f64, mask: i64) {
        if mask >= self.max_q_power {
            panic!("QUnit::PhaseParity mask out-of-bounds!");
        }
        
        if mask == 0 {
            return;
        }
        
        let phase_fac = Complex::new((radians / 2.0).cos(), (radians / 2.0).sin());
        
        if mask.is_power_of_two() {
            phase(Complex::new(1.0 / phase_fac.re, 1.0 / phase_fac.im), phase_fac, (mask as f64).log2() as usize);
            return;
        }
        
        let mut n_v = mask;
        let mut q_indices = Vec::new();
        
        while n_v != 0 {
            let v = n_v;
            n_v &= v - 1;
            q_indices.push(((v ^ n_v) & v).log2() as usize);
            to_perm_basis_prob(*q_indices.last().unwrap());
        }
        
        let mut flip_result = false;
        let mut e_indices = Vec::new();
        
        for i in 0..q_indices.len() {
            let shard = &mut self.shards[q_indices[i]];
            
            if shard.amp0 == 0.0 && shard.amp1 == 0.0 {
                continue;
            }
            
            if shard.amp0 == 1.0 && shard.amp1 == 0.0 {
                flip_result = !flip_result;
                continue;
            }
            
            e_indices.push(q_indices[i]);
        }
        
        if e_indices.is_empty() {
            if flip_result {
                phase(phase_fac, phase_fac, 0);
            } else {
                phase(Complex::new(1.0 / phase_fac.re, 1.0 / phase_fac.im), Complex::new(1.0 / phase_fac.re, 1.0 / phase_fac.im), 0);
            }
            return;
        }
        
        if e_indices.len() == 1 {
            if flip_result {
                phase(phase_fac, Complex::new(1.0 / phase_fac.re, 1.0 / phase_fac.im), (mask as f64).log2() as usize);
            } else {
                phase(Complex::new(1.0 / phase_fac.re, 1.0 / phase_fac.im), phase_fac, (mask as f64).log2() as usize);
            }
            return;
        }
        
        let unit = entangle(&e_indices);
        
        for i in 0..self.qubit_count {
            if self.shards[i].unit == unit {
                self.shards[i].make_dirty();
            }
        }
        
        let mut mapped_mask = 0;
        
        for i in 0..e_indices.len() {
            mapped_mask |= 1 << self.shards[e_indices[i]].mapped;
        }
        
        unit.phase_parity(if flip_result { -radians } else { radians }, mapped_mask);
    }
    
    pub fn prob_parity(&mut self, mask: i64) -> f64 {
        if mask >= self.max_q_power {
            panic!("QUnit::ProbParity mask out-of-bounds!");
        }
        
        if mask == 0 {
            return 0.0;
        }
        
        if mask.is_power_of_two() {
            return prob((mask as f64).log2() as usize);
        }
        
        let mut n_v = mask;
        let mut q_indices = Vec::new();
        
        while n_v != 0 {
            let v = n_v;
            n_v &= v - 1;
            q_indices.push(((v ^ n_v) & v).log2() as usize);
            revert_basis_2_qb(*q_indices.last().unwrap());
            let shard = &mut self.shards[*q_indices.last().unwrap()];
            
            if let Some(unit) = &shard.unit {
                if queued_phase(shard) {
                    revert_basis_1_qb(*q_indices.last().unwrap());
                }
            }
        }
        
        let mut units = HashMap::new();
        let mut odd_chance = 0.0;
        let mut n_odd_chance;
        
        for i in 0..q_indices.len() {
            let shard = &mut self.shards[q_indices[i]];
            
            if shard.unit.is_none() {
                n_odd_chance = if shard.pauli_basis != PauliZ {
                    (FRAC_1_SQRT_2 * (shard.amp0 - shard.amp1)).norm()
                } else {
                    shard.prob()
                };
                
                odd_chance = (odd_chance * (1.0 - n_odd_chance)) + ((1.0 - odd_chance) * n_odd_chance);
                continue;
            }
            
            revert_basis_1_qb(q_indices[i]);
            *units.entry(shard.unit.clone().unwrap()).or_insert(0) |= 1 << shard.mapped;
        }
        
        if q_indices.is_empty() {
            return odd_chance;
        }
        
        let mut result = 0.0;
        
        for (unit, mapped_mask) in units {
            let n_odd_chance = unit.prob_parity(mapped_mask);
            result = (result * (1.0 - n_odd_chance)) + ((1.0 - result) * n_odd_chance);
        }
        
        result
    }
    
    pub fn force_m_parity(&mut self, mask: i64, result: bool, do_force: bool) -> bool {
        if mask >= self.max_q_power {
            panic!("QUnit::ForceMParity mask out-of-bounds!");
        }
        
        if mask == 0 {
            return false;
        }
        
        if mask.is_power_of_two() {
            return force_m((mask as f64).log2() as usize, result, do_force);
        }
        
        let mut n_v = mask;
        let mut q_indices = Vec::new();
        
        while n_v != 0 {
            let v = n_v;
            n_v &= v - 1;
            q_indices.push(((v ^ n_v) & v).log2() as usize);
            to_perm_basis_prob(*q_indices.last().unwrap());
        }
        
        let mut flip_result = false;
        let mut e_indices = Vec::new();
        
        for i in 0..q_indices.len() {
            let shard = &mut self.shards[q_indices[i]];
            
            if shard.amp0 == 0.0 && shard.amp1 == 0.0 {
                continue;
            }
            
            if shard.amp0 == 1.0 && shard.amp1 == 0.0 {
                flip_result = !flip_result;
                continue;
            }
            
            e_indices.push(q_indices[i]);
        }
        
        if e_indices.is_empty() {
            return flip_result;
        }
        
        if e_indices.len() == 1 {
            return flip_result ^ force_m(e_indices[0], result ^ flip_result, do_force);
        }
        
        let unit = entangle(&e_indices);
        
        for i in 0..self.qubit_count {
            if self.shards[i].unit == unit {
                self.shards[i].make_dirty();
            }
        }
        
        let mut mapped_mask = 0;
        
        for i in 0..e_indices.len() {
            mapped_mask |= 1 << self.shards[e_indices[i]].mapped;
        }
        
        flip_result ^ unit.force_m_parity(mapped_mask, result ^ flip_result, do_force)
    }

    pub fn c_uniform_parity_rz(&self, c_controls: &[usize], mask: u64, angle: f64) {
        let max_q_power = self.max_q_power;
        if mask >= max_q_power {
            panic!("QUnit::CUniformParityRZ mask out-of-bounds!");
        }
        self.throw_if_qb_id_array_is_bad(
            c_controls,
            self.qubit_count,
            "QUnit::CUniformParityRZ parameter controls array values must be within allocated qubit bounds!",
        );
        let mut controls = Vec::new();
        let mut perm = 2usize.pow(c_controls.len() as u32) - 1;
        if self.trim_controls(c_controls, &mut controls, &mut perm) {
            return;
        }
        let mut n_v = mask;
        let mut q_indices = Vec::new();
        while n_v != 0 {
            let v = n_v;
            n_v &= v - 1;
            q_indices.push((v ^ n_v & v).log2());
        }
        let mut flip_result = false;
        let mut e_indices = Vec::new();
        for i in &q_indices {
            self.to_perm_basis(*i);
            if self.cached_zero(*i) {
                continue;
            }
            if self.cached_one(*i) {
                flip_result = !flip_result;
                continue;
            }
            e_indices.push(*i);
        }
        if e_indices.is_empty() {
            let cosine = angle.cos();
            let sine = angle.sin();
            let phase_fac = if flip_result {
                Complex::new(cosine, sine)
            } else {
                Complex::new(cosine, -sine)
            };
            if controls.is_empty() {
                self.phase(phase_fac, phase_fac, 0);
            } else {
                self.mc_phase(&controls, phase_fac, phase_fac, 0);
            }
            return;
        }
        if e_indices.len() == 1 {
            let cosine = angle.cos();
            let sine = angle.sin();
            let phase_fac = if flip_result {
                Complex::new(cosine, -sine)
            } else {
                Complex::new(cosine, sine)
            };
            if controls.is_empty() {
                self.phase(phase_fac.conj(), phase_fac, e_indices[0]);
            } else {
                self.mc_phase(&controls, phase_fac.conj(), phase_fac, e_indices[0]);
            }
            return;
        }
        for i in &e_indices {
            self.shards[*i].is_phase_dirty = true;
        }
        let unit = self.entangle(&e_indices);
        let mut mapped_mask = 0;
        for i in &e_indices {
            mapped_mask |= 1 << self.shards[*i].mapped;
        }
        if controls.is_empty() {
            std::dynamic_pointer_cast<QParity>(unit).uniform_parity_rz(mapped_mask, if flip_result { -angle } else { angle });
        } else {
            let mut ebits = Vec::with_capacity(controls.len());
            for i in &controls {
                ebits.push(i);
            }
            self.entangle(&ebits);
            let unit = self.entangle(&[controls[0], e_indices[0]]);
            let mut controls_mapped = Vec::with_capacity(controls.len());
            for i in &controls {
                let c_shard = &self.shards[*i];
                controls_mapped.push(c_shard.mapped);
                c_shard.is_phase_dirty = true;
            }
            std::dynamic_pointer_cast<QParity>(unit).c_uniform_parity_rz(
                &controls_mapped,
                mapped_mask,
                if flip_result { -angle } else { angle },
            );
        }
    }

    fn separate_bit(&mut self, value: bool, qubit: usize) -> bool {
        let shard = &mut self.shards[qubit];
        let unit = shard.unit.take();
        let mapped = shard.mapped;
        if let Some(unit) = unit {
            if unit.is_clifford() && !unit.try_separate(mapped) {
                return false;
            }
        }
        shard.unit = None;
        shard.mapped = 0;
        shard.is_prob_dirty = false;
        shard.is_phase_dirty = false;
        shard.amp0 = if value { ZERO_CMPLX } else { get_nonunitary_phase() };
        shard.amp1 = if value { get_nonunitary_phase() } else { ZERO_CMPLX };
        if unit.is_none() || unit.unwrap().get_qubit_count() == 1 {
            return true;
        }
        let prob = ONE_R1_F / 2 - unit.unwrap().prob(mapped);
        unit.unwrap().dispose(mapped, 1, if value { ONE_BCI } else { ZERO_BCI });
        if !unit.unwrap().is_binary_decision_tree() && (ONE_R1 / 2 - prob.abs() > FP_NORM_EPSILON) {
            unit.unwrap().update_running_norm();
            if !do_normalize {
                unit.unwrap().normalize_state();
            }
        }
        for s in &mut self.shards {
            if s.unit == unit && s.mapped > mapped {
                s.mapped -= 1;
            }
        }
        if unit.unwrap().get_qubit_count() != 1 {
            return true;
        }
        for partner_index in 0..qubit_count {
            let partner_shard = &mut self.shards[partner_index];
            if unit == partner_shard.unit {
                prob_base(partner_index);
                break;
            }
        }
        true
    }

    pub fn force_m(&mut self, qubit: usize, res: bool, do_force: bool, do_apply: bool) -> bool {
        if qubit >= qubit_count {
            panic!("QUnit::ForceM target parameter must be within allocated qubit bounds!");
        }
        if do_apply {
            revert_basis_1_qb(qubit);
            revert_basis_2_qb(qubit, ONLY_INVERT, ONLY_TARGETS);
        } else {
            to_perm_basis_measure(qubit);
        }
        let shard = &mut self.shards[qubit];
        let result;
        if shard.unit.is_none() {
            let prob = norm(shard.amp1) as real1_f;
            if do_force {
                result = res;
            } else if prob >= ONE_R1 {
                result = true;
            } else if prob <= ZERO_R1 {
                result = false;
            } else {
                result = rand() <= prob;
            }
        } else {
            result = shard.unit.unwrap().force_m(shard.mapped, res, do_force, do_apply);
        }
        if !do_apply {
            return result;
        }
        shard.is_prob_dirty = false;
        shard.is_phase_dirty = false;
        shard.amp0 = if result { ZERO_CMPLX } else { get_nonunitary_phase() };
        shard.amp1 = if result { get_nonunitary_phase() } else { ZERO_CMPLX };
        if shard.get_qubit_count() == 1 {
            shard.unit = None;
            shard.mapped = 0;
            if result {
                flush_1_eigenstate(qubit);
            } else {
                flush_0_eigenstate(qubit);
            }
            return result;
        }
        if let Some(unit) = shard.unit {
            for i in 0..qubit {
                if self.shards[i].unit == unit {
                    self.shards[i].make_dirty();
                }
            }
            for i in qubit + 1..qubit_count {
                if self.shards[i].unit == unit {
                    self.shards[i].make_dirty();
                }
            }
            separate_bit(result, qubit);
        }
        if result {
            flush_1_eigenstate(qubit);
        } else {
            flush_0_eigenstate(qubit);
        }
        result
    }

    pub fn force_m_reg(&mut self, start: usize, length: usize, result: bitCapInt, do_force: bool, do_apply: bool) -> bitCapInt {
        if is_bad_bit_range(start, length, qubit_count) {
            panic!("QUnit::ForceMReg range is out-of-bounds!");
        }
        if !do_force && do_apply && length == qubit_count {
            return m_all();
        }
        if !do_apply {
            to_perm_basis_measure(start, length);
        }
        QInterface::force_m_reg(start, length, result, do_force, do_apply)
    }

    pub fn m_all(&mut self) -> bitCapInt {
        for i in 0..qubit_count {
            revert_basis_1_qb(i);
        }
        for i in 0..qubit_count {
            let shard = &mut self.shards[i];
            shard.dump_phase_buffers();
            shard.clear_invert_phase();
        }
        if use_t_gadget && engines[0] == QINTERFACE_STABILIZER_HYBRID {
            for i in 0..qubit_count {
                let shard = &mut self.shards[i];
                if let Some(unit) = shard.unit {
                    if unit.is_clifford() {
                        unit.m_all();
                    }
                }
            }
        }
        for i in 0..qubit_count {
            if self.shards[i].is_invert_control() {
                m(i);
            }
        }
        let mut to_ret = ZERO_BCI;
        for i in 0..qubit_count {
            let to_find = self.shards[i].unit;
            if to_find.is_none() {
                let prob = norm(self.shards[i].amp1) as real1_f;
                if prob >= ONE_R1 || (prob > ZERO_R1 && rand() <= prob) {
                    self.shards[i].amp0 = ZERO_CMPLX;
                    self.shards[i].amp1 = get_nonunitary_phase();
                    bi_or_ip(&mut to_ret, pow2(i));
                } else {
                    self.shards[i].amp0 = get_nonunitary_phase();
                    self.shards[i].amp1 = ZERO_CMPLX;
                }
            } else if m(i) {
                bi_or_ip(&mut to_ret, pow2(i));
            }
        }
        let orig_fidelity = log_fidelity;
        set_permutation(to_ret);
        log_fidelity = orig_fidelity;
        to_ret
    }

    pub fn multi_shot_measure_mask(&mut self, q_powers: &[bitCapInt], shots: u32) -> std::collections::HashMap<bitCapInt, i32> {
        if shots == 0 {
            return std::collections::HashMap::new();
        }
        if q_powers.len() == self.shards.len() {
            for i in 0..qubit_count {
                revert_basis_1_qb(i);
            }
        } else {
            to_perm_basis_prob();
        }
        let mut q_indices = Vec::with_capacity(q_powers.len());
        let mut i_q_powers = std::collections::HashMap::new();
        for i in 0..q_powers.len() {
            let index = q_powers[i].log2();
            q_indices.push(index);
            i_q_powers.insert(index, pow2(i));
        }
        throw_if_qb_id_array_is_bad(&q_indices, qubit_count, "QInterface::MultiShotMeasureMask parameter qPowers array values must be within allocated qubit bounds!");
        let mut sub_q_powers = std::collections::HashMap::new();
        let mut sub_i_q_powers = std::collections::HashMap::new();
        let mut single_bits = Vec::new();
        for i in 0..q_powers.len() {
            let index = q_indices[i];
            let shard = &self.shards[index];
            if shard.unit.is_none() {
                single_bits.push(index);
                continue;
            }
            sub_q_powers.entry(shard.unit).or_insert_with(Vec::new).push(pow2(shard.mapped));
            sub_i_q_powers.entry(shard.unit).or_insert_with(Vec::new).push(*i_q_powers.get(&index).unwrap());
        }
        let mut combined_results = std::collections::HashMap::new();
        combined_results.insert(ZERO_BCI, shots as i32);
        for (sub_q_power, sub_q_powers) in sub_q_powers {
            let unit_results = sub_q_power.unwrap().multi_shot_measure_mask(&sub_q_powers, shots);
            let mut top_level_results = std::collections::HashMap::new();
            for (unit_result, count) in unit_results {
                let mut mask = ZERO_BCI;
                for i in 0..sub_q_powers.len() {
                    if bi_and_1(unit_result >> i) {
                        bi_or_ip(&mut mask, *sub_i_q_powers.get(&sub_q_power).unwrap().get(i).unwrap());
                    }
                }
                top_level_results.insert(mask, count);
            }
            if bi_compare_0(top_level_results.iter().next().unwrap().0) == 0 && top_level_results.get(&ZERO_BCI).unwrap() == &(shots as i32) {
                continue;
            }
            if bi_compare_0(combined_results.iter().next().unwrap().0) == 0 && combined_results.get(&ZERO_BCI).unwrap() == &(shots as i32) {
                std::mem::swap(&mut top_level_results, &mut combined_results);
                continue;
            }
            if combined_results.len() < top_level_results.len() {
                std::mem::swap(&mut top_level_results, &mut combined_results);
            }
            let mut n_combined_results = std::collections::HashMap::new();
            if top_level_results.len() == 1 {
                let pick_iter = top_level_results.iter().next().unwrap();
                for (combined_result, _) in &combined_results {
                    n_combined_results.insert(combined_result | pick_iter.0, *combined_result);
                }
                combined_results = n_combined_results;
                continue;
            }
            let mut shots_left = shots;
            for (combined_result, _) in &combined_results {
                for _ in 0..*combined_result {
                    let pick = (shots_left as f64 * rand()) as i32;
                    if shots_left <= pick {
                        pick = shots_left - 1;
                    }
                    shots_left -= 1;
                    let mut pick_iter = top_level_results.iter().next().unwrap();
                    let mut count = *pick_iter.1;
                    while pick > count {
                        pick_iter = top_level_results.iter().next().unwrap();
                        count += *pick_iter.1;
                    }
                    *n_combined_results.entry(combined_result | pick_iter.0).or_insert(0) += 1;
                    *pick_iter.1 -= 1;
                    if *pick_iter.1 == 0 {
                        top_level_results.remove(&pick_iter.0);
                    }
                }
            }
            combined_results = n_combined_results;
        }
        for i in 0..single_bits.len() {
            let index = single_bits[i];
            let prob = clamp_prob(norm(self.shards[index].amp1) as real1_f);
            if prob == ZERO_R1 {
                continue;
            }
            let mut n_combined_results = std::collections::HashMap::new();
            if prob == ONE_R1 {
                for (combined_result, _) in &combined_results {
                    n_combined_results.insert(combined_result | *i_q_powers.get(&index).unwrap(), *combined_result);
                }
            } else {
                for (combined_result, _) in &combined_results {
                    let zero_perm = *combined_result;
                    let one_perm = *combined_result | *i_q_powers.get(&index).unwrap();
                    for _ in 0..*combined_result {
                        if rand() > prob {
                            *n_combined_results.entry(zero_perm).or_insert(0) += 1;
                        } else {
                            *n_combined_results.entry(one_perm).or_insert(0) += 1;
                        }
                    }
                }
            }
            combined_results = n_combined_results;
        }
        if q_powers.len() != self.shards.len() {
            return combined_results;
        }
        let mut to_ret = std::collections::HashMap::new();
        for (combined_result, _) in &combined_results {
            let perm = *combined_result;
            for i in 0..q_indices.len() {
                let shard = &self.shards[q_indices[i]];
                let controls_shards = if bi_and_1(perm >> i) { &shard.controls_shards } else { &shard.anti_controls_shards };
                for (phase_shard, _) in controls_shards {
                    if !phase_shard.is_invert {
                        continue;
                    }
                    let partner = phase_shard.first;
                    let target = find_shard_index(partner);
                    for j in 0..q_indices.len() {
                        if q_indices[j] == target {
                            bi_xor_ip(&mut perm, pow2(j));
                            break;
                        }
                    }
                }
            }
            *to_ret.entry(perm).or_insert(0) += *combined_result;
        }
        to_ret
    }

    pub fn multi_shot_measure_mask(&mut self, q_powers: &[bitCapInt], shots: u32, shots_array: &mut [u64]) {
        if shots == 0 {
            return;
        }
        if q_powers.len() != self.shards.len() {
            to_perm_basis_prob();
            let unit = self.shards[q_powers[0].log2()].unit;
            if let Some(unit) = unit {
                let mut mapped_indices = Vec::with_capacity(q_powers.len());
                for j in 0..qubit_count {
                    if bi_compare(q_powers[0], pow2(j)) == 0 {
                        mapped_indices[0] = pow2(self.shards[j].mapped);
                        break;
                    }
                }
                for i in 1..q_powers.len() {
                    let qubit = q_powers[i].log2();
                    if qubit >= qubit_count {
                        panic!("QUnit::MultiShotMeasureMask parameter qPowers array values must be within allocated qubit bounds!");
                    }
                    if unit != self.shards[qubit].unit {
                        unit = None;
                        break;
                    }
                    for j in 0..qubit_count {
                        if bi_compare(q_powers[i], pow2(j)) == 0 {
                            mapped_indices[i] = pow2(self.shards[j].mapped);
                            break;
                        }
                    }
                }
                if let Some(unit) = unit {
                    unit.multi_shot_measure_mask(&mapped_indices, shots, shots_array);
                    return;
                }
            }
        }
        let results = self.multi_shot_measure_mask(q_powers, shots);
        let mut j = 0;
        let mut it = results.iter();
        while let Some((perm, count)) = it.next() {
            for _ in 0..*count {
                shots_array[j] = *perm as u64;
                j += 1;
            }
        }
    }

    pub fn set_reg(&mut self, start: usize, length: usize, value: u64) {
        self.m_reg(start, length);
        for i in 0..length {
            let i = i as u64;
            let shard = &mut self.shards[i + start];
            let bit = (value >> i) & 1;
            let is_non_zero = bit != 0;
            let phase = Complex::new(0.0, 0.0); // GetNonunitaryPhase() implementation not provided
            shard.amp1 = if is_non_zero {
                QEngineShard::new(shard.pauli_basis, Complex::zero(), phase)
            } else {
                QEngineShard::new(shard.pauli_basis, phase, Complex::zero())
            };
        }
    }

    fn either_i_swap(&mut self, qubit1: usize, qubit2: usize, is_inverse: bool) {
        if qubit1 >= self.qubit_count {
            panic!("QUnit::EitherISwap qubit index parameter must be within allocated qubit bounds!");
        }
        if qubit2 >= self.qubit_count {
            panic!("QUnit::EitherISwap qubit index parameter must be within allocated qubit bounds!");
        }
        if qubit1 == qubit2 {
            return;
        }
        let shard1 = &mut self.shards[qubit1];
        let shard2 = &mut self.shards[qubit2];
        let is_same_unit = shard1.pauli_basis == shard2.pauli_basis;
        let are_clifford = shard1.pauli_basis == PauliBasis::PauliZ
            && shard2.pauli_basis == PauliBasis::PauliZ;
        if is_same_unit || are_clifford {
            // Entangle() implementation not provided
            if is_inverse {
                // unit->IISwap(shard1.mapped, shard2.mapped) implementation not provided
            } else {
                // unit->ISwap(shard1.mapped, shard2.mapped) implementation not provided
            }
            shard1.make_dirty();
            shard2.make_dirty();
            if is_same_unit && !are_clifford {
                self.try_separate(qubit1);
                self.try_separate(qubit2);
            }
            return;
        }
        if is_inverse {
            // QInterface::IISwap(qubit1, qubit2) implementation not provided
        } else {
            // QInterface::ISwap(qubit1, qubit2) implementation not provided
        }
    }

    fn sqrt_swap(&mut self, qubit1: usize, qubit2: usize) {
        if qubit1 >= self.qubit_count {
            panic!("QUnit::SqrtSwap qubit index parameter must be within allocated qubit bounds!");
        }
        if qubit2 >= self.qubit_count {
            panic!("QUnit::SqrtSwap qubit index parameter must be within allocated qubit bounds!");
        }
        if qubit1 == qubit2 {
            return;
        }
        // RevertBasis2Qb(qubit1, ONLY_INVERT) implementation not provided
        // RevertBasis2Qb(qubit2, ONLY_INVERT) implementation not provided
        let shard1 = &mut self.shards[qubit1];
        let shard2 = &mut self.shards[qubit2];
        let is_same_unit = shard1.pauli_basis == shard2.pauli_basis;
        // Entangle() implementation not provided
        shard1.make_dirty();
        shard2.make_dirty();
        if is_same_unit {
            self.try_separate(qubit1);
            self.try_separate(qubit2);
        }
    }

    fn i_sqrt_swap(&mut self, qubit1: usize, qubit2: usize) {
        if qubit1 >= self.qubit_count {
            panic!("QUnit::ISqrtSwap qubit index parameter must be within allocated qubit bounds!");
        }
        if qubit2 >= self.qubit_count {
            panic!("QUnit::ISqrtSwap qubit index parameter must be within allocated qubit bounds!");
        }
        if qubit1 == qubit2 {
            return;
        }
        // RevertBasis2Qb(qubit1, ONLY_INVERT) implementation not provided
        // RevertBasis2Qb(qubit2, ONLY_INVERT) implementation not provided
        let shard1 = &mut self.shards[qubit1];
        let shard2 = &mut self.shards[qubit2];
        let is_same_unit = shard1.pauli_basis == shard2.pauli_basis;
        // Entangle() implementation not provided
        shard1.make_dirty();
        shard2.make_dirty();
        if is_same_unit {
            self.try_separate(qubit1);
            self.try_separate(qubit2);
        }
    }

    fn f_sim(&mut self, theta: f64, phi: f64, qubit1: usize, qubit2: usize) {
        let controls = vec![qubit1];
        let sin_theta = theta.sin();
        if sin_theta * sin_theta <= f64::EPSILON {
            // MCPhase(controls, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2) implementation not provided
            return;
        }
        let exp_i_phi = Complex::new(0.0, phi).exp();
        let was_same_unit = self.shards[qubit1].pauli_basis == self.shards[qubit2].pauli_basis
            && (self.shards[qubit1].pauli_basis != PauliBasis::PauliZ
                || !exp_i_phi.real.approx_eq(&1.0)
                || !exp_i_phi.imag.approx_eq(&0.0));
        let sin_theta_diff_neg = 1.0 + sin_theta;
        if !was_same_unit && sin_theta_diff_neg * sin_theta_diff_neg <= f64::EPSILON {
            self.i_swap(qubit1, qubit2);
            // MCPhase(controls, ONE_CMPLX, exp_i_phi, qubit2) implementation not provided
            return;
        }
        let sin_theta_diff_pos = 1.0 - sin_theta;
        if !was_same_unit && sin_theta_diff_pos * sin_theta_diff_pos <= f64::EPSILON {
            self.ii_swap(qubit1, qubit2);
            // MCPhase(controls, ONE_CMPLX, exp_i_phi, qubit2) implementation not provided
            return;
        }
        if qubit1 >= self.qubit_count {
            panic!("QUnit::FSim qubit index parameter must be within allocated qubit bounds!");
        }
        if qubit2 >= self.qubit_count {
            panic!("QUnit::FSim qubit index parameter must be within allocated qubit bounds!");
        }
        // RevertBasis2Qb(qubit1, ONLY_INVERT) implementation not provided
        // RevertBasis2Qb(qubit2, ONLY_INVERT) implementation not provided
        let shard1 = &mut self.shards[qubit1];
        let shard2 = &mut self.shards[qubit2];
        let is_same_unit = shard1.pauli_basis == shard2.pauli_basis;
        // Entangle() implementation not provided
        shard1.make_dirty();
        shard2.make_dirty();
        if is_same_unit && shard1.pauli_basis != PauliBasis::PauliZ {
            self.try_separate(qubit1);
            self.try_separate(qubit2);
        }
    }

    fn uniformly_controlled_single_bit(
        &mut self,
        controls: &[usize],
        qubit_index: usize,
        mtrxs: &[Complex],
        mtrx_skip_powers: &[u64],
        mtrx_skip_value_mask: u64,
    ) {
        if controls.is_empty() {
            self.mtrx(mtrxs, qubit_index);
            return;
        }
        if qubit_index >= self.qubit_count {
            panic!("QUnit::UniformlyControlledSingleBit qubitIndex is out-of-bounds!");
        }
        self.throw_if_qb_id_array_is_bad(controls);
        let mut trimmed_controls = Vec::new();
        let mut skip_powers = Vec::new();
        let mut skip_value_mask = 0;
        for (i, &control) in controls.iter().enumerate() {
            if !self.check_bits_permutation(control) {
                trimmed_controls.push(control);
            } else {
                skip_powers.push(1 << i);
                if self.shards[control].pauli_basis == PauliBasis::PauliZ {
                    skip_value_mask |= 1 << i;
                }
            }
        }
        if trimmed_controls.is_empty() {
            let control_perm = self.get_cached_permutation(controls);
            let mtrx = &mtrxs[(control_perm << 2)..((control_perm + 1) << 2)];
            self.mtrx(mtrx, qubit_index);
            return;
        }
        let mut bits = trimmed_controls.clone();
        bits.push(qubit_index);
        bits.sort_unstable();
        let mut ebits = bits.iter().collect::<Vec<_>>();
        let unit = self.entangle(&ebits);
        let mapped_controls = trimmed_controls
            .iter()
            .map(|&control| self.shards[control].mapped)
            .collect::<Vec<_>>();
        unit.uniformly_controlled_single_bit(
            &mapped_controls,
            self.shards[qubit_index].mapped,
            mtrxs,
            &skip_powers,
            skip_value_mask,
        );
        self.shards[qubit_index].make_dirty();
        if !is_reactive_separate || freeze_basis2_qb {
            return;
        }
        if bits.len() == 2 {
            self.try_separate(bits[0], bits[1]);
            return;
        }
        for i in 0..(bits.len() - 1) {
            for j in (i + 1)..bits.len() {
                self.try_separate(bits[i], bits[j]);
            }
        }
    }

    fn h(&mut self, target: usize) {
        if target >= self.qubit_count {
            panic!("QUnit::H qubit index parameter must be within allocated qubit bounds!");
        }
        let shard = &mut self.shards[target];
        let is_clifford = false; // Implementation not provided
        if is_clifford {
            // RevertBasis1Qb(target) implementation not provided
            // RevertBasis2Qb(target) implementation not provided
        } else {
            // RevertBasisY(target) implementation not provided
            // CommuteH(target) implementation not provided
        }
        shard.pauli_basis = if shard.pauli_basis == PauliBasis::PauliZ {
            PauliBasis::PauliX
        } else {
            PauliBasis::PauliZ
        };
        if is_clifford {
            // RevertBasis1Qb(target) implementation not provided
        }
    }

    fn s(&mut self, target: usize) {
        if target >= self.qubit_count {
            panic!("QUnit::S qubit index parameter must be within allocated qubit bounds!");
        }
        let shard = &mut self.shards[target];
        let is_clifford = false; // Implementation not provided
        if is_clifford {
            // RevertBasis1Qb(target) implementation not provided
            // RevertBasis2Qb(target) implementation not provided
        } else {
            shard.comm_phase(Complex::one(), Complex::i());
        }
        if shard.pauli_basis == PauliBasis::PauliY {
            shard.pauli_basis = PauliBasis::PauliX;
            self.x_base(target);
            return;
        }
        if shard.pauli_basis == PauliBasis::PauliX {
            shard.pauli_basis = PauliBasis::PauliY;
            return;
        }
        if shard.unit.is_some() {
            // shard.unit->S(shard.mapped) implementation not provided
        }
        shard.amp1 = Complex::i() * shard.amp1;
    }

    fn i_s(&mut self, target: usize) {
        if target >= self.qubit_count {
            panic!("QUnit::IS qubit index parameter must be within allocated qubit bounds!");
        }
        let shard = &mut self.shards[target];
        let is_clifford = false; // Implementation not provided
        if is_clifford {
            // RevertBasis1Qb(target) implementation not provided
            // RevertBasis2Qb(target) implementation not provided
        } else {
            shard.comm_phase(Complex::one(), -Complex::i());
        }
        if shard.pauli_basis == PauliBasis::PauliY {
            shard.pauli_basis = PauliBasis::PauliX;
            return;
        }
        if shard.pauli_basis == PauliBasis::PauliX {
            shard.pauli_basis = PauliBasis::PauliY;
            self.x_base(target);
            return;
        }
        if shard.unit.is_some() {
            // shard.unit->IS(shard.mapped) implementation not provided
        }
        shard.amp1 = -Complex::i() * shard.amp1;
    }

    fn x_base(&mut self, target: usize) {
        if target >= self.qubit_count {
            panic!("QUnit::XBase qubit index parameter must be within allocated qubit bounds!");
        }
        let shard = &mut self.shards[target];
        if shard.unit.is_some() {
            // shard.unit->X(shard.mapped) implementation not provided
        }
        std::mem::swap(&mut shard.amp0, &mut shard.amp1);
    }

    fn y_base(&mut self, target: usize) {
        if target >= self.qubit_count {
            panic!("QUnit::YBase qubit index parameter must be within allocated qubit bounds!");
        }
        let shard = &mut self.shards[target];
        if shard.unit.is_some() {
            // shard.unit->Y(shard.mapped) implementation not provided
        }
        let y0 = shard.amp0;
        shard.amp0 = -Complex::i() * shard.amp1;
        shard.amp1 = Complex::i() * y0;
    }

    fn z_base(&mut self, target: usize) {
        if target >= self.qubit_count {
            panic!("QUnit::ZBase qubit index parameter must be within allocated qubit bounds!");
        }
        let shard = &mut self.shards[target];
        if shard.unit.is_some() {
            // shard.unit->Z(shard.mapped) implementation not provided
        }
        shard.amp1 = -shard.amp1;
    }

    pub fn phase(&self, top_left: Complex, bottom_right: Complex, target: usize) -> Result<(), Box<dyn Error>> {
        if target >= self.qubit_count {
            return Err("QUnit::Phase qubit index parameter must be within allocated qubit bounds!".into());
        }
        if self.rand_global_phase || is_1_cmplx(top_left) {
            if is_norm_0(top_left - bottom_right) {
                return Ok(());
            }
            if is_norm_0((i_cmplx() * top_left) - bottom_right) {
                self.s(target)?;
                return Ok(());
            }
            if is_norm_0((i_cmplx() * top_left) + bottom_right) {
                self.is(target)?;
                return Ok(());
            }
        }
        let shard = &mut self.shards[target];
        let is_clifford = self.use_t_gadget && self.engines[0] == QINTERFACE_STABILIZER_HYBRID && (shard.unit.is_none() || shard.unit.is_clifford());
        if is_clifford {
            self.revert_basis_1_qb(target);
            self.revert_basis_2_qb(target);
        } else {
            shard.commute_phase(top_left, bottom_right);
        }
        if shard.pauli_basis == PauliZ {
            if let Some(unit) = &mut shard.unit {
                unit.phase(top_left, bottom_right, shard.mapped)?;
            }
            shard.amp0 *= top_left;
            shard.amp1 *= bottom_right;
            return Ok(());
        }
        let mut mtrx = [ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX];
        self.transform_phase(top_left, bottom_right, &mut mtrx);
        if let Some(unit) = &mut shard.unit {
            unit.mtrx(&mtrx, shard.mapped)?;
        }
        if self.dirty(shard) {
            shard.is_prob_dirty |= !is_phase_or_invert(&mtrx);
        }
        let y0 = shard.amp0;
        shard.amp0 = (mtrx[0] * y0) + (mtrx[1] * shard.amp1);
        shard.amp1 = (mtrx[2] * y0) + (mtrx[3] * shard.amp1);
        self.clamp_shard(target);
        Ok(())
    }

    pub fn invert(&self, top_right: Complex, bottom_left: Complex, target: usize) -> Result<(), Box<dyn Error>> {
        if target >= self.qubit_count {
            return Err("QUnit::Invert qubit index parameter must be within allocated qubit bounds!".into());
        }
        let shard = &mut self.shards[target];
        let is_clifford = self.use_t_gadget && self.engines[0] == QINTERFACE_STABILIZER_HYBRID && (shard.unit.is_none() || shard.unit.is_clifford());
        if is_clifford {
            self.revert_basis_1_qb(target);
            self.revert_basis_2_qb(target);
        } else {
            shard.flip_phase_anti();
            shard.commute_phase(top_right, bottom_left);
        }
        if shard.pauli_basis == PauliZ {
            if let Some(unit) = &mut shard.unit {
                unit.invert(top_right, bottom_left, shard.mapped)?;
            }
            let temp_amp1 = bottom_left * shard.amp0;
            shard.amp0 = top_right * shard.amp1;
            shard.amp1 = temp_amp1;
            return Ok(());
        }
        let mut mtrx = [ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX];
        if shard.pauli_basis == PauliX {
            self.transform_x_invert(top_right, bottom_left, &mut mtrx);
        } else {
            self.transform_y_invert(top_right, bottom_left, &mut mtrx);
        }
        if let Some(unit) = &mut shard.unit {
            unit.mtrx(&mtrx, shard.mapped)?;
        }
        if self.dirty(shard) {
            shard.is_prob_dirty |= !is_phase_or_invert(&mtrx);
        }
        let y0 = shard.amp0;
        shard.amp0 = (mtrx[0] * y0) + (mtrx[1] * shard.amp1);
        shard.amp1 = (mtrx[2] * y0) + (mtrx[3] * shard.amp1);
        self.clamp_shard(target);
        Ok(())
    }

    pub fn uc_phase(&self, l_controls: &Vec<bitLenInt>, top_left: complex, bottom_right: complex, target: bitLenInt, control_perm: bitCapInt) {
        throw_if_qb_id_array_is_bad(
            l_controls,
            self.qubit_count,
            "QUnit::UCPhase parameter controls array values must be within allocated qubit bounds!",
        );
        if is_1_cmplx(top_left) && is_1_cmplx(bottom_right) {
            return;
        }
        let mut control_vec: Vec<bitLenInt> = Vec::new();
        if self.trim_controls(l_controls, &mut control_vec, &mut control_perm) {
            return;
        }
        if control_vec.is_empty() {
            self.phase(top_left, bottom_right, target);
            return;
        }
        if control_vec.len() == 1 && is_norm_0(top_left - bottom_right) {
            if bi_compare_0(control_perm) != 0 {
                self.phase(ONE_CMPLX, bottom_right, control_vec[0]);
            } else {
                self.phase(top_left, ONE_CMPLX, control_vec[0]);
            }
            return;
        }
        if target >= self.qubit_count {
            panic!("QUnit::UCPhase qubit index parameter must be within allocated qubit bounds!");
        }
        if !self.freeze_basis2_qb && control_vec.len() == 1 {
            let control = control_vec[0];
            let c_shard = &mut self.shards[control];
            let t_shard = &mut self.shards[target];
            self.revert_basis2_qb(control, ONLY_INVERT, ONLY_TARGETS);
            let is_nonzero_ctrl_perm = bi_compare_0(control_perm) != 0;
            if is_nonzero_ctrl_perm {
                self.revert_basis2_qb(target, ONLY_INVERT, ONLY_TARGETS, ONLY_ANTI);
                self.revert_basis2_qb(target, ONLY_INVERT, ONLY_TARGETS, ONLY_CTRL, Vec::new(), vec![control]);
            } else {
                self.revert_basis2_qb(target, ONLY_INVERT, ONLY_TARGETS, ONLY_CTRL);
                self.revert_basis2_qb(target, ONLY_INVERT, ONLY_TARGETS, ONLY_ANTI, Vec::new(), vec![control]);
            }
            if !self.is_same_unit(c_shard, t_shard)
                && (!self.are_clifford(c_shard, t_shard)
                    || !((self.is_same(ONE_CMPLX, top_left) || self.is_same(-ONE_CMPLX, top_left))
                        && (self.is_same(ONE_CMPLX, bottom_right) || self.is_same(-ONE_CMPLX, bottom_right))))
            {
                if is_nonzero_ctrl_perm {
                    t_shard.add_phase_angles(c_shard, top_left, bottom_right);
                    self.optimize_pair_buffers(control, target, false);
                } else {
                    t_shard.add_anti_phase_angles(c_shard, bottom_right, top_left);
                    self.optimize_pair_buffers(control, target, true);
                }
                return;
            }
        }
        self.ctrled_phase_invert_wrap(self.uc_phase(CTRL_P_ARGS), self.uc_mtrx(CTRL_GEN_ARGS), false, top_left, bottom_right);
    }

    pub fn uc_invert(&self, l_controls: &Vec<bitLenInt>, top_right: complex, bottom_left: complex, target: bitLenInt, control_perm: bitCapInt) {
        throw_if_qb_id_array_is_bad(
            l_controls,
            self.qubit_count,
            "QUnit::UCInvert parameter controls array values must be within allocated qubit bounds!",
        );
        if target >= self.qubit_count {
            panic!("QUnit::UCInvert qubit index parameter must be within allocated qubit bounds!");
        }
        if is_1_cmplx(top_right) && is_1_cmplx(bottom_left) {
            if self.cached_plus(target) {
                return;
            }
        }
        let mut control_vec: Vec<bitLenInt> = Vec::new();
        if trim_controls(l_controls, &mut control_vec, &mut control_perm) {
            return;
        }
        if control_vec.is_empty() {
            self.invert(top_right, bottom_left, target);
            return;
        }
        if !self.freeze_basis2_qb && control_vec.len() == 1 {
            let control = control_vec[0];
            let c_shard = &mut self.shards[control];
            let t_shard = &mut self.shards[target];
            self.revert_basis2_qb(control, ONLY_INVERT, ONLY_TARGETS);
            let is_nonzero_ctrl_perm = bi_compare_0(control_perm) != 0;
            if is_nonzero_ctrl_perm {
                self.revert_basis2_qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, ONLY_ANTI);
                self.revert_basis2_qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, ONLY_CTRL, Vec::new(), vec![control]);
            } else {
                self.revert_basis2_qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, ONLY_CTRL);
                self.revert_basis2_qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, ONLY_ANTI, Vec::new(), vec![control]);
            }
            if !self.is_same_unit(c_shard, t_shard)
                && (!self.are_clifford(c_shard, t_shard)
                    || !(((self.is_same(ONE_CMPLX, top_right) || self.is_same(-ONE_CMPLX, top_right))
                        && (self.is_same(ONE_CMPLX, bottom_left) || self.is_same(-ONE_CMPLX, bottom_left)))
                        || ((self.is_same(I_CMPLX, top_right) || self.is_same(-I_CMPLX, top_right))
                            && (self.is_same(I_CMPLX, bottom_left) || self.is_same(-I_CMPLX, bottom_left)))))
            {
                if is_nonzero_ctrl_perm {
                    t_shard.add_inversion_angles(c_shard, top_right, bottom_left);
                    self.optimize_pair_buffers(control, target, false);
                } else {
                    t_shard.add_anti_inversion_angles(c_shard, bottom_left, top_right);
                    self.optimize_pair_buffers(control, target, true);
                }
                return;
            }
        }
        self.ctrled_phase_invert_wrap(self.uc_invert(CTRL_I_ARGS), self.uc_mtrx(CTRL_GEN_ARGS), true, top_right, bottom_left);
    }

    pub fn mtrx(&mut self, mtrx: &[complex], target: usize) {
        let shard = &mut self.shards[target];
        if is_norm_0(mtrx[1]) && is_norm_0(mtrx[2]) {
            self.phase(mtrx[0], mtrx[3], target);
            return;
        }
        if is_norm_0(mtrx[0]) && is_norm_0(mtrx[3]) {
            self.invert(mtrx[1], mtrx[2], target);
            return;
        }
        if (self.rand_global_phase || is_same(mtrx[0], complex::new(SQRT1_2_R1, 0.0))) && is_same(mtrx[0], mtrx[1]) &&
            is_same(mtrx[0], mtrx[2]) && is_same(mtrx[0], -mtrx[3]) {
            self.h(target);
            return;
        }
        if (self.rand_global_phase || is_same(mtrx[0], complex::new(SQRT1_2_R1, 0.0))) && is_same(mtrx[0], mtrx[1]) &&
            is_same(mtrx[0], -I_CMPLX * mtrx[2]) && is_same(mtrx[0], I_CMPLX * mtrx[3]) {
            self.h(target);
            self.s(target);
            return;
        }
        if (self.rand_global_phase || is_same(mtrx[0], complex::new(SQRT1_2_R1, 0.0))) && is_same(mtrx[0], I_CMPLX * mtrx[1]) &&
            is_same(mtrx[0], mtrx[2]) && is_same(mtrx[0], -I_CMPLX * mtrx[3]) {
            self.is_gate(target);
            self.h(target);
            return;
        }
        if target >= self.qubit_count {
            panic!("QUnit::Mtrx qubit index parameter must be within allocated qubit bounds!");
        }
        self.revert_basis_2_qb(target);
        let mut trns_mtrx = [complex::new(0.0, 0.0); 4];
        if shard.pauli_basis == PauliY {
            self.transform_y2x2(mtrx, &mut trns_mtrx);
        } else if shard.pauli_basis == PauliX {
            self.transform_x2x2(mtrx, &mut trns_mtrx);
        } else {
            trns_mtrx.copy_from_slice(mtrx);
        }
        if let Some(unit) = &mut shard.unit {
            unit.mtrx(&trns_mtrx, shard.mapped);
        }
        if dirty(shard) {
            shard.is_prob_dirty |= !is_phase_or_invert(&trns_mtrx);
        }
        let y0 = shard.amp0;
        shard.amp0 = (trns_mtrx[0] * y0) + (trns_mtrx[1] * shard.amp1);
        shard.amp1 = (trns_mtrx[2] * y0) + (trns_mtrx[3] * shard.amp1);
        self.clamp_shard(target);
    }

    pub fn uc_mtrx(&mut self, controls: &[bit_len_int], mtrx: &[complex], target: usize, control_perm: bit_cap_int) {
        if is_norm_0(mtrx[1]) && is_norm_0(mtrx[2]) {
            self.uc_phase(controls, mtrx[0], mtrx[3], target, control_perm);
            return;
        }
        if is_norm_0(mtrx[0]) && is_norm_0(mtrx[3]) {
            self.uc_invert(controls, mtrx[1], mtrx[2], target, control_perm);
            return;
        }
        self.throw_if_qb_id_array_is_bad(
            controls, self.qubit_count, "QUnit::UCMtrx parameter controls array values must be within allocated qubit bounds!");
        let mut control_vec = Vec::new();
        if self.trim_controls(controls, &mut control_vec, &mut control_perm) {
            return;
        }
        if control_vec.is_empty() {
            self.mtrx(mtrx, target);
            return;
        }
        if target >= self.qubit_count {
            panic!("QUnit::MCMtrx qubit index parameter must be within allocated qubit bounds!");
        }
        self.ctrled_gen_wrap(self.uc_mtrx(CTRL_GEN_ARGS));
    }

    fn trim_controls(&self, controls: &Vec<bitLenInt>, control_vec: &mut Vec<bitLenInt>, perm: &mut bitCapInt) -> bool {
        if controls.is_empty() {
            return false;
        }

        for i in 0..controls.len() {
            let anti = !bi_and_1(*perm >> i);
            if (anti && CACHED_ONE(controls[i])) || (!anti && CACHED_ZERO(controls[i])) {
                return true;
            }
        }

        for i in 0..controls.len() {
            let shard = &mut self.shards[controls[i]];
            if shard.pauli_basis != PauliZ || shard.is_invert_target() {
                continue;
            }
            self.prob_base(controls[i]);

            if IS_NORM_0(shard.amp1) {
                self.flush_0_eigenstate(controls[i]);
                if bi_and_1(*perm >> i) {
                    return true;
                }
            } else if IS_NORM_0(shard.amp0) {
                self.flush_1_eigenstate(controls[i]);
                if !bi_and_1(*perm >> i) {
                    return true;
                }
            }
        }

        for i in 0..controls.len() {
            let shard = &mut self.shards[controls[i]];
            if shard.pauli_basis == PauliZ || shard.is_invert_target() {
                continue;
            }
            self.revert_basis_1_qb(controls[i]);
            self.prob_base(controls[i]);

            if IS_NORM_0(shard.amp1) {
                self.flush_0_eigenstate(controls[i]);
                if bi_and_1(*perm >> i) {
                    return true;
                }
            } else if IS_NORM_0(shard.amp0) {
                self.flush_1_eigenstate(controls[i]);
                if !bi_and_1(*perm >> i) {
                    return true;
                }
            }
        }

        let mut out_perm = ZERO_BCI;
        for i in 0..controls.len() {
            let shard = &mut self.shards[controls[i]];
            self.to_perm_basis_prob(controls[i]);
            self.prob_base(controls[i]);
            let mut is_eigenstate = false;

            if IS_NORM_0(shard.amp1) {
                self.flush_0_eigenstate(controls[i]);
                if bi_and_1(*perm >> i) {
                    return true;
                }

                is_eigenstate = true;
            } else if IS_NORM_0(shard.amp0) {
                self.flush_1_eigenstate(controls[i]);
                if !bi_and_1(*perm >> i) {
                    return true;
                }

                is_eigenstate = true;
            }
            if !is_eigenstate {
                bi_or_ip(&mut out_perm, bi_and_1(*perm >> i) << control_vec.len());
                control_vec.push(controls[i]);
            }
        }
        *perm = out_perm;
        false
    }

    fn apply_either_controlled<CF: Fn(QInterfacePtr, &Vec<bitLenInt>)>(
        &self,
        control_vec: &Vec<bitLenInt>,
        targets: &Vec<bitLenInt>,
        cfn: CF,
        is_phase: bool,
    ) {
        for i in 0..control_vec.len() {
            self.to_perm_basis_prob(control_vec[i]);
        }
        if targets.len() > 1 {
            for i in 0..targets.len() {
                self.to_perm_basis(targets[i]);
            }
        } else if is_phase {
            self.revert_basis_2_qb(targets[0], ONLY_INVERT, ONLY_TARGETS);
        } else {
            self.revert_basis_2_qb(targets[0]);
        }
        let mut all_bits = Vec::with_capacity(control_vec.len() + targets.len());
        all_bits.extend_from_slice(control_vec);
        all_bits.extend_from_slice(targets);
        all_bits.sort();
        let mut all_bits_mapped = all_bits.clone();
        let mut ebits = Vec::with_capacity(all_bits_mapped.len());
        for i in 0..all_bits_mapped.len() {
            ebits.push(&mut all_bits_mapped[i]);
        }
        let unit = self.entangle_in_current_basis(ebits.as_mut_slice());
        for i in 0..control_vec.len() {
            let c = &mut control_vec[i];
            self.shards[*c].is_phase_dirty = true;
            *c = self.shards[*c].mapped;
        }
        for i in 0..targets.len() {
            let shard = &mut self.shards[targets[i]];
            shard.is_phase_dirty = true;
            shard.is_prob_dirty |= shard.pauli_basis != PauliZ || !is_phase;
        }

        cfn(unit, control_vec);

        if !self.is_reactive_separate || self.freeze_basis_2_qb {
            return;
        }

        if all_bits.len() == 2 {
            self.try_separate(all_bits[0], all_bits[1]);
            return;
        }

        for i in 0..(all_bits.len() - 1) {
            for j in (i + 1)..all_bits.len() {
                self.try_separate(all_bits[i], all_bits[j]);
            }
        }
    }

    fn convert_z_to_x(&mut self, i: bitLenInt) {
        let shard = &mut self.shards[i];

        shard.pauli_basis = if shard.pauli_basis == PauliX {
            PauliZ
        } else {
            PauliX
        };
        if let Some(unit) = &mut shard.unit {
            unit.h(shard.mapped);
        }
        if shard.is_phase_dirty || shard.is_prob_dirty {
            shard.is_prob_dirty = true;
            return;
        }
        let temp_amp1 = SQRT1_2_R1 * (shard.amp0 - shard.amp1);
        shard.amp0 = SQRT1_2_R1 * (shard.amp0 + shard.amp1);
        shard.amp1 = temp_amp1;
        self.clamp_shard(i);
    }

    fn convert_x_to_y(&mut self, i: bitLenInt) {
        let shard = &mut self.shards[i];
        shard.pauli_basis = PauliY;
        let mtrx = [
            ((ONE_R1 / 2) * (ONE_CMPLX - I_CMPLX)),
            ((ONE_R1 / 2) * (ONE_CMPLX + I_CMPLX)),
            ((ONE_R1 / 2) * (ONE_CMPLX + I_CMPLX)),
            ((ONE_R1 / 2) * (ONE_CMPLX - I_CMPLX)),
        ];
        if let Some(unit) = &mut shard.unit {
            unit.mtrx(mtrx, shard.mapped);
        }
        if shard.is_phase_dirty || shard.is_prob_dirty {
            shard.is_prob_dirty = true;
            return;
        }
        let y0 = shard.amp0;
        shard.amp0 = (mtrx[0] * y0) + (mtrx[1] * shard.amp1);
        shard.amp1 = (mtrx[2] * y0) + (mtrx[3] * shard.amp1);
        self.clamp_shard(i);
    }

    fn convert_y_to_z(&mut self, i: bitLenInt) {
        let shard = &mut self.shards[i];
        shard.pauli_basis = PauliZ;
        let mtrx = [
            complex(SQRT1_2_R1, ZERO_R1),
            complex(SQRT1_2_R1, ZERO_R1),
            complex(ZERO_R1, SQRT1_2_R1),
            complex(ZERO_R1, -SQRT1_2_R1),
        ];
        if let Some(unit) = &mut shard.unit {
            unit.mtrx(mtrx, shard.mapped);
        }
        if shard.is_phase_dirty || shard.is_prob_dirty {
            shard.is_prob_dirty = true;
            return;
        }
        let y0 = shard.amp0;
        shard.amp0 = (mtrx[0] * y0) + (mtrx[1] * shard.amp1);
        shard.amp1 = (mtrx[2] * y0) + (mtrx[3] * shard.amp1);
        self.clamp_shard(i);
    }

    fn convert_z_to_y(&mut self, i: bitLenInt) {
        let shard = &mut self.shards[i];
        shard.pauli_basis = PauliY;
        let mtrx = [
            complex(SQRT1_2_R1, ZERO_R1),
            complex(ZERO_R1, -SQRT1_2_R1),
            complex(SQRT1_2_R1, ZERO_R1),
            complex(ZERO_R1, SQRT1_2_R1),
        ];
        if let Some(unit) = &mut shard.unit {
            unit.mtrx(mtrx, shard.mapped);
        }
        if shard.is_phase_dirty || shard.is_prob_dirty {
            shard.is_prob_dirty = true;
            return;
        }
        let y0 = shard.amp0;
        shard.amp0 = (mtrx[0] * y0) + (mtrx[1] * shard.amp1);
        shard.amp1 = (mtrx[2] * y0) + (mtrx[3] * shard.amp1);
        self.clamp_shard(i);
    }

    fn shard_ai(&mut self, qubit: bitLenInt, azimuth: real1_f, inclination: real1_f) {
        let cosine_a = (cos(azimuth) as real1);
        let sine_a = (sin(azimuth) as real1);
        let cosine_i = (cos(inclination / 2) as real1);
        let sine_i = (sin(inclination / 2) as real1);
        let exp_a = complex(cosine_a, sine_a);
        let exp_neg_a = complex(cosine_a, -sine_a);
        let mtrx = [
            cosine_i,
            -exp_neg_a * sine_i,
            exp_a * sine_i,
            cosine_i,
        ];
        let shard = &mut self.shards[qubit];
        let y0 = shard.amp0;
        shard.amp0 = (mtrx[0] * y0) + (mtrx[1] * shard.amp1);
        shard.amp1 = (mtrx[2] * y0) + (mtrx[3] * shard.amp1);
        self.clamp_shard(qubit);
    }

    fn flush_0_eigenstate(&mut self, i: bitLenInt) {
        self.shards[i].dump_control_of();
        if self.rand_global_phase {
            self.shards[i].dump_same_phase_anti_control_of();
        }
        self.revert_basis_2_qb(i, INVERT_AND_PHASE, ONLY_CONTROLS, ONLY_ANTI);
    }

    fn flush_1_eigenstate(&mut self, i: bitLenInt) {
        self.shards[i].dump_anti_control_of();
        if self.rand_global_phase {
            self.shards[i].dump_same_phase_control_of();
        }
        self.revert_basis_2_qb(i, INVERT_AND_PHASE, ONLY_CONTROLS, ONLY_CTRL);
    }

    fn to_perm_basis(&mut self, i: bitLenInt) {
        self.revert_basis_1_qb(i);
        self.revert_basis_2_qb(i);
    }

    fn to_perm_basis_range(&mut self, start: bitLenInt, length: bitLenInt) {
        for i in 0..length {
            self.revert_basis_1_qb(start + i);
        }
        for i in 0..length {
            self.revert_basis_2_qb(start + i);
        }
    }

    fn to_perm_basis_prob(&mut self, i: bitLenInt) {
        self.revert_basis_1_qb(i);
        self.revert_basis_2_qb(i, ONLY_INVERT, ONLY_TARGETS);
    }

    fn to_perm_basis_prob_range(&mut self, start: bitLenInt, length: bitLenInt) {
        for i in 0..length {
            self.revert_basis_1_qb(start + i);
        }
        for i in 0..length {
            self.revert_basis_2_qb(start + i, ONLY_INVERT, ONLY_TARGETS);
        }
    }

    fn to_perm_basis_measure(&mut self, i: bitLenInt) {
        self.revert_basis_1_qb(i);
        self.revert_basis_2_qb(i, ONLY_INVERT);
        self.revert_basis_2_qb(i, ONLY_PHASE, ONLY_CONTROLS);
        self.shards[i].dump_multi_bit();
    }

    fn to_perm_basis_measure_range(&mut self, start: bitLenInt, length: bitLenInt) {
        if start == 0 && length == self.qubit_count {
            self.to_perm_basis_all_measure();
            return;
        }
        let mut except_bits = HashSet::new();
        for i in 0..length {
            except_bits.insert(start + i);
        }
        for i in 0..length {
            self.revert_basis_1_qb(start + i);
        }
        for i in 0..length {
            self.revert_basis_2_qb(start + i, ONLY_INVERT);
            self.revert_basis_2_qb(start + i, ONLY_PHASE, ONLY_CONTROLS, CTRL_AND_ANTI, except_bits);
            self.shards[start + i].dump_multi_bit();
        }
    }

    fn to_perm_basis_all_measure(&mut self) {
        for i in 0..self.qubit_count {
            self.revert_basis_1_qb(i);
        }
        for i in 0..self.qubit_count {
            self.shards[i].clear_invert_phase();
            self.revert_basis_2_qb(i, ONLY_INVERT);
            self.shards[i].dump_multi_bit();
        }
    }
}
