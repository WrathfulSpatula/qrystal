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
}
