use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

pub trait QInterface {
    fn set_quantum_state(&mut self, state: &[Complex]);
    fn get_quantum_state(&self) -> Vec<Complex>;
    fn set_amplitude(&mut self, perm: u64, amp: Complex);
    fn set_permutation(&mut self, perm: u64, phase_fac: Complex);
    fn get_amplitude_or_prob(&self, perm: u64, is_prob: bool) -> Complex;
    fn prob_base(&self, qubit: usize) -> f64;
    fn prob(&self, qubit: usize) -> f64;
    fn prob_all(&self, perm: u64) -> f64;
    fn prob_all_rdm(&self, round_rz: bool, perm: u64) -> f64;
    fn sum_sqr_diff(&self, to_compare: &Rc<RefCell<dyn QInterface>>) -> f64;
    fn expectation_bits_factorized(
        &self,
        bits: &[usize],
        perms: &[u64],
        offset: u64,
    ) -> f64;
    fn expectation_bits_factorized_rdm(
        &self,
        round_rz: bool,
        bits: &[usize],
        perms: &[u64],
        offset: u64,
    ) -> f64;
    fn update_running_norm(&mut self, norm_thresh: f64);
    fn normalize_state(&mut self, nrm: f64, norm_thresh: f64, phase_arg: f64);
    fn finish(&mut self);
    fn is_finished(&self) -> bool;
    fn dump(&mut self);
    fn is_clifford(&self, qubit: usize) -> bool;
    fn get_unitary_fidelity(&self) -> f64;
    fn reset_unitary_fidelity(&mut self);
    fn set_sdrp(&mut self, sdrp: f64);
    fn clone_qinterface(&self) -> Rc<RefCell<dyn QInterface>>;
}

pub trait QParity {
    fn anti_cnot(&mut self, control: usize, target: usize);
    fn cnot(&mut self, control: usize, target: usize);
    fn either_iswap(&mut self, qubit1: usize, qubit2: usize, is_inverse: bool);
    fn iswap(&mut self, qubit1: usize, qubit2: usize);
    fn iiswap(&mut self, qubit1: usize, qubit2: usize);
    fn mc_phase(
        &mut self,
        controls: &[usize],
        top_left: Complex,
        bottom_right: Complex,
        target: usize,
    );
    fn mc_invert(
        &mut self,
        controls: &[usize],
        top_right: Complex,
        bottom_left: Complex,
        target: usize,
    );
    fn mac_phase(
        &mut self,
        controls: &[usize],
        top_left: Complex,
        bottom_right: Complex,
        target: usize,
    );
    fn mac_invert(
        &mut self,
        controls: &[usize],
        top_right: Complex,
        bottom_left: Complex,
        target: usize,
    );
    fn uc_phase(
        &mut self,
        controls: &[usize],
        top_left: Complex,
        bottom_right: Complex,
        target: usize,
        control_perm: u64,
    );
    fn uc_invert(
        &mut self,
        controls: &[usize],
        top_right: Complex,
        bottom_left: Complex,
        target: usize,
        control_perm: u64,
    );
    fn mtrx(&mut self, mtrx: &[Complex], qubit: usize);
    fn mc_mtrx(&mut self, controls: &[usize], mtrx: &[Complex], target: usize);
    fn mac_mtrx(&mut self, controls: &[usize], mtrx: &[Complex], target: usize);
    fn set_reg(&mut self, start: usize, length: usize, value: u64);
    fn swap(&mut self, qubit1: usize, qubit2: usize);
    fn iswap(&mut self, qubit1: usize, qubit2: usize);
    fn iiswap(&mut self, qubit1: usize, qubit2: usize);
    fn prob_rdm(&self, qubit: usize) -> f64;
    fn cprob_rdm(&self, control: usize, target: usize) -> f64;
    fn acprob_rdm(&self, control: usize, target: usize) -> f64;
    fn set_amplitude(&mut self, perm: u64, amp: Complex);
    fn set_permutation(&mut self, perm: u64, phase_fac: Complex);
    fn compose(&mut self, to_copy: Rc<RefCell<dyn QInterface>>) -> usize;
    fn decompose(&mut self, start: usize, dest: Rc<RefCell<dyn QInterface>>);
    fn decompose(&mut self, start: usize, length: usize) -> Rc<RefCell<dyn QInterface>>;
    fn dispose(&mut self, start: usize, length: usize);
    fn dispose(&mut self, start: usize, length: usize, disposed_perm: u64);
    fn phase(&mut self, top_left: Complex, bottom_right: Complex, qubit_index: usize);
    fn invert(&mut self, top_right: Complex, bottom_left: Complex, qubit_index: usize);
    fn mcmtrx(&mut self, controls: &[usize], mtrx: &[Complex], target: usize);
    fn macmtrx(&mut self, controls: &[usize], mtrx: &[Complex], target: usize);
    fn set_t_injection(&mut self, use_gadget: bool);
    fn set_reactive_separate(&mut self, is_agg_sep: bool);
    fn set_device(&mut self, d_id: i64);
    fn get_device(&self) -> i64;
    fn parallel_unit_apply(
        &mut self,
        fn_ptr: fn(
            Rc<RefCell<dyn QInterface>>,
            f64,
            f64,
            f64,
            i64,
        ) -> bool,
        param1: f64,
        param2: f64,
        param3: f64,
        param4: i64,
    ) -> bool;
    fn entangle_range(&mut self, start: usize, length: usize, is_for_prob: bool) -> Rc<RefCell<dyn QInterface>>;
    fn entangle_all(&mut self, is_for_prob: bool) -> Rc<RefCell<dyn QInterface>>;
    fn order_contiguous(&mut self, unit: Rc<RefCell<dyn QInterface>>);
    fn detach(&mut self, start: usize, length: usize, dest: Option<Rc<RefCell<dyn QInterface>>>);
    fn phase(&mut self, top_left: Complex, bottom_right: Complex, qubit_index: usize);
    fn invert(&mut self, top_right: Complex, bottom_left: Complex, qubit_index: usize);
    fn mcmtrx(&mut self, controls: &[usize], mtrx: &[Complex], target: usize);
    fn macmtrx(&mut self, controls: &[usize], mtrx: &[Complex], target: usize);
    fn set_reg(&mut self, start: usize, length: usize, value: u64);
    fn swap(&mut self, qubit1: usize, qubit2: usize);
    fn iswap(&mut self, qubit1: usize, qubit2: usize);
    fn iiswap(&mut self, qubit1: usize, qubit2: usize);
    fn prob(&self, qubit: usize) -> f64;
    fn prob_all(&self, perm: u64) -> f64;
    fn prob_all_rdm(&self, round_rz: bool, perm: u64) -> f64;
    fn sum_sqr_diff(&self, to_compare: &Rc<RefCell<dyn QInterface>>) -> f64;
    fn expectation_bits_factorized(
        &self,
        bits: &[usize],
        perms: &[u64],
        offset: u64,
    ) -> f64;
    fn expectation_bits_factorized_rdm(
        &self,
        round_rz: bool,
        bits: &[usize],
        perms: &[u64],
        offset: u64,
    ) -> f64;
    fn update_running_norm(&mut self, norm_thresh: f64);
    fn normalize_state(&mut self, nrm: f64, norm_thresh: f64, phase_arg: f64);
    fn finish(&mut self);
    fn is_finished(&self) -> bool;
    fn dump(&mut self);
    fn is_clifford(&self, qubit: usize) -> bool;
    fn get_unitary_fidelity(&self) -> f64;
    fn reset_unitary_fidelity(&mut self);
    fn set_sdrp(&mut self, sdrp: f64);
    fn clone_qinterface(&self) -> Rc<RefCell<dyn QInterface>>;
}

pub struct QEngineShard {
    pub unit: Option<Rc<RefCell<dyn QInterface>>>,
    pub pauli_basis: Pauli,
    pub mapped: usize,
    pub amp0: Complex,
    pub amp1: Complex,
    pub is_phase_dirty: bool,
    pub is_prob_dirty: bool,
    pub found: bool,
}

pub struct QEngineShardMap {
    pub shards: Vec<QEngineShard>,
}

pub struct QInterfaceEngine {
    pub engine: Rc<RefCell<dyn QInterface>>,
    pub qubit_count: usize,
}

pub struct QUnit {
    freeze_basis_2_qb: bool,
    use_host_ram: bool,
    is_sparse: bool,
    is_reactive_separate: bool,
    use_t_gadget: bool,
    threshold_qubits: usize,
    separability_threshold: f64,
    log_fidelity: f64,
    dev_id: i64,
    phase_factor: Complex,
    shards: QEngineShardMap,
    device_ids: Vec<i64>,
    engines: Vec<QInterfaceEngine>,
}

impl QUnit {
    pub fn new(
        eng: Vec<QInterfaceEngine>,
        q_bit_count: usize,
        init_state: u64,
        rgp: Option<Rc<RefCell<dyn QInterface>>>,
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
        separation_thresh: f64,
    ) -> Self {
        Self {
            freeze_basis_2_qb: false,
            use_host_ram: use_host_mem,
            is_sparse: use_sparse_state_vec,
            is_reactive_separate: false,
            use_t_gadget: false,
            threshold_qubits: qubit_threshold,
            separability_threshold: separation_thresh,
            log_fidelity: 0.0,
            dev_id: device_id,
            phase_factor: phase_fac,
            shards: QEngineShardMap { shards: vec![] },
            device_ids: dev_ids,
            engines: eng,
        }
    }
}

impl QInterface for QUnit {
    fn set_quantum_state(&mut self, state: &[Complex]) {
        let mut perm = 0;
        for i in 0..state.len() {
            self.set_amplitude(perm, state[i]);
            perm += 1;
        }
    }

    fn get_quantum_state(&self) -> Vec<Complex> {
        let mut state = vec![];
        for i in 0..self.shards.shards.len() {
            let shard = &self.shards.shards[i];
            if let Some(unit) = &shard.unit {
                let unit_state = unit.borrow().get_quantum_state();
                state.extend(unit_state);
            } else {
                state.push(shard.amp0);
                state.push(shard.amp1);
            }
        }
        state
    }

    fn set_amplitude(&mut self, perm: u64, amp: Complex) {
        if perm >= self.shards.shards.len() as u64 {
            panic!("QUnit::SetAmplitude argument out-of-bounds!");
        }
        self.entangle_all(false);
        self.shards.shards[0].unit.as_mut().unwrap().borrow_mut().set_amplitude(perm, amp);
    }

    fn set_permutation(&mut self, perm: u64, phase_fac: Complex) {
        if perm >= self.shards.shards.len() as u64 {
            panic!("QUnit::SetPermutation argument out-of-bounds!");
        }
        self.entangle_all(false);
        self.shards.shards[0].unit.as_mut().unwrap().borrow_mut().set_permutation(perm, phase_fac);
    }

    fn get_amplitude_or_prob(&self, perm: u64, is_prob: bool) -> Complex {
        if perm >= self.shards.shards.len() as u64 {
            panic!("QUnit::GetAmplitudeOrProb argument out-of-bounds!");
        }
        let shard = &self.shards.shards[perm as usize];
        if let Some(unit) = &shard.unit {
            unit.borrow().get_amplitude_or_prob(perm, is_prob)
        } else {
            if is_prob {
                if perm == 0 {
                    shard.amp0.norm()
                } else {
                    shard.amp1.norm()
                }
            } else {
                if perm == 0 {
                    shard.amp0
                } else {
                    shard.amp1
                }
            }
        }
    }

    fn prob_base(&self, qubit: usize) -> f64 {
        if qubit >= self.shards.shards.len() {
            panic!("QUnit::ProbBase target parameter must be within allocated qubit bounds!");
        }
        self.to_perm_basis_prob(qubit);
        self.prob(qubit)
    }

    fn prob(&self, qubit: usize) -> f64 {
        if qubit >= self.shards.shards.len() {
            panic!("QUnit::Prob target parameter must be within allocated qubit bounds!");
        }
        self.to_perm_basis_prob(qubit);
        self.prob_base(qubit)
    }

    fn prob_all(&self, perm: u64) -> f64 {
        self.clamp_prob(self.norm(self.get_amplitude_or_prob(perm, true)))
    }

    fn prob_all_rdm(&self, round_rz: bool, perm: u64) -> f64 {
        if let Some(unit) = &self.shards.shards[0].unit {
            self.order_contiguous(unit.clone());
            unit.borrow().prob_all_rdm(round_rz, perm)
        } else {
            let clone = self.clone_qinterface();
            let unit = clone.borrow_mut().entangle_all(true);
            clone.borrow_mut().order_contiguous(unit.clone());
            unit.borrow().prob_all_rdm(round_rz, perm)
        }
    }

    fn sum_sqr_diff(&self, to_compare: &Rc<RefCell<dyn QInterface>>) -> f64 {
        let to_compare = to_compare.borrow();
        if self.shards.shards.len() != to_compare.shards.shards.len() {
            panic!("QUnit::SumSqrDiff() must be called with QUnits of the same size!");
        }
        let mut sum = 0.0;
        for i in 0..self.shards.shards.len() {
            let shard = &self.shards.shards[i];
            let to_compare_shard = &to_compare.shards.shards[i];
            if let Some(unit) = &shard.unit {
                if let Some(to_compare_unit) = &to_compare_shard.unit {
                    sum += unit.borrow().sum_sqr_diff(to_compare_unit);
                } else {
                    sum += unit.borrow().sum_sqr_diff(to_compare_shard.amp0, to_compare_shard.amp1);
                }
            } else {
                if let Some(to_compare_unit) = &to_compare_shard.unit {
                    sum += to_compare_unit.borrow().sum_sqr_diff(shard.amp0, shard.amp1);
                } else {
                    sum += self.sum_sqr_diff_single(shard.amp0, shard.amp1, to_compare_shard.amp0, to_compare_shard.amp1);
                }
            }
        }
        sum
    }

    fn expectation_bits_factorized(
        &self,
        bits: &[usize],
        perms: &[u64],
        offset: u64,
    ) -> f64 {
        self.expectation_factorized(false, false, bits, perms, vec![], offset, false)
    }

    fn expectation_bits_factorized_rdm(
        &self,
        round_rz: bool,
        bits: &[usize],
        perms: &[u64],
        offset: u64,
    ) -> f64 {
        self.expectation_factorized(true, false, bits, perms, vec![], offset, round_rz)
    }

    fn update_running_norm(&mut self, norm_thresh: f64) {
        for i in 0..self.shards.shards.len() {
            let shard = &mut self.shards.shards[i];
            if let Some(unit) = &shard.unit {
                unit.borrow_mut().update_running_norm(norm_thresh);
            }
        }
    }

    fn normalize_state(&mut self, nrm: f64, norm_thresh: f64, phase_arg: f64) {
        for i in 0..self.shards.shards.len() {
            let shard = &mut self.shards.shards[i];
            if let Some(unit) = &shard.unit {
                unit.borrow_mut().normalize_state(nrm, norm_thresh, phase_arg);
            }
        }
    }

    fn finish(&mut self) {
        for i in 0..self.shards.shards.len() {
            let shard = &mut self.shards.shards[i];
            if shard.unit.is_none() {
                if shard.amp1.norm() <= FP_NORM_EPSILON {
                    shard.unit = Some(self.make_engine(1, 0));
                } else if shard.amp0.norm() <= FP_NORM_EPSILON {
                    shard.unit = Some(self.make_engine(1, 1));
                } else {
                    let bit_state = [shard.amp0, shard.amp1];
                    shard.unit = Some(self.make_engine(1, 0));
                    shard.unit.as_mut().unwrap().borrow_mut().set_quantum_state(&bit_state);
                }
            }
        }
    }

    fn is_finished(&self) -> bool {
        for i in 0..self.shards.shards.len() {
            if let Some(unit) = &self.shards.shards[i].unit {
                if !unit.borrow().is_finished() {
                    return false;
                }
            }
        }
        true
    }

    fn dump(&mut self) {
        for shard in &mut self.shards.shards {
            shard.unit = None;
        }
    }

    fn is_clifford(&self, qubit: usize) -> bool {
        self.shards.shards[qubit].is_clifford()
    }

    fn get_unitary_fidelity(&self) -> f64 {
        self.log_fidelity.exp()
    }

    fn reset_unitary_fidelity(&mut self) {
        self.log_fidelity = 0.0;
    }

    fn set_sdrp(&mut self, sdrp: f64) {
        self.separability_threshold = sdrp;
    }

    fn clone_qinterface(&self) -> Rc<RefCell<dyn QInterface>> {
        Rc::new(RefCell::new(Self {
            freeze_basis_2_qb: self.freeze_basis_2_qb,
            use_host_ram: self.use_host_ram,
            is_sparse: self.is_sparse,
            is_reactive_separate: self.is_reactive_separate,
            use_t_gadget: self.use_t_gadget,
            threshold_qubits: self.threshold_qubits,
            separability_threshold: self.separability_threshold,
            log_fidelity: self.log_fidelity,
            dev_id: self.dev_id,
            phase_factor: self.phase_factor,
            shards: self.shards.clone(),
            device_ids: self.device_ids.clone(),
            engines: self.engines.clone(),
        }))
    }
}

impl QParity for QUnit {
    fn anti_cnot(&mut self, control: usize, target: usize) {
        self.shards.shards[control].unit.as_mut().unwrap().borrow_mut().anti_cnot(control, target);
        self.shards.shards[target].unit.as_mut().unwrap().borrow_mut().prob_rdm(target);
    }

    fn cnot(&mut self, control: usize, target: usize) {
        self.shards.shards[control].unit.as_mut().unwrap().borrow_mut().cnot(control, target);
        self.shards.shards[target].unit.as_mut().unwrap().borrow_mut().prob_rdm(target);
    }

    fn either_iswap(&mut self, qubit1: usize, qubit2: usize, is_inverse: bool) {
        self.shards.shards[qubit1].unit.as_mut().unwrap().borrow_mut().either_iswap(qubit1, qubit2, is_inverse);
        self.shards.shards[qubit2].unit.as_mut().unwrap().borrow_mut().prob_rdm(qubit2);
    }

    fn iswap(&mut self, qubit1: usize, qubit2: usize) {
        self.shards.shards[qubit1].unit.as_mut().unwrap().borrow_mut().iswap(qubit1, qubit2);
        self.shards.shards[qubit2].unit.as_mut().unwrap().borrow_mut().prob_rdm(qubit2);
    }

    fn iiswap(&mut self, qubit1: usize, qubit2: usize) {
        self.shards.shards[qubit1].unit.as_mut().unwrap().borrow_mut().iiswap(qubit1, qubit2);
        self.shards.shards[qubit2].unit.as_mut().unwrap().borrow_mut().prob_rdm(qubit2);
    }

    fn mc_phase(
        &mut self,
        controls: &[usize],
        top_left: Complex,
        bottom_right: Complex,
        target: usize,
    ) {
        let m = 2usize.pow(controls.len() as u32) - 1;
        self.shards.shards[controls[0]].unit.as_mut().unwrap().borrow_mut().mc_phase(
            controls,
            top_left,
            bottom_right,
            target,
            m as u64,
        );
        self.shards.shards[target].unit.as_mut().unwrap().borrow_mut().prob_rdm(target);
    }

    fn mc_invert(
        &mut self,
        controls: &[usize],
        top_right: Complex,
        bottom_left: Complex,
        target: usize,
    ) {
        let m = 2usize.pow(controls.len() as u32) - 1;
        self.shards.shards[controls[0]].unit.as_mut().unwrap().borrow_mut().mc_invert(
            controls,
            top_right,
            bottom_left,
            target,
            m as u64,
        );
        self.shards.shards[target].unit.as_mut().unwrap().borrow_mut().prob_rdm(target);
    }

    fn mac_phase(
        &mut self,
        controls: &[usize],
        top_left: Complex,
        bottom_right: Complex,
        target: usize,
    ) {
        self.shards.shards[controls[0]].unit.as_mut().unwrap().borrow_mut().mac_phase(
            controls,
            top_left,
            bottom_right,
            target,
        );
        self.shards.shards[target].unit.as_mut().unwrap().borrow_mut().prob_rdm(target);
    }

    fn mac_invert(
        &mut self,
        controls: &[usize],
        top_right: Complex,
        bottom_left: Complex,
        target: usize,
    ) {
        self.shards.shards[controls[0]].unit.as_mut().unwrap().borrow_mut().mac_invert(
            controls,
            top_right,
            bottom_left,
            target,
        );
        self.shards.shards[target].unit.as_mut().unwrap().borrow_mut().prob_rdm(target);
    }

    fn uc_phase(
        &mut self,
        controls: &[usize],
        top_left: Complex,
        bottom_right: Complex,
        target: usize,
        control_perm: u64,
    ) {
        self.shards.shards[controls[0]].unit.as_mut().unwrap().borrow_mut().uc_phase(
            controls,
            top_left,
            bottom_right,
            target,
            control_perm,
        );
        self.shards.shards[target].unit.as_mut().unwrap().borrow_mut().prob_rdm(target);
    }

    fn uc_invert(
        &mut self,
        controls: &[usize],
        top_right: Complex,
        bottom_left: Complex,
        target: usize,
        control_perm: u64,
    ) {
        self.shards.shards[controls[0]].unit.as_mut().unwrap().borrow_mut().uc_invert(
            controls,
            top_right,
            bottom_left,
            target,
            control_perm,
        );
        self.shards.shards[target].unit.as_mut().unwrap().borrow_mut().prob_rdm(target);
    }

    fn mtrx(&mut self, mtrx: &[Complex], qubit: usize) {
        self.shards.shards[qubit].unit.as_mut().unwrap().borrow_mut().mtrx(mtrx, qubit);
        self.shards.shards[qubit].unit.as_mut().unwrap().borrow_mut().prob_rdm(qubit);
    }

    fn mc_mtrx(&mut self, controls: &[usize], mtrx: &[Complex], target: usize) {
        let m = 2usize.pow(controls.len() as u32) - 1;
        self.shards.shards[controls[0]].unit.as_mut().unwrap().borrow_mut().mc_mtrx(
            controls,
            mtrx,
            target,
            m as u64,
        );
        self.shards.shards[target].unit.as_mut().unwrap().borrow_mut().prob_rdm(target);
    }

    fn mac_mtrx(&mut self, controls: &[usize], mtrx: &[Complex], target: usize) {
        self.shards.shards[controls[0]].unit.as_mut().unwrap().borrow_mut().mac_mtrx(
            controls,
            mtrx,
            target,
        );
        self.shards.shards[target].unit.as_mut().unwrap().borrow_mut().prob_rdm(target);
    }

    fn set_reg(&mut self, start: usize, length: usize, value: u64) {
        self.shards.shards[start].unit.as_mut().unwrap().borrow_mut().set_reg(start, length, value);
        for i in start..start + length {
            self.shards.shards[i].unit.as_mut().unwrap().borrow_mut().prob_rdm(i);
        }
    }

    fn swap(&mut self, qubit1: usize, qubit2: usize) {
        self.shards.shards[qubit1].unit.as_mut().unwrap().borrow_mut().swap(qubit1, qubit2);
        self.shards.shards[qubit2].unit.as_mut().unwrap().borrow_mut().prob_rdm(qubit2);
    }

    fn iswap(&mut self, qubit1: usize, qubit2: usize) {
        self.shards.shards[qubit1].unit.as_mut().unwrap().borrow_mut().iswap(qubit1, qubit2);
        self.shards.shards[qubit2].unit.as_mut().unwrap().borrow_mut().prob_rdm(qubit2);
    }

    fn iiswap(&mut self, qubit1: usize, qubit2: usize) {
        self.shards.shards[qubit1].unit.as_mut().unwrap().borrow_mut().iiswap(qubit1, qubit2);
        self.shards.shards[qubit2].unit.as_mut().unwrap().borrow_mut().prob_rdm(qubit2);
    }

    fn prob_rdm(&self, qubit: usize) -> f64 {
        let shard = &self.shards.shards[qubit];
        if let Some(unit) = &shard.unit {
            unit.borrow().prob_rdm(qubit)
        } else {
            self.prob(qubit)
        }
    }

    fn cprob_rdm(&self, control: usize, target: usize) -> f64 {
        self.anti_cnot(control, target);
        let prob = self.prob_rdm(target);
        self.anti_cnot(control, target);
        prob
    }

    fn acprob_rdm(&self, control: usize, target: usize) -> f64 {
        self.cnot(control, target);
        let prob = self.prob_rdm(target);
        self.cnot(control, target);
        prob
    }

    fn set_amplitude(&mut self, perm: u64, amp: Complex) {
        if perm >= self.shards.shards.len() as u64 {
            panic!("QUnit::SetAmplitude argument out-of-bounds!");
        }
        self.entangle_all(false);
        self.shards.shards[0].unit.as_mut().unwrap().borrow_mut().set_amplitude(perm, amp);
    }

    fn set_permutation(&mut self, perm: u64, phase_fac: Complex) {
        if perm >= self.shards.shards.len() as u64 {
            panic!("QUnit::SetPermutation argument out-of-bounds!");
        }
        self.entangle_all(false);
        self.shards.shards[0].unit.as_mut().unwrap().borrow_mut().set_permutation(perm, phase_fac);
    }

    fn compose(&mut self, to_copy: Rc<RefCell<dyn QInterface>>) -> usize {
        let to_copy = to_copy.borrow();
        if self.shards.shards.len() < to_copy.shards.shards.len() {
            panic!("QUnit::Compose start index is out-of-bounds!");
        }
        let clone = to_copy.clone_qinterface();
        let start = self.shards.shards.len();
        self.shards.shards.extend_from_slice(&to_copy.shards.shards);
        self.set_qubit_count(self.get_qubit_count() + to_copy.get_qubit_count());
        start
    }

    fn decompose(&mut self, start: usize, dest: Rc<RefCell<dyn QInterface>>) {
        let dest = dest.borrow();
        self.detach(start, dest.get_qubit_count(), Some(dest.clone_qinterface()));
    }

    fn decompose(&mut self, start: usize, length: usize) -> Rc<RefCell<dyn QInterface>> {
        let dest = self.clone_qinterface();
        self.detach(start, length, Some(dest.clone_qinterface()));
        dest
    }

    fn dispose(&mut self, start: usize, length: usize) {
        self.detach(start, length, None);
    }

    fn dispose(&mut self, start: usize, length: usize, disposed_perm: u64) {
        self.detach(start, length, None);
    }

    fn phase(&mut self, top_left: Complex, bottom_right: Complex, qubit_index: usize) {
        self.shards.shards[qubit_index].unit.as_mut().unwrap().borrow_mut().phase(top_left, bottom_right, qubit_index);
        self.shards.shards[qubit_index].unit.as_mut().unwrap().borrow_mut().prob_rdm(qubit_index);
    }

    fn invert(&mut self, top_right: Complex, bottom_left: Complex, qubit_index: usize) {
        self.shards.shards[qubit_index].unit.as_mut().unwrap().borrow_mut().invert(top_right, bottom_left, qubit_index);
        self.shards.shards[qubit_index].unit.as_mut().unwrap().borrow_mut().prob_rdm(qubit_index);
    }

    fn mcmtrx(&mut self, controls: &[usize], mtrx: &[Complex], target: usize) {
        let m = 2usize.pow(controls.len() as u32) - 1;
        self.shards.shards[controls[0]].unit.as_mut().unwrap().borrow_mut().mcmtrx(
            controls,
            mtrx,
            target,
            m as u64,
        );
        self.shards.shards[target].unit.as_mut().unwrap().borrow_mut().prob_rdm(target);
    }

    fn macmtrx(&mut self, controls: &[usize], mtrx: &[Complex], target: usize) {
        self.shards.shards[controls[0]].unit.as_mut().unwrap().borrow_mut().macmtrx(
            controls,
            mtrx,
            target,
        );
        self.shards.shards[target].unit.as_mut().unwrap().borrow_mut().prob_rdm(target);
    }

    fn set_reg(&mut self, start: usize, length: usize, value: u64) {
        self.shards.shards[start].unit.as_mut().unwrap().borrow_mut().set_reg(start, length, value);
        for i in start..start + length {
            self.shards.shards[i].unit.as_mut().unwrap().borrow_mut().prob_rdm(i);
        }
    }

    fn swap(&mut self, qubit1: usize, qubit2: usize) {
        self.shards.shards[qubit1].unit.as_mut().unwrap().borrow_mut().swap(qubit1, qubit2);
        self.shards.shards[qubit2].unit.as_mut().unwrap().borrow_mut().prob_rdm(qubit2);
    }

    fn iswap(&mut self, qubit1: usize, qubit2: usize) {
        self.shards.shards[qubit1].unit.as_mut().unwrap().borrow_mut().iswap(qubit1, qubit2);
        self.shards.shards[qubit2].unit.as_mut().unwrap().borrow_mut().prob_rdm(qubit2);
    }

    fn iiswap(&mut self, qubit1: usize, qubit2: usize) {
        self.shards.shards[qubit1].unit.as_mut().unwrap().borrow_mut().iiswap(qubit1, qubit2);
        self.shards.shards[qubit2].unit.as_mut().unwrap().borrow_mut().prob_rdm(qubit2);
    }

    fn prob(&self, qubit: usize) -> f64 {
        if qubit >= self.shards.shards.len() {
            panic!("QUnit::Prob target parameter must be within allocated qubit bounds!");
        }
        self.to_perm_basis_prob(qubit);
        self.prob_base(qubit)
    }

    fn prob_all(&self, perm: u64) -> f64 {
        self.clamp_prob(self.norm(self.get_amplitude_or_prob(perm, true)))
    }

    fn prob_all_rdm(&self, round_rz: bool, perm: u64) -> f64 {
        if let Some(unit) = &self.shards.shards[0].unit {
            self.order_contiguous(unit.clone());
            unit.borrow().prob_all_rdm(round_rz, perm)
        } else {
            let clone = self.clone_qinterface();
            let unit = clone.borrow_mut().entangle_all(true);
            clone.borrow_mut().order_contiguous(unit.clone());
            unit.borrow().prob_all_rdm(round_rz, perm)
        }
    }

    fn sum_sqr_diff(&self, to_compare: &Rc<RefCell<dyn QInterface>>) -> f64 {
        let to_compare = to_compare.borrow();
        if self.shards.shards.len() != to_compare.shards.shards.len() {
            panic!("QUnit::SumSqrDiff() must be called with QUnits of the same size!");
        }
        let mut sum = 0.0;
        for i in 0..self.shards.shards.len() {
            let shard = &self.shards.shards[i];
            let to_compare_shard = &to_compare.shards.shards[i];
            if let Some(unit) = &shard.unit {
                if let Some(to_compare_unit) = &to_compare_shard.unit {
                    sum += unit.borrow().sum_sqr_diff(to_compare_unit);
                } else {
                    sum += unit.borrow().sum_sqr_diff(to_compare_shard.amp0, to_compare_shard.amp1);
                }
            } else {
                if let Some(to_compare_unit) = &to_compare_shard.unit {
                    sum += to_compare_unit.borrow().sum_sqr_diff(shard.amp0, shard.amp1);
                } else {
                    sum += self.sum_sqr_diff_single(shard.amp0, shard.amp1, to_compare_shard.amp0, to_compare_shard.amp1);
                }
            }
        }
        sum
    }

    fn expectation_bits_factorized(
        &self,
        bits: &[usize],
        perms: &[u64],
        offset: u64,
    ) -> f64 {
        self.expectation_factorized(false, false, bits, perms, vec![], offset, false)
    }

    fn expectation_bits_factorized_rdm(
        &self,
        round_rz: bool,
        bits: &[usize],
        perms: &[u64],
        offset: u64,
    ) -> f64 {
        self.expectation_factorized(true, false, bits, perms, vec![], offset, round_rz)
    }

    fn update_running_norm(&mut self, norm_thresh: f64) {
        for i in 0..self.shards.shards.len() {
            let shard = &mut self.shards.shards[i];
            if let Some(unit) = &shard.unit {
                unit.borrow_mut().update_running_norm(norm_thresh);
            }
        }
    }

    fn normalize_state(&mut self, nrm: f64, norm_thresh: f64, phase_arg: f64) {
        for i in 0..self.shards.shards.len() {
            let shard = &mut self.shards.shards[i];
            if let Some(unit) = &shard.unit {
                unit.borrow_mut().normalize_state(nrm, norm_thresh, phase_arg);
            }
        }
    }

    fn finish(&mut self) {
        for i in 0..self.shards.shards.len() {
            let shard = &mut self.shards.shards[i];
            if shard.unit.is_none() {
                if shard.amp1.norm() <= FP_NORM_EPSILON {
                    shard.unit = Some(self.make_engine(1, 0));
                } else if shard.amp0.norm() <= FP_NORM_EPSILON {
                    shard.unit = Some(self.make_engine(1, 1));
                } else {
                    let bit_state = [shard.amp0, shard.amp1];
                    shard.unit = Some(self.make_engine(1, 0));
                    shard.unit.as_mut().unwrap().borrow_mut().set_quantum_state(&bit_state);
                }
            }
        }
    }

    fn is_finished(&self) -> bool {
        for i in 0..self.shards.shards.len() {
            if let Some(unit) = &self.shards.shards[i].unit {
                if !unit.borrow().is_finished() {
                    return false;
                }
            }
        }
        true
    }

    fn dump(&mut self) {
        for shard in &mut self.shards.shards {
            shard.unit = None;
        }
    }

    fn is_clifford(&self, qubit: usize) -> bool {
        self.shards.shards[qubit].is_clifford()
    }

    fn get_unitary_fidelity(&self) -> f64 {
        self.log_fidelity.exp()
    }

    fn reset_unitary_fidelity(&mut self) {
        self.log_fidelity = 0.0;
    }

    fn set_sdrp(&mut self, sdrp: f64) {
        self.separability_threshold = sdrp;
    }

    fn clone_qinterface(&self) -> Rc<RefCell<dyn QInterface>> {
        Rc::new(RefCell::new(Self {
            freeze_basis_2_qb: self.freeze_basis_2_qb,
            use_host_ram: self.use_host_ram,
            is_sparse: self.is_sparse,
            is_reactive_separate: self.is_reactive_separate,
            use_t_gadget: self.use_t_gadget,
            threshold_qubits: self.threshold_qubits,
            separability_threshold: self.separability_threshold,
            log_fidelity: self.log_fidelity,
            dev_id: self.dev_id,
            phase_factor: self.phase_factor,
            shards: self.shards.clone(),
            device_ids: self.device_ids.clone(),
            engines: self.engines.clone(),
        }))
    }
}

impl QUnit {
    fn make_engine(&self, length: usize, perm: u64) -> Rc<RefCell<dyn QInterface>> {
        let engine = self.engines[perm as usize].engine.clone();
        let mut engine = engine.borrow_mut();
        engine.set_qubit_count(length);
        engine
    }

    fn set_qubit_count(&mut self, qubit_count: usize) {
        for i in 0..self.shards.shards.len() {
            let shard = &mut self.shards.shards[i];
            if let Some(unit) = &shard.unit {
                unit.borrow_mut().set_qubit_count(qubit_count);
            }
        }
    }

    fn to_perm_basis_prob(&mut self, qubit: usize) {
        for i in 0..qubit {
            self.shards.shards[i].unit.as_mut().unwrap().borrow_mut().to_perm_basis_prob(qubit);
        }
    }

    fn clamp_prob(&self, prob: f64) -> f64 {
        if prob < 0.0 {
            0.0
        } else if prob > 1.0 {
            1.0
        } else {
            prob
        }
    }

    fn norm(&self, amp: Complex) -> f64 {
        amp.norm()
    }

    fn sum_sqr_diff_single(
        &self,
        amp0: Complex,
        amp1: Complex,
        to_compare_amp0: Complex,
        to_compare_amp1: Complex,
    ) -> f64 {
        let diff0 = amp0 - to_compare_amp0;
        let diff1 = amp1 - to_compare_amp1;
        diff0.norm() + diff1.norm()
    }

    fn order_contiguous(&mut self, unit: Rc<RefCell<dyn QInterface>>) {
        let mut unit = unit.borrow_mut();
        unit.order_contiguous();
    }

    fn detach(&mut self, start: usize, length: usize, dest: Option<Rc<RefCell<dyn QInterface>>>) {
        let mut dest = dest;
        if let Some(dest) = &mut dest {
            let mut dest = dest.borrow_mut();
            dest.set_qubit_count(length);
        }
        for i in start..start + length {
            let shard = &mut self.shards.shards[i];
            if let Some(unit) = &shard.unit {
                if let Some(dest) = &mut dest {
                    let mut unit = unit.borrow_mut();
                    unit.detach(0, length, Some(dest.clone_qinterface()));
                } else {
                    let mut unit = unit.borrow_mut();
                    unit.detach(0, length, None);
                }
            }
        }
        if let Some(dest) = &mut dest {
            let mut dest = dest.borrow_mut();
            dest.finish();
        }
        self.shards.shards.drain(start..start + length);
        self.set_qubit_count(self.get_qubit_count() - length);
    }

    fn to_perm_basis_prob(&mut self, qubit: usize) {
        for i in 0..qubit {
            self.shards.shards[i].unit.as_mut().unwrap().borrow_mut().to_perm_basis_prob(qubit);
        }
    }

    fn set_qubit_count(&mut self, qubit_count: usize) {
        for i in 0..self.shards.shards.len() {
            let shard = &mut self.shards.shards[i];
            if let Some(unit) = &shard.unit {
                unit.borrow_mut().set_qubit_count(qubit_count);
            }
        }
    }
}


