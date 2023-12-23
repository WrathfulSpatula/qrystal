use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;

struct QStabilizer {
    // implementation of QStabilizer
}

type QStabilizerPtr = Rc<RefCell<QStabilizer>>;

struct CliffordShard {
    mapped: u32,
    unit: QStabilizerPtr,
}

impl CliffordShard {
    fn new(m: u32, u: QStabilizerPtr) -> Self {
        Self {
            mapped: m,
            unit: u,
        }
    }
}

struct QUnitClifford {
    phase_offset: Complex,
    shards: Vec<CliffordShard>,
}

type QUnitCliffordPtr = Rc<RefCell<QUnitClifford>>;

impl QUnitClifford {
    fn combine_phase_offsets(&mut self, unit: QStabilizerPtr) {
        if rand_global_phase {
            return;
        }
        self.phase_offset *= unit.borrow().get_phase_offset();
        unit.borrow_mut().reset_phase_offset();
    }

    fn entangle_in_current_basis(
        &mut self,
        first: impl Iterator<Item = &u32>,
        last: impl Iterator<Item = &u32>,
    ) -> QStabilizerPtr {
        // implementation of entangle_in_current_basis
    }

    fn entangle_all(&mut self) -> QStabilizerPtr {
        if self.qubit_count == 0 {
            return make_stabilizer(0);
        }
        let bits: Vec<u32> = (0..self.qubit_count).collect();
        let ebits: Vec<&u32> = bits.iter().collect();
        let to_ret = self.entangle_in_current_basis(ebits.iter(), ebits.iter().copied());
        order_contiguous(to_ret);
        to_ret
    }

    fn c_gate(
        &mut self,
        control: u32,
        target: u32,
        mtrx: Option<&[Complex]>,
        fn: impl Fn(QStabilizerPtr, u32, u32, Option<&[Complex]>),
    ) {
        let bits = vec![control, target];
        let ebits: Vec<&u32> = bits.iter().collect();
        let unit = self.entangle_in_current_basis(ebits.iter(), ebits.iter().copied());
        fn(unit, bits[0], bits[1], mtrx);
        self.combine_phase_offsets(unit);
        self.try_separate(control);
        self.try_separate(target);
    }

    fn clone_body(&self, copy_ptr: QUnitCliffordPtr) -> QInterfacePtr {
        // implementation of clone_body
    }

    fn throw_if_qubit_invalid(&self, t: u32, method_name: &str) {
        if t >= self.qubit_count {
            panic!(
                "{} target qubit index parameter must be within allocated qubit bounds!",
                method_name
            );
        }
    }

    fn throw_if_qubit_set_invalid(
        &self,
        controls: &[u32],
        t: u32,
        method_name: &str,
    ) -> u32 {
        if t >= self.qubit_count {
            panic!(
                "{} target qubit index parameter must be within allocated qubit bounds!",
                method_name
            );
        }
        if controls.len() > 1 {
            panic!("{} can only have one control qubit!", method_name);
        }
        let c = controls[0];
        if c >= self.qubit_count {
            panic!(
                "{} control qubit index parameter must be within allocated qubit bounds!",
                method_name
            );
        }
        controls[0]
    }

    // implementation of other methods
}

fn make_stabilizer(length: u32, perm: u64, phase_fac: Complex) -> QStabilizerPtr {
    // implementation of make_stabilizer
}

fn set_quantum_state(&self, input_state: &[Complex]) {
    // implementation of set_quantum_state
}

fn set_amplitude(&self, perm: u64, amp: Complex) {
    panic!("QUnitClifford::SetAmplitude() not implemented!");
}

fn cnot(&mut self, c: u32, t: u32) {
    self.c_gate(c, t, None, |unit, c, t, _| {
        unit.borrow_mut().cnot(c, t);
    });
}

fn cy(&mut self, c: u32, t: u32) {
    self.c_gate(c, t, None, |unit, c, t, _| {
        unit.borrow_mut().cy(c, t);
    });
}

fn cz(&mut self, c: u32, t: u32) {
    self.c_gate(c, t, None, |unit, c, t, _| {
        unit.borrow_mut().cz(c, t);
    });
}

fn anti_cnot(&mut self, c: u32, t: u32) {
    self.c_gate(c, t, None, |unit, c, t, _| {
        unit.borrow_mut().anti_cnot(c, t);
    });
}

fn anti_cy(&mut self, c: u32, t: u32) {
    self.c_gate(c, t, None, |unit, c, t, _| {
        unit.borrow_mut().anti_cy(c, t);
    });
}

fn anti_cz(&mut self, c: u32, t: u32) {
    self.c_gate(c, t, None, |unit, c, t, _| {
        unit.borrow_mut().anti_cz(c, t);
    });
}

fn h(&mut self, t: u32) {
    self.throw_if_qubit_invalid(t, "QUnitClifford::H");
    let shard = &mut self.shards[t as usize];
    shard.unit.borrow_mut().h(shard.mapped);
}

fn s(&mut self, t: u32) {
    self.throw_if_qubit_invalid(t, "QUnitClifford::S");
    let shard = &mut self.shards[t as usize];
    shard.unit.borrow_mut().s(shard.mapped);
    self.combine_phase_offsets(shard.unit);
}

fn is_(&mut self, t: u32) {
    self.throw_if_qubit_invalid(t, "QUnitClifford::IS");
    let shard = &mut self.shards[t as usize];
    shard.unit.borrow_mut().is_(shard.mapped);
    self.combine_phase_offsets(shard.unit);
}

fn z(&mut self, t: u32) {
    self.throw_if_qubit_invalid(t, "QUnitClifford::Z");
    let shard = &mut self.shards[t as usize];
    shard.unit.borrow_mut().z(shard.mapped);
    self.combine_phase_offsets(shard.unit);
}

fn x(&mut self, t: u32) {
    self.throw_if_qubit_invalid(t, "QUnitClifford::X");
    let shard = &mut self.shards[t as usize];
    shard.unit.borrow_mut().x(shard.mapped);
}

fn y(&mut self, t: u32) {
    self.throw_if_qubit_invalid(t, "QUnitClifford::Y");
    let shard = &mut self.shards[t as usize];
    shard.unit.borrow_mut().y(shard.mapped);
    self.combine_phase_offsets(shard.unit);
}

fn swap(&mut self, qubit1: u32, qubit2: u32) {
    self.throw_if_qubit_invalid(qubit1, "QUnitClifford::Swap");
    self.throw_if_qubit_invalid(qubit2, "QUnitClifford::Swap");
    if qubit1 == qubit2 {
        return;
    }
    self.shards.swap(qubit1 as usize, qubit2 as usize);
}

fn iswap(&mut self, c: u32, t: u32) {
    self.c_gate(c, t, None, |unit, c, t, _| {
        unit.borrow_mut().iswap(c, t);
    });
}

fn iiswap(&mut self, c: u32, t: u32) {
    self.c_gate(c, t, None, |unit, c, t, _| {
        unit.borrow_mut().iiswap(c, t);
    });
}

fn force_m(&mut self, t: u32, result: bool, do_force: bool, do_apply: bool) -> bool {
    // implementation of force_m
}

fn m_all(&mut self) -> u64 {
    let to_ret = QInterface::m_all(self);
    self.set_permutation(to_ret);
    to_ret
}

fn multi_shot_measure_mask(
    &mut self,
    q_powers: &[u64],
    shots: u32,
) -> HashMap<u64, i32> {
    // implementation of multi_shot_measure_mask
}

fn multi_shot_measure_mask_into(
    &mut self,
    q_powers: &[u64],
    shots: u32,
    shots_array: &mut [u64],
) {
    // implementation of multi_shot_measure_mask_into
}

fn get_quantum_state(&self, state_vec: &mut [Complex]) {
    // implementation of get_quantum_state
}

fn get_quantum_state_into(&self, eng: QInterfacePtr) {
    // implementation of get_quantum_state_into
}

fn get_quantum_state_map(&self) -> HashMap<u64, Complex> {
    // implementation of get_quantum_state_map
}

fn get_probs(&self, output_probs: &mut [f64]) {
    // implementation of get_probs
}

fn get_amplitude(&self, perm: u64) -> Complex {
    // implementation of get_amplitude
}

fn get_amplitudes(&self, perms: &[u64]) -> Vec<Complex> {
    // implementation of get_amplitudes
}

fn is_separable_z(&self, t: u32) -> bool {
    self.throw_if_qubit_invalid(t, "QUnitClifford::IsSeparableZ");
    let shard = &self.shards[t as usize];
    shard.unit.borrow().is_separable_z(shard.mapped)
}

fn is_separable_x(&self, t: u32) -> bool {
    self.throw_if_qubit_invalid(t, "QUnitClifford::IsSeparableX");
    let shard = &self.shards[t as usize];
    shard.unit.borrow().is_separable_x(shard.mapped)
}

fn is_separable_y(&self, t: u32) -> bool {
    self.throw_if_qubit_invalid(t, "QUnitClifford::IsSeparableY");
    let shard = &self.shards[t as usize];
    shard.unit.borrow().is_separable_y(shard.mapped)
}

fn is_separable(&self, t: u32) -> u8 {
    self.throw_if_qubit_invalid(t, "QUnitClifford::IsSeparable");
    let shard = &self.shards[t as usize];
    shard.unit.borrow().is_separable(shard.mapped)
}

fn can_decompose_dispose(&self, start: u32, length: u32) -> bool {
    QUnitClifford::clone(self).entangle_all().can_decompose_dispose(start, length)
}

fn compose(&mut self, to_copy: QUnitCliffordPtr) -> u32 {
    self.compose(to_copy, self.qubit_count)
}

fn compose_qinterface(&mut self, to_copy: QInterfacePtr) -> u32 {
    self.compose(to_copy.as_any().downcast_ref::<QUnitCliffordPtr>().unwrap().clone())
}

fn compose(&mut self, to_copy: QUnitCliffordPtr, start: u32) -> u32 {
    if start > self.qubit_count {
        panic!("QUnit::Compose start index is out-of-bounds!");
    }
    let clone = to_copy.borrow().clone();
    self.shards.splice(start as usize..start as usize, clone.shards.iter().cloned());
    self.set_qubit_count(self.qubit_count + to_copy.borrow().get_qubit_count());
    start
}

fn decompose(&mut self, start: u32, dest: QInterfacePtr) {
    self.decompose(start, dest.as_any().downcast_ref::<QUnitCliffordPtr>().unwrap().clone());
}

fn decompose_qunitclifford(&mut self, start: u32, dest: QUnitCliffordPtr) {
    self.detach(start, dest.get_qubit_count(), Some(dest));
}

fn decompose_qinterface(&mut self, start: u32, dest: QInterfacePtr) {
    self.decompose(start, dest.as_any().downcast_ref::<QUnitCliffordPtr>().unwrap().clone());
}

fn decompose(&mut self, start: u32, length: u32) -> QInterfacePtr {
    let dest = QUnitClifford::new(
        length,
        0,
        rand_generator,
        phase_offset,
        do_normalize,
        rand_global_phase,
        false,
        0,
        use_rdrand,
    );
    self.decompose(start, dest.clone());
    dest
}

fn dispose(&mut self, start: u32, length: u32) {
    self.detach(start, length, None);
}

fn dispose_with_perm(&mut self, start: u32, length: u32, disposed_perm: u64) {
    self.detach(start, length, None);
}

fn allocate(&mut self, start: u32, length: u32) -> u32 {
    if length == 0 {
        return start;
    }
    if start > self.qubit_count {
        panic!("QUnitClifford::Allocate() cannot start past end of register!");
    }
    if self.qubit_count == 0 {
        self.set_qubit_count(length);
        self.set_permutation(0);
        return 0;
    }
    let n_qubits = QUnitClifford::new(
        length,
        0,
        rand_generator,
        phase_offset,
        do_normalize,
        rand_global_phase,
        false,
        0,
        use_rdrand,
    );
    self.compose(n_qubits, start)
}

fn normalize_state(&mut self, nrm: f64, norm_thresh: f64, phase_arg: f64) {
    if !rand_global_phase {
        self.phase_offset *= Complex::from_polar(1.0, phase_arg);
    }
}

fn update_running_norm(&mut self, norm_thresh: f64) {
    // implementation of update_running_norm
}

fn sum_sqr_diff(&self, to_compare: QUnitCliffordPtr) -> f64 {
    // implementation of sum_sqr_diff
}

fn approx_compare(&self, to_compare: QUnitCliffordPtr, error_tol: f64) -> bool {
    if self.as_ptr() == to_compare.as_ptr() {
        return true;
    }
    self.clone().entangle_all().approx_compare(
        to_compare.clone().entangle_all(),
        error_tol,
    )
}

fn prob(&self, qubit: u32) -> f64 {
    self.throw_if_qubit_invalid(qubit, "QUnitClifford::Prob");
    let shard = &self.shards[qubit as usize];
    shard.unit.borrow().prob(shard.mapped)
}

fn mtrx(&mut self, mtrx: &[Complex], t: u32) {
    self.throw_if_qubit_invalid(t, "QUnitClifford::Mtrx");
    let shard = &mut self.shards[t as usize];
    shard.unit.borrow_mut().mtrx(mtrx, shard.mapped);
    self.combine_phase_offsets(shard.unit);
}

fn phase(&mut self, top_left: Complex, bottom_right: Complex, t: u32) {
    self.throw_if_qubit_invalid(t, "QUnitClifford::Phase");
    let shard = &mut self.shards[t as usize];
    shard.unit.borrow_mut().phase(top_left, bottom_right, shard.mapped);
    self.combine_phase_offsets(shard.unit);
}

fn invert(&mut self, top_right: Complex, bottom_left: Complex, t: u32) {
    self.throw_if_qubit_invalid(t, "QUnitClifford::Invert");
    let shard = &mut self.shards[t as usize];
    shard.unit.borrow_mut().invert(top_right, bottom_left, shard.mapped);
    self.combine_phase_offsets(shard.unit);
}

fn mc_phase(
    &mut self,
    controls: &[u32],
    top_left: Complex,
    bottom_right: Complex,
    t: u32,
) {
    if controls.is_empty() {
        self.phase(top_left, bottom_right, t);
        return;
    }
    let c = self.throw_if_qubit_set_invalid(controls, t, "QUnitClifford::MCPhase");
    let mtrx = [top_left, Complex::zero(), Complex::zero(), bottom_right];
    self.c_gate(c, t, Some(&mtrx), |unit, c, t, mtrx| {
        unit.borrow_mut().mc_phase(&[c], mtrx[0], mtrx[3], t);
    });
}

fn mac_phase(
    &mut self,
    controls: &[u32],
    top_left: Complex,
    bottom_right: Complex,
    t: u32,
) {
    if controls.is_empty() {
        self.phase(top_left, bottom_right, t);
        return;
    }
    let c = self.throw_if_qubit_set_invalid(controls, t, "QUnitClifford::MACPhase");
    let mtrx = [top_left, Complex::zero(), Complex::zero(), bottom_right];
    self.c_gate(c, t, Some(&mtrx), |unit, c, t, mtrx| {
        unit.borrow_mut().mac_phase(&[c], mtrx[0], mtrx[3], t);
    });
}

fn mc_invert(
    &mut self,
    controls: &[u32],
    top_right: Complex,
    bottom_left: Complex,
    t: u32,
) {
    if controls.is_empty() {
        self.invert(top_right, bottom_left, t);
        return;
    }
    let c = self.throw_if_qubit_set_invalid(controls, t, "QUnitClifford::MCInvert");
    let mtrx = [Complex::zero(), top_right, bottom_left, Complex::zero()];
    self.c_gate(c, t, Some(&mtrx), |unit, c, t, mtrx| {
        unit.borrow_mut().mc_invert(&[c], mtrx[1], mtrx[2], t);
    });
}

fn mac_invert(
    &mut self,
    controls: &[u32],
    top_right: Complex,
    bottom_left: Complex,
    t: u32,
) {
    if controls.is_empty() {
        self.invert(top_right, bottom_left, t);
        return;
    }
    let c = self.throw_if_qubit_set_invalid(controls, t, "QUnitClifford::MACInvert");
    let mtrx = [Complex::zero(), top_right, bottom_left, Complex::zero()];
    self.c_gate(c, t, Some(&mtrx), |unit, c, t, mtrx| {
        unit.borrow_mut().mac_invert(&[c], mtrx[1], mtrx[2], t);
    });
}

fn mc_mtrx(&mut self, controls: &[u32], mtrx: &[Complex], t: u32) {
    if controls.is_empty() {
        self.mtrx(mtrx, t);
        return;
    }
    let c = self.throw_if_qubit_set_invalid(controls, t, "QUnitClifford::MCMtrx");
    self.c_gate(c, t, Some(mtrx), |unit, c, t, mtrx| {
        unit.borrow_mut().mc_mtrx(&[c], mtrx, t);
    });
}

fn mac_mtrx(&mut self, controls: &[u32], mtrx: &[Complex], t: u32) {
    if controls.is_empty() {
        self.mtrx(mtrx, t);
        return;
    }
    let c = self.throw_if_qubit_set_invalid(controls, t, "QUnitClifford::MACMtrx");
    self.c_gate(c, t, Some(mtrx), |unit, c, t, mtrx| {
        unit.borrow_mut().mac_mtrx(&[c], mtrx, t);
    });
}

fn fsim(&mut self, theta: f64, phi: f64, c: u32, t: u32) {
    self.throw_if_qubit_invalid(c, "QUnitClifford::FSim");
    self.throw_if_qubit_invalid(t, "QUnitClifford::FSim");
    let mtrx = [theta, phi, 0.0, 0.0];
    self.c_gate(c, t, Some(&mtrx), |unit, c, t, mtrx| {
        unit.borrow_mut().fsim(mtrx[0], mtrx[1], c, t);
    });
}

fn try_separate(&mut self, qubit: u32) -> bool {
    // implementation of try_separate
}

fn try_separate_qubits(&mut self, qubits: &[u32], ignored: f64) -> bool {
    for &qubit in qubits {
        if !self.try_separate(qubit) {
            return false;
        }
    }
    true
}

fn try_separate_qubits_with_ignored(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}

fn try_separate_qubits_with_ignored2(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}

fn try_separate_qubits_with_ignored3(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}

fn try_separate_qubits_with_ignored4(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}

fn try_separate_qubits_with_ignored5(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}

fn try_separate_qubits_with_ignored6(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}

fn try_separate_qubits_with_ignored7(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}

fn try_separate_qubits_with_ignored8(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}

fn try_separate_qubits_with_ignored9(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}

fn try_separate_qubits_with_ignored10(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}

fn try_separate_qubits_with_ignored11(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}

fn try_separate_qubits_with_ignored12(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}

fn try_separate_qubits_with_ignored13(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}

fn try_separate_qubits_with_ignored14(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}

fn try_separate_qubits_with_ignored15(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}

fn try_separate_qubits_with_ignored16(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}

fn try_separate_qubits_with_ignored17(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}

fn try_separate_qubits_with_ignored18(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}

fn try_separate_qubits_with_ignored19(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}

fn try_separate_qubits_with_ignored20(&mut self, qubits: &[u32], ignored: f64) -> bool {
    self.try_separate_qubits(qubits, ignored)
}


