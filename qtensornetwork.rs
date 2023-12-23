use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

struct QInterfaceEngine {
    // implementation details
}

struct QCircuit {
    // implementation details
}

struct QInterface {
    // implementation details
}

type QInterfacePtr = Rc<RefCell<QInterface>>;
type QCircuitPtr = Rc<RefCell<QCircuit>>;

struct QTensorNetwork {
    use_host_ram: bool,
    is_sparse: bool,
    is_reactive_separate: bool,
    use_t_gadget: bool,
    is_near_clifford: bool,
    dev_id: i64,
    global_phase: Complex,
    layer_stack: QInterfacePtr,
    device_ids: Vec<i64>,
    engines: Vec<QInterfaceEngine>,
    circuit: Vec<QCircuitPtr>,
    measurements: Vec<HashMap<bitLenInt, bool>>,
}

impl QTensorNetwork {
    fn get_circuit(&mut self, target: bitLenInt, controls: Vec<bitLenInt>) -> QCircuitPtr {
        for (i, m) in self.measurements.iter().enumerate().rev() {
            let l = self.measurements.len() - (i + 1);
            let m = &self.measurements[l];
            if m.contains_key(&target) {
                if self.circuit.len() == l {
                    self.circuit.push(Rc::new(RefCell::new(QCircuit::new())));
                }
                return self.circuit[l].clone();
            }
            for control in &controls {
                if m.contains_key(control) {
                    if self.circuit.len() == l {
                        self.circuit.push(Rc::new(RefCell::new(QCircuit::new())));
                    }
                    return self.circuit[l].clone();
                }
            }
        }
        self.circuit[0].clone()
    }

    fn check_qubit_count(&self, target: bitLenInt) {
        if target >= self.qubit_count {
            panic!("QTensorNetwork qubit index values must be within allocated qubit bounds!");
        }
    }

    fn check_qubit_count_with_controls(&self, target: bitLenInt, controls: &[bitLenInt]) {
        self.check_qubit_count(target);
        throw_if_qb_id_array_is_bad(
            controls,
            self.qubit_count,
            "QTensorNetwork qubit index values must be within allocated qubit bounds!",
        );
    }

    fn run_measurement_layer(&self, layer_id: usize) {
        let m_layer = &self.measurements[layer_id];
        let mut bits = Vec::with_capacity(m_layer.len());
        let mut values = Vec::with_capacity(m_layer.len());
        for (bit, value) in m_layer {
            bits.push(*bit);
            values.push(*value);
        }
        self.layer_stack.borrow_mut().force_m(&bits, &values);
    }

    fn get_threshold_qb(&self) -> bitLenInt {
        30
    }

    fn make_layer_stack(&self, qubits: Option<Vec<bitLenInt>>) {
        // implementation details
    }

    fn run_as_amplitudes<F>(&self, fn: F, qubits: Option<Vec<bitLenInt>>)
    where
        F: Fn(QInterfacePtr),
    {
        if let Some(qubits) = qubits {
            let max_qb = self.get_threshold_qb();
            if self.qubit_count <= max_qb {
                self.make_layer_stack(None);
                fn(self.layer_stack.clone());
            } else {
                self.make_layer_stack(Some(qubits));
                let ls = self.layer_stack.clone();
                self.layer_stack = None;
                fn(ls);
            }
        } else {
            self.make_layer_stack(None);
            fn(self.layer_stack.clone());
        }
    }

    fn get_unitary_fidelity(&self) -> f64 {
        let mut to_ret = 0.0;
        self.run_as_amplitudes(|ls| {
            to_ret = ls.borrow().get_unitary_fidelity();
        }, None);
        to_ret
    }

    fn set_device(&mut self, d_id: i64) {
        self.dev_id = d_id;
    }

    fn finish(&self) {
        if let Some(layer_stack) = &self.layer_stack {
            layer_stack.borrow_mut().finish();
        }
    }

    fn is_finished(&self) -> bool {
        self.layer_stack.is_none() || self.layer_stack.borrow().is_finished()
    }

    fn dump(&self) {
        if let Some(layer_stack) = &self.layer_stack {
            layer_stack.borrow().dump();
        }
    }

    fn update_running_norm(&self, norm_thresh: f32) {
        if let Some(layer_stack) = &self.layer_stack {
            layer_stack.borrow_mut().update_running_norm(norm_thresh);
        }
    }

    fn normalize_state(&self, nrm: f32, norm_thresh: f32, phase_arg: f32) {
        if let Some(layer_stack) = &self.layer_stack {
            layer_stack.borrow_mut().normalize_state(nrm, norm_thresh, phase_arg);
        }
    }

    fn sum_sqr_diff(&self, to_compare: QInterfacePtr) -> f32 {
        let mut to_ret = 0.0;
        to_compare.borrow().make_layer_stack(None);
        self.run_as_amplitudes(|ls| {
            to_ret = ls.borrow().sum_sqr_diff(to_compare.borrow().layer_stack.clone());
        }, None);
        to_ret
    }

    fn set_permutation(&mut self, init_state: bitCapInt, phase_fac: Complex) {
        self.circuit.clear();
        self.measurements.clear();
        self.layer_stack = None;
        self.circuit.push(Rc::new(RefCell::new(QCircuit::new())));
        for i in 0..self.qubit_count {
            if (pow2(i) & init_state) != 0 {
                self.x(i);
            }
        }
        if phase_fac == Complex::default() && self.rand_global_phase {
            let angle = rand() * 2.0 * PI;
            self.global_phase = Complex::new(angle.cos(), angle.sin());
        }
    }

    fn clone(&self) -> QInterfacePtr {
        // implementation details
    }

    fn get_quantum_state(&self, state: &mut [Complex]) {
        self.run_as_amplitudes(|ls| {
            ls.borrow().get_quantum_state(state);
        }, None);
    }

    fn set_quantum_state(&self, state: &[Complex]) {
        panic!("QTensorNetwork::SetQuantumState() not implemented!");
    }

    fn set_quantum_state_qinterface(&self, eng: QInterfacePtr) {
        panic!("QTensorNetwork::SetQuantumState() not implemented!");
    }

    fn get_probs(&self, output_probs: &mut [f32]) {
        self.run_as_amplitudes(|ls| {
            ls.borrow().get_probs(output_probs);
        }, None);
    }

    fn get_amplitude(&self, perm: bitCapInt) -> Complex {
        let mut to_ret = Complex::default();
        self.run_as_amplitudes(|ls| {
            to_ret = ls.borrow().get_amplitude(perm);
        }, None);
        to_ret
    }

    fn set_amplitude(&self, perm: bitCapInt, amp: Complex) {
        panic!("QTensorNetwork::SetAmplitude() not implemented!");
    }

    fn compose(&self, to_copy: QInterfacePtr, start: bitLenInt) -> bitLenInt {
        panic!("QTensorNetwork::Compose() not implemented!");
    }

    fn decompose_qinterface(&self, start: bitLenInt, dest: QInterfacePtr) {
        panic!("QTensorNetwork::Decompose() not implemented!");
    }

    fn decompose(&self, start: bitLenInt, length: bitLenInt) -> QInterfacePtr {
        panic!("QTensorNetwork::Decompose() not implemented!");
    }

    fn dispose(&self, start: bitLenInt, length: bitLenInt) {
        panic!("QTensorNetwork::Dispose() not implemented!");
    }

    fn dispose_with_disposed_perm(&self, start: bitLenInt, length: bitLenInt, disposed_perm: bitCapInt) {
        panic!("QTensorNetwork::Dispose() not implemented!");
    }

    fn allocate(&mut self, start: bitLenInt, length: bitLenInt) -> bitLenInt {
        if start > self.qubit_count {
            panic!("QTensorNetwork::Allocate() 'start' argument is out-of-bounds!");
        }
        if length == 0 {
            return start;
        }
        let moved_qubits = self.qubit_count - start;
        self.set_qubit_count(self.qubit_count + length);
        if moved_qubits == 0 {
            return start;
        }
        for i in 0..moved_qubits {
            let q = start + moved_qubits - (i + 1);
            self.swap(q, q + length);
        }
        start
    }

    fn prob(&self, qubit: bitLenInt) -> f32 {
        let mut to_ret = 0.0;
        self.run_as_amplitudes(|ls| {
            to_ret = ls.borrow().prob(qubit);
        }, Some(vec![qubit]));
        to_ret
    }

    fn prob_all(&self, full_register: bitCapInt) -> f32 {
        let mut to_ret = 0.0;
        self.run_as_amplitudes(|ls| {
            to_ret = ls.borrow().prob_all(full_register);
        }, None);
        to_ret
    }

    fn force_m(&self, qubit: bitLenInt, result: bool, do_force: bool, do_apply: bool) -> bool {
        // implementation details
    }

    fn m_all(&self) -> bitCapInt {
        let mut to_ret = 0;
        let max_qb = self.get_threshold_qb();
        if self.qubit_count <= max_qb {
            self.make_layer_stack(None);
            to_ret = self.layer_stack.borrow().m_all();
        } else {
            for i in 0..self.qubit_count {
                if self.m(i) {
                    bi_or_ip(&mut to_ret, pow2(i));
                }
            }
        }
        self.set_permutation(to_ret);
        to_ret
    }

    fn multi_shot_measure_mask(&self, q_powers: &[bitCapInt], shots: u32) -> HashMap<bitCapInt, i32> {
        let mut qubits = Vec::new();
        for q_pow in q_powers {
            qubits.push(log2(*q_pow));
        }
        let mut to_ret = HashMap::new();
        self.run_as_amplitudes(|ls| {
            to_ret = ls.borrow().multi_shot_measure_mask(q_powers, shots);
        }, Some(qubits));
        to_ret
    }

    fn multi_shot_measure_mask_into(&self, q_powers: &[bitCapInt], shots: u32, shots_array: &mut [u64]) {
        let mut qubits = Vec::new();
        for q_pow in q_powers {
            qubits.push(log2(*q_pow));
        }
        self.run_as_amplitudes(|ls| {
            ls.borrow().multi_shot_measure_mask_into(q_powers, shots, shots_array);
        }, Some(qubits));
    }

    fn mtrx(&self, mtrx: &[Complex], target: bitLenInt) {
        self.check_qubit_count(target);
        self.layer_stack = None;
        self.get_circuit(target, vec![]).borrow_mut().append_gate(target, mtrx);
    }

    fn mc_mtrx(&self, controls: &[bitLenInt], mtrx: &[Complex], target: bitLenInt) {
        self.check_qubit_count_with_controls(target, controls);
        self.layer_stack = None;
        let m = pow2(controls.len()) - 1;
        self.get_circuit(target, controls).borrow_mut().append_gate(target, mtrx, controls.iter().cloned().collect(), m);
    }

    fn mac_mtrx(&self, controls: &[bitLenInt], mtrx: &[Complex], target: bitLenInt) {
        self.check_qubit_count_with_controls(target, controls);
        self.layer_stack = None;
        self.get_circuit(target, controls).borrow_mut().append_gate(target, mtrx, controls.iter().cloned().collect(), 0);
    }

    fn mc_phase(&self, controls: &[bitLenInt], top_left: Complex, bottom_right: Complex, target: bitLenInt) {
        self.check_qubit_count_with_controls(target, controls);
        self.layer_stack = None;
        let mut l_mtrx = vec![Complex::default(); 4];
        l_mtrx[0] = top_left;
        l_mtrx[3] = bottom_right;
        let m = pow2(controls.len()) - 1;
        self.get_circuit(target, controls).borrow_mut().append_gate(target, &l_mtrx, controls.iter().cloned().collect(), m);
    }

    fn mac_phase(&self, controls: &[bitLenInt], top_left: Complex, bottom_right: Complex, target: bitLenInt) {
        self.check_qubit_count_with_controls(target, controls);
        self.layer_stack = None;
        let mut l_mtrx = vec![Complex::default(); 4];
        l_mtrx[0] = top_left;
        l_mtrx[3] = bottom_right;
        self.get_circuit(target, controls).borrow_mut().append_gate(target, &l_mtrx, controls.iter().cloned().collect(), 0);
    }

    fn mc_invert(&self, controls: &[bitLenInt], top_right: Complex, bottom_left: Complex, target: bitLenInt) {
        self.check_qubit_count_with_controls(target, controls);
        self.layer_stack = None;
        let mut l_mtrx = vec![Complex::default(); 4];
        l_mtrx[1] = top_right;
        l_mtrx[2] = bottom_left;
        let m = pow2(controls.len()) - 1;
        self.get_circuit(target, controls).borrow_mut().append_gate(target, &l_mtrx, controls.iter().cloned().collect(), m);
    }

    fn mac_invert(&self, controls: &[bitLenInt], top_right: Complex, bottom_left: Complex, target: bitLenInt) {
        self.check_qubit_count_with_controls(target, controls);
        self.layer_stack = None;
        let mut l_mtrx = vec![Complex::default(); 4];
        l_mtrx[1] = top_right;
        l_mtrx[2] = bottom_left;
        self.get_circuit(target, controls).borrow_mut().append_gate(target, &l_mtrx, controls.iter().cloned().collect(), 0);
    }

    fn f_sim(&self, theta: f32, phi: f32, qubit1: bitLenInt, qubit2: bitLenInt) {
        // implementation details
    }
}


