use std::collections::HashMap;
use std::f64::consts::PI;
use std::rc::Rc;
use std::cell::RefCell;
use rand::Rng;

type Complex = num_complex::Complex<f64>;
type QInterfacePtr = Rc<RefCell<QInterface>>;

enum Pauli {
    PauliI = 0,
    PauliX = 1,
    PauliY = 3,
    PauliZ = 2,
}

enum QInterfaceEngine {
    QINTERFACE_CPU = 0,
    QINTERFACE_OPENCL,
    QINTERFACE_CUDA,
    QINTERFACE_HYBRID,
    QINTERFACE_BDT,
    QINTERFACE_BDT_HYBRID,
    QINTERFACE_STABILIZER,
    QINTERFACE_STABILIZER_HYBRID,
    QINTERFACE_QPAGER,
    QINTERFACE_QUNIT,
    QINTERFACE_QUNIT_MULTI,
    QINTERFACE_QUNIT_CLIFFORD,
    QINTERFACE_TENSOR_NETWORK,
    QINTERFACE_OPTIMAL_SCHROEDINGER = QINTERFACE_QPAGER,
    QINTERFACE_OPTIMAL_BASE = QINTERFACE_HYBRID,
    QINTERFACE_OPTIMAL = QINTERFACE_QUNIT,
    QINTERFACE_OPTIMAL_MULTI = QINTERFACE_QUNIT_MULTI,
    QINTERFACE_MAX,
}

struct QInterface {
    do_normalize: bool,
    rand_global_phase: bool,
    use_rdrand: bool,
    qubit_count: usize,
    random_seed: u32,
    amplitude_floor: f64,
    max_q_power: u64,
    rand_generator: Box<dyn Rng>,
    hardware_rand_generator: Option<RdRandom>,
}

impl QInterface {
    fn new(n: usize, rgp: Option<Box<dyn Rng>>, do_norm: bool, use_hardware_rng: bool, random_global_phase: bool, norm_thresh: f64) -> Self {
        let qubit_count = n;
        let max_q_power = 2u64.pow(qubit_count as u32);
        let rand_generator = match rgp {
            Some(rgp) => rgp,
            None => Box::new(rand::thread_rng()),
        };
        QInterface {
            do_normalize: do_norm,
            rand_global_phase: random_global_phase,
            use_rdrand: use_hardware_rng,
            qubit_count,
            random_seed: 0,
            amplitude_floor: norm_thresh,
            max_q_power,
            rand_generator,
            hardware_rand_generator: None,
        }
    }

    fn set_random_seed(&mut self, seed: u32) {
        self.random_seed = seed;
        if let Some(rand_generator) = self.rand_generator.as_mut().downcast_mut::<rand::rngs::StdRng>() {
            *rand_generator = rand::SeedableRng::seed_from_u64(seed as u64);
        }
    }

    fn set_quantum_state(&mut self, input_state: &[Complex]) {
        // TODO: Implement
    }

    fn get_quantum_state(&self, output_state: &mut [Complex]) {
        // TODO: Implement
    }

    fn get_probs(&self, output_probs: &mut [f64]) {
        // TODO: Implement
    }

    fn get_amplitude(&self, perm: u64) -> Complex {
        // TODO: Implement
        Complex::new(0.0, 0.0)
    }

    fn set_amplitude(&mut self, perm: u64, amp: Complex) {
        // TODO: Implement
    }

    fn set_permutation(&mut self, perm: u64, phase_fac: Complex) {
        // TODO: Implement
    }

    fn compose(&mut self, to_copy: QInterfacePtr) -> usize {
        self.compose(to_copy, self.qubit_count)
    }

    fn compose_no_clone(&mut self, to_copy: QInterfacePtr) -> usize {
        self.compose(to_copy)
    }

    fn compose_multiple(to_copy: Vec<QInterfacePtr>) -> HashMap<QInterfacePtr, usize> {
        // TODO: Implement
        HashMap::new()
    }

    fn compose(&mut self, to_copy: QInterfacePtr, start: usize) -> usize {
        // TODO: Implement
        0
    }

    fn decompose(&mut self, start: usize, dest: QInterfacePtr) {
        // TODO: Implement
    }

    fn decompose(&mut self, start: usize, length: usize) -> QInterfacePtr {
        // TODO: Implement
        Rc::new(RefCell::new(QInterface::new(0, None, false, false, false, 0.0)))
    }

    fn dispose(&mut self, start: usize, length: usize) {
        // TODO: Implement
    }

    fn dispose(&mut self, start: usize, length: usize, disposed_perm: u64) {
        // TODO: Implement
    }

    fn allocate(&mut self, length: usize) -> usize {
        self.allocate(self.qubit_count, length)
    }

    fn allocate(&mut self, start: usize, length: usize) -> usize {
        // TODO: Implement
        0
    }

    fn mtrx(&mut self, mtrx: &[Complex], qubit_index: usize) {
        // TODO: Implement
    }

    fn mc_mtrx(&mut self, controls: &[usize], mtrx: &[Complex], target: usize) {
        // TODO: Implement
    }

    fn mac_mtrx(&mut self, controls: &[usize], mtrx: &[Complex], target: usize) {
        if mtrx[1].norm() == 0.0 && mtrx[2].norm() == 0.0 {
            self.mac_phase(controls, mtrx[0], mtrx[3], target);
        } else if mtrx[0].norm() == 0.0 && mtrx[3].norm() == 0.0 {
            self.mac_invert(controls, mtrx[1], mtrx[2], target);
        } else {
            self.mac_wrapper(controls, |lc| self.mc_mtrx(lc, mtrx, target));
        }
    }

    fn uc_mtrx(&mut self, controls: &[usize], mtrx: &[Complex], target: usize, control_perm: u64) {
        // TODO: Implement
    }

    fn phase(&mut self, top_left: Complex, bottom_right: Complex, qubit_index: usize) {
        if (self.rand_global_phase || (Complex::new(1.0, 0.0) - top_left).norm() == 0.0) && (top_left - bottom_right).norm() == 0.0 {
            return;
        }
        let mtrx = [top_left, Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), bottom_right];
        self.mtrx(&mtrx, qubit_index);
    }

    fn invert(&mut self, top_right: Complex, bottom_left: Complex, qubit_index: usize) {
        let mtrx = [Complex::new(0.0, 0.0), top_right, bottom_left, Complex::new(0.0, 0.0)];
        self.mtrx(&mtrx, qubit_index);
    }
}


