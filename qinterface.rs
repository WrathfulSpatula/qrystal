use std::collections::HashMap;
use std::f64::consts::PI;
use std::rc::Rc;
use std::cell::RefCell;
use rand::Rng;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::rc::Rc;

type RdRandom = rand::rngs::StdRng;
type Complex = num_complex::Complex<f64>;
type QInterfacePtr = Rc<dyn QInterface>;
type QrackRandGenPtr = Option<Rc<dyn QrackRandGen>>;

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

trait ParallelFor {}

trait QInterface: ParallelFor {
    fn set_qubit_count(&mut self, qb: usize);
    fn norm_helper(c: Complex) -> f64;
    fn clamp_prob(to_clamp: f64) -> f64;
    fn get_nonunitary_phase(&self) -> Complex;
    fn mac_wrapper<F>(&mut self, controls: &[usize], fn: F)
    where
        F: Fn(&[usize]);
    fn sample_clone(&self, q_powers: &[usize]) -> usize;
    fn finish(&mut self);
    fn is_finished(&self) -> bool;
    fn dump(&self);
    fn is_binary_decision_tree(&self) -> bool;
    fn is_clifford(&self) -> bool;
    fn is_clifford_qubit(&self, qubit: usize) -> bool;
    fn is_open_cl(&self) -> bool;
    fn set_random_seed(&mut self, seed: u32);
    fn set_concurrency(&mut self, threads_per_engine: u32);
    fn get_qubit_count(&self) -> usize;
    fn get_max_q_power(&self) -> usize;
    fn get_is_arbitrary_global_phase(&self) -> bool;
    fn rand(&self) -> f64;
    fn set_permutation(&mut self, perm: usize, phase_fac: Complex);
    fn compose(&mut self, to_copy: QInterfacePtr) -> usize;
    fn compose_no_clone(&mut self, to_copy: QInterfacePtr) -> usize;
    fn compose_multiple(&mut self, to_copy: Vec<QInterfacePtr>) -> HashMap<QInterfacePtr, usize>;
    fn compose_with_start(&mut self, to_copy: QInterfacePtr, start: usize) -> usize;
    fn allocate(&mut self, length: usize) -> usize;
    fn allocate_with_start(&mut self, start: usize, length: usize) -> usize;
    fn mac_mtrx(&mut self, controls: &[usize], mtrx: &[Complex], target: usize);
    fn uc_mtrx(&mut self, controls: &[usize], mtrx: &[Complex], target: usize, control_perm: usize);
    fn phase(&mut self, top_left: Complex, bottom_right: Complex, qubit_index: usize);
    fn invert(&mut self, top_right: Complex, bottom_left: Complex, qubit_index: usize);
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
        control_perm: usize,
    );
    fn uc_invert(
        &mut self,
        controls: &[usize],
        top_right: Complex,
        bottom_left: Complex,
        target: usize,
        control_perm: usize,
    );
    fn uniformly_controlled_single_bit(
        &mut self,
        controls: &[usize],
        qubit_index: usize,
        mtrxs: &[Complex],
    );
    fn ccnot(&mut self, control1: usize, control2: usize, target: usize);
    fn anti_ccnot(&mut self, control1: usize, control2: usize, target: usize);
    fn cnot(&mut self, control: usize, target: usize);
    fn anti_cnot(&mut self, control: usize, target: usize);
    fn cy(&mut self, control: usize, target: usize);
    fn anti_cy(&mut self, control: usize, target: usize);
    fn ccy(&mut self, control1: usize, control2: usize, target: usize);
    fn anti_ccy(&mut self, control1: usize, control2: usize, target: usize);
    fn cz(&mut self, control: usize, target: usize);
    fn anti_cz(&mut self, control: usize, target: usize);
    fn ccz(&mut self, control1: usize, control2: usize, target: usize);
    fn anti_ccz(&mut self, control1: usize, control2: usize, target: usize);
    fn u(&mut self, target: usize, theta: f64, phi: f64, lambda: f64);
    fn u2(&mut self, target: usize, phi: f64, lambda: f64);
    fn iu2(&mut self, target: usize, phi: f64, lambda: f64);
    fn h(&mut self, qubit: usize);
    fn sqrt_h(&mut self, qubit: usize);
    fn sh(&mut self, qubit: usize);
    fn his(&mut self, qubit: usize);
    fn m(&mut self, qubit_index: usize) -> bool;
    fn force_m(&mut self, qubit: usize, result: bool, do_force: bool, do_apply: bool) -> bool;
    fn s(&mut self, qubit: usize);
    fn is(&mut self, qubit: usize);
    fn t(&mut self, qubit: usize);
    fn it(&mut self, qubit: usize);
    fn phase_root_n(&mut self, n: usize, qubit: usize);
    fn i_phase_root_n(&mut self, n: usize, qubit: usize);
    fn x(&mut self, qubit: usize);
    fn y(&mut self, qubit: usize);
    fn z(&mut self, qubit: usize);
    fn sqrt_x(&mut self, qubit: usize);
    fn isqrt_x(&mut self, qubit: usize);
    fn sqrt_y(&mut self, qubit: usize);
    fn isqrt_y(&mut self, qubit: usize);
    fn sqrt_w(&mut self, qubit: usize);
    fn isqrt_w(&mut self, qubit: usize);
    fn ch(&mut self, control: usize, target: usize);
    fn anti_ch(&mut self, control: usize, target: usize);
    fn cs(&mut self, control: usize, target: usize);
    fn anti_cs(&mut self, control: usize, target: usize);
    fn cis(&mut self, control: usize, target: usize);
    fn anti_cis(&mut self, control: usize, target: usize);
    fn ct(&mut self, control: usize, target: usize);
    fn cit(&mut self, control: usize, target: usize);
    fn c_phase_root_n(&mut self, n: usize, control: usize, target: usize);
    fn anti_c_phase_root_n(&mut self, n: usize, control: usize, target: usize);
    fn ci_phase_root_n(&mut self, n: usize, control: usize, target: usize);
    fn anti_ci_phase_root_n(&mut self, n: usize, control: usize, target: usize);
    fn phase_flip(&mut self);
    fn set_reg(&mut self, start: usize, length: usize, value: usize);
    fn m_reg(&mut self, start: usize, length: usize) -> usize;
    fn m_all(&mut self) -> usize;
    fn force_m_reg(
        &mut self,
        start: usize,
        length: usize,
        result: usize,
        do_force: bool,
        do_apply: bool,
    ) -> usize;
    fn m(&mut self, bits: &[usize]) -> usize;
    fn force_m(&mut self, bits: &[usize], values: &[bool], do_apply: bool) -> usize;
}

struct QInterfaceImpl {
    do_normalize: bool,
    rand_global_phase: bool,
    use_rdrand: bool,
    qubit_count: usize,
    random_seed: u32,
    amplitude_floor: f64,
    max_q_power: usize,
    rand_generator: QrackRandGenPtr,
    rand_distribution: rand::distributions::Uniform<f64>,
    hardware_rand_generator: Option<RdRandom>,
}

impl QInterfaceImpl {
    pub fn new(
        n: u32,
        rgp: Option<qrack_rand_gen_ptr>,
        do_norm: bool,
        use_hardware_rng: bool,
        random_global_phase: bool,
        norm_thresh: f64,
    ) -> Self {
        let do_normalize = do_norm;
        let rand_global_phase = random_global_phase;
        let use_rdrand = use_hardware_rng;
        let qubit_count = n;
        let amplitude_floor = norm_thresh;
        let max_q_power = 2u64.pow(qubit_count);
        let rand_distribution = rand::distributions::Uniform::new(0.0, 1.0);
        let hardware_rand_generator = if use_hardware_rng {
            let hardware_rand_generator = RdRandom::new();
            if !hardware_rand_generator.supports_rdrand() {
                None
            } else {
                Some(hardware_rand_generator)
            }
        } else {
            None
        };
        let rand_generator = if let Some(rgp) = rgp {
            rgp
        } else if let Some(hardware_rand_generator) = &hardware_rand_generator {
            use_rdrand = true;
            Arc::new(Mutex::new(qrack_rand_gen::new_with_rng(hardware_rand_generator)))
        } else {
            let random_seed = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs() as u32;
            Arc::new(Mutex::new(qrack_rand_gen::new_with_seed(random_seed)))
        };

        Self {
            do_normalize,
            rand_global_phase,
            use_rdrand,
            qubit_count,
            amplitude_floor,
            max_q_power,
            rand_distribution,
            hardware_rand_generator,
            rand_generator,
        }
    }
}

impl QInterface for QInterfaceImpl {
    fn set_qubit_count(&mut self, qb: usize) {
        self.qubit_count = qb;
        self.max_q_power = 1 << qb;
    }

    fn norm_helper(c: Complex) -> f64 {
        c.norm()
    }

    fn clamp_prob(to_clamp: f64) -> f64 {
        if to_clamp < 0.0 {
            0.0
        } else if to_clamp > 1.0 {
            1.0
        } else {
            to_clamp
        }
    }

    fn get_nonunitary_phase(&self) -> Complex {
        if self.rand_global_phase {
            let angle = self.rand() * 2.0 * PI;
            Complex::new(angle.cos(), angle.sin())
        } else {
            Complex::new(1.0, 0.0)
        }
    }

    fn mac_wrapper<F>(&mut self, controls: &[usize], fn: F)
    where
        F: Fn(&[usize]),
    {
        let x_mask = controls.iter().fold(0, |acc, &control| acc | (1 << control));
        self.x_mask(x_mask);
        fn(controls);
        self.x_mask(x_mask);
    }

    fn sample_clone(&self, q_powers: &[usize]) -> usize {
        let clone = self.clone();
        let raw_sample = clone.m_all();
        let mut sample = 0;
        for (i, &q_power) in q_powers.iter().enumerate() {
            if raw_sample & q_power != 0 {
                sample |= 1 << i;
            }
        }
        sample
    }

    fn finish(&mut self) {}

    fn is_finished(&self) -> bool {
        true
    }

    fn dump(&self) {}

    fn is_binary_decision_tree(&self) -> bool {
        false
    }

    fn is_clifford(&self) -> bool {
        false
    }

    fn is_clifford_qubit(&self, qubit: usize) -> bool {
        false
    }

    fn is_open_cl(&self) -> bool {
        false
    }

    fn set_random_seed(&mut self, seed: u32) {
        if let Some(rand_generator) = &mut self.rand_generator {
            rand_generator.seed(seed);
        }
    }

    fn set_concurrency(&mut self, threads_per_engine: u32) {
        self.set_concurrency_level(threads_per_engine);
    }

    fn get_qubit_count(&self) -> usize {
        self.qubit_count
    }

    fn get_max_q_power(&self) -> usize {
        self.max_q_power
    }

    fn get_is_arbitrary_global_phase(&self) -> bool {
        self.rand_global_phase
    }

    fn rand(&self) -> f64 {
        if let Some(hardware_rand_generator) = &self.hardware_rand_generator {
            hardware_rand_generator.gen()
        } else {
            self.rand_distribution.sample(&mut self.rand_generator.as_ref().unwrap())
        }
    }

    fn set_permutation(&mut self, perm: usize, phase_fac: Complex) {}

    fn compose(&mut self, to_copy: QInterfacePtr) -> usize {
        self.compose_with_start(to_copy, self.qubit_count)
    }

    fn compose_no_clone(&mut self, to_copy: QInterfacePtr) -> usize {
        self.compose(to_copy)
    }

    fn compose_multiple(&mut self, to_copy: Vec<QInterfacePtr>) -> HashMap<QInterfacePtr, usize> {
        let mut result = HashMap::new();
        let mut start = self.qubit_count;
        for q_interface in to_copy {
            let length = self.compose_with_start(q_interface, start);
            result.insert(q_interface, start);
            start += length;
        }
        result
    }

    fn compose_with_start(&mut self, to_copy: QInterfacePtr, start: usize) -> usize {
        let qubit_count = to_copy.get_qubit_count();
        let max_q_power = to_copy.get_max_q_power();
        let mut qubits = Vec::with_capacity(qubit_count);
        for i in 0..qubit_count {
            qubits.push(start + i);
        }
        self.uniformly_controlled_single_bit(&qubits, start, &[Complex::new(1.0, 0.0)]);
        max_q_power
    }

    fn allocate(&mut self, length: usize) -> usize {
        self.allocate_with_start(self.qubit_count, length)
    }

    fn allocate_with_start(&mut self, start: usize, length: usize) -> usize {
        let end = start + length;
        for i in start..end {
            self.h(i);
        }
        length
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

    fn uc_mtrx(&mut self, controls: &[usize], mtrx: &[Complex], target: usize, control_perm: usize) {
        if mtrx[1].norm() == 0.0 && mtrx[2].norm() == 0.0 {
            self.uc_phase(controls, mtrx[0], mtrx[3], target, control_perm);
        } else if mtrx[0].norm() == 0.0 && mtrx[3].norm() == 0.0 {
            self.uc_invert(controls, mtrx[1], mtrx[2], target, control_perm);
        } else {
            self.mac_wrapper(controls, |lc| self.uc_mtrx(lc, mtrx, target, control_perm));
        }
    }

    fn phase(&mut self, top_left: Complex, bottom_right: Complex, qubit_index: usize) {
        if self.rand_global_phase || (Complex::new(1.0, 0.0) - top_left).norm() == 0.0
            || (top_left - bottom_right).norm() == 0.0
        {
            return;
        }
        let mtrx = [top_left, Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), bottom_right];
        self.mtrx(&mtrx, qubit_index);
    }

    fn invert(&mut self, top_right: Complex, bottom_left: Complex, qubit_index: usize) {
        let mtrx = [Complex::new(0.0, 0.0), top_right, bottom_left, Complex::new(0.0, 0.0)];
        self.mtrx(&mtrx, qubit_index);
    }

    fn mc_phase(
        &mut self,
        controls: &[usize],
        top_left: Complex,
        bottom_right: Complex,
        target: usize,
    ) {
        if (Complex::new(1.0, 0.0) - top_left).norm() == 0.0
            && (Complex::new(1.0, 0.0) - bottom_right).norm() == 0.0
        {
            return;
        }
        let mtrx = [top_left, Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), bottom_right];
        self.mc_mtrx(controls, &mtrx, target);
    }

    fn mc_invert(
        &mut self,
        controls: &[usize],
        top_right: Complex,
        bottom_left: Complex,
        target: usize,
    ) {
        let mtrx = [Complex::new(0.0, 0.0), top_right, bottom_left, Complex::new(0.0, 0.0)];
        self.mc_mtrx(controls, &mtrx, target);
    }

    fn mac_phase(
        &mut self,
        controls: &[usize],
        top_left: Complex,
        bottom_right: Complex,
        target: usize,
    ) {
        if (Complex::new(1.0, 0.0) - top_left).norm() == 0.0
            && (Complex::new(1.0, 0.0) - bottom_right).norm() == 0.0
        {
            return;
        }
        self.mac_wrapper(controls, |lc| {
            self.mc_phase(lc, top_left, bottom_right, target);
        });
    }

    fn mac_invert(
        &mut self,
        controls: &[usize],
        top_right: Complex,
        bottom_left: Complex,
        target: usize,
    ) {
        self.mac_wrapper(controls, |lc| {
            self.mc_invert(lc, top_right, bottom_left, target);
        });
    }

    fn uc_phase(
        &mut self,
        controls: &[usize],
        top_left: Complex,
        bottom_right: Complex,
        target: usize,
        control_perm: usize,
    ) {
        if (Complex::new(1.0, 0.0) - top_left).norm() == 0.0
            && (Complex::new(1.0, 0.0) - bottom_right).norm() == 0.0
        {
            return;
        }
        let mtrx = [top_left, Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), bottom_right];
        self.uc_mtrx(controls, &mtrx, target, control_perm);
    }

    fn uc_invert(
        &mut self,
        controls: &[usize],
        top_right: Complex,
        bottom_left: Complex,
        target: usize,
        control_perm: usize,
    ) {
        let mtrx = [Complex::new(0.0, 0.0), top_right, bottom_left, Complex::new(0.0, 0.0)];
        self.uc_mtrx(controls, &mtrx, target, control_perm);
    }

    fn uniformly_controlled_single_bit(
        &mut self,
        controls: &[usize],
        qubit_index: usize,
        mtrxs: &[Complex],
    ) {
        self.uniformly_controlled_single_bit_with_skip(
            controls,
            qubit_index,
            mtrxs,
            &[],
            0,
        );
    }

    fn uniformly_controlled_single_bit_with_skip(
        &mut self,
        controls: &[usize],
        qubit_index: usize,
        mtrxs: &[Complex],
        mtrx_skip_powers: &[usize],
        mtrx_skip_value_mask: usize,
    ) {
        let mut mtrx = vec![Complex::new(0.0, 0.0); 4];
        for (i, &m) in mtrxs.iter().enumerate() {
            mtrx[i] = m;
        }
        self.mac_wrapper(controls, |lc| {
            self.mc_mtrx_with_skip(
                lc,
                &mtrx,
                qubit_index,
                mtrx_skip_powers,
                mtrx_skip_value_mask,
            );
        });
    }

    fn ccnot(&mut self, control1: usize, control2: usize, target: usize) {
        let controls = [control1, control2];
        self.mc_invert(&controls, Complex::new(1.0, 0.0), Complex::new(1.0, 0.0), target);
    }

    fn anti_ccnot(&mut self, control1: usize, control2: usize, target: usize) {
        let controls = [control1, control2];
        self.mac_invert(&controls, Complex::new(1.0, 0.0), Complex::new(1.0, 0.0), target);
    }

    fn cnot(&mut self, control: usize, target: usize) {
        let controls = [control];
        self.mc_invert(&controls, Complex::new(1.0, 0.0), Complex::new(1.0, 0.0), target);
    }

    fn anti_cnot(&mut self, control: usize, target: usize) {
        let controls = [control];
        self.mac_invert(&controls, Complex::new(1.0, 0.0), Complex::new(1.0, 0.0), target);
    }

    fn cy(&mut self, control: usize, target: usize) {
        let controls = [control];
        self.mc_invert(&controls, Complex::new(0.0, 1.0), Complex::new(0.0, -1.0), target);
    }

    fn anti_cy(&mut self, control: usize, target: usize) {
        let controls = [control];
        self.mac_invert(&controls, Complex::new(0.0, 1.0), Complex::new(0.0, -1.0), target);
    }

    fn ccy(&mut self, control1: usize, control2: usize, target: usize) {
        let controls = [control1, control2];
        self.mc_invert(&controls, Complex::new(0.0, 1.0), Complex::new(0.0, -1.0), target);
    }

    fn anti_ccy(&mut self, control1: usize, control2: usize, target: usize) {
        let controls = [control1, control2];
        self.mac_invert(&controls, Complex::new(0.0, 1.0), Complex::new(0.0, -1.0), target);
    }

    fn cz(&mut self, control: usize, target: usize) {
        let controls = [control];
        self.mc_phase(&controls, Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0), target);
    }

    fn anti_cz(&mut self, control: usize, target: usize) {
        let controls = [control];
        self.mac_phase(&controls, Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0), target);
    }

    fn ccz(&mut self, control1: usize, control2: usize, target: usize) {
        let controls = [control1, control2];
        self.mc_phase(&controls, Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0), target);
    }

    fn anti_ccz(&mut self, control1: usize, control2: usize, target: usize) {
        let controls = [control1, control2];
        self.mac_phase(&controls, Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0), target);
    }

    fn u(&mut self, target: usize, theta: f64, phi: f64, lambda: f64) {
        let c = Complex::new(theta.cos(), theta.sin());
        let s = Complex::new((theta / 2.0).cos(), (theta / 2.0).sin());
        let mtrx = [
            Complex::new(c * phi.cos(), c * phi.sin()),
            Complex::new(-s * lambda.sin(), s * lambda.cos()),
            Complex::new(s * lambda.cos(), s * lambda.sin()),
            Complex::new(c * phi.cos(), -c * phi.sin()),
        ];
        self.mtrx(&mtrx, target);
    }

    fn u2(&mut self, target: usize, phi: f64, lambda: f64) {
        self.u(target, PI / 2.0, phi, lambda);
    }

    fn iu2(&mut self, target: usize, phi: f64, lambda: f64) {
        self.u(target, PI / 2.0, -lambda - PI, -phi + PI);
    }

    fn h(&mut self, qubit: usize) {
        let c_sqrt1_2 = Complex::new(1.0 / 2.0f64.sqrt(), 0.0);
        let c_sqrt1_2_neg = Complex::new(-1.0 / 2.0f64.sqrt(), 0.0);
        let mtrx = [
            c_sqrt1_2,
            c_sqrt1_2,
            c_sqrt1_2,
            c_sqrt1_2_neg,
        ];
        self.mtrx(&mtrx, qubit);
    }

    fn sqrt_h(&mut self, qubit: usize) {
        let m00 = Complex::new((1.0 + 2.0f64.sqrt()) / (2.0 * 2.0f64.sqrt()), (-1.0 + 2.0f64.sqrt()) / (2.0 * 2.0f64.sqrt()));
        let m01 = Complex::new(1.0 / 2.0f64.sqrt(), -1.0 / 2.0f64.sqrt());
        let m10 = m01;
        let m11 = Complex::new((-1.0 + 2.0f64.sqrt()) / (2.0 * 2.0f64.sqrt()), (1.0 + 2.0f64.sqrt()) / (2.0 * 2.0f64.sqrt()));
        let mtrx = [
            m00,
            m01,
            m10,
            m11,
        ];
        self.mtrx(&mtrx, qubit);
    }

    fn sh(&mut self, qubit: usize) {
        let c_sqrt1_2 = Complex::new(1.0 / 2.0f64.sqrt(), 0.0);
        let c_i_sqrt1_2 = Complex::new(0.0, 1.0 / 2.0f64.sqrt());
        let c_i_sqrt1_2_neg = Complex::new(0.0, -1.0 / 2.0f64.sqrt());
        let mtrx = [
            c_sqrt1_2,
            c_sqrt1_2,
            c_i_sqrt1_2,
            c_i_sqrt1_2_neg,
        ];
        self.mtrx(&mtrx, qubit);
    }

    fn his(&mut self, qubit: usize) {
        let c_sqrt1_2 = Complex::new(1.0 / 2.0f64.sqrt(), 0.0);
        let c_i_sqrt1_2 = Complex::new(0.0, 1.0 / 2.0f64.sqrt());
        let c_i_sqrt1_2_neg = Complex::new(0.0, -1.0 / 2.0f64.sqrt());
        let mtrx = [
            c_sqrt1_2,
            c_i_sqrt1_2_neg,
            c_sqrt1_2,
            c_i_sqrt1_2,
        ];
        self.mtrx(&mtrx, qubit);
    }

    fn m(&mut self, qubit_index: usize) -> bool {
        self.force_m(qubit_index, false, false, false)
    }

    fn force_m(&mut self, qubit: usize, result: bool, do_force: bool, do_apply: bool) -> bool {
        false
    }

    fn s(&mut self, qubit: usize) {
        self.phase(Complex::new(1.0, 0.0), Complex::new(0.0, 1.0), qubit);
    }

    fn is(&mut self, qubit: usize) {
        self.phase(Complex::new(1.0, 0.0), Complex::new(0.0, -1.0), qubit);
    }

    fn t(&mut self, qubit: usize) {
        self.phase(Complex::new(1.0, 0.0), Complex::new(1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt()), qubit);
    }

    fn it(&mut self, qubit: usize) {
        self.phase(Complex::new(1.0, 0.0), Complex::new(1.0 / 2.0f64.sqrt(), -1.0 / 2.0f64.sqrt()), qubit);
    }

    fn phase_root_n(&mut self, n: usize, qubit: usize) {
        if n == 0 {
            return;
        }
        self.phase(Complex::new(1.0, 0.0), (-1.0f64).powf(1.0 / (1 << (n - 1))) as f64, qubit);
    }

    fn i_phase_root_n(&mut self, n: usize, qubit: usize) {
        if n == 0 {
            return;
        }
        self.phase(Complex::new(1.0, 0.0), (-1.0f64).powf(-1.0 / (1 << (n - 1))) as f64, qubit);
    }

    fn x(&mut self, qubit: usize) {
        self.invert(Complex::new(1.0, 0.0), Complex::new(1.0, 0.0), qubit);
    }

    fn y(&mut self, qubit: usize) {
        self.invert(Complex::new(0.0, 1.0), Complex::new(0.0, -1.0), qubit);
    }

    fn z(&mut self, qubit: usize) {
        self.phase(Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0), qubit);
    }

    fn sqrt_x(&mut self, qubit: usize) {
        let c_sqrt1_2 = Complex::new(1.0 / 2.0f64.sqrt(), 0.0);
        let c_sqrt1_2_neg = Complex::new(-1.0 / 2.0f64.sqrt(), 0.0);
        let mtrx = [
            c_sqrt1_2,
            c_sqrt1_2_neg,
            c_sqrt1_2,
            c_sqrt1_2,
        ];
        self.mtrx(&mtrx, qubit);
    }

    fn isqrt_x(&mut self, qubit: usize) {
        let c_sqrt1_2 = Complex::new(1.0 / 2.0f64.sqrt(), 0.0);
        let c_sqrt1_2_neg = Complex::new(-1.0 / 2.0f64.sqrt(), 0.0);
        let mtrx = [
            c_sqrt1_2_neg,
            c_sqrt1_2,
            c_sqrt1_2,
            c_sqrt1_2_neg,
        ];
        self.mtrx(&mtrx, qubit);
    }

    fn sqrt_y(&mut self, qubit: usize) {
        let c_sqrt1_2 = Complex::new(1.0 / 2.0f64.sqrt(), 0.0);
        let c_sqrt1_2_neg = Complex::new(-1.0 / 2.0f64.sqrt(), 0.0);
        let mtrx = [
            c_sqrt1_2,
            c_sqrt1_2_neg,
            c_sqrt1_2,
            c_sqrt1_2,
        ];
        self.mtrx(&mtrx, qubit);
    }

    fn isqrt_y(&mut self, qubit: usize) {
        let c_sqrt1_2 = Complex::new(1.0 / 2.0f64.sqrt(), 0.0);
        let c_sqrt1_2_neg = Complex::new(-1.0 / 2.0f64.sqrt(), 0.0);
        let mtrx = [
            c_sqrt1_2_neg,
            c_sqrt1_2_neg,
            c_sqrt1_2,
            c_sqrt1_2,
        ];
        self.mtrx(&mtrx, qubit);
    }

    fn sqrt_w(&mut self, qubit: usize) {
        let diag = Complex::new(1.0 / 2.0f64.sqrt(), 0.0);
        let m01 = Complex::new(-1.0 / 2.0f64.sqrt(), -1.0 / 2.0f64.sqrt());
        let m10 = Complex::new(1.0 / 2.0f64.sqrt(), -1.0 / 2.0f64.sqrt());
        let mtrx = [
            diag,
            m01,
            m10,
            diag,
        ];
        self.mtrx(&mtrx, qubit);
    }

    fn isqrt_w(&mut self, qubit: usize) {
        let diag = Complex::new(1.0 / 2.0f64.sqrt(), 0.0);
        let m01 = Complex::new(1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt());
        let m10 = Complex::new(-1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt());
        let mtrx = [
            diag,
            m01,
            m10,
            diag,
        ];
        self.mtrx(&mtrx, qubit);
    }

    fn ch(&mut self, control: usize, target: usize) {
        let controls = [control];
        let c_sqrt1_2 = Complex::new(1.0 / 2.0f64.sqrt(), 0.0);
        let c_sqrt1_2_neg = Complex::new(-1.0 / 2.0f64.sqrt(), 0.0);
        let mtrx = [
            c_sqrt1_2,
            c_sqrt1_2,
            c_sqrt1_2,
            c_sqrt1_2_neg,
        ];
        self.mc_mtrx(&controls, &mtrx, target);
    }

    fn anti_ch(&mut self, control: usize, target: usize) {
        let controls = [control];
        let c_sqrt1_2 = Complex::new(1.0 / 2.0f64.sqrt(), 0.0);
        let c_sqrt1_2_neg = Complex::new(-1.0 / 2.0f64.sqrt(), 0.0);
        let mtrx = [
            c_sqrt1_2,
            c_sqrt1_2,
            c_sqrt1_2,
            c_sqrt1_2_neg,
        ];
        self.mac_mtrx(&controls, &mtrx, target);
    }

    fn cs(&mut self, control: usize, target: usize) {
        let controls = [control];
        self.mc_phase(&controls, Complex::new(1.0, 0.0), Complex::new(0.0, 1.0), target);
    }

    fn anti_cs(&mut self, control: usize, target: usize) {
        let controls = [control];
        self.mac_phase(&controls, Complex::new(1.0, 0.0), Complex::new(0.0, 1.0), target);
    }

    fn cis(&mut self, control: usize, target: usize) {
        let controls = [control];
        self.mc_phase(&controls, Complex::new(1.0, 0.0), Complex::new(0.0, -1.0), target);
    }

    fn anti_cis(&mut self, control: usize, target: usize) {
        let controls = [control];
        self.mac_phase(&controls, Complex::new(1.0, 0.0), Complex::new(0.0, -1.0), target);
    }

    fn ct(&mut self, control: usize, target: usize) {
        let controls = [control];
        self.mc_phase(&controls, Complex::new(1.0, 0.0), Complex::new(1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt()), target);
    }

    fn cit(&mut self, control: usize, target: usize) {
        let controls = [control];
        self.mc_phase(&controls, Complex::new(1.0, 0.0), Complex::new(1.0 / 2.0f64.sqrt(), -1.0 / 2.0f64.sqrt()), target);
    }

    fn c_phase_root_n(&mut self, n: usize, control: usize, target: usize) {
        let controls = [control];
        if n == 0 {
            return;
        }
        self.mc_phase(&controls, Complex::new(1.0, 0.0), (-1.0f64).powf(1.0 / (1 << (n - 1))) as f64, target);
    }

    fn anti_c_phase_root_n(&mut self, n: usize, control: usize, target: usize) {
        let controls = [control];
        if n == 0 {
            return;
        }
        self.mac_phase(&controls, Complex::new(1.0, 0.0), (-1.0f64).powf(1.0 / (1 << (n - 1))) as f64, target);
    }

    fn ci_phase_root_n(&mut self, n: usize, control: usize, target: usize) {
        let controls = [control];
        if n == 0 {
            return;
        }
        self.mc_phase(&controls, Complex::new(1.0, 0.0), (-1.0f64).powf(-1.0 / (1 << (n - 1))) as f64, target);
    }

    fn anti_ci_phase_root_n(&mut self, n: usize, control: usize, target: usize) {
        let controls = [control];
        if n == 0 {
            return;
        }
        self.mac_phase(&controls, Complex::new(1.0, 0.0), (-1.0f64).powf(-1.0 / (1 << (n - 1))) as f64, target);
    }

    fn phase_flip(&mut self) {
        self.phase(Complex::new(-1.0, 0.0), Complex::new(-1.0, 0.0), 0);
    }

    fn set_reg(&mut self, start: usize, length: usize, value: usize) {}

    fn m_reg(&mut self, start: usize, length: usize) -> usize {
        force_m_reg(start, length, ZERO_BCI, false, true);
    }

    fn m_all(&mut self) -> usize {
        m_reg(0, self.qubit_count)
    }

    fn force_m_reg(
        &mut self,
        start: usize,
        length: usize,
        result: usize,
        do_force: bool,
        do_apply: bool,
    ) -> usize {
        let mut res: u64 = 0;
        for bit in 0..length {
            let power = 1 << bit;
            if force_m(start + bit, (power & result) != 0, do_force, do_apply) {
                res |= power;
            }
        }
        res
    }

    fn m(&mut self, bits: &[usize]) -> usize {
        self.force_m(bits, Vec::<bool>::new())
    }

    fn force_m(&mut self, bits: &[usize], values: &[bool], do_apply: bool) -> usize {
        if !values.is_empty() && bits.len() != values.len() {
            panic!("QInterface::ForceM() boolean values vector length does not match bit vector length!");
        }
        if !values.is_empty() {
            let mut result = 0;
            for (bit, &value) in bits.iter().zip(values.iter()) {
                if self.force_m(*bit, value, true, do_apply) {
                    result |= pow2(*bit);
                }
            }
            result
        } else if do_apply {
            let mut result = 0;
            for &bit in bits {
                if self.m(bit) {
                    result |= pow2(bit);
                }
            }
            result
        } else {
            let q_powers: Vec<u64> = bits.iter().map(|&bit| self.pow2(bit)).collect();
            *self.multi_shot_measure_mask(&q_powers, 1).keys().next().unwrap()
        }
    }

    fn reverse(&mut self, first: usize, last: usize) {
        let mut first = first;
        let mut last = last;
        while last > 0 && first < last - 1 {
            last -= 1;
            self.swap(first, last);
            first += 1;
        }
    }

    fn prob(&self, qubit_index: usize) -> f64 {
        unimplemented!()
    }

    fn c_prob(&mut self, control: usize, target: usize) -> f64 {
        self.anti_cnot(control, target);
        let prob = self.prob(target);
        self.anti_cnot(control, target);
        prob
    }

    fn ac_prob(&mut self, control: usize, target: usize) -> f64 {
        self.cnot(control, target);
        let prob = self.prob(target);
        self.cnot(control, target);
        prob
    }

    fn prob_all(&self, full_register: usize) -> f64 {
        Self::clamp_prob(self.get_amplitude(full_register).norm())
    }

    fn prob_reg(&self, start: usize, length: usize, permutation: usize) -> f64 {
        unimplemented!()
    }

    fn prob_mask(&self, mask: usize, permutation: usize) -> f64 {
        unimplemented!()
    }

    fn prob_mask_all(&self, mask: usize, probs_array: &mut [f64]) {
        unimplemented!()
    }

    fn prob_bits_all(&self, bits: &[usize], probs_array: &mut [f64]) {
        unimplemented!()
    }

    fn expectation_bits_all(&self, bits: &[usize], offset: usize) -> f64 {
        unimplemented!()
    }

    fn expectation_bits_factorized(
        &self,
        bits: &[usize],
        perms: &[usize],
        offset: usize,
    ) -> f64 {
        unimplemented!()
    }

    fn expectation_floats_factorized(
        &self,
        bits: &[usize],
        weights: &[f64],
    ) -> f64 {
        unimplemented!()
    }

    fn prob_rdm(&self, qubit_index: usize) -> f64 {
        self.prob(qubit_index)
    }

    fn prob_all_rdm(&self, round_rz: bool, full_register: usize) -> f64 {
        self.prob_all(full_register)
    }

    fn prob_mask_rdm(&self, round_rz: bool, mask: usize, permutation: usize) -> f64 {
        self.prob_mask(mask, permutation)
    }

    fn expectation_bits_all_rdm(
        &self,
        round_rz: bool,
        bits: &[usize],
        offset: usize,
    ) -> f64 {
        self.expectation_bits_all(bits, offset)
    }

    fn multi_shot_measure_mask(
        &mut self,
        q_powers: &[usize],
        shots: u32,
    ) -> HashMap<usize, i32> {
        unimplemented!()
    }

    fn set_bit(&mut self, qubit: usize, value: bool) {
        if value != self.m(qubit) {
            self.x(qubit);
        }
    }

    fn approx_compare(&self, to_compare: QInterfacePtr, error_tol: f64) -> bool {
        self.sum_sqr_diff(to_compare) <= error_tol
    }
    
    fn get_max_size(&self) -> usize {
        2usize.pow(std::mem::size_of::<usize>() as u32 * 8)
    }

    fn first_nonzero_phase(&self) -> f64 {
        let mut perm = 0usize;
        loop {
            let amp = self.get_amplitude(perm);
            perm += 1;
            if amp.norm() > 1e-6 {
                return amp.arg();
            }
        }
    }

    pub fn set_permutation(&mut self, perm: u64, _ignored: Complex) {
        let measured = self.m_all();
        for i in 0..self.qubit_count {
            if (perm ^ measured) >> i & 1 != 0 {
                self.x(i);
            }
        }
    }

    pub fn qft(&mut self, start: u32, length: u32, try_separate: bool) {
        if length == 0 {
            return;
        }
        let end = start + length - 1;
        for i in 0..length {
            let h_bit = end - i;
            for j in 0..i {
                let c = h_bit;
                let t = h_bit + 1 + j;
                self.c_phase_root_n(j + 2, c, t);
                if try_separate {
                    self.try_separate(c, t);
                }
            }
            self.h(h_bit);
        }
    }

    pub fn iqft(&mut self, start: u32, length: u32, try_separate: bool) {
        if length == 0 {
            return;
        }
        for i in 0..length {
            for j in 0..i {
                let c = start + i - (j + 1);
                let t = start + i;
                self.ci_phase_root_n(j + 2, c, t);
                if try_separate {
                    self.try_separate(c, t);
                }
            }
            self.h(start + i);
        }
    }

    pub fn qftr(&mut self, qubits: &[u32], try_separate: bool) {
        if qubits.is_empty() {
            return;
        }
        let end = qubits.len() - 1;
        for (i, &qubit) in qubits.iter().enumerate() {
            self.h(qubits[end - i]);
            for j in 0..(qubits.len() - 1 - i) {
                self.c_phase_root_n(j + 2, qubits[(end - i) - (j + 1)], qubits[end - i]);
            }
            if try_separate {
                self.try_separate(qubits[end - i]);
            }
        }
    }

    pub fn iqftr(&mut self, qubits: &[u32], try_separate: bool) {
        if qubits.is_empty() {
            return;
        }
        for (i, &qubit) in qubits.iter().enumerate() {
            for j in 0..i {
                self.ci_phase_root_n(j + 2, qubits[i - (j + 1)], qubits[i]);
            }
            self.h(qubits[i]);
            if try_separate {
                self.try_separate(qubits[i]);
            }
        }
    }

    pub fn set_reg(&mut self, start: u32, length: u32, value: u64) {
        if length == 1 {
            self.set_bit(start, value & 1 != 0);
            return;
        }
        if start == 0 && length == self.qubit_count {
            self.set_permutation(value, Complex::new(0.0, 0.0));
            return;
        }
        let reg_val = self.m_reg(start, length);
        for i in 0..length {
            if (reg_val >> i & 1 == 0) != (value >> i & 1 == 0) {
                self.x(start + i);
            }
        }
    }

    pub fn force_m_reg(
        &mut self,
        start: u32,
        length: u32,
        result: u64,
        do_force: bool,
        do_apply: bool,
    ) -> u64 {
        let mut res = 0;
        for bit in 0..length {
            let power = 1 << bit;
            if self.force_m(start + bit, result & power != 0, do_force, do_apply) {
                res |= power;
            }
        }
        res
    }

    pub fn force_m(&mut self, bits: &[u32], values: &[bool], do_apply: bool) -> u64 {
        if values.len() != bits.len() {
            panic!("QInterface::ForceM() boolean values vector length does not match bit vector length!");
        }
        if !values.is_empty() {
            let mut result = 0;
            for (bit, &value) in bits.iter().zip(values) {
                if self.force_m(*bit, value, true, do_apply) {
                    result |= 1 << bit;
                }
            }
            result
        } else if do_apply {
            let mut result = 0;
            for &bit in bits {
                if self.m(bit) {
                    result |= 1 << bit;
                }
            }
            result
        } else {
            let q_powers: Vec<_> = bits.iter().map(|&bit| 1 << bit).collect();
            self.multi_shot_measure_mask(&q_powers, 1).keys().next().copied().unwrap_or(0)
        }
    }

    pub fn prob_reg(&mut self, start: u32, length: u32, permutation: u64) -> f64 {
        let start_mask = (1 << start) - 1;
        let max_lcv = self.max_q_power >> length;
        let p = permutation;
        let mut prob = 0.0;
        for lcv in 0..max_lcv {
            let mut i = lcv & start_mask;
            i |= ((lcv ^ i) | p) << length;
            prob += self.prob_all(i);
        }
        prob.clamp(0.0, 1.0)
    }

    pub fn prob_mask(&mut self, mask: u64, permutation: u64) -> f64 {
        if self.max_q_power - 1 == mask {
            return self.prob_all(permutation);
        }
        let mut prob = 0.0;
        for lcv in 0..self.max_q_power {
            if lcv & mask == permutation {
                prob += self.prob_all(lcv);
            }
        }
        prob.clamp(0.0, 1.0)
    }

    pub fn rol(&mut self, shift: u32, start: u32, length: u32) {
        if length < 2 {
            return;
        }
        let shift = shift % length;
        if shift == 0 {
            return;
        }
        let end = start + length;
        self.reverse(start, end);
        self.reverse(start, start + shift);
        self.reverse(start + shift, end);
    }

    pub fn ror(&mut self, shift: u32, start: u32, length: u32) {
        if length < 2 {
            return;
        }
        let shift = shift % length;
        if shift == 0 {
            return;
        }
        let end = start + length;
        self.reverse(start + shift, end);
        self.reverse(start, start + shift);
        self.reverse(start, end);
    }

    pub fn asl(&mut self, shift: u32, start: u32, length: u32) {
        if length == 0 || shift == 0 {
            return;
        }
        if shift >= length {
            self.set_reg(start, length, 0);
        } else {
            let end = start + length;
            self.swap(end - 1, end - 2);
            self.rol(shift, start, length);
            self.set_reg(start, shift, 0);
            self.swap(end - 1, end - 2);
        }
    }

    pub fn asr(&mut self, shift: u32, start: u32, length: u32) {
        if length == 0 || shift == 0 {
            return;
        }
        if shift >= length {
            self.set_reg(start, length, 0);
        } else {
            let end = start + length;
            self.swap(end - 1, end - 2);
            self.ror(shift, start, length);
            self.set_reg(end - shift - 1, shift, 0);
            self.swap(end - 1, end - 2);
        }
    }

    pub fn lsl(&mut self, shift: u32, start: u32, length: u32) {
        if length == 0 || shift == 0 {
            return;
        }
        if shift >= length {
            self.set_reg(start, length, 0);
        } else {
            self.rol(shift, start, length);
            self.set_reg(start, shift, 0);
        }
    }

    pub fn lsr(&mut self, shift: u32, start: u32, length: u32) {
        if length == 0 || shift == 0 {
            return;
        }
        if shift >= length {
            self.set_reg(start, length, 0);
        } else {
            self.set_reg(start, shift, 0);
            self.ror(shift, start, length);
        }
    }

    pub fn compose(&mut self, to_copy: QInterfacePtr, start: u32) -> u32 {
        if start == self.qubit_count {
            return self.compose(to_copy);
        }
        let orig_size = self.qubit_count;
        self.rol(orig_size - start, 0, self.qubit_count);
        let result = self.compose(to_copy);
        self.ror(orig_size - start, 0, self.qubit_count);
        result
    }

    pub fn compose(&mut self, to_copy: Vec<QInterfacePtr>) -> HashMap<QInterfacePtr, u32> {
        let mut ret = HashMap::new();
        for q in to_copy {
            ret.insert(q, self.compose(q));
        }
        ret
    }

    pub fn prob_mask_all(&mut self, mask: u64, probs_array: &mut [f64]) {
        let mut v = mask;
        let mut bit_powers = Vec::new();
        while v != 0 {
            let old_v = v;
            v &= v - 1;
            bit_powers.push((v ^ old_v) & old_v);
        }
        probs_array.iter_mut().for_each(|prob| *prob = 0.0);
        for lcv in 0..self.max_q_power {
            let mut i = 0;
            for (p, &bit_power) in bit_powers.iter().enumerate() {
                if lcv & bit_power != 0 {
                    i |= 1 << p;
                }
            }
            probs_array[i] += self.prob_all(lcv);
        }
    }

    pub fn prob_bits_all(&mut self, bits: &[u32], probs_array: &mut [f64]) {
        if bits.len() == self.qubit_count && bits.iter().enumerate().all(|(i, &bit)| bit == i as u32) {
            self.get_probs(probs_array);
            return;
        }
        probs_array.iter_mut().for_each(|prob| *prob = 0.0);
        let bit_powers: Vec<_> = bits.iter().map(|&bit| 1 << bit).collect();
        for lcv in 0..self.max_q_power {
            let mut ret_index = 0;
            for (p, &bit_power) in bit_powers.iter().enumerate() {
                if lcv & bit_power != 0 {
                    ret_index |= 1 << p;
                }
            }
            probs_array[ret_index] += self.prob_all(lcv);
        }
    }

    pub fn expectation_bits_factorized(
        &mut self,
        bits: &[u32],
        perms: &[u64],
        offset: u64,
    ) -> f64 {
        if perms.len() < (bits.len() << 1) {
            panic!("QInterface::ExpectationBitsFactorized() must supply at least twice as many 'perms' as bits!");
        }
        self.finish();
        let temp_do_norm = self.do_normalize;
        self.do_normalize = false;
        let mut unit_copy = self.clone();
        self.do_normalize = temp_do_norm;
        unit_copy.decompose(bits[0]);
        unit_copy.compose(bits[0]);
        let did_separate = unit_copy.approx_compare(self, error_tol);
        if did_separate {
            self.dispose(bits[0], self.get_qubit_count());
        }
        expectation
    }

    pub fn expectation_floats_factorized(&mut self, bits: &[u32], weights: &[f64]) -> f64 {
        if weights.len() < (bits.len() << 1) {
            panic!("QInterface::ExpectationFloatsFactorized() must supply at least twice as many weights as bits!");
        }
        self.finish();
        let temp_do_norm = self.do_normalize;
        self.do_normalize = false;
        let mut unit_copy = self.clone();
        self.do_normalize = temp_do_norm;
        unit_copy.decompose(bits[0]);
        unit_copy.compose(bits[0]);
        let did_separate = unit_copy.approx_compare(self, error_tol);
        if did_separate {
            self.dispose(bits[0], self.get_qubit_count());
        }
        expectation
    }

    pub fn multi_shot_measure_mask(
        &mut self,
        q_powers: &[u64],
        shots: u32,
    ) -> HashMap<u64, u32> {
        if shots == 0 {
            return HashMap::new();
        }
        let mut results = HashMap::new();
        let results_mutex = Arc::new(Mutex::new(results));
        (0..shots).into_par_iter().for_each(|shot| {
            let sample = self.sample_clone(q_powers);
            let mut results = results_mutex.lock().unwrap();
            *results.entry(sample).or_insert(0) += 1;
        });
        results_mutex.into_inner().unwrap()
    }

    pub fn multi_shot_measure_mask(
        &mut self,
        q_powers: &[u64],
        shots: u32,
        shots_array: &mut [u64],
    ) {
        if shots == 0 {
            return;
        }
        (0..shots).into_par_iter().for_each(|shot| {
            shots_array[shot as usize] = self.sample_clone(q_powers) as u64;
        });
    }

    pub fn try_decompose(&mut self, start: u32, dest: QInterfacePtr, error_tol: f64) -> bool {
        self.finish();
        let temp_do_norm = self.do_normalize;
        self.do_normalize = false;
        let unit_copy = self.clone();
        self.do_normalize = temp_do_norm;
        unit_copy.decompose(start, dest);
        unit_copy.compose(dest, start);
        let did_separate = unit_copy.approx_compare(self, error_tol);
        if did_separate {
            self.dispose(start, dest.get_qubit_count());
        }
        did_separate
    }
    
    pub fn gate(&mut self, start: usize, length: usize) {
        for bit in 0..length {
            self.gate(start + bit);
        }
    }

    pub fn gate(&mut self, qubit1: usize, qubit2: usize, length: usize) {
        for bit in 0..length {
            self.gate(qubit1 + bit, qubit2 + bit);
        }
    }

    pub fn gate(&mut self, qubit1: usize, qubit2: usize, qubit3: usize, length: usize) {
        for bit in 0..length {
            self.gate(qubit1 + bit, qubit2 + bit, qubit3 + bit);
        }
    }

    pub fn gate(&mut self, qInputStart: usize, classicalInput: usize, outputStart: usize, length: usize) {
        for i in 0..length {
            self.gate(qInputStart + i, classicalInput as bitCapIntOcl, outputStart + i);
        }
    }

    pub fn gate(&mut self, radians: f64, start: usize, length: usize) {
        for bit in 0..length {
            self.gate(radians, start + bit);
        }
    }

    pub fn gate(&mut self, numerator: i32, denominator: i32, start: usize, length: usize) {
        for bit in 0..length {
            self.gate(numerator, denominator, start + bit);
        }
    }

    pub fn gate(&mut self, control: usize, target: usize, length: usize) {
        for bit in 0..length {
            self.gate(control + bit, target + bit);
        }
    }

    pub fn gate(&mut self, control1: usize, control2: usize, target: usize, length: usize) {
        for bit in 0..length {
            self.gate(control1 + bit, control2 + bit, target + bit);
        }
    }

    pub fn gate(&mut self, radians: f64, control: usize, target: usize, length: usize) {
        for bit in 0..length {
            self.gate(radians, control + bit, target + bit);
        }
    }

    pub fn gate(&mut self, numerator: i32, denominator: i32, control: usize, target: usize, length: usize) {
        for bit in 0..length {
            self.gate(numerator, denominator, control + bit, target + bit);
        }
    }

    pub fn gate(&mut self, start: usize, length: usize) {
        for bit in 0..length {
            self.gate(start + bit);
        }
    }

    pub fn gate(&mut self, start: usize, length: usize, theta: f64, phi: f64, lambda: f64) {
        for bit in 0..length {
            self.gate(start + bit, theta, phi, lambda);
        }
    }

    pub fn gate(&mut self, start: usize, length: usize, phi: f64, lambda: f64) {
        for bit in 0..length {
            self.gate(start + bit, phi, lambda);
        }
    }

    pub fn phase_root_n(&mut self, n: usize, start: usize, length: usize) {
        for bit in 0..length {
            self.phase_root_n(n, start + bit);
        }
    }

    pub fn i_phase_root_n(&mut self, n: usize, start: usize, length: usize) {
        for bit in 0..length {
            self.i_phase_root_n(n, start + bit);
        }
    }

    pub fn c_phase_root_n(&mut self, n: usize, control: usize, target: usize, length: usize) {
        if n == 0 {
            return;
        }
        if n == 1 {
            self.cz(control, target, length);
            return;
        }
        for bit in 0..length {
            self.c_phase_root_n(n, control + bit, target + bit);
        }
    }

    pub fn ci_phase_root_n(&mut self, n: usize, control: usize, target: usize, length: usize) {
        if n == 0 {
            return;
        }
        if n == 1 {
            self.cz(control, target, length);
            return;
        }
        for bit in 0..length {
            self.ci_phase_root_n(n, control + bit, target + bit);
        }
    }

    pub fn u(&mut self, start: usize, length: usize, theta: f64, phi: f64, lambda: f64) {
        for bit in 0..length {
            self.u(start + bit, theta, phi, lambda);
        }
    }

    pub fn u2(&mut self, start: usize, length: usize, phi: f64, lambda: f64) {
        for bit in 0..length {
            self.u2(start + bit, phi, lambda);
        }
    }

    pub fn r_t_dyad(&mut self, numerator: i32, denom_power: i32, qubit: usize) {
        self.rt(dyad_angle(numerator, denom_power), qubit);
    }

    pub fn exp_dyad(&mut self, numerator: i32, denom_power: i32, qubit: usize) {
        self.exp(dyad_angle(numerator, denom_power), qubit);
    }

    pub fn exp_x_dyad(&mut self, numerator: i32, denom_power: i32, qubit: usize) {
        self.exp_x(dyad_angle(numerator, denom_power), qubit);
    }

    pub fn exp_y_dyad(&mut self, numerator: i32, denom_power: i32, qubit: usize) {
        self.exp_y(dyad_angle(numerator, denom_power), qubit);
    }

    pub fn exp_z_dyad(&mut self, numerator: i32, denom_power: i32, qubit: usize) {
        self.exp_z(dyad_angle(numerator, denom_power), qubit);
    }

    pub fn r_x_dyad(&mut self, numerator: i32, denom_power: i32, qubit: usize) {
        self.r_x(dyad_angle(numerator, denom_power), qubit);
    }

    pub fn r_y_dyad(&mut self, numerator: i32, denom_power: i32, qubit: usize) {
        self.r_y(dyad_angle(numerator, denom_power), qubit);
    }

    pub fn r_z_dyad(&mut self, numerator: i32, denom_power: i32, qubit: usize) {
        self.r_z(dyad_angle(numerator, denom_power), qubit);
    }

    pub fn cr_t_dyad(&mut self, numerator: i32, denom_power: i32, control: usize, target: usize) {
        self.crt(dyad_angle(numerator, denom_power), control, target);
    }

    pub fn cr_x_dyad(&mut self, numerator: i32, denom_power: i32, control: usize, target: usize) {
        self.crx(dyad_angle(numerator, denom_power), control, target);
    }

    pub fn cr_y_dyad(&mut self, numerator: i32, denom_power: i32, control: usize, target: usize) {
        self.cry(dyad_angle(numerator, denom_power), control, target);
    }

    pub fn cr_z_dyad(&mut self, numerator: i32, denom_power: i32, control: usize, target: usize) {
        self.crz(dyad_angle(numerator, denom_power), control, target);
    }
    
    fn ucmtrx(&self, controls: &[usize], mtrx: &[Complex<f64>], target: usize, control_perm: u64) {
        let set_count = controls.iter().filter(|&c| (control_perm >> c) & 1 == 1).count();
        if (set_count << 1) > controls.len() {
            for &c in controls {
                if ((control_perm >> c) & 1) == 0 {
                    self.x(c);
                }
            }
            self.mcmtrx(controls, mtrx, target);
            for &c in controls {
                if ((control_perm >> c) & 1) == 0 {
                    self.x(c);
                }
            }
            return;
        }
        for &c in controls {
            if ((control_perm >> c) & 1) == 1 {
                self.x(c);
            }
        }
        self.macmtrx(controls, mtrx, target);
        for &c in controls {
            if ((control_perm >> c) & 1) == 1 {
                self.x(c);
            }
        }
    }

    fn uniformly_controlled_single_bit(
        &self,
        controls: &[usize],
        qubit_index: usize,
        mtrxs: &[Complex<f64>],
        mtrx_skip_powers: &[u64],
        mtrx_skip_value_mask: u64,
    ) {
        for &control in controls {
            self.x(control);
        }
        let max_i = 2usize.pow(controls.len() as u32) - 1;
        for lcv in 0..max_i {
            let index = self.push_apart_bits(lcv, mtrx_skip_powers) | mtrx_skip_value_mask;
            self.mcmtrx(controls, &mtrxs[(index * 4) as usize..], qubit_index);
            let lcv_diff = lcv ^ (lcv + 1);
            for bit_pos in 0..controls.len() {
                if ((lcv_diff >> bit_pos) & 1) == 1 {
                    self.x(controls[bit_pos]);
                }
            }
        }
        let index = self.push_apart_bits(max_i, mtrx_skip_powers) | mtrx_skip_value_mask;
        self.mcmtrx(controls, &mtrxs[(index * 4) as usize..], qubit_index);
    }

    fn zero_phase_flip(&self, start: usize, length: usize) {
        if length == 0 {
            return;
        }
        if length == 1 {
            self.phase(-1.0, 1.0, start);
            return;
        }
        let controls: Vec<usize> = (start..start + length - 1).collect();
        self.mac_phase(&controls, -1.0, 1.0, start + controls.len());
    }

    fn x_mask(&self, mask: u64) {
        let mut v = mask;
        while mask != 0 {
            v = v & (v - 1);
            self.x((mask ^ v).log2() as usize);
            mask = v;
        }
    }

    fn y_mask(&self, mask: u64) {
        let bit = mask.log2() as usize;
        if 2usize.pow(bit as u32) == mask {
            self.y(bit);
            return;
        }
        self.z_mask(mask);
        self.x_mask(mask);
        if self.rand_global_phase() {
            return;
        }
        let mut parity = 0;
        let mut v = mask;
        while v != 0 {
            v = v & (v - 1);
            parity = (parity + 1) & 3;
        }
        match parity {
            1 => self.phase(0.0, 1.0, 0),
            2 => self.phase_flip(),
            3 => self.phase(0.0, -1.0, 0),
            _ => (),
        }
    }

    fn z_mask(&self, mask: u64) {
        let mut v = mask;
        while mask != 0 {
            v = v & (v - 1);
            self.z((mask ^ v).log2() as usize);
            mask = v;
        }
    }

    fn swap(&self, q1: usize, q2: usize) {
        if q1 == q2 {
            return;
        }
        self.cnot(q1, q2);
        self.cnot(q2, q1);
        self.cnot(q1, q2);
    }

    fn iswap(&self, q1: usize, q2: usize) {
        if q1 == q2 {
            return;
        }
        self.swap(q1, q2);
        self.cz(q1, q2);
        self.s(q1);
        self.s(q2);
    }

    fn iiswap(&self, q1: usize, q2: usize) {
        if q1 == q2 {
            return;
        }
        self.is(q2);
        self.is(q1);
        self.cz(q1, q2);
        self.swap(q1, q2);
    }

    fn sqrt_swap(&self, q1: usize, q2: usize) {
        if q1 == q2 {
            return;
        }
        self.cnot(q1, q2);
        self.h(q1);
        self.it(q2);
        self.t(q1);
        self.h(q2);
        self.h(q1);
        self.cnot(q1, q2);
        self.h(q1);
        self.h(q2);
        self.it(q1);
        self.h(q1);
        self.cnot(q1, q2);
        self.is(q1);
        self.s(q2);
    }

    fn isqrt_swap(&self, q1: usize, q2: usize) {
        if q1 == q2 {
            return;
        }
        self.is(q2);
        self.s(q1);
        self.cnot(q1, q2);
        self.h(q1);
        self.t(q1);
        self.h(q2);
        self.h(q1);
        self.cnot(q1, q2);
        self.h(q1);
        self.h(q2);
        self.it(q1);
        self.t(q2);
        self.h(q1);
        self.cnot(q1, q2);
    }

    fn cswap(&self, controls: &[usize], q1: usize, q2: usize) {
        if controls.is_empty() {
            self.swap(q1, q2);
            return;
        }
        if q1 == q2 {
            return;
        }
        let mut l_controls = controls.to_vec();
        l_controls.push(q1);
        self.mc_invert(&l_controls, 1.0, 1.0, q2);
        l_controls[controls.len()] = q2;
        self.mc_invert(&l_controls, 1.0, 1.0, q1);
        l_controls[controls.len()] = q1;
        self.mc_invert(&l_controls, 1.0, 1.0, q2);
    }

    fn anti_cswap(&self, controls: &[usize], q1: usize, q2: usize) {
        let mut m = 0;
        for &control in controls {
            m |= 2usize.pow(control as u32);
        }
        self.x_mask(m);
        self.cswap(controls, q1, q2);
        self.x_mask(m);
    }

    fn csqrt_swap(&self, controls: &[usize], q1: usize, q2: usize) {
        if controls.is_empty() {
            self.sqrt_swap(q1, q2);
            return;
        }
        if q1 == q2 {
            return;
        }
        let mut l_controls = controls.to_vec();
        l_controls.push(q1);
        self.mc_invert(&l_controls, 1.0, 1.0, q2);
        let had = [C_SQRT1_2, C_SQRT1_2, C_SQRT1_2, C_SQRT1_2_NEG];
        self.mcmtrx(controls, &had, q1);
        let it = [1.0, 0.0, 0.0, C_SQRT_N_I];
        self.mcmtrx(controls, &it, q2);
        let t = [1.0, 0.0, 0.0, C_SQRT_I];
        self.mcmtrx(controls, &t, q1);
        self.mcmtrx(controls, &had, q2);
        self.mcmtrx(controls, &had, q1);
        self.mc_invert(&l_controls, 1.0, 1.0, q2);
        self.mcmtrx(controls, &had, q1);
        self.mcmtrx(controls, &had, q2);
        self.mcmtrx(controls, &it, q1);
        self.mcmtrx(controls, &had, q1);
        self.mc_invert(&l_controls, 1.0, 1.0, q2);
        let is = [1.0, 0.0, 0.0, -1.0];
        self.mcmtrx(controls, &is, q1);
        let s = [1.0, 0.0, 0.0, 1.0];
        self.mcmtrx(controls, &s, q2);
    }

    fn cisqrt_swap(&self, controls: &[usize], q1: usize, q2: usize) {
        if q1 == q2 {
            return;
        }
        let mut l_controls = controls.to_vec();
        l_controls.push(q1);
        let is = [1.0, 0.0, 0.0, -1.0];
        self.mcmtrx(controls, &is, q2);
        let s = [1.0, 0.0, 0.0, 1.0];
        self.mcmtrx(controls, &s, q1);
        self.mc_invert(&l_controls, 1.0, 1.0, q2);
        let had = [C_SQRT1_2, C_SQRT1_2, C_SQRT1_2, C_SQRT1_2_NEG];
        self.mcmtrx(controls, &had, q1);
        let t = [1.0, 0.0, 0.0, C_SQRT_I];
        self.mcmtrx(controls, &t, q1);
        self.mcmtrx(controls, &had, q2);
        self.mcmtrx(controls, &had, q1);
        self.mc_invert(&l_controls, 1.0, 1.0, q2);
        self.mcmtrx(controls, &had, q1);
        self.mcmtrx(controls, &had, q2);
        let it = [1.0, 0.0, 0.0, C_SQRT_N_I];
        self.mcmtrx(controls, &it, q1);
        self.mcmtrx(controls, &t, q2);
        self.mcmtrx(controls, &had, q1);
        self.mc_invert(&l_controls, 1.0, 1.0, q2);
    }

    fn anti_cisqrt_swap(&self, controls: &[usize], q1: usize, q2: usize) {
        let mut m = 0;
        for &control in controls {
            m |= 2usize.pow(control as u32);
        }
        self.x_mask(m);
        self.cisqrt_swap(controls, q1, q2);
        self.x_mask(m);
    }

    fn phase_parity(&self, radians: f64, mask: u64) {
        if mask == 0 {
            return;
        }
        let qubits: Vec<usize> = (0..64).filter(|&i| (mask >> i) & 1 == 1).collect();
        let end = qubits.len() - 1;
        for i in 0..end {
            self.cnot(qubits[i], qubits[i + 1]);
        }
        let cosine = (radians / 2.0).cos();
        let sine = (radians / 2.0).sin();
        self.phase(cosine - I_CMPLX_NEG * sine, cosine + I_CMPLX_NEG * sine, qubits[end]);
        for i in (0..end).rev() {
            self.cnot(qubits[i], qubits[i + 1]);
        }
    }

    fn time_evolve(&self, h: Hamiltonian, time_diff: f64) {
        if time_diff.abs() <= f64::EPSILON {
            return;
        }
        for op in h {
            let op_mtrx = op.matrix;
            let max_j = if op.uniform {
                4 * 2usize.pow(op.controls.len() as u32)
            } else {
                4
            };
            let mut mtrx = vec![Complex::new(0.0, 0.0); max_j];
            for j in 0..max_j {
                mtrx[j] = op_mtrx[j] * (-time_diff);
            }
            if !op.toggles.is_empty() {
                for (j, &toggle) in op.controls.iter().enumerate() {
                    if op.toggles[j] {
                        self.x(toggle);
                    }
                }
            }
            if op.uniform {
                let mut exp_mtrx = vec![Complex::new(0.0, 0.0); max_j];
                for j in 0..2usize.pow(op.controls.len() as u32) {
                    self.exp2x2(&mtrx[j * 4..], &mut exp_mtrx[j * 4..]);
                }
                self.uniformly_controlled_single_bit(&op.controls, op.target_bit, &exp_mtrx);
            } else {
                let times_i = [
                    I_CMPLX_NEG * mtrx[0],
                    I_CMPLX_NEG * mtrx[1],
                    I_CMPLX_NEG * mtrx[2],
                    I_CMPLX_NEG * mtrx[3],
                ];
                let mut to_apply = [Complex::new(0.0, 0.0); 4];
                self.exp2x2(&times_i, &mut to_apply);
                if op.controls.is_empty() {
                    self.mtrx(&to_apply, op.target_bit);
                } else if op.anti {
                    self.macmtrx(&op.controls, &to_apply, op.target_bit);
                } else {
                    self.mcmtrx(&op.controls, &to_apply, op.target_bit);
                }
            }
            if !op.toggles.is_empty() {
                for (j, &toggle) in op.controls.iter().enumerate() {
                    if op.toggles[j] {
                        self.x(toggle);
                    }
                }
            }
        }
    }

    fn depolarizing_channel_weak_1_qb(&self, qubit: usize, lambda: f64) {
        if lambda <= 0.0 {
            return;
        }
        self.h(qubit);
        let ancilla = self.allocate(1);
        self.cry(2.0 * (lambda.powf(1.0 / 4.0)).asin(), qubit, ancilla);
        self.m(ancilla);
        self.dispose(ancilla, 1);
        self.h(qubit);
        self.try_separate(qubit);
    }

    fn depolarizing_channel_strong_1_qb(&self, qubit: usize, lambda: f64) -> usize {
        self.h(qubit);
        let ancilla = self.allocate(1);
        self.cry(2.0 * (lambda.powf(1.0 / 4.0)).asin(), qubit, ancilla);
        self.h(qubit);
        ancilla
    }
    
    pub fn and(&mut self, input_bit1: i32, input_bit2: i32, output_bit: i32) -> Result<(), Box<dyn Error>> {
        if input_bit1 == input_bit2 && input_bit2 == output_bit {
            return Ok(());
        }
        if input_bit1 == output_bit || input_bit2 == output_bit {
            return Err("Invalid AND arguments.".into());
        }
        if input_bit1 == input_bit2 {
            self.cnot(input_bit1, output_bit)?;
        } else {
            self.ccnot(input_bit1, input_bit2, output_bit)?;
        }
        Ok(())
    }

    pub fn or(&mut self, input_bit1: i32, input_bit2: i32, output_bit: i32) -> Result<(), Box<dyn Error>> {
        if input_bit1 == input_bit2 && input_bit2 == output_bit {
            return Ok(());
        }
        if input_bit1 == output_bit || input_bit2 == output_bit {
            return Err("Invalid OR arguments.".into());
        }
        self.x(output_bit)?;
        if input_bit1 == input_bit2 {
            self.anti_cnot(input_bit1, output_bit)?;
        } else {
            self.anti_ccnot(input_bit1, input_bit2, output_bit)?;
        }
        Ok(())
    }

    pub fn xor(&mut self, input_bit1: i32, input_bit2: i32, output_bit: i32) -> Result<(), Box<dyn Error>> {
        if input_bit1 == input_bit2 && input_bit2 == output_bit {
            self.set_bit(output_bit, false)?;
            return Ok(());
        }
        if input_bit1 == output_bit {
            self.cnot(input_bit2, output_bit)?;
        } else if input_bit2 == output_bit {
            self.cnot(input_bit1, output_bit)?;
        } else {
            self.cnot(input_bit1, output_bit)?;
            self.cnot(input_bit2, output_bit)?;
        }
        Ok(())
    }

    pub fn nand(&mut self, input_bit1: i32, input_bit2: i32, output_bit: i32) -> Result<(), Box<dyn Error>> {
        self.and(input_bit1, input_bit2, output_bit)?;
        self.x(output_bit)?;
        Ok(())
    }

    pub fn nor(&mut self, input_bit1: i32, input_bit2: i32, output_bit: i32) -> Result<(), Box<dyn Error>> {
        self.or(input_bit1, input_bit2, output_bit)?;
        self.x(output_bit)?;
        Ok(())
    }

    pub fn xnor(&mut self, input_bit1: i32, input_bit2: i32, output_bit: i32) -> Result<(), Box<dyn Error>> {
        self.xor(input_bit1, input_bit2, output_bit)?;
        self.x(output_bit)?;
        Ok(())
    }

    pub fn cland(&mut self, input_qbit: i32, input_classical_bit: bool, output_bit: i32) -> Result<(), Box<dyn Error>> {
        if input_classical_bit && input_qbit != output_bit {
            self.cnot(input_qbit, output_bit)?;
        }
        Ok(())
    }

    pub fn clor(&mut self, input_qbit: i32, input_classical_bit: bool, output_bit: i32) -> Result<(), Box<dyn Error>> {
        if input_classical_bit {
            self.x(output_bit)?;
        } else if input_qbit != output_bit {
            self.cnot(input_qbit, output_bit)?;
        }
        Ok(())
    }

    pub fn clxor(&mut self, input_qbit: i32, input_classical_bit: bool, output_bit: i32) -> Result<(), Box<dyn Error>> {
        if input_qbit != output_bit {
            if input_classical_bit {
                self.x(output_bit)?;
            }
            self.cnot(input_qbit, output_bit)?;
        } else if input_classical_bit {
            self.x(output_bit)?;
        }
        Ok(())
    }

    pub fn clnand(&mut self, input_qbit: i32, input_classical_bit: bool, output_bit: i32) -> Result<(), Box<dyn Error>> {
        self.cland(input_qbit, input_classical_bit, output_bit)?;
        self.x(output_bit)?;
        Ok(())
    }

    pub fn clnor(&mut self, input_qbit: i32, input_classical_bit: bool, output_bit: i32) -> Result<(), Box<dyn Error>> {
        self.clor(input_qbit, input_classical_bit, output_bit)?;
        self.x(output_bit)?;
        Ok(())
    }

    pub fn clxnor(&mut self, input_qbit: i32, input_classical_bit: bool, output_bit: i32) -> Result<(), Box<dyn Error>> {
        self.clxor(input_qbit, input_classical_bit, output_bit)?;
        self.x(output_bit)?;
        Ok(())
    }

    pub fn inc(&mut self, to_add: u64, start: usize, length: usize) {
        if length == 0 {
            return;
        }
        if length == 1 {
            if to_add & 1 != 0 {
                self.x(start);
            }
            return;
        }
        let mut bits: Vec<usize> = Vec::with_capacity(length);
        for i in 0..length {
            bits.push(start + i);
        }
        let length_min_1 = length - 1;
        for i in 0..length {
            if (to_add >> i) & 1 == 0 {
                continue;
            }
            self.x(start + i);
            for j in 0..(length_min_1 - i) {
                let target = start + ((i + j + 1) % length);
                self.mac_invert(&bits[i..(i + j + 2)], 1.0, 1.0, target);
            }
        }
    }

    pub fn dec(&mut self, to_sub: u64, start: usize, length: usize) {
        let inv_to_sub = 2u64.pow(length as u32) - to_sub;
        self.inc(inv_to_sub, start, length);
    }

    pub fn incdecc(&mut self, to_add: u64, start: usize, length: usize, carry_index: usize) {
        if length == 0 {
            return;
        }
        let mut bits: Vec<usize> = Vec::with_capacity(length + 1);
        for i in 0..length {
            bits.push(start + i);
        }
        bits.push(carry_index);
        for i in 0..length {
            if (to_add >> i) & 1 == 0 {
                continue;
            }
            self.x(start + i);
            for j in 0..(length - i) {
                let target = start + (((i + j + 1) == length) as usize) * carry_index
                    + (((i + j + 1) != length) as usize) * ((i + j + 1) % length);
                self.mac_invert(&bits[i..(i + j + 2)], 1.0, 1.0, target);
            }
        }
    }

    pub fn incc(&mut self, to_add: u64, start: usize, length: usize, carry_index: usize) {
        let has_carry = self.m(carry_index);
        if has_carry {
            self.x(carry_index);
            to_add += 1;
        }
        self.incdecc(to_add, start, length, carry_index);
    }
    
    pub fn u(&mut self, target: i32, theta: f64, phi: f64, lambda: f64) {
        let cos0 = Complex::new((theta / 2.0).cos(), 0.0);
        let sin0 = Complex::new((theta / 2.0).sin(), 0.0);
        let u_gate = [
            Complex::new(cos0.re, 0.0),
            sin0 * Complex::new((-lambda).cos(), (-lambda).sin()),
            sin0 * Complex::new(phi.cos(), phi.sin()),
            cos0 * Complex::new((phi + lambda).cos(), (phi + lambda).sin()),
        ];
        self.mtrx(&u_gate, target);
    }

    pub fn cu(&mut self, controls: &Vec<i32>, target: i32, theta: f64, phi: f64, lambda: f64) {
        let cos0 = Complex::new((theta / 2.0).cos(), 0.0);
        let sin0 = Complex::new((theta / 2.0).sin(), 0.0);
        let u_gate = [
            Complex::new(cos0.re, 0.0),
            sin0 * Complex::new((-lambda).cos(), (-lambda).sin()),
            sin0 * Complex::new(phi.cos(), phi.sin()),
            cos0 * Complex::new((phi + lambda).cos(), (phi + lambda).sin()),
        ];
        self.mcmtrx(controls, &u_gate, target);
    }

    pub fn anti_cu(&mut self, controls: &Vec<i32>, target: i32, theta: f64, phi: f64, lambda: f64) {
        let cos0 = Complex::new((theta / 2.0).cos(), 0.0);
        let sin0 = Complex::new((theta / 2.0).sin(), 0.0);
        let u_gate = [
            Complex::new(cos0.re, 0.0),
            sin0 * Complex::new((-lambda).cos(), (-lambda).sin()),
            sin0 * Complex::new(phi.cos(), phi.sin()),
            cos0 * Complex::new((phi + lambda).cos(), (phi + lambda).sin()),
        ];
        self.macmtrx(controls, &u_gate, target);
    }

    pub fn ai(&mut self, target: i32, azimuth: f64, inclination: f64) {
        let cosine_a = azimuth.cos();
        let sine_a = azimuth.sin();
        let cosine_i = (inclination / 2.0).cos();
        let sine_i = (inclination / 2.0).sin();
        let mtrx = [
            Complex::new(cosine_i, 0.0),
            Complex::new(-cosine_a * sine_i, sine_a * sine_i),
            Complex::new(cosine_a * sine_i, sine_a * sine_i),
            Complex::new(cosine_i, 0.0),
        ];
        self.mtrx(&mtrx, target);
    }

    pub fn iai(&mut self, target: i32, azimuth: f64, inclination: f64) {
        let cosine_a = azimuth.cos();
        let sine_a = azimuth.sin();
        let cosine_i = (inclination / 2.0).cos();
        let sine_i = (inclination / 2.0).sin();
        let mtrx = [
            Complex::new(cosine_i, 0.0),
            Complex::new(-cosine_a * sine_i, sine_a * sine_i),
            Complex::new(cosine_a * sine_i, sine_a * sine_i),
            Complex::new(cosine_i, 0.0),
        ];
        let mut inv_mtrx = [Complex::new(0.0, 0.0); 4];
        self.inv2x2(&mtrx, &mut inv_mtrx);
        self.mtrx(&inv_mtrx, target);
    }

    pub fn cai(&mut self, control: i32, target: i32, azimuth: f64, inclination: f64) {
        let cosine_a = azimuth.cos();
        let sine_a = azimuth.sin();
        let cosine_i = (inclination / 2.0).cos();
        let sine_i = (inclination / 2.0).sin();
        let mtrx = [
            Complex::new(cosine_i, 0.0),
            Complex::new(-cosine_a * sine_i, sine_a * sine_i),
            Complex::new(cosine_a * sine_i, sine_a * sine_i),
            Complex::new(cosine_i, 0.0),
        ];
        let controls = vec![control];
        self.mcmtrx(&controls, &mtrx, target);
    }

    pub fn ciai(&mut self, control: i32, target: i32, azimuth: f64, inclination: f64) {
        let cosine_a = azimuth.cos();
        let sine_a = azimuth.sin();
        let cosine_i = (inclination / 2.0).cos();
        let sine_i = (inclination / 2.0).sin();
        let mtrx = [
            Complex::new(cosine_i, 0.0),
            Complex::new(-cosine_a * sine_i, sine_a * sine_i),
            Complex::new(cosine_a * sine_i, sine_a * sine_i),
            Complex::new(cosine_i, 0.0),
        ];
        let controls = vec![control];
        let mut inv_mtrx = [Complex::new(0.0, 0.0); 4];
        self.inv2x2(&mtrx, &mut inv_mtrx);
        self.mcmtrx(&controls, &inv_mtrx, target);
    }

    pub fn anti_cai(&mut self, control: i32, target: i32, azimuth: f64, inclination: f64) {
        let cosine_a = azimuth.cos();
        let sine_a = azimuth.sin();
        let cosine_i = (inclination / 2.0).cos();
        let sine_i = (inclination / 2.0).sin();
        let mtrx = [
            Complex::new(cosine_i, 0.0),
            Complex::new(-cosine_a * sine_i, sine_a * sine_i),
            Complex::new(cosine_a * sine_i, sine_a * sine_i),
            Complex::new(cosine_i, 0.0),
        ];
        let controls = vec![control];
        self.macmtrx(&controls, &mtrx, target);
    }

    pub fn anti_ciai(&mut self, control: i32, target: i32, azimuth: f64, inclination: f64) {
        let cosine_a = azimuth.cos();
        let sine_a = azimuth.sin();
        let cosine_i = (inclination / 2.0).cos();
        let sine_i = (inclination / 2.0).sin();
        let mtrx = [
            Complex::new(cosine_i, 0.0),
            Complex::new(-cosine_a * sine_i, sine_a * sine_i),
            Complex::new(cosine_a * sine_i, sine_a * sine_i),
            Complex::new(cosine_i, 0.0),
        ];
        let controls = vec![control];
        let mut inv_mtrx = [Complex::new(0.0, 0.0); 4];
        self.inv2x2(&mtrx, &mut inv_mtrx);
        self.macmtrx(&controls, &inv_mtrx, target);
    }

    pub fn uniformly_controlled_ry(&mut self, controls: &Vec<i32>, qubit_index: i32, angles: &[f64]) {
        let perm_count = 2i32.pow(controls.len() as u32);
        let mut pauli_rys = vec![Complex::new(0.0, 0.0); 4 * perm_count as usize];
        for i in 0..perm_count {
            let cosine = (angles[i as usize] / 2.0).cos();
            let sine = (angles[i as usize] / 2.0).sin();
            pauli_rys[0 + 4 * i as usize] = Complex::new(cosine, 0.0);
            pauli_rys[1 + 4 * i as usize] = Complex::new(-sine, 0.0);
            pauli_rys[2 + 4 * i as usize] = Complex::new(sine, 0.0);
            pauli_rys[3 + 4 * i as usize] = Complex::new(cosine, 0.0);
        }
        self.uniformly_controlled_single_bit(controls, qubit_index, &pauli_rys);
    }

    pub fn uniformly_controlled_rz(&mut self, controls: &Vec<i32>, qubit_index: i32, angles: &[f64]) {
        let perm_count = 2i32.pow(controls.len() as u32);
        let mut pauli_rzs = vec![Complex::new(0.0, 0.0); 4 * perm_count as usize];
        for i in 0..perm_count {
            let cosine = (angles[i as usize] / 2.0).cos();
            let sine = (angles[i as usize] / 2.0).sin();
            pauli_rzs[0 + 4 * i as usize] = Complex::new(cosine, -sine);
            pauli_rzs[1 + 4 * i as usize] = Complex::new(0.0, 0.0);
            pauli_rzs[2 + 4 * i as usize] = Complex::new(0.0, 0.0);
            pauli_rzs[3 + 4 * i as usize] = Complex::new(cosine, sine);
        }
        self.uniformly_controlled_single_bit(controls, qubit_index, &pauli_rzs);
    }

    pub fn rt(&mut self, radians: f64, qubit: i32) {
        let phase_fac = Complex::new((radians / 2.0).cos(), (radians / 2.0).sin());
        self.phase(Complex::new(1.0, 0.0), phase_fac, qubit);
    }

    pub fn rx(&mut self, radians: f64, qubit: i32) {
        let cosine = (radians / 2.0).cos();
        let sine = (radians / 2.0).sin();
        let pauli_rx = [
            Complex::new(cosine, 0.0),
            Complex::new(0.0, -sine),
            Complex::new(0.0, -sine),
            Complex::new(cosine, 0.0),
        ];
        self.mtrx(&pauli_rx, qubit);
    }

    pub fn ry(&mut self, radians: f64, qubit: i32) {
        let cosine = (radians / 2.0).cos();
        let sine = (radians / 2.0).sin();
        let pauli_ry = [
            Complex::new(cosine, 0.0),
            Complex::new(-sine, 0.0),
            Complex::new(sine, 0.0),
            Complex::new(cosine, 0.0),
        ];
        self.mtrx(&pauli_ry, qubit);
    }

    pub fn rz(&mut self, radians: f64, qubit: i32) {
        let cosine = (radians / 2.0).cos();
        let sine = (radians / 2.0).sin();
        self.phase(Complex::new(cosine, -sine), Complex::new(cosine, sine), qubit);
    }

    pub fn crz(&mut self, radians: f64, control: i32, target: i32) {
        let cosine = (radians / 2.0).cos();
        let sine = (radians / 2.0).sin();
        let controls = vec![control];
        self.mcphase(&controls, Complex::new(cosine, -sine), Complex::new(cosine, sine), target);
    }

    pub fn cry(&mut self, radians: f64, control: i32, target: i32) {
        let cosine = (radians / 2.0).cos();
        let sine = (radians / 2.0).sin();
        let pauli_ry = [
            Complex::new(cosine, 0.0),
            Complex::new(-sine, 0.0),
            Complex::new(sine, 0.0),
            Complex::new(cosine, 0.0),
        ];
        let controls = vec![control];
        self.mcmtrx(&controls, &pauli_ry, target);
    }
}

fn dyad_angle(numerator: i32, denom_power: i32) -> f64 {
    (-PI * numerator as f64 * 2.0) / f64::powi(2.0, denom_power)
}
