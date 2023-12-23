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
    fn new(
        n: usize,
        rgp: QrackRandGenPtr,
        do_norm: bool,
        use_hardware_rng: bool,
        random_global_phase: bool,
        norm_thresh: f64,
    ) -> Self {
        Self {
            do_normalize: do_norm,
            rand_global_phase: random_global_phase,
            use_rdrand: use_hardware_rng,
            qubit_count: n,
            random_seed: 0,
            amplitude_floor: norm_thresh,
            max_q_power: 1 << n,
            rand_generator: rgp,
            rand_distribution: rand::distributions::Uniform::new(0.0, 1.0),
            hardware_rand_generator: None,
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
        // TODO: Implement
        0
    }

    fn m_all(&mut self) -> usize {
        // TODO: Implement
        0
    }

    fn force_m_reg(
        &mut self,
        start: usize,
        length: usize,
        result: usize,
        do_force: bool,
        do_apply: bool,
    ) -> usize {
        // TODO: Implement
        0
    }

    fn m(&mut self, bits: &[usize]) -> usize {
        // TODO: Implement
        0
    }

    fn force_m(&mut self, bits: &[usize], values: &[bool], do_apply: bool) -> usize {
        // TODO: Implement
        0
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

    fn sum_sqr_diff(&self, to_compare: QInterfacePtr) -> f64 {
        unimplemented!()
    }

    fn try_decompose(
        &mut self,
        start: usize,
        dest: QInterfacePtr,
        error_tol: f64,
    ) -> bool {
        unimplemented!()
    }

    fn update_running_norm(&mut self, norm_thresh: f64) {
        unimplemented!()
    }

    fn normalize_state(&mut self, nrm: f64, norm_thresh: f64, phase_arg: f64) {
        unimplemented!()
    }

    fn try_separate(&mut self, qubits: &[usize], error_tol: f64) -> bool {
        unimplemented!()
    }

    fn try_separate_1(&mut self, qubit: usize) -> bool {
        unimplemented!()
    }

    fn try_separate_2(&mut self, qubit1: usize, qubit2: usize) -> bool {
        unimplemented!()
    }

    fn get_unitary_fidelity(&self) -> f64 {
        unimplemented!()
    }

    fn reset_unitary_fidelity(&mut self) {
        unimplemented!()
    }

    fn set_sdrp(&mut self, sdrp: f64) {
        unimplemented!()
    }

    fn set_reactive_separate(&mut self, is_agg_sep: bool) {
        unimplemented!()
    }

    fn get_reactive_separate(&self) -> bool {
        unimplemented!()
    }

    fn set_t_injection(&mut self, use_gadget: bool) {
        unimplemented!()
    }

    fn get_t_injection(&self) -> bool {
        unimplemented!()
    }

    fn clone_qinterface(&self) -> QInterfacePtr {
        Rc::new(RefCell::new(Box::new(*self)))
    }

    fn set_device(&mut self, d_id: i64) {
        unimplemented!()
    }

    fn get_device(&self) -> i64 {
        unimplemented!()
    }

    fn depolarizing_channel_weak_1_qb(&mut self, qubit: usize, lambda: f64) {
        unimplemented!()
    }

    fn depolarizing_channel_strong_1_qb(&mut self, qubit: usize, lambda: f64) -> usize {
        unimplemented!()
    }
}
