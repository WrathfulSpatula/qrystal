use std::cmp::Ordering;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::ops::{Add, Mul, Sub};
use std::rc::Rc;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
struct Complex {
    real: f64,
    imag: f64,
}

impl Complex {
    fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            real: self.real - other.real,
            imag: self.imag - other.imag,
        }
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }
}

#[derive(Clone, Debug)]
struct AmplitudeEntry {
    permutation: u64,
    amplitude: Complex,
}

impl AmplitudeEntry {
    fn new(permutation: u64, amplitude: Complex) -> Self {
        Self {
            permutation,
            amplitude,
        }
    }
}

#[derive(Clone, Debug)]
struct QStabilizer {
    raw_rand_bools: u32,
    raw_rand_bools_remaining: u32,
    phase_offset: f64,
    max_state_map_cache_qubit_count: u64,
    is_unitarity_broken: bool,
    r: Vec<u8>,
    x: Vec<Vec<bool>>,
    z: Vec<Vec<bool>>,
}

impl QStabilizer {
    fn new(
        n: u64,
        perm: u64,
        phase_fac: Complex,
        do_norm: bool,
        random_global_phase: bool,
        use_hardware_rng: bool,
    ) -> Self {
        Self {
            raw_rand_bools: 0,
            raw_rand_bools_remaining: 0,
            phase_offset: 0.0,
            max_state_map_cache_qubit_count: 0,
            is_unitarity_broken: false,
            r: vec![],
            x: vec![],
            z: vec![],
        }
    }

    fn set_phase_offset(&mut self, phase_arg: f64) {
        self.phase_offset = phase_arg;
        let is_neg = self.phase_offset < 0.0;
        if is_neg {
            self.phase_offset = -self.phase_offset;
        }
        self.phase_offset -= (self.phase_offset / (2.0 * PI)).floor() * (2.0 * PI);
        if self.phase_offset > PI {
            self.phase_offset -= 2.0 * PI;
        }
        if is_neg {
            self.phase_offset = -self.phase_offset;
        }
    }

    fn get_qubit_count(&self) -> u64 {
        self.x.len() as u64
    }

    fn get_max_q_power(&self) -> u64 {
        2u64.pow(self.get_qubit_count() as u32)
    }

    fn set_permutation(&mut self, perm: u64, phase_fac: Complex) {}

    fn set_random_seed(&mut self, seed: u32) {}

    fn rand(&mut self) -> bool {
        if self.raw_rand_bools_remaining == 0 {
            self.raw_rand_bools = 0;
            self.raw_rand_bools_remaining = 32;
        }
        self.raw_rand_bools_remaining -= 1;
        ((self.raw_rand_bools >> self.raw_rand_bools_remaining) & 1) != 0
    }

    fn clear(&mut self) {
        self.x.clear();
        self.z.clear();
        self.r.clear();
        self.phase_offset = 0.0;
    }

    fn rowcopy(&mut self, i: usize, k: usize) {
        if i == k {
            return;
        }
        self.x[i] = self.x[k].clone();
        self.z[i] = self.z[k].clone();
        self.r[i] = self.r[k];
    }

    fn rowswap(&mut self, i: usize, k: usize) {
        if i == k {
            return;
        }
        self.x.swap(i, k);
        self.z.swap(i, k);
        self.r.swap(i, k);
    }

    fn rowset(&mut self, i: usize, b: u64) {
        self.x[i] = vec![false; self.get_qubit_count() as usize];
        self.z[i] = vec![false; self.get_qubit_count() as usize];
        self.r[i] = 0;
        if b < self.get_qubit_count() {
            self.x[i][b as usize] = true;
        } else {
            let b = b - self.get_qubit_count();
            self.z[i][b as usize] = true;
        }
    }

    fn rowmult(&mut self, i: usize, k: usize) {
        self.r[i] = self.clifford(i, k);
        for j in 0..self.get_qubit_count() as usize {
            self.x[i][j] ^= self.x[k][j];
            self.z[i][j] ^= self.z[k][j];
        }
    }

    fn clifford(&self, i: usize, k: usize) -> u8 {
        0
    }

    fn seed(&self, g: usize) {}

    fn get_basis_amp(&self, nrm: f64) -> AmplitudeEntry {
        AmplitudeEntry::new(0, Complex::new(0.0, 0.0))
    }

    fn set_basis_state(&self, nrm: f64, state_vec: &mut [Complex]) {}

    fn set_basis_prob(&self, nrm: f64, output_probs: &mut [f64]) {}

    fn get_expectation(
        &self,
        nrm: f64,
        bit_powers: &[u64],
        perms: &[u64],
        offset: u64,
    ) -> f64 {
        0.0
    }

    fn get_expectation_floats_factorized(
        &self,
        nrm: f64,
        bits: &[u64],
        weights: &[f64],
    ) -> f64 {
        0.0
    }

    fn decompose_dispose(&self, start: usize, length: usize, to_copy: &mut QStabilizer) {}

    fn approx_compare_helper(
        &self,
        to_compare: &QStabilizer,
        error_tol: f64,
        is_discrete: bool,
    ) -> f64 {
        0.0
    }

    fn gaussian(&self) -> u64 {
        0
    }

    fn perm_count(&self) -> u64 {
        2u64.pow(self.gaussian() as u32)
    }

    fn set_quantum_state(&self, input_state: &[Complex]) {}

    fn get_quantum_state(&self, state_vec: &mut [Complex]) {}

    fn get_probs(&self, output_probs: &mut [f64]) {}

    fn get_amplitude(&self, perm: u64) -> Complex {
        Complex::new(0.0, 0.0)
    }

    fn get_amplitudes(&self, perms: &[u64]) -> Vec<Complex> {
        vec![]
    }

    fn get_any_amplitude(&self) -> AmplitudeEntry {
        AmplitudeEntry::new(0, Complex::new(0.0, 0.0))
    }

    fn get_qubit_amplitude(&self, t: usize, m: bool) -> AmplitudeEntry {
        AmplitudeEntry::new(0, Complex::new(0.0, 0.0))
    }

    fn expectation_bits_factorized(
        &self,
        bits: &[u64],
        perms: &[u64],
        offset: u64,
    ) -> f64 {
        0.0
    }

    fn prob_perm_rdm(&self, perm: u64, ancillae_start: usize) -> f64 {
        0.0
    }

    fn prob_mask(&self, mask: u64, permutation: u64) -> f64 {
        0.0
    }

    fn is_separable_z(&self, target: usize) -> bool {
        false
    }

    fn is_separable_x(&self, target: usize) -> bool {
        false
    }

    fn is_separable_y(&self, target: usize) -> bool {
        false
    }

    fn is_separable(&self, target: usize) -> u8 {
        0
    }

    fn compose(&self, to_copy: &QStabilizer, start: usize) -> usize {
        0
    }

    fn decompose(&self, start: usize, dest: &mut QStabilizer) {}

    fn dispose(&self, start: usize, length: usize) {}

    fn can_decompose_dispose(&self, start: usize, length: usize) -> bool {
        false
    }

    fn allocate(&self, start: usize, length: usize) -> usize {
        0
    }

    fn normalize_state(&self, nrm: f64, norm_thresh: f64, phase_arg: f64) {}

    fn update_running_norm(&self, norm_thresh: f64) {}

    fn sum_sqr_diff(&self, to_compare: &QStabilizer) -> f64 {
        0.0
    }

    fn approx_compare(&self, to_compare: &QStabilizer, error_tol: f64) -> bool {
        false
    }

    fn global_phase_compare(&self, to_compare: &QStabilizer, error_tol: f64) -> bool {
        false
    }

    fn prob(&self, qubit: usize) -> f64 {
        0.0
    }

    fn mtrx(&self, mtrx: &[Complex], target: usize) {}

    fn phase(&self, top_left: Complex, bottom_right: Complex, target: usize) {}

    fn invert(&self, top_right: Complex, bottom_left: Complex, target: usize) {}

    fn mc_phase(
        &self,
        controls: &[usize],
        top_left: Complex,
        bottom_right: Complex,
        target: usize,
    ) {
    }

    fn mac_phase(
        &self,
        controls: &[usize],
        top_left: Complex,
        bottom_right: Complex,
        target: usize,
    ) {
    }

    fn mc_invert(
        &self,
        controls: &[usize],
        top_right: Complex,
        bottom_left: Complex,
        target: usize,
    ) {
    }

    fn mac_invert(
        &self,
        controls: &[usize],
        top_right: Complex,
        bottom_left: Complex,
        target: usize,
    ) {
    }

    fn mc_mtrx(&self, controls: &[usize], mtrx: &[Complex], target: usize) {}

    fn mac_mtrx(&self, controls: &[usize], mtrx: &[Complex], target: usize) {}

    fn fsim(&self, theta: f64, phi: f64, qubit1: usize, qubit2: usize) {}

    fn try_separate(&self, qubits: &[usize], ignored: f64) -> bool {
        false
    }

    fn try_separate_qubit(&self, qubit: usize) -> bool {
        false
    }

    fn try_separate_qubits(&self, qubit1: usize, qubit2: usize) -> bool {
        false
    }
}

fn main() {
    let mut q_stabilizer = QStabilizer::new(
        0,
        0,
        Complex::new(0.0, 0.0),
        false,
        true,
        true,
    );
    q_stabilizer.set_phase_offset(0.0);
    q_stabilizer.get_qubit_count();
    q_stabilizer.get_max_q_power();
    q_stabilizer.set_permutation(0, Complex::new(0.0, 0.0));
    q_stabilizer.set_random_seed(0);
    q_stabilizer.rand();
    q_stabilizer.clear();
    q_stabilizer.rowcopy(0, 0);
    q_stabilizer.rowswap(0, 0);
    q_stabilizer.rowset(0, 0);
    q_stabilizer.rowmult(0, 0);
    q_stabilizer.clifford(0, 0);
    q_stabilizer.seed(0);
    q_stabilizer.get_basis_amp(0.0);
    q_stabilizer.set_basis_state(0.0, &mut []);
    q_stabilizer.set_basis_prob(0.0, &mut []);
    q_stabilizer.get_expectation(0.0, &[], &[], 0);
    q_stabilizer.get_expectation_floats_factorized(0.0, &[], &[]);
    q_stabilizer.decompose_dispose(0, 0, &mut QStabilizer::new(
        0,
        0,
        Complex::new(0.0, 0.0),
        false,
        true,
        true,
    ));
    q_stabilizer.approx_compare_helper(
        &QStabilizer::new(0, 0, Complex::new(0.0, 0.0), false, true, true),
        0.0,
        false,
    );
    q_stabilizer.gaussian();
    q_stabilizer.perm_count();
    q_stabilizer.set_quantum_state(&[]);
    q_stabilizer.get_quantum_state(&mut []);
    q_stabilizer.get_probs(&mut []);
    q_stabilizer.get_amplitude(0);
    q_stabilizer.get_amplitudes(&[]);
    q_stabilizer.get_any_amplitude();
    q_stabilizer.get_qubit_amplitude(0, false);
    q_stabilizer.expectation_bits_factorized(&[], &[], 0);
    q_stabilizer.prob_perm_rdm(0, 0);
    q_stabilizer.prob_mask(0, 0);
    q_stabilizer.is_separable_z(0);
    q_stabilizer.is_separable_x(0);
    q_stabilizer.is_separable_y(0);
    q_stabilizer.is_separable(0);
    q_stabilizer.compose(&QStabilizer::new(0, 0, Complex::new(0.0, 0.0), false, true, true), 0);
    q_stabilizer.decompose(0, &mut QStabilizer::new(0, 0, Complex::new(0.0, 0.0), false, true, true));
    q_stabilizer.dispose(0, 0);
    q_stabilizer.can_decompose_dispose(0, 0);
    q_stabilizer.allocate(0, 0);
    q_stabilizer.normalize_state(0.0, 0.0, 0.0);
    q_stabilizer.update_running_norm(0.0);
    q_stabilizer.sum_sqr_diff(&QStabilizer::new(0, 0, Complex::new(0.0, 0.0), false, true, true));
    q_stabilizer.approx_compare(
        &QStabilizer::new(0, 0, Complex::new(0.0, 0.0), false, true, true),
        0.0,
    );
    q_stabilizer.global_phase_compare(
        &QStabilizer::new(0, 0, Complex::new(0.0, 0.0), false, true, true),
        0.0,
    );
    q_stabilizer.prob(0);
    q_stabilizer.mtrx(&[], 0);
    q_stabilizer.phase(Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), 0);
    q_stabilizer.invert(Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), 0);
    q_stabilizer.mc_phase(&[], Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), 0);
    q_stabilizer.mac_phase(&[], Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), 0);
    q_stabilizer.mc_invert(&[], Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), 0);
    q_stabilizer.mac_invert(&[], Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), 0);
    q_stabilizer.mc_mtrx(&[], &[], 0);
    q_stabilizer.mac_mtrx(&[], &[], 0);
    q_stabilizer.fsim(0.0, 0.0, 0, 0);
    q_stabilizer.try_separate(&[], 0.0);
    q_stabilizer.try_separate_qubit(0);
    q_stabilizer.try_separate_qubits(0, 0);
}


