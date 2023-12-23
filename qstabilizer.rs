//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// Adapted from:
//
// CHP: CNOT-Hadamard-Phase
// Stabilizer Quantum Computer Simulator
// by Scott Aaronson
// Last modified June 30, 2004
//
// Thanks to Simon Anders and Andrew Cross for bugfixes
//
// https://www.scottaaronson.com/chp/
//
// Daniel Strano and the Qrack contributers appreciate Scott Aaronson's open sharing of the CHP code, and we hope that
// vm6502q/qrack is one satisfactory framework by which CHP could be adapted to enter the C++ STL. Our project
// philosophy aims to raise the floor of decentralized quantum computing technology access across all modern platforms,
// for all people, not commercialization.
//
// This was heavily adapted for Qrystal, from Qrack, by Daniel Strano making extensive use of https://codeconvert.ai
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::ops::{AddAssign, BitAndAssign, BitOrAssign, Not};
use std::sync::Arc;

type bitCapInt = u64;
type bitLenInt = u64;
type real1 = f64;
type complex = num_complex::Complex<real1>;

const ZERO_BCI: bitCapInt = 0;
const ZERO_R1: real1 = 0.0;
const ONE_BCI: bitCapInt = 1;
const ONE_R1: real1 = 1.0;
const PI_R1: real1 = PI;
const I_CMPLX: complex = complex::new(ZERO_R1, ONE_R1);
const CMPLX_DEFAULT_ARG: complex = complex::new(ZERO_R1, ZERO_R1);
const REAL1_EPSILON: real1 = 1e-10;
const FP_NORM_EPSILON_F: real1 = 1e-10;
const REAL1_DEFAULT_ARG: real1 = ZERO_R1;
const TRYDECOMPOSE_EPSILON: real1 = 1e-10;
const IS_NORM_0: fn(real1) -> bool = |r| r.abs() <= REAL1_EPSILON;
const C_SQRT1_2: Complex<f64> = Complex::new(std::f64::consts::SQRT_1_2, 0.0);
const C_I_SQRT1_2: Complex<f64> = Complex::new(0.0, std::f64::consts::SQRT_1_2);

struct AmplitudeEntry {
    permutation: bitCapInt,
    amplitude: complex,
}

impl AmplitudeEntry {
    fn new(permutation: bitCapInt, amplitude: complex) -> Self {
        Self {
            permutation,
            amplitude,
        }
    }
}

struct QStabilizer {
    qubit_count: bitLenInt,
    raw_rand_bools: u32,
    raw_rand_bools_remaining: u32,
    phase_offset: real1,
    max_state_map_cache_qubit_count: bitLenInt,
    is_unitarity_broken: bool,
    r: Vec<u8>,
    x: Vec<Vec<bool>>,
    z: Vec<Vec<bool>>,
}

impl QStabilizer {
    fn new(
        n: bitLenInt,
        perm: bitCapInt,
        rgp: Option<Arc<dyn rand::RngCore>>,
        phase_fac: complex,
        do_norm: bool,
        random_global_phase: bool,
        ignored2: bool,
        ignored3: i64,
        use_hardware_rng: bool,
        ignored4: bool,
        ignored5: real1,
        ignored6: Vec<i64>,
        ignored7: bitLenInt,
        ignored8: real1,
    ) -> Self {
        let qubit_count = n;
        let raw_rand_bools = 0;
        let raw_rand_bools_remaining = 0;
        let phase_offset = ZERO_R1;
        let max_state_map_cache_qubit_count = if let Ok(max_cpu_qb) = std::env::var("QRACK_MAX_CPU_QB") {
            max_cpu_qb.parse().unwrap_or(28 - if QBCAPPOW < FPPOW { 1 } else { 1 + QBCAPPOW - FPPOW })
        } else {
            28 - if QBCAPPOW < FPPOW { 1 } else { 1 + QBCAPPOW - FPPOW }
        };
        let is_unitarity_broken = false;
        let r = vec![0; (n << 1) + 1];
        let x = vec![vec![false; n as usize]; (n << 1) + 1];
        let z = vec![vec![false; n as usize]; (n << 1) + 1];

        Self {
            qubit_count,
            raw_rand_bools,
            raw_rand_bools_remaining,
            phase_offset,
            max_state_map_cache_qubit_count,
            is_unitarity_broken,
            r,
            x,
            z,
        }
    }

    fn dispatch(&self, fn_: impl FnOnce()) {
        fn_();
    }

    fn par_for(&self, fn_: impl Fn(bitLenInt)) {
        let max_lcv = self.qubit_count << 1;
        for i in 0..max_lcv {
            fn_(i);
        }
    }

    fn set_phase_offset(&mut self, phase_arg: real1) {
        self.phase_offset = phase_arg;
        let is_neg = self.phase_offset < 0.0;
        if is_neg {
            self.phase_offset = -self.phase_offset;
        }
        self.phase_offset -= (self.phase_offset / (2.0 * PI_R1)).floor() * (2.0 * PI_R1);
        if self.phase_offset > PI_R1 {
            self.phase_offset -= 2.0 * PI_R1;
        }
        if is_neg {
            self.phase_offset = -self.phase_offset;
        }
    }

    fn get_qubit_count(&self) -> bitLenInt {
        self.qubit_count
    }

    fn get_max_q_power(&self) -> bitCapInt {
        2u64.pow(self.qubit_count as u32)
    }

    fn reset_phase_offset(&mut self) {
        self.phase_offset = ZERO_R1;
    }

    fn get_phase_offset(&self) -> complex {
        complex::from_polar(&ONE_R1, &self.phase_offset)
    }

    fn set_permutation(&mut self, perm: bitCapInt, phase_fac: complex) {
        self.dump();
        self.is_unitarity_broken = false;
        if phase_fac != CMPLX_DEFAULT_ARG {
            self.phase_offset = phase_fac.arg();
        } else if self.random_global_phase {
            self.phase_offset = 2.0 * PI_R1 * (self.rand() as real1) - PI_R1;
        } else {
            self.phase_offset = ZERO_R1;
        }
        let row_count = self.qubit_count << 1;
        self.r.iter_mut().for_each(|r| *r = 0);
        for i in 0..row_count {
            self.x[i].iter_mut().for_each(|x| *x = false);
            self.z[i].iter_mut().for_each(|z| *z = false);
            if i < self.qubit_count {
                self.x[i][i] = true;
            } else {
                let j = i - self.qubit_count;
                self.z[i][j] = true;
            }
        }
        if perm == ZERO_BCI {
            return;
        }
        for j in 0..self.qubit_count {
            if (perm >> j) & 1 == 1 {
                self.x(j);
            }
        }
    }

    fn clifford(&self, i: bitLenInt, k: bitLenInt) -> u8 {
        let xi = &self.x[i];
        let zi = &self.z[i];
        let xk = &self.x[k];
        let zk = &self.z[k];
        let mut e = 0;
        for j in 0..self.qubit_count {
            if xk[j] && !zk[j] {
                e += (xi[j] && zi[j]) as u8;
                e -= (!xi[j] && zi[j]) as u8;
            }
            if xk[j] && zk[j] {
                e += (!xi[j] && zi[j]) as u8;
                e -= (xi[j] && !zi[j]) as u8;
            }
            if !xk[j] && zk[j] {
                e += (xi[j] && !zi[j]) as u8;
                e -= (xi[j] && zi[j]) as u8;
            }
        }
        (e + self.r[i] + self.r[k]) & 0x3
    }

    fn gaussian(&mut self) -> bitLenInt {
        let n = self.qubit_count;
        let max_lcv = n << 1;
        let mut i = n;
        let mut k;
        for j in 0..n {
            for k in i..max_lcv {
                if !self.x[k][j] {
                    continue;
                }
                self.rowswap(i, k);
                self.rowswap(i - n, k - n);
                for k2 in (i + 1)..max_lcv {
                    if self.x[k2][j] {
                        self.rowmult(k2, i);
                        self.rowmult(i - n, k2 - n);
                    }
                }
                i += 1;
                break;
            }
        }
        let g = i - n;
        for j in 0..n {
            for k in i..max_lcv {
                if !self.z[k][j] {
                    continue;
                }
                self.rowswap(i, k);
                self.rowswap(i - n, k - n);
                for k2 in (i + 1)..max_lcv {
                    if self.z[k2][j] {
                        self.rowmult(k2, i);
                        self.rowmult(i - n, k2 - n);
                    }
                }
                i += 1;
                break;
            }
        }
        g
    }

    fn seed(&mut self, g: bitLenInt) {
        let elem_count = self.qubit_count << 1;
        let mut min = 0;
        self.r[elem_count] = 0;
        self.x[elem_count].iter_mut().for_each(|x| *x = false);
        self.z[elem_count].iter_mut().for_each(|z| *z = false);
        for i in (0..elem_count - 1).rev() {
            let mut f = self.r[i];
            for j in (0..self.qubit_count).rev() {
                if self.z[i][j] {
                    min = j;
                    if self.x[elem_count][j] {
                        f = (f + 2) & 0x3;
                    }
                }
            }
            if f == 2 {
                let j = min;
                self.x[elem_count][j] = !self.x[elem_count][j];
            }
        }
    }

    fn get_basis_amp(&self, nrm: real1) -> AmplitudeEntry {
        let elem_count = self.qubit_count << 1;
        let e = self.r[elem_count];
        let x_row = &self.x[elem_count];
        let z_row = &self.z[elem_count];
        let mut amp = complex::new(nrm, ZERO_R1);
        if e & 1 != 0 {
            amp *= I_CMPLX;
        }
        if e & 2 != 0 {
            amp *= -ONE_CMPLX;
        }
        amp *= complex::from_polar(&ONE_R1, &self.phase_offset);
        let mut perm = ZERO_BCI;
        for j in 0..self.qubit_count {
            if x_row[j] {
                perm |= 1 << j;
            }
        }
        AmplitudeEntry::new(perm, amp)
    }

    fn set_basis_state(&self, nrm: real1, state_vec: &mut [complex]) {
        let entry = self.get_basis_amp(nrm);
        state_vec[entry.permutation as usize] = entry.amplitude;
    }

    fn set_basis_state_qinterface(&self, nrm: real1, eng: &mut dyn QInterface) {
        let entry = self.get_basis_amp(nrm);
        eng.set_amplitude(entry.permutation, entry.amplitude);
    }

    fn set_basis_state_map(&self, nrm: real1, state_map: &mut HashMap<bitCapInt, complex>) {
        let entry = self.get_basis_amp(nrm);
        state_map.insert(entry.permutation, entry.amplitude);
    }

    fn set_basis_prob(&self, nrm: real1, output_probs: &mut [real1]) {
        let entry = self.get_basis_amp(nrm);
        output_probs[entry.permutation as usize] = entry.amplitude.norm();
    }

    fn get_expectation(
        &self,
        nrm: f64,
        bit_powers: &Vec<u64>,
        perms: &Vec<u64>,
        offset: u64,
    ) -> f64 {
        let entry = self.get_basis_amp(nrm);
        let mut ret_index = 0;
        for b in 0..bit_powers.len() {
            ret_index += if entry.permutation & bit_powers[b] != 0 {
                perms[(b << 1) | 1]
            } else {
                perms[b << 1]
            };
        }
        (offset + ret_index) as f64 * norm(entry.amplitude)
    }

    fn get_expectation(
        &self,
        nrm: f64,
        bit_powers: &Vec<u64>,
        weights: &Vec<f64>,
    ) -> f64 {
        let entry = self.get_basis_amp(nrm);
        let mut weight = 0.0;
        for b in 0..bit_powers.len() {
            weight += if entry.permutation & bit_powers[b] != 0 {
                weights[(b << 1) | 1]
            } else {
                weights[b << 1]
            };
        }
        weight * norm(entry.amplitude)
    }

    fn get_expectation_weights(
        &self,
        nrm: real1,
        bit_powers: &[bitCapInt],
        weights: &[real1],
    ) -> real1 {
        let entry = self.get_basis_amp(nrm);
        let mut weight = ZERO_R1;
        for (b, &bit_power) in bit_powers.iter().enumerate() {
            weight += if (entry.permutation & bit_power) != 0 {
                weights[(b << 1) + 1]
            } else {
                weights[b << 1]
            };
        }
        weight * entry.amplitude.norm()
    }

    fn rowcopy(&mut self, i: bitLenInt, k: bitLenInt) {
        if i == k {
            return;
        }
        self.x[i] = self.x[k].clone();
        self.z[i] = self.z[k].clone();
        self.r[i] = self.r[k];
    }

    fn rowswap(&mut self, i: bitLenInt, k: bitLenInt) {
        if i == k {
            return;
        }
        self.x.swap(i as usize, k as usize);
        self.z.swap(i as usize, k as usize);
        self.r.swap(i as usize, k as usize);
    }

    fn rowset(&mut self, i: bitLenInt, b: bitLenInt) {
        self.x[i].iter_mut().for_each(|x| *x = false);
        self.z[i].iter_mut().for_each(|z| *z = false);
        self.r[i] = 0;
        if b < self.qubit_count {
            self.x[i][b as usize] = true;
        } else {
            let b = b - self.qubit_count;
            self.z[i][b as usize] = true;
        }
    }

    fn rowmult(&mut self, i: bitLenInt, k: bitLenInt) {
        self.r[i] = self.clifford(i, k);
        for j in 0..self.qubit_count {
            self.x[i][j] ^= self.x[k][j];
            self.z[i][j] ^= self.z[k][j];
        }
    }

    fn get_quantum_state(&self, state_vec: &mut [Complex<f64>]) {
        self.finish();
        let g = self.gaussian();
        let perm_count = 2u64.pow(g);
        let mut perm_count_min1 = perm_count;
        perm_count_min1 -= 1;
        let elem_count = qubit_count << 1;
        let nrm = (1.0 / (perm_count as f64)).sqrt();
        self.seed(g);
        state_vec.fill(Complex::new(0.0, 0.0));
        self.set_basis_state(nrm, state_vec);
        for t in 0..perm_count_min1 {
            let t2 = t ^ (t + 1);
            for i in 0..g {
                if t2 & (1 << i) != 0 {
                    self.rowmult(elem_count, qubit_count + i);
                }
            }
            self.set_basis_state(nrm, state_vec);
        }
    }

    fn get_quantum_state(&self, eng: &mut QInterfacePtr) {
        self.finish();
        let g = self.gaussian();
        let perm_count = 2u64.pow(g);
        let mut perm_count_min1 = perm_count;
        perm_count_min1 -= 1;
        let elem_count = qubit_count << 1;
        let nrm = (1.0 / (perm_count as f64)).sqrt();
        self.seed(g);
        eng.set_permutation(0);
        eng.set_amplitude(0, Complex::new(0.0, 0.0));
        self.set_basis_state(nrm, eng);
        for t in 0..perm_count_min1 {
            let t2 = t ^ (t + 1);
            for i in 0..g {
                if t2 & (1 << i) != 0 {
                    self.rowmult(elem_count, qubit_count + i);
                }
            }
            self.set_basis_state(nrm, eng);
        }
    }

    fn get_quantum_state(&self) -> std::collections::HashMap<u64, Complex<f64>> {
        self.finish();
        let g = self.gaussian();
        let perm_count = 2u64.pow(g);
        let mut perm_count_min1 = perm_count;
        perm_count_min1 -= 1;
        let elem_count = qubit_count << 1;
        let nrm = (1.0 / (perm_count as f64)).sqrt();
        self.seed(g);
        let mut state_map = std::collections::HashMap::new();
        self.set_basis_state(nrm, &mut state_map);
        for t in 0..perm_count_min1 {
            let t2 = t ^ (t + 1);
            for i in 0..g {
                if t2 & (1 << i) != 0 {
                    self.rowmult(elem_count, qubit_count + i);
                }
            }
            self.set_basis_state(nrm, &mut state_map);
        }
        state_map
    }


    fn set_quantum_state(&mut self, input_state: &[complex]) {
        let mut perm = ZERO_BCI;
        for (i, &state) in input_state.iter().enumerate() {
            if state.norm() > REAL1_EPSILON {
                perm |= 1 << i;
            }
        }
        self.set_permutation(perm, CMPLX_DEFAULT_ARG);
    }

    fn get_probs(&self, output_probs: &mut [f64]) {
        self.finish();
        let g = self.gaussian();
        let perm_count = 2u64.pow(g);
        let mut perm_count_min1 = perm_count;
        perm_count_min1 -= 1;
        let elem_count = qubit_count << 1;
        let nrm = (1.0 / (perm_count as f64)).sqrt();
        self.seed(g);
        output_probs.fill(0.0);
        self.set_basis_prob(nrm, output_probs);
        for t in 0..perm_count_min1 {
            let t2 = t ^ (t + 1);
            for i in 0..g {
                if t2 & (1 << i) != 0 {
                    self.rowmult(elem_count, qubit_count + i);
                }
            }
            self.set_basis_prob(nrm, output_probs);
        }
    }

    fn get_amplitudes(&self, perms: &Vec<u64>) -> Vec<Complex<f64>> {
        let prms: std::collections::HashSet<u64> = perms.iter().cloned().collect();
        let mut amps: std::collections::HashMap<u64, Complex<f64>> = std::collections::HashMap::new();
        self.finish();
        let g = self.gaussian();
        let perm_count = 2u64.pow(g);
        let mut perm_count_min1 = perm_count;
        perm_count_min1 -= 1;
        let elem_count = qubit_count << 1;
        let nrm = (1.0 / (perm_count as f64)).sqrt();
        self.seed(g);
        let entry = get_basis_amp(nrm);
        if prms.contains(&entry.permutation) {
            amps.insert(entry.permutation, entry.amplitude);
        }
        if amps.len() < perms.len() {
            for t in 0..perm_count_min1 {
                let t2 = t ^ (t + 1);
                for i in 0..g {
                    if t2 & (1 << i) != 0 {
                        self.rowmult(elem_count, qubit_count + i);
                    }
                }
                let entry = self.get_basis_amp(nrm);
                if prms.contains(&entry.permutation) {
                    amps.insert(entry.permutation, entry.amplitude);
                    if amps.len() >= perms.len() {
                        break;
                    }
                }
            }
        }
        let mut to_ret = Vec::with_capacity(perms.len());
        for perm in perms {
            to_ret.push(amps[perm]);
        }
        to_ret
    }

    fn get_any_amplitude(&self) -> AmplitudeEntry {
        self.finish();
        let g = self.gaussian();
        let nrm = (1.0 / (2u64.pow(g) as f64)).sqrt();
        self.seed(g);
        self.get_basis_amp(nrm)
    }

    fn set_rand_global_phase(&mut self, is_rand: bool) {
        self.random_global_phase = is_rand;
    }

    fn get_qubit_amplitude(&self, t: u64, m: bool) -> AmplitudeEntry {
        let t_pow = 2u64.pow(t);
        let m_pow = if m { t_pow } else { 0 };
        self.finish();
        let g = self.gaussian();
        let perm_count = 2u64.pow(g);
        let mut perm_count_min1 = perm_count;
        perm_count_min1 -= 1;
        let elem_count = self.qubit_count << 1;
        let nrm = (1.0 / (perm_count as f64)).sqrt();
        self.seed(g);
        let entry = self.get_basis_amp(nrm);
        if entry.permutation & t_pow == m_pow {
            return entry;
        }
        for t in 0..perm_count_min1 {
            let t2 = t ^ (t + 1);
            for i in 0..g {
                if t2 & (1 << i) != 0 {
                    self.rowmult(elem_count, self.qubit_count + i);
                }
            }
            let entry = self.get_basis_amp(nrm);
            if entry.permutation & t_pow == m_pow {
                return entry;
            }
        }
        AmplitudeEntry { permutation: 0, amplitude: Complex::new(0.0, 0.0) }
    }

    fn expectation_bits_factorized(
        &self,
        bits: &Vec<u64>,
        perms: &Vec<u64>,
        offset: u64,
    ) -> f64 {
        if perms.len() < (bits.len() << 1) {
            panic!("QStabilizer::ExpectationBitsFactorized must supply at least twice as many weights as bits!");
        }
        self.throw_if_qb_id_array_is_bad(bits, self.qubit_count, "QStabilizer::ExpectationBitsAllRdm parameter qubits vector values must be within allocated qubit bounds!");
        let bit_powers: Vec<u64> = bits.iter().map(|&bit| 2u64.pow(bit)).collect();
        self.finish();
        let g = self.gaussian();
        let perm_count = 2u64.pow(g);
        let mut perm_count_min1 = perm_count;
        perm_count_min1 -= 1;
        let elem_count = self.qubit_count << 1;
        let nrm = (1.0 / (perm_count as f64)).sqrt();
        self.seed(g);
        let expectation = self.get_expectation(nrm, &bit_powers, perms, offset);
        for t in 0..perm_count_min1 {
            let t2 = t ^ (t + 1);
            for i in 0..g {
                if t2 & (1 << i) != 0 {
                    self.rowmult(elem_count, self.qubit_count + i);
                }
            }
            let expectation = self.get_expectation(nrm, &bit_powers, perms, offset);
        }
        expectation
    }

    fn expectation_floats_factorized(
        &self,
        bits: &Vec<u64>,
        weights: &Vec<f64>,
    ) -> f64 {
        if weights.len() < (bits.len() << 1) {
            panic!("QStabilizer::ExpectationFloatsFactorized() must supply at least twice as many weights as bits!");
        }
        self.throw_if_qb_id_array_is_bad(bits, self.qubit_count, "QStabilizer::ExpectationFloatsFactorized() parameter qubits vector values must be within allocated qubit bounds!");
        let bit_powers: Vec<u64> = bits.iter().map(|&bit| 2u64.pow(bit)).collect();
        self.finish();
        let g = self.gaussian();
        let perm_count = 2u64.pow(g);
        let mut perm_count_min1 = perm_count;
        perm_count_min1 -= 1;
        let elem_count = self.qubit_count << 1;
        let nrm = (1.0 / (perm_count as f64)).sqrt();
        self.seed(g);
        let expectation = self.get_expectation(nrm, &bit_powers, weights);
        for t in 0..perm_count_min1 {
            let t2 = t ^ (t + 1);
            for i in 0..g {
                if t2 & (1 << i) != 0 {
                    self.rowmult(elem_count, self.qubit_count + i);
                }
            }
            let expectation = self.get_expectation(nrm, &bit_powers, weights);
        }
        expectation
    }

    fn prob_perm_rdm(&self, perm: u64, ancillae_start: u64) -> f64 {
        if ancillae_start > self.qubit_count {
            panic!("QStabilizer::ProbPermRDM ancillaeStart is out-of-bounds!");
        }
        if ancillae_start == self.qubit_count {
            return self.prob_all(perm);
        }
        let qubit_mask = 2u64.pow(ancillae_start) - 1;
        let perm = perm & qubit_mask;
        self.finish();
        let g = self.gaussian();
        let perm_count = 2u64.pow(g);
        let mut perm_count_min1 = perm_count;
        perm_count_min1 -= 1;
        let elem_count = self.qubit_count << 1;
        let nrm = (1.0 / (perm_count as f64)).sqrt();
        self.seed(g);
        let first_amp = self.get_basis_amp(nrm);
        let prob = if first_amp.permutation & qubit_mask == perm {
            self.norm(first_amp.amplitude)
        } else {
            0.0
        };
        for t in 0..perm_count_min1 {
            let t2 = t ^ (t + 1);
            for i in 0..g {
                if t2 & (1 << i) != 0 {
                    self.rowmult(elem_count, self.qubit_count + i);
                }
            }
            let amp = self.get_basis_amp(nrm);
            if perm == amp.permutation & qubit_mask {
                prob += self.norm(amp.amplitude);
            }
        }
        prob
    }

    fn prob_mask(&self, mask: u64, perm: u64) -> f64 {
        self.finish();
        let g = self.gaussian();
        let perm_count = 2u64.pow(g);
        let mut perm_count_min1 = perm_count;
        perm_count_min1 -= 1;
        let elem_count = self.qubit_count << 1;
        let nrm = (1.0 / (perm_count as f64)).sqrt();
        self.seed(g);
        let first_amp = self.get_basis_amp(nrm);
        let prob = if first_amp.permutation & mask == perm {
            self.norm(first_amp.amplitude)
        } else {
            0.0
        };
        for t in 0..perm_count_min1 {
            let t2 = t ^ (t + 1);
            for i in 0..g {
                if t2 & (1 << i) != 0 {
                    self.rowmult(elem_count, self.qubit_count + i);
                }
            }
            let amp = self.get_basis_amp(nrm);
            if perm == amp.permutation & mask {
                prob += self.norm(amp.amplitude);
            }
        }
        prob
    }

    fn cnot(&self, c: u64, t: u64) {
        if !self.rand_global_phase {
            self.h(t);
            self.cz(c, t);
            self.h(t);
            return;
        }
        for i in 0..self.x.len() {
            if self.x[i][c] {
                self.x[i][t] = !self.x[i][t];
            }
            if self.z[i][t] {
                self.z[i][c] = !self.z[i][c];
                if self.x[i][c] && (self.x[i][t] == self.z[i][c]) {
                    self.r[i] = (self.r[i] + 2) & 0x3;
                }
            }
        }
    }

    fn anti_cnot(&self, c: u64, t: u64) {
        if !self.rand_global_phase {
            self.h(t);
            self.anti_cz(c, t);
            self.h(t);
            return;
        }
        for i in 0..self.x.len() {
            if self.x[i][c] {
                self.x[i][t] = !self.x[i][t];
            }
            if self.z[i][t] {
                self.z[i][c] = !self.z[i][c];
                if !self.x[i][c] || (self.x[i][t] != self.z[i][c]) {
                    self.r[i] = (self.r[i] + 2) & 0x3;
                }
            }
        }
    }

    fn cy(&self, c: u64, t: u64) {
        if !self.rand_global_phase {
            self.is(t);
            self.cnot(c, t);
            self.s(t);
            return;
        }
        for i in 0..self.x.len() {
            self.z[i][t] ^= self.x[i][t];
            if self.x[i][c] {
                self.x[i][t] = !self.x[i][t];
            }
            if self.z[i][t] {
                if self.x[i][c] && (self.x[i][t] == self.z[i][c]) {
                    self.r[i] = (self.r[i] + 2) & 0x3;
                }
                self.z[i][c] = !self.z[i][c];
            }
            self.z[i][t] ^= self.x[i][t];
        }
    }

    fn anti_cy(&self, c: u64, t: u64) {
        if !self.rand_global_phase {
            self.is(t);
            self.anti_cnot(c, t);
            self.s(t);
            return;
        }
        for i in 0..self.x.len() {
            self.z[i][t] ^= self.x[i][t];
            if self.x[i][c] {
                self.x[i][t] = !self.x[i][t];
            }
            if self.z[i][t] {
                if !self.x[i][c] || (self.x[i][t] != self.z[i][c]) {
                    self.r[i] = (self.r[i] + 2) & 0x3;
                }
                self.z[i][c] = !self.z[i][c];
            }
            self.z[i][t] ^= self.x[i][t];
        }
    }

    fn swap(&self, c: u64, t: u64) {
        if c == t {
            return;
        }
        if !self.rand_global_phase {
            QInterface::swap(c, t);
            return;
        }
        for i in 0..self.x.len() {
            self.x[i].swap(c, t);
            self.z[i].swap(c, t);
        }
    }

    fn iswap(&self, c: u64, t: u64) {
        if c == t {
            return;
        }
        if !self.rand_global_phase {
            QInterface::iswap(c, t);
            return;
        }
        for i in 0..self.x.len() {
            self.x[i].swap(c, t);
            self.z[i].swap(c, t);
            if self.x[i][t] {
                self.z[i][c] = !self.z[i][c];
                if !self.x[i][c] && self.z[i][t] {
                    self.r[i] = (self.r[i] + 2) & 0x3;
                }
            }
            if self.x[i][c] {
                self.z[i][t] = !self.z[i][t];
                if self.z[i][c] && !self.x[i][t] {
                    self.r[i] = (self.r[i] + 2) & 0x3;
                }
            }
            self.z[i][c] ^= self.x[i][c];
            self.z[i][t] ^= self.x[i][t];
        }
    }

    fn iiswap(&self, c: u64, t: u64) {
        if c == t {
            return;
        }
        if !self.rand_global_phase {
            QInterface::iiswap(c, t);
            return;
        }
        for i in 0..self.x.len() {
            self.z[i][c] ^= self.x[i][c];
            self.z[i][t] ^= self.x[i][t];
            if self.x[i][t] {
                self.z[i][c] = !self.z[i][c];
                if self.z[i][c] && !self.x[i][t] {
                    self.r[i] = (self.r[i] + 2) & 0x3;
                }
            }
            if self.x[i][c] {
                self.z[i][t] = !self.z[i][t];
                if !self.x[i][c] && self.z[i][t] {
                    self.r[i] = (self.r[i] + 2) & 0x3;
                }
            }
            self.x[i].swap(c, t);
            self.z[i].swap(c, t);
        }
    }

    fn force_m(&mut self, t: bitLenInt, result: bool, do_force: bool, do_apply: bool) -> bool {
        let mut ret = false;
        self.par_for(|i| {
            if i == t {
                let x = self.x[i][0];
                let z = self.z[i][0];
                if (x && !z) != result {
                    if do_force {
                        self.x[i].iter_mut().for_each(|x| *x ^= true);
                    }
                    ret = true;
                }
                if (x && z) != result {
                    if do_force {
                        self.x[i].iter_mut().for_each(|x| *x ^= true);
                        self.z[i].iter_mut().for_each(|z| *z ^= true);
                    }
                    ret = true;
                }
                if (!x && z) != result {
                    if do_force {
                        self.z[i].iter_mut().for_each(|z| *z ^= true);
                    }
                    ret = true;
                }
            }
        });
        if do_apply && ret {
            self.apply();
        }
        ret
    }

    fn h(&mut self, t: usize) {
        let clone = if self.rand_global_phase {
            None
        } else {
            Some(self.clone())
        };
        self.par_for(t, |i| {
            self.x[i][t] = std::mem::replace(&mut self.z[i][t], self.x[i][t]);
            if self.x[i][t] && self.z[i][t] {
                self.r[i] = (self.r[i] + 2) & 0x3;
            }
        });
        if self.rand_global_phase {
            return;
        }
        let o_is_sep_z = clone.as_ref().unwrap().is_separable_z(t);
        let n_is_sep_z = self.is_separable_z(t);
        let t_pow = 2usize.pow(t as u32);
        let g = self.gaussian();
        let perm_count = 2usize.pow(g as u32);
        let perm_count_min1 = perm_count - 1;
        let elem_count = self.qubit_count << 1;
        let nrm = (1.0 / (perm_count as f64)).sqrt();
        self.seed(g);
        let entry = self.get_basis_amp(nrm);
        if n_is_sep_z || (entry.permutation & t_pow == 0) {
            let o_amp = clone.unwrap().get_amplitude(if o_is_sep_z { entry.permutation } else { entry.permutation & !t_pow });
            if o_amp.norm() > std::f64::EPSILON {
                self.set_phase_offset(self.phase_offset + o_amp.arg() - entry.amplitude.arg());
                return;
            }
        }
        for t in 0..perm_count_min1 {
            let t2 = t ^ (t + 1);
            for i in 0..g {
                if t2 & (1 << i) != 0 {
                    self.rowmult(elem_count, self.qubit_count + i);
                }
            }
            let entry = self.get_basis_amp(nrm);
            if n_is_sep_z || (entry.permutation & t_pow == 0) {
                let o_amp = clone.unwrap().get_amplitude(if o_is_sep_z { entry.permutation } else { entry.permutation & !t_pow });
                if o_amp.norm() > std::f64::EPSILON {
                    self.set_phase_offset(self.phase_offset + o_amp.arg() - entry.amplitude.arg());
                    return;
                }
            }
        }
    }

    fn x(&mut self, t: usize) {
        if !self.rand_global_phase {
            self.h(t);
            self.z(t);
            self.h(t);
            return;
        }
        self.par_for(t, |i| {
            if self.z[i][t] {
                self.r[i] = (self.r[i] + 2) & 0x3;
            }
        });
    }

    fn y(&mut self, t: usize) {
        if !self.rand_global_phase && self.is_separable_z(t) {
            self.is(t);
            self.x(t);
            self.s(t);
            return;
        }
        self.par_for(t, |i| {
            if self.z[i][t] ^ self.x[i][t] {
                self.r[i] = (self.r[i] + 2) & 0x3;
            }
        });
    }

    fn z(&mut self, t: usize) {
        if !self.rand_global_phase && self.is_separable_z(t) {
            if self.m(t) {
                self.set_phase_offset(self.phase_offset + PI);
            }
            return;
        }
        let amp_entry = if self.rand_global_phase {
            AmplitudeEntry(0, Complex::new(0.0, 0.0))
        } else {
            self.get_qubit_amplitude(t, false)
        };
        self.par_for(t, |i| {
            if self.x[i][t] {
                self.r[i] = (self.r[i] + 2) & 0x3;
            }
        });
        if self.rand_global_phase {
            self.set_phase_offset(self.phase_offset + amp_entry.amplitude.arg() - self.get_amplitude(amp_entry.permutation).arg());
        }
    }

    fn s(&mut self, t: usize) {
        if !self.rand_global_phase && self.is_separable_z(t) {
            if self.m(t) {
                self.set_phase_offset(self.phase_offset + PI / 2);
            }
            return;
        }
        let amp_entry = if self.rand_global_phase {
            AmplitudeEntry(0, Complex::new(0.0, 0.0))
        } else {
            self.get_qubit_amplitude(t, false)
        };
        self.par_for(t, |i| {
            if self.x[i][t] && self.z[i][t] {
                self.r[i] = (self.r[i] + 2) & 0x3;
            }
            self.z[i][t] ^= self.x[i][t];
        });
        if self.rand_global_phase {
            return;
        }
        if self.rand_global_phase {
            self.set_phase_offset(self.phase_offset + amp_entry.amplitude.arg() - self.get_amplitude(amp_entry.permutation).arg());
        }
    }

    fn is(&mut self, t: usize) {
        if !self.rand_global_phase && self.is_separable_z(t) {
            if self.m(t) {
                self.set_phase_offset(self.phase_offset - PI / 2);
            }
            return;
        }
        let amp_entry = if self.rand_global_phase {
            AmplitudeEntry(0, Complex::new(0.0, 0.0))
        } else {
            self.get_qubit_amplitude(t, false)
        };
        self.par_for(t, |i| {
            self.z[i][t] ^= self.x[i][t];
            if self.x[i][t] && self.z[i][t] {
                self.r[i] = (self.r[i] + 2) & 0x3;
            }
        });
        if self.rand_global_phase {
            return;
        }
        if self.rand_global_phase {
            self.set_phase_offset(self.phase_offset + amp_entry.amplitude.arg() - self.get_amplitude(amp_entry.permutation).arg());
        }
    }

    fn is_separable_z(&self, t: usize) -> bool {
        if t >= self.qubit_count {
            panic!("QStabilizer::IsSeparableZ qubit index is out-of-bounds!");
        }
        self.finish();
        let n = self.qubit_count;
        for p in 0..n {
            if self.x[p + n][t] {
                return false;
            }
        }
        true
    }

    fn is_separable_x(&mut self, t: usize) -> bool {
        self.h(t);
        let is_separable = self.is_separable_z(t);
        self.h(t);
        is_separable
    }

    fn is_separable_y(&mut self, t: usize) -> bool {
        self.is(t);
        let is_separable = self.is_separable_x(t);
        self.s(t);
        is_separable
    }

    fn is_separable(&mut self, t: usize) -> u8 {
        if self.is_separable_z(t) {
            return 1;
        }
        if self.is_separable_x(t) {
            return 2;
        }
        if self.is_separable_y(t) {
            return 3;
        }
        0
    }

    fn force_m(&mut self, t: usize, result: bool, do_force: bool, do_apply: bool) -> bool {
        if t >= self.qubit_count {
            panic!("QStabilizer::ForceM qubit index is out-of-bounds!");
        }
        if do_force && !do_apply {
            return result;
        }
        self.finish();
        let elem_count = self.qubit_count << 1;
        let n = self.qubit_count;
        let mut p = 0;
        while p < n {
            if self.x[p + n][t] {
                break;
            }
            p += 1;
        }
        if p < n {
            if !do_force {
                return rand::thread_rng().gen();
            }
            if !do_apply {
                return result;
            }
            self.is_unitarity_broken = true;
            let clone = if self.rand_global_phase {
                None
            } else {
                Some(self.clone())
            };
            self.rowcopy(p, p + n);
            self.rowset(p + n, t + n);
            self.r[p + n] = if result { 2 } else { 0 };
            for i in 0..p {
                if self.x[i][t] {
                    self.rowmult(i, p);
                }
            }
            for i in p + 1..elem_count {
                if self.x[i][t] {
                    self.rowmult(i, p);
                }
            }
            if self.rand_global_phase {
                return result;
            }
            let g = self.gaussian();
            let perm_count = 2usize.pow(g as u32);
            let perm_count_min1 = perm_count - 1;
            let elem_count = self.qubit_count << 1;
            let nrm = (1.0 / (perm_count as f64)).sqrt();
            self.seed(g);
            let entry = self.get_basis_amp(nrm);
            let o_amp = clone.unwrap().get_amplitude(if entry.permutation & (1 << t) != 0 { entry.permutation } else { entry.permutation & !(1 << t) });
            if o_amp.norm() > std::f64::EPSILON {
                self.set_phase_offset(self.phase_offset + o_amp.arg() - entry.amplitude.arg());
                return result;
            }
            for t in 0..perm_count_min1 {
                let t2 = t ^ (t + 1);
                for i in 0..g {
                    if t2 & (1 << i) != 0 {
                        self.rowmult(elem_count, self.qubit_count + i);
                    }
                }
                let entry = self.get_basis_amp(nrm);
                let o_amp = clone.unwrap().get_amplitude(if entry.permutation & (1 << t) != 0 { entry.permutation } else { entry.permutation & !(1 << t) });
                if o_amp.norm() > std::f64::EPSILON {
                    self.set_phase_offset(self.phase_offset + o_amp.arg() - entry.amplitude.arg());
                    return result;
                }
            }
            return result;
        }
        let mut m = 0;
        while m < n {
            if self.x[m][t] {
                break;
            }
            m += 1;
        }
        if m >= n {
            return false;
        }
        self.rowcopy(elem_count, m + n);
        for i in m + 1..n {
            if self.x[i][t] {
                self.rowmult(elem_count, i + n);
            }
        }
        if do_force && (result != (self.r[elem_count] != 0)) {
            panic!("QStabilizer::ForceM() forced a measurement with 0 probability!");
        }
        self.r[elem_count] != 0
    }

    fn compose(&mut self, to_copy: &mut QStabilizer, start: usize) -> usize {
        if start > self.qubit_count {
            panic!("QStabilizer::Compose start index is out-of-bounds!");
        }
        to_copy.finish();
        self.finish();
        self.set_phase_offset(self.phase_offset + to_copy.phase_offset);
        let row_count = (self.qubit_count << 1) + 1;
        let length = to_copy.qubit_count;
        let n_qubit_count = self.qubit_count + length;
        let end_length = self.qubit_count - start;
        let second_start = self.qubit_count + start;
        let d_len = length << 1;
        let row = vec![false; length];
        for i in 0..row_count {
            self.x[i].splice(start..start, row.iter().cloned());
            self.z[i].splice(start..start, row.iter().cloned());
        }
        self.x.splice(second_start..second_start, to_copy.x[length..d_len].iter().cloned());
        self.z.splice(second_start..second_start, to_copy.z[length..d_len].iter().cloned());
        self.r.splice(second_start..second_start, to_copy.r[length..d_len].iter().cloned());
        for i in 0..length {
            let offset = second_start + i;
            self.x[offset].splice(..start, vec![false; start].iter().cloned());
            self.x[offset].splice(end_length.., vec![false; end_length].iter().cloned());
            self.z[offset].splice(..start, vec![false; start].iter().cloned());
            self.z[offset].splice(end_length.., vec![false; end_length].iter().cloned());
        }
        self.x.splice(start..start, to_copy.x[..length].iter().cloned());
        self.z.splice(start..start, to_copy.z[..length].iter().cloned());
        self.r.splice(start..start, to_copy.r[..length].iter().cloned());
        for i in 0..length {
            let offset = start + i;
            self.x[offset].splice(..start, vec![false; start].iter().cloned());
            self.x[offset].splice(end_length.., vec![false; end_length].iter().cloned());
            self.z[offset].splice(..start, vec![false; start].iter().cloned());
            self.z[offset].splice(end_length.., vec![false; end_length].iter().cloned());
        }
        self.set_qubit_count(n_qubit_count);
        start
    }

    fn decompose(&mut self, start: usize, length: usize) -> QStabilizer {
        let mut dest = QStabilizer::new(length, 0, false);
        self.decompose_dispose(start, length, &mut dest);
        dest
    }

    fn can_decompose_dispose(&mut self, start: usize, length: usize) -> bool {
        if start + length > self.qubit_count {
            panic!("QStabilizer::CanDecomposeDispose range is out-of-bounds!");
        }
        if self.qubit_count == 1 {
            return true;
        }
        self.finish();
        self.gaussian();
        let end = start + length;
        for i in 0..start {
            let i2 = i + self.qubit_count;
            for j in start..end {
                if self.x[i][j] || self.z[i][j] || self.x[i2][j] || self.z[i2][j] {
                    return false;
                }
            }
        }
        for i in end..self.qubit_count {
            let i2 = i + self.qubit_count;
            for j in start..end {
                if self.x[i][j] || self.z[i][j] || self.x[i2][j] || self.z[i2][j] {
                    return false;
                }
            }
        }
        for i in start..end {
            let i2 = i + self.qubit_count;
            for j in 0..start {
                if self.x[i][j] || self.z[i][j] || self.x[i2][j] || self.z[i2][j] {
                    return false;
                }
            }
            for j in end..self.qubit_count {
                if self.x[i][j] || self.z[i][j] || self.x[i2][j] || self.z[i2][j] {
                    return false;
                }
            }
        }
        true
    }

    fn decompose_dispose(&mut self, start: usize, length: usize, dest: &mut QStabilizer) {
        if start + length > self.qubit_count {
            panic!("QStabilizer::DecomposeDispose range is out-of-bounds!");
        }
        if length == 0 {
            return;
        }
        if dest.is_some() {
            dest.unwrap().dump();
        }
        self.finish();
        let amp_entry = if self.rand_global_phase || dest.is_some() {
            AmplitudeEntry(0, Complex::new(1.0, 0.0))
        } else {
            self.get_any_amplitude()
        };
        self.gaussian();
        let end = start + length;
        let n_qubit_count = self.qubit_count - length;
        let second_start = self.qubit_count + start;
        let second_end = self.qubit_count + end;
        if dest.is_some() {
            for i in 0..length {
                let j = start + i;
                dest.unwrap().x[i].splice(.., self.x[j][start..end].iter().cloned());
                dest.unwrap().z[i].splice(.., self.z[j][start..end].iter().cloned());
                let j = self.qubit_count + start + i;
                dest.unwrap().x[i + length].splice(.., self.x[j][start..end].iter().cloned());
                dest.unwrap().z[i + length].splice(.., self.z[j][start..end].iter().cloned());
            }
            let j = start;
            dest.unwrap().r.splice(.., self.r[j..j + length].iter().cloned());
            let j = self.qubit_count + start;
            dest.unwrap().r.splice(length.., self.r[j..j + length].iter().cloned());
        }
        self.x.splice(second_start..second_end, std::iter::empty());
        self.z.splice(second_start..second_end, std::iter::empty());
        self.r.splice(second_start..second_end, std::iter::empty());
        self.x.splice(start..end, std::iter::empty());
        self.z.splice(start..end, std::iter::empty());
        self.r.splice(start..end, std::iter::empty());
        self.set_qubit_count(n_qubit_count);
        let row_count = (self.qubit_count << 1) + 1;
        for i in 0..row_count {
            self.x[i].splice(start..end, std::iter::empty());
            self.z[i].splice(start..end, std::iter::empty());
        }
        if self.rand_global_phase || dest.is_some() {
            return;
        }
        let start_mask = (1 << start) - 1;
        let end_mask = ((1 << self.qubit_count) - 1) ^ ((1 << (start + length)) - 1);
        let n_perm = (amp_entry.permutation & start_mask) | ((amp_entry.permutation & end_mask) >> length);
        self.set_phase_offset(self.phase_offset + amp_entry.amplitude.arg() - self.get_amplitude(n_perm).arg());
    }

    fn prob(&mut self, qubit: usize) -> f64 {
        if self.is_separable_z(qubit) {
            return if self.m(qubit) { 1.0 } else { 0.0 };
        }
        0.5
    }

    fn is_separable_z(&self, target: bitLenInt) -> bool {
        let mut ret = true;
        self.par_for(|i| {
            if i == target {
                if self.x[i].iter().any(|&x| x) {
                    ret = false;
                }
            }
        });
        ret
    }

    fn is_separable_x(&self, target: bitLenInt) -> bool {
        let mut ret = true;
        self.par_for(|i| {
            if i == target {
                if self.z[i].iter().any(|&z| z) {
                    ret = false;
                }
            }
        });
        ret
    }

    fn is_separable_y(&self, target: bitLenInt) -> bool {
        let mut ret = true;
        self.par_for(|i| {
            if i == target {
                if self.x[i].iter().any(|&x| x) || self.z[i].iter().any(|&z| z) {
                    ret = false;
                }
            }
        });
        ret
    }

    fn is_separable(&self, target: bitLenInt) -> u8 {
        let mut ret = 0;
        self.par_for(|i| {
            if i == target {
                if self.x[i].iter().any(|&x| x) {
                    ret |= 1;
                }
                if self.z[i].iter().any(|&z| z) {
                    ret |= 2;
                }
            }
        });
        ret
    }

    fn compose(&mut self, to_copy: &QStabilizer, start: bitLenInt) -> bitLenInt {
        let length = to_copy.qubit_count;
        if length == 0 {
            return start;
        }
        if start > self.qubit_count {
            panic!("QStabilizer::Allocate() cannot start past end of register!");
        }
        if self.qubit_count == 0 {
            self.set_qubit_count(length);
            self.set_permutation(ZERO_BCI);
            return 0;
        }
        let mut n_qubits = QStabilizer::new(
            length,
            ZERO_BCI,
            self.rand_generator.clone(),
            CMPLX_DEFAULT_ARG,
            false,
            self.random_global_phase,
            false,
            -1,
            self.hardware_rand_generator.is_some(),
            false,
            REAL1_EPSILON,
            vec![],
            0,
            FP_NORM_EPSILON_F,
        );
        let ret = self.compose(&mut n_qubits, start);
        *self = n_qubits;
        ret
    }

    fn decompose(&mut self, start: bitLenInt, dest: &mut dyn QInterface) {
        let length = dest.get_qubit_count();
        self.decompose_dispose(start, length, dest);
    }

    fn decompose_qinterface(&mut self, start: bitLenInt, length: bitLenInt) -> Box<dyn QInterface> {
        let dest = QStabilizer::new(
            length,
            ZERO_BCI,
            self.rand_generator.clone(),
            CMPLX_DEFAULT_ARG,
            false,
            self.random_global_phase,
            false,
            -1,
            self.hardware_rand_generator.is_some(),
            false,
            REAL1_EPSILON,
            vec![],
            0,
            FP_NORM_EPSILON_F,
        );
        self.decompose_dispose(start, length, &mut *dest);
        dest
    }

    fn dispose(&mut self, start: bitLenInt, length: bitLenInt) {
        self.decompose_dispose(start, length, None);
    }

    fn can_decompose_dispose(&self, start: bitLenInt, length: bitLenInt) -> bool {
        let mut ret = true;
        self.par_for(|i| {
            if i >= start && i < start + length {
                if self.x[i].iter().any(|&x| x) || self.z[i].iter().any(|&z| z) {
                    ret = false;
                }
            }
        });
        ret
    }

    fn allocate(&mut self, start: bitLenInt, length: bitLenInt) -> bitLenInt {
        if length == 0 {
            return start;
        }
        if start > self.qubit_count {
            panic!("QStabilizer::Allocate() cannot start past end of register!");
        }
        if self.qubit_count == 0 {
            self.set_qubit_count(length);
            self.set_permutation(ZERO_BCI);
            return 0;
        }
        let mut n_qubits = QStabilizer::new(
            length,
            ZERO_BCI,
            self.rand_generator.clone(),
            CMPLX_DEFAULT_ARG,
            false,
            self.random_global_phase,
            false,
            -1,
            self.hardware_rand_generator.is_some(),
            false,
            REAL1_EPSILON,
            vec![],
            0,
            FP_NORM_EPSILON_F,
        );
        let ret = self.compose(&mut n_qubits, start);
        *self = n_qubits;
        ret
    }

    fn normalize_state(&mut self, nrm: real1, norm_thresh: real1, phase_arg: real1) {
        if !self.random_global_phase {
            self.set_phase_offset(self.phase_offset + phase_arg);
        }
    }

    fn update_running_norm(&mut self, norm_thresh: real1) {}

    fn sum_sqr_diff(&mut self, to_compare: &QStabilizer) -> real1 {
        self.approx_compare_helper(to_compare, TRYDECOMPOSE_EPSILON, true)
    }

    fn approx_compare(&mut self, to_compare: &QStabilizer, error_tol: real1) -> bool {
        error_tol >= self.approx_compare_helper(to_compare, error_tol, true)
    }

    fn global_phase_compare(&mut self, to_compare: &QStabilizer, error_tol: real1) -> bool {
        let this_amp_entry = self.get_any_amplitude();
        let arg_diff = (this_amp_entry.amplitude.arg() - to_compare.get_amplitude(this_amp_entry.permutation).arg()) / (2.0 * PI_R1);
        let arg_diff = arg_diff - arg_diff.floor();
        let arg_diff = if arg_diff > 0.5 { arg_diff - 1.0 } else { arg_diff };
        if FP_NORM_EPSILON >= arg_diff.abs() {
            return false;
        }
        error_tol >= self.approx_compare_helper(to_compare, error_tol, true)
    }

    fn prob(&mut self, qubit: bitLenInt) -> real1 {
        let mut ret = ZERO_R1;
        self.par_for(|i| {
            if i == qubit {
                let x = self.x[i][0];
                let z = self.z[i][0];
                if x && !z {
                    ret += 1.0;
                }
                if x && z {
                    ret += 1.0;
                }
                if !x && z {
                    ret += 1.0;
                }
            }
        });
        ret
    }

    fn mtrx(&self, mtrx: &[Complex], target: usize) {
        if mtrx[1].norm() == 0.0 && mtrx[2].norm() == 0.0 {
            self.phase(mtrx[0], mtrx[3], target);
            return;
        }
        if mtrx[0].norm() == 0.0 && mtrx[3].norm() == 0.0 {
            self.invert(mtrx[1], mtrx[2], target);
            return;
        }
        if mtrx[0] == mtrx[1] && mtrx[0] == mtrx[2] && mtrx[0] == -mtrx[3] {
            self.h(target);
            self.set_phase_offset(self.phase_offset + mtrx[0].arg());
            return;
        }
        if mtrx[0] == mtrx[1] && mtrx[0] == -mtrx[2] && mtrx[0] == mtrx[3] {
            self.x(target);
            self.h(target);
            self.set_phase_offset(self.phase_offset + mtrx[0].arg());
            return;
        }
        if mtrx[0] == -mtrx[1] && mtrx[0] == mtrx[2] && mtrx[0] == mtrx[3] {
            self.h(target);
            self.x(target);
            self.set_phase_offset(self.phase_offset + mtrx[0].arg());
            return;
        }
        if mtrx[0] == -mtrx[1] && mtrx[0] == -mtrx[2] && mtrx[0] == -mtrx[3] {
            self.x(target);
            self.h(target);
            self.x(target);
            self.set_phase_offset(self.phase_offset + mtrx[0].arg() + PI);
            return;
        }
        if mtrx[0] == mtrx[1] && mtrx[0] == -I * mtrx[2] && mtrx[0] == I * mtrx[3] {
            self.h(target);
            self.s(target);
            self.set_phase_offset(self.phase_offset + mtrx[0].arg());
            return;
        }
        if mtrx[0] == mtrx[1] && mtrx[0] == I * mtrx[2] && mtrx[0] == -I * mtrx[3] {
            self.h(target);
            self.is(target);
            self.set_phase_offset(self.phase_offset + mtrx[0].arg());
            return;
        }
        if mtrx[0] == -mtrx[1] && mtrx[0] == I * mtrx[2] && mtrx[0] == I * mtrx[3] {
            self.h(target);
            self.x(target);
            self.is(target);
            self.set_phase_offset(self.phase_offset + mtrx[0].arg());
            return;
        }
        if mtrx[0] == -mtrx[1] && mtrx[0] == -I * mtrx[2] && mtrx[0] == -I * mtrx[3] {
            self.h(target);
            self.x(target);
            self.s(target);
            self.set_phase_offset(self.phase_offset + mtrx[0].arg());
            return;
        }
        if mtrx[0] == I * mtrx[1] && mtrx[0] == mtrx[2] && mtrx[0] == -I * mtrx[3] {
            self.is(target);
            self.h(target);
            self.set_phase_offset(self.phase_offset + mtrx[0].arg());
            return;
        }
        if mtrx[0] == -I * mtrx[1] && mtrx[0] == mtrx[2] && mtrx[0] == I * mtrx[3] {
            self.s(target);
            self.h(target);
            self.set_phase_offset(self.phase_offset + mtrx[0].arg());
            return;
        }
        if mtrx[0] == -I * mtrx[1] && mtrx[0] == -mtrx[2] && mtrx[0] == -I * mtrx[3] {
            self.is(target);
            self.h(target);
            self.x(target);
            self.z(target);
            self.set_phase_offset(self.phase_offset + mtrx[0].arg());
            return;
        }
        if mtrx[0] == I * mtrx[1] && mtrx[0] == -mtrx[2] && mtrx[0] == I * mtrx[3] {
            self.s(target);
            self.h(target);
            self.x(target);
            self.z(target);
            self.set_phase_offset(self.phase_offset + mtrx[0].arg());
            return;
        }
        if mtrx[0] == I * mtrx[1] && mtrx[0] == I * mtrx[2] && mtrx[0] == mtrx[3] {
            self.is(target);
            self.h(target);
            self.is(target);
            self.set_phase_offset(self.phase_offset + mtrx[0].arg());
            return;
        }
        if mtrx[0] == -I * mtrx[1] && mtrx[0] == -I * mtrx[2] && mtrx[0] == mtrx[3] {
            self.s(target);
            self.h(target);
            self.s(target);
            self.set_phase_offset(self.phase_offset + mtrx[0].arg());
            return;
        }
        if mtrx[0] == I * mtrx[1] && mtrx[0] == -I * mtrx[2] && mtrx[0] == -mtrx[3] {
            self.is(target);
            self.h(target);
            self.s(target);
            self.set_phase_offset(self.phase_offset + mtrx[0].arg());
            return;
        }
        if mtrx[0] == -I * mtrx[1] && mtrx[0] == I * mtrx[2] && mtrx[0] == -mtrx[3] {
            self.s(target);
            self.h(target);
            self.is(target);
            self.set_phase_offset(self.phase_offset + mtrx[0].arg());
            return;
        }
        panic!("QStabilizer::Mtrx() not implemented for non-Clifford/Pauli cases!");
    }

    fn phase(&self, top_left: Complex, bottom_right: Complex, target: usize) {
        if top_left == bottom_right {
            self.set_phase_offset(self.phase_offset + top_left.arg());
            return;
        }
        if top_left == -bottom_right {
            self.z(target);
            self.set_phase_offset(self.phase_offset + top_left.arg());
            return;
        }
        if top_left == -I * bottom_right {
            self.s(target);
            self.set_phase_offset(self.phase_offset + top_left.arg());
            return;
        }
        if top_left == I * bottom_right {
            self.is(target);
            self.set_phase_offset(self.phase_offset + top_left.arg());
            return;
        }
        if self.is_separable_z(target) {
            if self.m(target) {
                self.phase(bottom_right, bottom_right, target);
            } else {
                self.phase(top_left, top_left, target);
            }
            return;
        }
        panic!("QStabilizer::Phase() not implemented for non-Clifford/Pauli cases!");
    }

    fn invert(&self, top_right: Complex, bottom_left: Complex, target: usize) {
        if top_right == bottom_left {
            self.x(target);
            self.set_phase_offset(self.phase_offset + top_right.arg());
            return;
        }
        if top_right == -bottom_left {
            self.y(target);
            self.set_phase_offset(self.phase_offset + top_right.arg() + PI / 2.0);
            return;
        }
        if top_right == -I * bottom_left {
            self.x(target);
            self.s(target);
            self.set_phase_offset(self.phase_offset + top_right.arg());
            return;
        }
        if top_right == I * bottom_left {
            self.x(target);
            self.is(target);
            self.set_phase_offset(self.phase_offset + top_right.arg());
            return;
        }
        if self.is_separable_z(target) {
            if self.m(target) {
                self.invert(top_right, top_right, target);
            } else {
                self.invert(bottom_left, bottom_left, target);
            }
            return;
        }
        panic!("QStabilizer::Invert() not implemented for non-Clifford/Pauli cases!");
    }

    fn mc_phase(&self, controls: &[usize], top_left: Complex, bottom_right: Complex, target: usize) {
        if top_left.norm() == 0.0 && bottom_right.norm() == 0.0 {
            return;
        }
        if controls.is_empty() {
            self.phase(top_left, bottom_right, target);
            return;
        }
        if controls.len() > 1 {
            panic!("QStabilizer::MCPhase() not implemented for non-Clifford/Pauli cases! (Too many controls)");
        }
        let control = controls[0];
        if top_left == Complex::one() {
            if bottom_right == Complex::one() {
                return;
            } else if bottom_right == -Complex::one() {
                self.cz(control, target);
                return;
            }
        } else if top_left == -Complex::one() {
            if bottom_right == Complex::one() {
                self.cnot(control, target);
                self.cz(control, target);
                self.cnot(control, target);
                return;
            } else if bottom_right == -Complex::one() {
                self.cz(control, target);
                self.cnot(control, target);
                self.cz(control, target);
                self.cnot(control, target);
                return;
            }
        } else if top_left == Complex::i() {
            if bottom_right == Complex::i() {
                self.cz(control, target);
                self.cy(control, target);
                self.cnot(control, target);
                return;
            } else if bottom_right == -Complex::i() {
                self.cy(control, target);
                self.cnot(control, target);
                return;
            }
        } else if top_left == -Complex::i() {
            if bottom_right == Complex::i() {
                self.cnot(control, target);
                self.cy(control, target);
                return;
            } else if bottom_right == -Complex::i() {
                self.cy(control, target);
                self.cz(control, target);
                self.cnot(control, target);
                return;
            }
        }
        panic!("QStabilizer::MCPhase() not implemented for non-Clifford/Pauli cases! (Non-Clifford/Pauli target payload)");
    }

    fn mac_phase(&self, controls: &[usize], top_left: Complex, bottom_right: Complex, target: usize) {
        if top_left.norm() == 0.0 && bottom_right.norm() == 0.0 {
            return;
        }
        if controls.is_empty() {
            self.phase(top_left, bottom_right, target);
            return;
        }
        if controls.len() > 1 {
            panic!("QStabilizer::MACPhase() not implemented for non-Clifford/Pauli cases! (Too many controls)");
        }
        let control = controls[0];
        if top_left == Complex::one() {
            if bottom_right == Complex::one() {
                return;
            } else if bottom_right == -Complex::one() {
                self.anti_cz(control, target);
                return;
            }
        } else if top_left == -Complex::one() {
            if bottom_right == Complex::one() {
                self.anti_cnot(control, target);
                self.anti_cz(control, target);
                self.anti_cnot(control, target);
                return;
            } else if bottom_right == -Complex::one() {
                self.anti_cz(control, target);
                self.anti_cnot(control, target);
                self.anti_cz(control, target);
                self.anti_cnot(control, target);
                return;
            }
        } else if top_left == Complex::i() {
            if bottom_right == Complex::i() {
                self.anti_cz(control, target);
                self.anti_cy(control, target);
                self.anti_cnot(control, target);
                return;
            } else if bottom_right == -Complex::i() {
                self.anti_cy(control, target);
                self.anti_cnot(control, target);
                return;
            }
        } else if top_left == -Complex::i() {
            if bottom_right == Complex::i() {
                self.anti_cnot(control, target);
                self.anti_cy(control, target);
                return;
            } else if bottom_right == -Complex::i() {
                self.anti_cy(control, target);
                self.anti_cz(control, target);
                return;
            }
        }
        panic!("QStabilizer::MACPhase() not implemented for non-Clifford/Pauli cases! (Non-Clifford/Pauli target payload)");
    }

    fn mc_invert(&self, controls: &[usize], top_right: Complex, bottom_left: Complex, target: usize) {
        if controls.is_empty() {
            self.invert(top_right, bottom_left, target);
            return;
        }
        if controls.len() > 1 {
            panic!("QStabilizer::MCInvert() not implemented for non-Clifford/Pauli cases! (Too many controls)");
        }
        let control = controls[0];
        if top_right == Complex::one() {
            if bottom_left == Complex::one() {
                self.cnot(control, target);
                return;
            } else if bottom_left == -Complex::one() {
                self.cnot(control, target);
                self.cz(control, target);
                return;
            }
        } else if top_right == -Complex::one() {
            if bottom_left == Complex::one() {
                self.cz(control, target);
                self.cnot(control, target);
                return;
            } else if bottom_left == -Complex::one() {
                self.cz(control, target);
                self.cnot(control, target);
                self.cz(control, target);
                return;
            }
        } else if top_right == Complex::i() {
            if bottom_left == Complex::i() {
                self.cz(control, target);
                self.cy(control, target);
                return;
            } else if bottom_left == -Complex::i() {
                self.cz(control, target);
                self.cy(control, target);
                self.cz(control, target);
                return;
            }
        } else if top_right == -Complex::i() {
            if bottom_left == Complex::i() {
                self.cy(control, target);
                return;
            } else if bottom_left == -Complex::i() {
                self.cy(control, target);
                self.cz(control, target);
                return;
            }
        }
        panic!("QStabilizer::MCInvert() not implemented for non-Clifford/Pauli cases! (Non-Clifford/Pauli target payload)");
    }

    fn mac_invert(&self, controls: &[usize], top_right: Complex, bottom_left: Complex, target: usize) {
        if controls.is_empty() {
            self.invert(top_right, bottom_left, target);
            return;
        }
        if controls.len() > 1 {
            panic!("QStabilizer::MACInvert() not implemented for non-Clifford/Pauli cases! (Too many controls)");
        }
        let control = controls[0];
        if top_right == Complex::one() {
            if bottom_left == Complex::one() {
                self.anti_cnot(control, target);
                return;
            } else if bottom_left == -Complex::one() {
                self.anti_cnot(control, target);
                self.anti_cz(control, target);
                return;
            }
        } else if top_right == -Complex::one() {
            if bottom_left == Complex::one() {
                self.anti_cz(control, target);
                self.anti_cnot(control, target);
                return;
            } else if bottom_left == -Complex::one() {
                self.anti_cz(control, target);
                self.anti_cnot(control, target);
                self.anti_cz(control, target);
                return;
            }
        } else if top_right == Complex::i() {
            if bottom_left == Complex::i() {
                self.anti_cz(control, target);
                self.anti_cy(control, target);
                return;
            } else if bottom_left == -Complex::i() {
                self.anti_cz(control, target);
                self.anti_cy(control, target);
                self.anti_cz(control, target);
                return;
            }
        } else if top_right == -Complex::i() {
            if bottom_left == Complex::i() {
                self.anti_cy(control, target);
                return;
            } else if bottom_left == -Complex::i() {
                self.anti_cy(control, target);
                self.anti_cz(control, target);
                return;
            }
        }
        panic!("QStabilizer::MACInvert() not implemented for non-Clifford/Pauli cases! (Non-Clifford/Pauli target payload)");
    }

    fn f_sim(&self, theta: f64, phi: f64, qubit1: usize, qubit2: usize) {
        let controls = vec![qubit1];
        let sin_theta = theta.sin();
        if sin_theta == 0.0 {
            self.mc_phase(&controls, Complex::one(), Complex::new(0.0, phi), qubit2);
            return;
        }
        if sin_theta == 1.0 {
            self.iswap(qubit1, qubit2);
            self.mc_phase(&controls, Complex::one(), Complex::new(0.0, phi), qubit2);
            return;
        }
        panic!("QStabilizer::FSim() not implemented for non-Clifford/Pauli cases!");
    }

    fn try_separate(&self, qubits: &[usize], ignored: f64) -> bool {
        let mut l_qubits = qubits.to_vec();
        l_qubits.sort();
        for (i, &qubit) in l_qubits.iter().enumerate() {
            self.swap(qubit, i);
        }
        let to_ret = self.can_decompose_dispose(0, l_qubits.len());
        let last = l_qubits.len() - 1;
        for (i, &qubit) in l_qubits.iter().enumerate() {
            self.swap(qubit, last - i);
        }
        to_ret
    }

    fn try_separate(&self, qubit: bitLenInt) -> bool {
        self.can_decompose_dispose(qubit, 1)
    }

    fn try_separate_2(&mut self, qubit1: bitLenInt, qubit2: bitLenInt) -> bool {
        if qubit2 < qubit1 {
            std::mem::swap(&mut qubit1, &mut qubit2);
        }
        self.swap(qubit1, 0);
        self.swap(qubit2, 1);
        let to_ret = self.can_decompose_dispose(0, 2);
        self.swap(qubit2, 1);
        self.swap(qubit1, 0);
        to_ret
    }
}

trait QInterface {
    fn get_qubit_count(&self) -> bitLenInt;
    fn set_qubit_count(&mut self, n: bitLenInt);
    fn get_amplitude(&self, perm: bitCapInt) -> complex;
    fn set_amplitude(&mut self, perm: bitCapInt, amp: complex);
    fn apply(&mut self);
    fn dump(&self);
    fn finish(&mut self);
    fn rand(&mut self) -> bool;
    fn set_random_seed(&mut self, seed: u32);
    fn set_device(&mut self, d_id: i64);
    fn clear(&mut self);
    fn clone_empty(&self) -> Box<dyn QInterface>;
    fn is_clifford(&self) -> bool;
    fn is_clifford_qubit(&self, qubit: bitLenInt) -> bool;
    fn get_qubit_count(&self) -> bitLenInt;
    fn get_max_q_power(&self) -> bitCapInt;
    fn reset_phase_offset(&mut self);
    fn get_phase_offset(&self) -> complex;
    fn set_permutation(&mut self, perm: bitCapInt, phase_fac: complex);
    fn set_random_seed(&mut self, seed: u32);
    fn set_device(&mut self, d_id: i64);
    fn rand(&mut self) -> bool;
    fn clear(&mut self);
    fn rowcopy(&mut self, i: bitLenInt, k: bitLenInt);
    fn rowswap(&mut self, i: bitLenInt, k: bitLenInt);
    fn rowset(&mut self, i: bitLenInt, b: bitLenInt);
    fn rowmult(&mut self, i: bitLenInt, k: bitLenInt);
    fn set_quantum_state(&mut self, input_state: &[complex]);
    fn set_amplitude(&mut self, perm: bitCapInt, amp: complex);
    fn set_rand_global_phase(&mut self, is_rand: bool);
    fn cnot(&mut self, control: bitLenInt, target: bitLenInt);
    fn cy(&mut self, control: bitLenInt, target: bitLenInt);
    fn cz(&mut self, control: bitLenInt, target: bitLenInt);
    fn anti_cnot(&mut self, control: bitLenInt, target: bitLenInt);
    fn anti_cy(&mut self, control: bitLenInt, target: bitLenInt);
    fn anti_cz(&mut self, control: bitLenInt, target: bitLenInt);
    fn h(&mut self, qubit_index: bitLenInt);
    fn x(&mut self, qubit_index: bitLenInt);
    fn y(&mut self, qubit_index: bitLenInt);
    fn z(&mut self, qubit_index: bitLenInt);
    fn s(&mut self, qubit_index: bitLenInt);
    fn is(&mut self, qubit_index: bitLenInt);
    fn swap(&mut self, qubit_index1: bitLenInt, qubit_index2: bitLenInt);
    fn iswap(&mut self, qubit_index1: bitLenInt, qubit_index2: bitLenInt);
    fn iiswap(&mut self, qubit_index1: bitLenInt, qubit_index2: bitLenInt);
    fn force_m(&mut self, t: bitLenInt, result: bool, do_force: bool, do_apply: bool) -> bool;
    fn is_separable_z(&self, target: bitLenInt) -> bool;
    fn is_separable_x(&self, target: bitLenInt) -> bool;
    fn is_separable_y(&self, target: bitLenInt) -> bool;
    fn is_separable(&self, target: bitLenInt) -> u8;
    fn compose(&mut self, to_copy: &QStabilizer, start: bitLenInt) -> bitLenInt;
    fn decompose(&mut self, start: bitLenInt, dest: &mut dyn QInterface);
    fn decompose_qinterface(&mut self, start: bitLenInt, length: bitLenInt) -> Box<dyn QInterface>;
    fn dispose(&mut self, start: bitLenInt, length: bitLenInt);
    fn can_decompose_dispose(&self, start: bitLenInt, length: bitLenInt) -> bool;
    fn allocate(&mut self, start: bitLenInt, length: bitLenInt) -> bitLenInt;
    fn normalize_state(&mut self, nrm: real1, norm_thresh: real1, phase_arg: real1);
    fn update_running_norm(&mut self, norm_thresh: real1);
    fn sum_sqr_diff(&mut self, to_compare: &QStabilizer) -> real1;
    fn approx_compare(&mut self, to_compare: &QStabilizer, error_tol: real1) -> bool;
    fn global_phase_compare(&mut self, to_compare: &QStabilizer, error_tol: real1) -> bool;
    fn prob(&mut self, qubit: bitLenInt) -> real1;
    fn mtrx(&mut self, mtrx: &[complex], target: bitLenInt);
    fn phase(&mut self, top_left: complex, bottom_right: complex, target: bitLenInt);
    fn invert(&mut self, top_right: complex, bottom_left: complex, target: bitLenInt);
    fn mcphase(&mut self, controls: &[bitLenInt], top_left: complex, bottom_right: complex, target: bitLenInt);
    fn macphase(&mut self, controls: &[bitLenInt], top_left: complex, bottom_right: complex, target: bitLenInt);
    fn mcinvert(&mut self, controls: &[bitLenInt], top_right: complex, bottom_left: complex, target: bitLenInt);
    fn macinvert(&mut self, controls: &[bitLenInt], top_right: complex, bottom_left: complex, target: bitLenInt);
    fn mcmtrx(&mut self, controls: &[bitLenInt], mtrx: &[complex], target: bitLenInt);
    fn macmtrx(&mut self, controls: &[bitLenInt], mtrx: &[complex], target: bitLenInt);
    fn try_separate(&self, qubit: bitLenInt) -> bool;
    fn try_separate_2(&mut self, qubit1: bitLenInt, qubit2: bitLenInt) -> bool;
}

impl QInterface for QStabilizer {
    fn get_qubit_count(&self) -> bitLenInt {
        self.qubit_count
    }

    fn set_qubit_count(&mut self, n: bitLenInt) {
        self.qubit_count = n;
    }

   fn get_amplitude(&self, perm: bitCapInt) -> Complex<f64> {
        self.finish();
        let g = self.gaussian();
        let perm_count = 2u64.pow(g);
        let mut perm_count_min1 = perm_count;
        perm_count_min1 -= 1;
        let elem_count = qubit_count << 1;
        let nrm = (1.0 / (perm_count as f64)).sqrt();
        self.seed(g);
        let entry = self.get_basis_amp(nrm);
        if entry.permutation == perm {
            return entry.amplitude;
        }
        for t in 0..perm_count_min1 {
            let t2 = t ^ (t + 1);
            for i in 0..g {
                if t2 & (1 << i) != 0 {
                    self.rowmult(elem_count, qubit_count + i);
                }
            }
            let entry = self.get_basis_amp(nrm);
            if entry.permutation == perm {
                return entry.amplitude;
            }
        }
        Complex::new(0.0, 0.0)
    }

    fn set_amplitude(&mut self, perm: bitCapInt, amp: complex) {
        self.set_permutation(perm, amp);
    }

    fn apply(&mut self) {}

    fn dump(&self) {}

    fn finish(&mut self) {}

    fn rand(&mut self) -> bool {
        if let Some(rand_generator) = &mut self.rand_generator {
            if self.raw_rand_bools_remaining == 0 {
                self.raw_rand_bools = rand_generator.next_u32();
                self.raw_rand_bools_remaining = 32;
            }
            self.raw_rand_bools_remaining -= 1;
            (self.raw_rand_bools >> self.raw_rand_bools_remaining) & 1 == 1
        } else {
            rand::random()
        }
    }

    fn set_random_seed(&mut self, seed: u32) {
        if let Some(rand_generator) = &mut self.rand_generator {
            rand_generator.seed(seed);
        }
    }

    fn set_device(&mut self, _d_id: i64) {}

    fn clear(&mut self) {
        self.x.iter_mut().for_each(|x| x.iter_mut().for_each(|x| *x = false));
        self.z.iter_mut().for_each(|z| z.iter_mut().for_each(|z| *z = false));
        self.r.iter_mut().for_each(|r| *r = 0);
        self.phase_offset = ZERO_R1;
        self.qubit_count = 0;
    }

    fn clone_empty(&self) -> Box<dyn QInterface> {
        Box::new(QStabilizer::new(
            self.qubit_count,
            ZERO_BCI,
            self.rand_generator.clone(),
            CMPLX_DEFAULT_ARG,
            false,
            self.random_global_phase,
            false,
            -1,
            self.hardware_rand_generator.is_some(),
            false,
            REAL1_EPSILON,
            vec![],
            0,
            FP_NORM_EPSILON_F,
        ))
    }

    fn is_clifford(&self) -> bool {
        true
    }

    fn is_clifford_qubit(&self, qubit: bitLenInt) -> bool {
        true
    }

    fn get_qubit_count(&self) -> bitLenInt {
        self.qubit_count
    }

    fn get_max_q_power(&self) -> bitCapInt {
        2u64.pow(self.qubit_count as u32)
    }

    fn reset_phase_offset(&mut self) {
        self.phase_offset = ZERO_R1;
    }

    fn get_phase_offset(&self) -> complex {
        complex::from_polar(&ONE_R1, &self.phase_offset)
    }

    fn set_permutation(&mut self, perm: bitCapInt, phase_fac: complex) {
        self.set_permutation(perm, phase_fac);
    }

    fn set_random_seed(&mut self, seed: u32) {
        self.set_random_seed(seed);
    }

    fn set_device(&mut self, _d_id: i64) {}

    fn rand(&mut self) -> bool {
        self.rand()
    }

    fn clear(&mut self) {
        self.clear();
    }

    fn rowcopy(&mut self, i: bitLenInt, k: bitLenInt) {
        self.rowcopy(i, k);
    }

    fn rowswap(&mut self, i: bitLenInt, k: bitLenInt) {
        self.rowswap(i, k);
    }

    fn rowset(&mut self, i: bitLenInt, b: bitLenInt) {
        self.rowset(i, b);
    }

    fn rowmult(&mut self, i: bitLenInt, k: bitLenInt) {
        self.rowmult(i, k);
    }

    fn set_quantum_state(&mut self, input_state: &[complex]) {
        self.set_quantum_state(input_state);
    }

    fn set_amplitude(&mut self, perm: bitCapInt, amp: complex) {
        self.set_amplitude(perm, amp);
    }

    fn set_rand_global_phase(&mut self, is_rand: bool) {
        self.set_rand_global_phase(is_rand);
    }

    fn cnot(&mut self, control: bitLenInt, target: bitLenInt) {
        self.cnot(control, target);
    }

    fn cy(&mut self, control: bitLenInt, target: bitLenInt) {
        self.cy(control, target);
    }

    fn cz(&mut self, control: bitLenInt, target: bitLenInt) {
        self.cz(control, target);
    }

    fn anti_cnot(&mut self, control: bitLenInt, target: bitLenInt) {
        self.anti_cnot(control, target);
    }

    fn anti_cy(&mut self, control: bitLenInt, target: bitLenInt) {
        self.anti_cy(control, target);
    }

    fn anti_cz(&mut self, control: bitLenInt, target: bitLenInt) {
        self.anti_cz(control, target);
    }

    fn h(&mut self, qubit_index: bitLenInt) {
        self.h(qubit_index);
    }

    fn x(&mut self, qubit_index: bitLenInt) {
        self.x(qubit_index);
    }

    fn y(&mut self, qubit_index: bitLenInt) {
        self.y(qubit

