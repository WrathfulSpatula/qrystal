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
        nrm: real1,
        bit_powers: &[bitCapInt],
        perms: &[bitCapInt],
        offset: bitCapInt,
    ) -> real1 {
        let entry = self.get_basis_amp(nrm);
        let mut ret_index = ZERO_BCI;
        for (b, &bit_power) in bit_powers.iter().enumerate() {
            if (entry.permutation & bit_power) != 0 {
                ret_index += if (perms[(b << 1) + 1] & entry.permutation) != 0 {
                    perms[(b << 1) + 1]
                } else {
                    perms[b << 1]
                };
            }
        }
        (offset + ret_index).to_f64().unwrap() * entry.amplitude.norm()
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

    fn set_quantum_state(&mut self, input_state: &[complex]) {
        let mut perm = ZERO_BCI;
        for (i, &state) in input_state.iter().enumerate() {
            if state.norm() > REAL1_EPSILON {
                perm |= 1 << i;
            }
        }
        self.set_permutation(perm, CMPLX_DEFAULT_ARG);
    }

    fn set_rand_global_phase(&mut self, is_rand: bool) {
        self.random_global_phase = is_rand;
    }

    fn cnot(&mut self, control: bitLenInt, target: bitLenInt) {
        self.par_for(|i| {
            if i == control {
                self.x[i].iter_mut().for_each(|x| *x ^= self.x[target]);
                self.z[i].iter_mut().for_each(|z| *z ^= self.z[target]);
            }
        });
    }

    fn cy(&mut self, control: bitLenInt, target: bitLenInt) {
        self.par_for(|i| {
            if i == control {
                self.x[i].iter_mut().for_each(|x| *x ^= self.x[target]);
                self.z[i].iter_mut().for_each(|z| *z ^= self.z[target]);
            } else if i == target {
                self.z[i].iter_mut().for_each(|z| *z ^= self.x[control]);
            }
        });
    }

    fn cz(&mut self, control: bitLenInt, target: bitLenInt) {
        self.par_for(|i| {
            if i == control {
                self.x[i].iter_mut().for_each(|x| *x ^= self.x[target]);
                self.z[i].iter_mut().for_each(|z| *z ^= self.z[target]);
            } else if i == target {
                self.x[i].iter_mut().for_each(|x| *x ^= self.z[control]);
            }
        });
    }

    fn anti_cnot(&mut self, control: bitLenInt, target: bitLenInt) {
        self.par_for(|i| {
            if i == control {
                self.x[i].iter_mut().for_each(|x| *x ^= self.x[target]);
                self.z[i].iter_mut().for_each(|z| *z ^= self.z[target]);
            }
        });
    }

    fn anti_cy(&mut self, control: bitLenInt, target: bitLenInt) {
        self.par_for(|i| {
            if i == control {
                self.x[i].iter_mut().for_each(|x| *x ^= self.x[target]);
                self.z[i].iter_mut().for_each(|z| *z ^= self.z[target]);
            } else if i == target {
                self.z[i].iter_mut().for_each(|z| *z ^= self.x[control]);
            }
        });
    }

    fn anti_cz(&mut self, control: bitLenInt, target: bitLenInt) {
        self.par_for(|i| {
            if i == control {
                self.x[i].iter_mut().for_each(|x| *x ^= self.x[target]);
                self.z[i].iter_mut().for_each(|z| *z ^= self.z[target]);
            } else if i == target {
                self.x[i].iter_mut().for_each(|x| *x ^= self.z[control]);
            }
        });
    }

    fn h(&mut self, qubit_index: bitLenInt) {
        self.par_for(|i| {
            if i == qubit_index {
                self.x[i].iter_mut().for_each(|x| *x ^= self.z[i]);
                self.z[i].iter_mut().for_each(|z| *z ^= self.x[i]);
            }
        });
    }

    fn x(&mut self, qubit_index: bitLenInt) {
        self.par_for(|i| {
            if i == qubit_index {
                self.x[i].iter_mut().for_each(|x| *x ^= true);
            }
        });
    }

    fn y(&mut self, qubit_index: bitLenInt) {
        self.par_for(|i| {
            if i == qubit_index {
                self.x[i].iter_mut().for_each(|x| *x ^= true);
                self.z[i].iter_mut().for_each(|z| *z ^= true);
            }
        });
    }

    fn z(&mut self, qubit_index: bitLenInt) {
        self.par_for(|i| {
            if i == qubit_index {
                self.z[i].iter_mut().for_each(|z| *z ^= true);
            }
        });
    }

    fn s(&mut self, qubit_index: bitLenInt) {
        self.par_for(|i| {
            if i == qubit_index {
                self.z[i].iter_mut().for_each(|z| *z ^= self.x[i]);
            }
        });
    }

    fn is(&mut self, qubit_index: bitLenInt) {
        self.par_for(|i| {
            if i == qubit_index {
                self.x[i].iter_mut().for_each(|x| *x ^= self.z[i]);
            }
        });
    }

    fn swap(&mut self, qubit_index1: bitLenInt, qubit_index2: bitLenInt) {
        self.par_for(|i| {
            if i == qubit_index1 {
                self.x[i].iter_mut().for_each(|x| *x ^= self.x[qubit_index2]);
                self.z[i].iter_mut().for_each(|z| *z ^= self.z[qubit_index2]);
            } else if i == qubit_index2 {
                self.x[i].iter_mut().for_each(|x| *x ^= self.x[qubit_index1]);
                self.z[i].iter_mut().for_each(|z| *z ^= self.z[qubit_index1]);
            }
        });
    }

    fn iswap(&mut self, qubit_index1: bitLenInt, qubit_index2: bitLenInt) {
        self.par_for(|i| {
            if i == qubit_index1 {
                self.x[i].iter_mut().for_each(|x| *x ^= self.x[qubit_index2]);
                self.z[i].iter_mut().for_each(|z| *z ^= self.z[qubit_index2]);
            } else if i == qubit_index2 {
                self.x[i].iter_mut().for_each(|x| *x ^= self.z[qubit_index1]);
                self.z[i].iter_mut().for_each(|z| *z ^= self.x[qubit_index1]);
            }
        });
    }

    fn iiswap(&mut self, qubit_index1: bitLenInt, qubit_index2: bitLenInt) {
        self.par_for(|i| {
            if i == qubit_index1 {
                self.x[i].iter_mut().for_each(|x| *x ^= self.z[qubit_index2]);
                self.z[i].iter_mut().for_each(|z| *z ^= self.x[qubit_index2]);
            } else if i == qubit_index2 {
                self.x[i].iter_mut().for_each(|x| *x ^= self.z[qubit_index1]);
                self.z[i].iter_mut().for_each(|z| *z ^= self.x[qubit_index1]);
            }
        });
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

    fn mtrx(&mut self, mtrx: &[complex], target: bitLenInt) {
        if IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2]) {
            self.mcphase(&[], mtrx[0], mtrx[3], target);
            return;
        }
        if IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3]) {
            self.mcinvert(&[], mtrx[1], mtrx[2], target);
            return;
        }
        panic!("QStabilizer::MCMtrx() not implemented for non-Clifford/Pauli cases!");
    }

    fn phase(&mut self, top_left: complex, bottom_right: complex, target: bitLenInt) {
        self.par_for(|i| {
            if i == target {
                self.x[i].iter_mut().for_each(|x| *x ^= self.z[i]);
                self.z[i].iter_mut().for_each(|z| *z ^= self.x[i]);
            }
        });
    }

    fn invert(&mut self, top_right: complex, bottom_left: complex, target: bitLenInt) {
        self.par_for(|i| {
            if i == target {
                self.x[i].iter_mut().for_each(|x| *x ^= self.x[i]);
                self.z[i].iter_mut().for_each(|z| *z ^= self.z[i]);
            }
        });
    }

    fn mcphase(&mut self, controls: &[bitLenInt], top_left: complex, bottom_right: complex, target: bitLenInt) {
        self.par_for(|i| {
            if i == target {
                self.x[i].iter_mut().for_each(|x| *x ^= self.z[i]);
                self.z[i].iter_mut().for_each(|z| *z ^= self.x[i]);
            }
        });
    }

    fn macphase(&mut self, controls: &[bitLenInt], top_left: complex, bottom_right: complex, target: bitLenInt) {
        self.par_for(|i| {
            if i == target {
                self.x[i].iter_mut().for_each(|x| *x ^= self.z[i]);
                self.z[i].iter_mut().for_each(|z| *z ^= self.x[i]);
            }
        });
    }

    fn mcinvert(&mut self, controls: &[bitLenInt], top_right: complex, bottom_left: complex, target: bitLenInt) {
        self.par_for(|i| {
            if i == target {
                self.x[i].iter_mut().for_each(|x| *x ^= self.x[i]);
                self.z[i].iter_mut().for_each(|z| *z ^= self.z[i]);
            }
        });
    }

    fn macinvert(&mut self, controls: &[bitLenInt], top_right: complex, bottom_left: complex, target: bitLenInt) {
        self.par_for(|i| {
            if i == target {
                self.x[i].iter_mut().for_each(|x| *x ^= self.x[i]);
                self.z[i].iter_mut().for_each(|z| *z ^= self.z[i]);
            }
        });
    }

    fn mcmtrx(&mut self, controls: &[bitLenInt], mtrx: &[complex], target: bitLenInt) {
        if IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2]) {
            self.mcphase(controls, mtrx[0], mtrx[3], target);
            return;
        }
        if IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3]) {
            self.mcinvert(controls, mtrx[1], mtrx[2], target);
            return;
        }
        panic!("QStabilizer::MCMtrx() not implemented for non-Clifford/Pauli cases!");
    }

    fn macmtrx(&mut self, controls: &[bitLenInt], mtrx: &[complex], target: bitLenInt) {
        if IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2]) {
            self.macphase(controls, mtrx[0], mtrx[3], target);
            return;
        }
        if IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3]) {
            self.macinvert(controls, mtrx[1], mtrx[2], target);
            return;
        }
        panic!("QStabilizer::MACMtrx() not implemented for non-Clifford/Pauli cases!");
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

    fn get_amplitude(&self, perm: bitCapInt) -> complex {
        let elem_count = self.qubit_count << 1;
        let mut ret = complex::new(ZERO_R1, ZERO_R1);
        for i in 0..elem_count {
            if i < self.qubit_count {
                if (perm >> i) & 1 == 1 {
                    ret *= I_CMPLX;
                }
            } else {
                let j = i - self.qubit_count;
                if (perm >> j) & 1 == 1 {
                    ret *= -ONE_CMPLX;
                }
            }
        }
        ret *= complex::from_polar(&ONE_R1, &self.phase_offset);
        ret
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

