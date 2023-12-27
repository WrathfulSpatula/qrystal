use std::cmp::Ordering;
use std::f64::consts::PI;
use std::mem;
use std::ptr;
use std::slice;
use std::vec::Vec;

mod QInterface;

trait QEngine: QInterface {
    use_host_ram: bool,
    running_norm: f64,
    max_q_power_ocl: u64,
    qubit_count: usize,
    rgp: Option<*mut qrack_rand_gen>,
    do_norm: bool,
    random_global_phase: bool,
    use_host_mem: bool,
    use_hardware_rng: bool,
    norm_thresh: f64,
    qubit_states: Vec<Complex>,

    pub fn new(
        qubit_count: usize,
        rgp: Option<*mut qrack_rand_gen>,
        do_norm: bool,
        random_global_phase: bool,
        use_host_mem: bool,
        use_hardware_rng: bool,
        norm_thresh: f64,
    ) -> Self {
        let mut qubit_states = Vec::with_capacity(1 << qubit_count);
        unsafe {
            qubit_states.set_len(1 << qubit_count);
            ptr::write_bytes(qubit_states.as_mut_ptr(), 0, 1 << qubit_count);
        }

        Self {
            use_host_ram: use_host_mem,
            running_norm: 1.0,
            max_q_power_ocl: 1 << qubit_count,
            qubit_count,
            rgp,
            do_norm,
            random_global_phase,
            use_host_mem,
            use_hardware_rng,
            norm_thresh,
            qubit_states,
        }
    }

    pub fn set_qubit_count(&mut self, qb: usize) {
        self.qubit_count = qb;
        self.max_q_power_ocl = 1 << qb;
    }

    pub fn get_running_norm(&self) -> f64 {
        self.finish();
        self.running_norm
    }

    pub fn reset_host_ptr(&mut self) {
        self.switch_host_ptr(self.use_host_ram);
    }

    pub fn z_mask(&mut self, mask: u64) {
        self.phase_parity(PI, mask);
    }

    pub fn force_m(&mut self, qubit_index: usize, result: bool, do_force: bool, do_apply: bool) -> bool {
        let one_chance = self.prob(qubit_index);
        let result = if !do_force {
            if one_chance >= 1.0 {
                true
            } else if one_chance <= 0.0 {
                false
            } else {
                result
            }
        } else {
            result
        };

        let nrmlzr = if result {
            one_chance
        } else {
            1.0 - one_chance
        };

        if nrmlzr <= 0.0 {
            panic!("QEngine::ForceM() forced a measurement result with 0 probability!");
        }

        if do_apply && (1.0 - nrmlzr) > f64::EPSILON {
            let q_power = 1 << qubit_index;
            self.apply_m(q_power, result, self.get_nonunitary_phase() / nrmlzr.sqrt());
        }

        result
    }

    pub fn apply_m(&mut self, q_power: u64, result: bool, nrm: Complex) {
        let power_test = if result { q_power } else { 0 };
        self.apply_m(q_power, power_test, nrm);
    }

    pub fn mtrx(&mut self, mtrx: &[Complex], qubit: usize) {
        if self.is_identity(mtrx, false) {
            return;
        }

        let q_powers = vec![1 << qubit];
        self.apply_2x2(0, q_powers[0], mtrx, 1, &q_powers, false);
    }

    pub fn m_c_mtrx(&mut self, controls: &[usize], mtrx: &[Complex], target: usize, is_anti: bool) {
        if controls.is_empty() {
            self.mtrx(mtrx, target);
            return;
        }

        if self.is_identity(mtrx, true) {
            return;
        }

        if is_anti {
            self.apply_anti_controlled_2x2(controls, target, mtrx);
        } else {
            self.apply_controlled_2x2(controls, target, mtrx);
        }

        if self.do_norm && !(self.is_phase(mtrx) || self.is_invert(mtrx)) {
            self.update_running_norm();
        }
    }

    pub fn u_c_mtrx(
        &mut self,
        controls: &[usize],
        mtrx: &[Complex],
        target: usize,
        control_perm: u64,
    ) {
        if controls.is_empty() {
            self.mtrx(mtrx, target);
            return;
        }

        if self.is_identity(mtrx, true) {
            return;
        }

        let mut q_powers_sorted = Vec::with_capacity(controls.len() + 1);
        let target_mask = 1 << target;
        let mut full_mask = 0;
        for i in 0..controls.len() {
            q_powers_sorted.push(1 << controls[i]);
            if control_perm & (1 << i) != 0 {
                full_mask |= 1 << controls[i];
            }
        }

        let control_mask = full_mask;
        q_powers_sorted.push(target_mask);
        full_mask |= target_mask;
        q_powers_sorted.sort_unstable();

        self.apply_2x2(
            control_mask,
            full_mask,
            mtrx,
            controls.len() + 1,
            &q_powers_sorted,
            false,
        );
    }

    pub fn c_swap(&mut self, controls: &[usize], qubit1: usize, qubit2: usize) {
        if controls.is_empty() {
            self.swap(qubit1, qubit2);
            return;
        }

        if qubit1 == qubit2 {
            return;
        }

        if qubit2 < qubit1 {
            self.swap(qubit1, qubit2);
            return;
        }

        let pauli_x = [
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        let mut skip_mask = 0;
        let mut q_powers_sorted = Vec::with_capacity(controls.len() + 2);
        for i in 0..controls.len() {
            q_powers_sorted.push(1 << controls[i]);
            skip_mask |= 1 << controls[i];
        }
        q_powers_sorted.push(1 << qubit1);
        q_powers_sorted.push(1 << qubit2);
        q_powers_sorted.sort_unstable();

        self.apply_2x2(
            skip_mask | (1 << qubit1),
            skip_mask | (1 << qubit2),
            &pauli_x,
            controls.len() + 2,
            &q_powers_sorted,
            false,
        );
    }

    pub fn anti_c_swap(&mut self, controls: &[usize], qubit1: usize, qubit2: usize) {
        if controls.is_empty() {
            self.swap(qubit1, qubit2);
            return;
        }

        if qubit1 == qubit2 {
            return;
        }

        if qubit2 < qubit1 {
            self.swap(qubit1, qubit2);
            return;
        }

        let pauli_x = [
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        let mut q_powers_sorted = Vec::with_capacity(controls.len() + 2);
        for i in 0..controls.len() {
            q_powers_sorted.push(1 << controls[i]);
        }
        q_powers_sorted.push(1 << qubit1);
        q_powers_sorted.push(1 << qubit2);
        q_powers_sorted.sort_unstable();

        self.apply_2x2(
            1 << qubit1,
            1 << qubit2,
            &pauli_x,
            controls.len() + 2,
            &q_powers_sorted,
            false,
        );
    }

    pub fn c_sqrt_swap(&mut self, controls: &[usize], qubit1: usize, qubit2: usize) {
        if controls.is_empty() {
            self.sqrt_swap(qubit1, qubit2);
            return;
        }

        if qubit1 == qubit2 {
            return;
        }

        if qubit2 < qubit1 {
            self.sqrt_swap(qubit1, qubit2);
            return;
        }

        let sqrt_x = [
            Complex::new(0.5, 0.5),
            Complex::new(0.5, -0.5),
            Complex::new(0.5, -0.5),
            Complex::new(0.5, 0.5),
        ];

        let mut skip_mask = 0;
        let mut q_powers_sorted = Vec::with_capacity(controls.len() + 2);
        for i in 0..controls.len() {
            q_powers_sorted.push(1 << controls[i]);
            skip_mask |= 1 << controls[i];
        }
        q_powers_sorted.push(1 << qubit1);
        q_powers_sorted.push(1 << qubit2);
        q_powers_sorted.sort_unstable();

        self.apply_2x2(
            skip_mask | (1 << qubit1),
            skip_mask | (1 << qubit2),
            &sqrt_x,
            controls.len() + 2,
            &q_powers_sorted,
            false,
        );
    }

    pub fn anti_c_sqrt_swap(&mut self, controls: &[usize], qubit1: usize, qubit2: usize) {
        if controls.is_empty() {
            self.sqrt_swap(qubit1, qubit2);
            return;
        }

        if qubit1 == qubit2 {
            return;
        }

        if qubit2 < qubit1 {
            self.sqrt_swap(qubit1, qubit2);
            return;
        }

        let sqrt_x = [
            Complex::new(0.5, 0.5),
            Complex::new(0.5, -0.5),
            Complex::new(0.5, -0.5),
            Complex::new(0.5, 0.5),
        ];

        let mut q_powers_sorted = Vec::with_capacity(controls.len() + 2);
        for i in 0..controls.len() {
            q_powers_sorted.push(1 << controls[i]);
        }
        q_powers_sorted.push(1 << qubit1);
        q_powers_sorted.push(1 << qubit2);
        q_powers_sorted.sort_unstable();

        self.apply_2x2(
            1 << qubit1,
            1 << qubit2,
            &sqrt_x,
            controls.len() + 2,
            &q_powers_sorted,
            false,
        );
    }

    pub fn c_i_sqrt_swap(&mut self, controls: &[usize], qubit1: usize, qubit2: usize) {
        if controls.is_empty() {
            self.i_sqrt_swap(qubit1, qubit2);
            return;
        }

        if qubit1 == qubit2 {
            return;
        }

        if qubit2 < qubit1 {
            self.i_sqrt_swap(qubit1, qubit2);
            return;
        }

        let i_sqrt_x = [
            Complex::new(0.5, -0.5),
            Complex::new(0.5, 0.5),
            Complex::new(0.5, 0.5),
            Complex::new(0.5, -0.5),
        ];

        let mut skip_mask = 0;
        let mut q_powers_sorted = Vec::with_capacity(controls.len() + 2);
        for i in 0..controls.len() {
            q_powers_sorted.push(1 << controls[i]);
            skip_mask |= 1 << controls[i];
        }
        q_powers_sorted.push(1 << qubit1);
        q_powers_sorted.push(1 << qubit2);
        q_powers_sorted.sort_unstable();

        self.apply_2x2(
            skip_mask | (1 << qubit1),
            skip_mask | (1 << qubit2),
            &i_sqrt_x,
            controls.len() + 2,
            &q_powers_sorted,
            false,
        );
    }

    pub fn iswap(&mut self, qubit1: usize, qubit2: usize) {
        if qubit1 == qubit2 {
            return;
        }

        let mut q_powers_sorted = vec![1 << qubit1, 1 << qubit2];
        q_powers_sorted.sort_unstable();

        let mut qubit_states = Vec::with_capacity(self.qubit_states.len());
        unsafe {
            qubit_states.set_len(self.qubit_states.len());
            ptr::copy_nonoverlapping(
                self.qubit_states.as_ptr(),
                qubit_states.as_mut_ptr(),
                self.qubit_states.len(),
            );
        }

        for i in 0..(1 << self.qubit_count) {
            let i1 = i ^ (1 << qubit1);
            let i2 = i ^ (1 << qubit2);
            let temp = qubit_states[i1];
            qubit_states[i1] = qubit_states[i2];
            qubit_states[i2] = temp;
            qubit_states[i1].im *= -1.0;
            qubit_states[i2].im *= -1.0;
        }

        self.qubit_states = qubit_states;
    }

    pub fn iiswap(&mut self, qubit1: usize, qubit2: usize) {
        if qubit1 == qubit2 {
            return;
        }

        let mut q_powers_sorted = vec![1 << qubit1, 1 << qubit2];
        q_powers_sorted.sort_unstable();

        let mut qubit_states = Vec::with_capacity(self.qubit_states.len());
        unsafe {
            qubit_states.set_len(self.qubit_states.len());
            ptr::copy_nonoverlapping(
                self.qubit_states.as_ptr(),
                qubit_states.as_mut_ptr(),
                self.qubit_states.len(),
            );
        }

        for i in 0..(1 << self.qubit_count) {
            let i1 = i ^ (1 << qubit1);
            let i2 = i ^ (1 << qubit2);
            let temp = qubit_states[i1];
            qubit_states[i1] = qubit_states[i2];
            qubit_states[i2] = temp;
            qubit_states[i1].re *= -1.0;
            qubit_states[i2].re *= -1.0;
        }

        self.qubit_states = qubit_states;
    }

    pub fn prob_all(&mut self, full_register: u64) -> f64 {
        if self.do_norm {
            self.normalize_state();
        }

        let amplitude = self.get_amplitude(full_register);
        amplitude.norm_sqr()
    }

    pub fn c_prob(&mut self, control: usize, target: usize) -> f64 {
        self.ctrl_or_anti_prob(true, control, target)
    }

    pub fn ac_prob(&mut self, control: usize, target: usize) -> f64 {
        self.ctrl_or_anti_prob(false, control, target)
    }

    pub fn prob_reg(&mut self, start: usize, length: usize, permutation: u64) -> f64 {
        let mut reg_mask = 0;
        for i in 0..length {
            reg_mask |= 1 << (start + i);
        }

        self.prob_mask(reg_mask, permutation)
    }

    pub fn get_expectation(&mut self, value_start: usize, value_length: usize) -> f64 {
        let mut expectation = 0.0;
        let mut mask = 0;
        for i in 0..value_length {
            mask |= 1 << (value_start + i);
        }

        for i in 0..(1 << self.qubit_count) {
            let value = i & mask;
            let amplitude = self.get_amplitude(i);
            expectation += amplitude.norm_sqr() * (if value != 0 { 1.0 } else { -1.0 });
        }

        expectation
    }

    pub fn decompose(&mut self, start: usize, length: usize) -> Self {
        let mut dest = Self::new(
            length,
            self.rgp,
            self.do_norm,
            self.random_global_phase,
            self.use_host_mem,
            self.use_hardware_rng,
            self.norm_thresh,
        );
        dest.set_qubit_count(length);
        self.decompose(start, &mut dest);
        dest
    }

    pub fn decompose(&mut self, start: usize, dest: &mut Self) {
        let mut qubit_states = Vec::with_capacity(self.qubit_states.len());
        unsafe {
            qubit_states.set_len(self.qubit_states.len());
            ptr::copy_nonoverlapping(
                self.qubit_states.as_ptr(),
                qubit_states.as_mut_ptr(),
                self.qubit_states.len(),
            );
        }

        let mut qubit_states_ptr = qubit_states.as_mut_ptr();
        let qubit_states_len = qubit_states.len();

        let mut dest_qubit_states = Vec::with_capacity(dest.qubit_states.len());
        unsafe {
            dest_qubit_states.set_len(dest.qubit_states.len());
            ptr::copy_nonoverlapping(
                dest.qubit_states.as_ptr(),
                dest_qubit_states.as_mut_ptr(),
                dest.qubit_states.len(),
            );
        }

        let mut dest_qubit_states_ptr = dest_qubit_states.as_mut_ptr();
        let dest_qubit_states_len = dest_qubit_states.len();

        unsafe {
            decompose(
                qubit_states_ptr,
                qubit_states_len,
                start,
                dest_qubit_states_ptr,
                dest_qubit_states_len,
            );
        }

        self.qubit_states = qubit_states;
        dest.qubit_states = dest_qubit_states;
    }

    pub fn normalize_state(&mut self) {
        let mut qubit_states = Vec::with_capacity(self.qubit_states.len());
        unsafe {
            qubit_states.set_len(self.qubit_states.len());
            ptr::copy_nonoverlapping(
                self.qubit_states.as_ptr(),
                qubit_states.as_mut_ptr(),
                self.qubit_states.len(),
            );
        }

        let mut qubit_states_ptr = qubit_states.as_mut_ptr();
        let qubit_states_len = qubit_states.len();

        unsafe {
            normalize_state(qubit_states_ptr, qubit_states_len);
        }

        self.qubit_states = qubit_states;
    }

    pub fn prob_mask(&mut self, mask: u64, permutation: u64) -> f64 {
        if self.do_norm {
            self.normalize_state();
        }

        let mut qubit_states = Vec::with_capacity(self.qubit_states.len());
        unsafe {
            qubit_states.set_len(self.qubit_states.len());
            ptr::copy_nonoverlapping(
                self.qubit_states.as_ptr(),
                qubit_states.as_mut_ptr(),
                self.qubit_states.len(),
            );
        }

        let mut qubit_states_ptr = qubit_states.as_mut_ptr();
        let qubit_states_len = qubit_states.len();

        let prob = unsafe { prob_mask(qubit_states_ptr, qubit_states_len, mask, permutation) };

        self.qubit_states = qubit_states;

        prob
    }

    pub fn prob_mask_all(&mut self, mask: u64, probs_array: &mut [f64]) {
        if self.do_norm {
            self.normalize_state();
        }

        let mut qubit_states = Vec::with_capacity(self.qubit_states.len());
        unsafe {
            qubit_states.set_len(self.qubit_states.len());
            ptr::copy_nonoverlapping(
                self.qubit_states.as_ptr(),
                qubit_states.as_mut_ptr(),
                self.qubit_states.len(),
            );
        }

        let mut qubit_states_ptr = qubit_states.as_mut_ptr();
        let qubit_states_len = qubit_states.len();

        let mut probs_array_ptr = probs_array.as_mut_ptr();
        let probs_array_len = probs_array.len();

        unsafe {
            prob_mask_all(
                qubit_states_ptr,
                qubit_states_len,
                mask,
                probs_array_ptr,
                probs_array_len,
            );
        }

        self.qubit_states = qubit_states;
    }

    fn anti_c_i_sqrt_swap(&self, controls: &[usize], qubit1: usize, qubit2: usize) {
        if controls.is_empty() {
            self.i_sqrt_swap(qubit1, qubit2);
            return;
        }
        if qubit1 == qubit2 {
            return;
        }
        let mut q_powers_sorted = controls.iter().map(|&control| 1 << control).collect::<Vec<_>>();
        q_powers_sorted.push(1 << qubit1);
        q_powers_sorted.push(1 << qubit2);
        q_powers_sorted.sort();
        self.apply_2x2(1 << qubit1, 1 << qubit2, &[complex(0.5, -0.5), complex(0.5, 0.5), complex(0.5, 0.5), complex(0.5, -0.5)], controls.len() + 2, &q_powers_sorted, false);
    }

    
    pub fn apply_2x2(
        &mut self,
        offset1: u64,
        offset2: u64,
        mtrx: &[Complex],
        bit_count: usize,
        q_powers_sorted: &[u64],
        do_calc_norm: bool,
    ) {
        let mut qubit_states = Vec::with_capacity(self.qubit_states.len());
        unsafe {
            qubit_states.set_len(self.qubit_states.len());
            ptr::copy_nonoverlapping(
                self.qubit_states.as_ptr(),
                qubit_states.as_mut_ptr(),
                self.qubit_states.len(),
            );
        }

        let mut qubit_states_ptr = qubit_states.as_mut_ptr();
        let qubit_states_len = qubit_states.len();

        let mut q_powers_sorted = q_powers_sorted.to_vec();
        q_powers_sorted.sort_unstable();

        let mut q_powers_sorted_ptr = q_powers_sorted.as_mut_ptr();
        let q_powers_sorted_len = q_powers_sorted.len();

        unsafe {
            apply_2x2(
                qubit_states_ptr,
                qubit_states_len,
                offset1,
                offset2,
                mtrx.as_ptr(),
                bit_count,
                q_powers_sorted_ptr,
                q_powers_sorted_len,
                do_calc_norm,
                self.norm_thresh,
            );
        }

        self.qubit_states = qubit_states;
    }

    fn apply_controlled_2x2(&self, controls: &[usize], target: usize, mtrx: &[complex]) {
        let mut q_powers_sorted = controls.iter().map(|&control| 1 << control).collect::<Vec<_>>();
        let target_mask = 1 << target;
        let mut full_mask = 0;
        for &control in controls {
            full_mask |= 1 << control;
        }
        let control_mask = full_mask;
        q_powers_sorted.push(target_mask);
        full_mask |= target_mask;
        q_powers_sorted.sort();
        self.apply_2x2(control_mask, full_mask, mtrx, controls.len() + 1, &q_powers_sorted, false);
    }

    fn apply_anti_controlled_2x2(&self, controls: &[usize], target: usize, mtrx: &[complex]) {
        let mut q_powers_sorted = controls.iter().map(|&control| 1 << control).collect::<Vec<_>>();
        let target_mask = 1 << target;
        q_powers_sorted.push(target_mask);
        q_powers_sorted.sort();
        self.apply_2x2(0, target_mask, mtrx, controls.len() + 1, &q_powers_sorted, false);
    }

    fn swap(&self, qubit1: usize, qubit2: usize) {
        if qubit1 == qubit2 {
            return;
        }
        let pauli_x = [complex(0.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(0.0, 0.0)];
        let q_powers_sorted = [1 << qubit1, 1 << qubit2];
        self.apply_2x2(q_powers_sorted[0], q_powers_sorted[1], &pauli_x, 2, &q_powers_sorted, false);
    }

    fn i_swap(&self, qubit1: usize, qubit2: usize) {
        if qubit1 == qubit2 {
            return;
        }
        let pauli_x = [complex(0.0, 0.0), complex(0.0, 1.0), complex(0.0, 1.0), complex(0.0, 0.0)];
        let q_powers_sorted = [1 << qubit1, 1 << qubit2];
        self.apply_2x2(q_powers_sorted[0], q_powers_sorted[1], &pauli_x, 2, &q_powers_sorted, false);
    }

    fn ii_swap(&self, qubit1: usize, qubit2: usize) {
        if qubit1 == qubit2 {
            return;
        }
        let pauli_x = [complex(0.0, 0.0), complex(0.0, -1.0), complex(0.0, -1.0), complex(0.0, 0.0)];
        let q_powers_sorted = [1 << qubit1, 1 << qubit2];
        self.apply_2x2(q_powers_sorted[0], q_powers_sorted[1], &pauli_x, 2, &q_powers_sorted, false);
    }

    fn sqrt_swap(&self, qubit1: usize, qubit2: usize) {
        if qubit1 == qubit2 {
            return;
        }
        let sqrt_x = [complex(0.5, 0.5), complex(0.5, -0.5), complex(0.5, -0.5), complex(0.5, 0.5)];
        let q_powers_sorted = [1 << qubit1, 1 << qubit2];
        self.apply_2x2(q_powers_sorted[0], q_powers_sorted[1], &sqrt_x, 2, &q_powers_sorted, false);
    }

    fn i_sqrt_swap(&self, qubit1: usize, qubit2: usize) {
        if qubit1 == qubit2 {
            return;
        }
        let i_sqrt_x = [complex(0.5, -0.5), complex(0.5, 0.5), complex(0.5, 0.5), complex(0.5, -0.5)];
        let q_powers_sorted = [1 << qubit1, 1 << qubit2];
        self.apply_2x2(q_powers_sorted[0], q_powers_sorted[1], &i_sqrt_x, 2, &q_powers_sorted, false);
    }

    fn f_sim(&self, theta: f64, phi: f64, qubit1: usize, qubit2: usize) {
        if qubit2 < qubit1 {
            self.swap(qubit1, qubit2);
        }
        let sin_theta = theta.sin();
        if sin_theta * sin_theta > f64::EPSILON {
            let cos_theta = theta.cos();
            let f_sim_swap = [complex(cos_theta, 0.0), complex(0.0, -sin_theta), complex(0.0, -sin_theta), complex(cos_theta, 0.0)];
            let q_powers_sorted = [1 << qubit1, 1 << qubit2];
            self.apply_2x2(q_powers_sorted[0], q_powers_sorted[1], &f_sim_swap, 2, &q_powers_sorted, false);
        }
        let controls = vec![qubit1];
        self.mc_phase(&controls, complex(1.0, 0.0), complex(0.0, phi).exp(), qubit2);
    }

    fn ctrl_or_anti_prob(&self, control_state: bool, control: usize, target: usize) -> f64 {
        if control_state {
            self.anti_c_not(control, target);
        } else {
            self.c_not(control, target);
        }
        let prob = self.prob(target);
        if control_state {
            self.anti_c_not(control, target);
        } else {
            self.c_not(control, target);
        }
        prob
    }

    fn prob_reg_all(&self, start: usize, length: usize, probs_array: &mut [f64]) {
        let length_mask = (1 << length) - 1;
        probs_array.iter_mut().for_each(|prob| *prob = 0.0);
        for i in 0..(1 << self.qubit_count) {
            let reg = (i >> start) & length_mask;
            probs_array[reg] += self.prob_all(i);
        }
    }

    fn force_m_reg(&self, start: usize, length: usize, result: u64, do_force: bool, do_apply: bool) -> u64 {
        if start + length > self.qubit_count {
            panic!("QEngine::ForceMReg range is out-of-bounds!");
        }
        if length == 1 {
            return if self.force_m(start, result & 1, do_force, do_apply) { 1 } else { 0 };
        }
        let length_power = 1 << length;
        let reg_mask = (length_power - 1) << start;
        let mut nrmlzr = 1.0;
        if do_force {
            nrmlzr = self.prob_mask(reg_mask, result << start);
        } else {
            let mut lcv = 0;
            let mut prob_array = vec![0.0; length_power];
            self.prob_reg_all(start, length, &mut prob_array);
            let prob = self.rand();
            let mut lower_prob = 0.0;
            let mut result = length_power - 1;
            while lower_prob < prob && lcv < length_power {
                lower_prob += prob_array[lcv];
                if prob_array[lcv] > 0.0 {
                    nrmlzr = prob_array[lcv];
                    result = lcv;
                }
                lcv += 1;
            }
        }
        if do_apply {
            let result_ptr = result << start;
            let nrm = self.get_nonunitary_phase() / (nrmlzr.sqrt());
            self.apply_m(reg_mask, result_ptr, nrm);
        }
        result
    }

    fn multi_shot_measure_mask(&self, q_powers: &[u64], shots: u32) -> HashMap<u64, i32> {
        if shots == 0 {
            return HashMap::new();
        }
        let bit_map = q_powers.iter().map(|&q_power| (q_power as f64).log2() as usize).collect::<Vec<_>>();
        let mask_max_q_power = 1 << q_powers.len();
        let mut mask_probs_vec = vec![0.0; mask_max_q_power];
        self.prob_bits_all(&bit_map, &mut mask_probs_vec);
        let dist = rand::distributions::WeightedIndex::new(&mask_probs_vec).unwrap();
        let mut results = HashMap::new();
        let mut rng = rand::thread_rng();
        for _ in 0..shots {
            *results.entry(dist.sample(&mut rng) as u64).or_insert(0) += 1;
        }
        results
    }

    fn multi_shot_measure_mask(&self, q_powers: &[u64], shots: u32, shots_array: &mut [u64]) {
        if shots == 0 {
            return;
        }
        let bit_map = q_powers.iter().map(|&q_power| (q_power as f64).log2() as usize).collect::<Vec<_>>();
        let mask_max_q_power = 1 << q_powers.len();
        let mut mask_probs_vec = vec![0.0; mask_max_q_power];
        self.prob_bits_all(&bit_map, &mut mask_probs_vec);
        let dist = rand::distributions::WeightedIndex::new(&mask_probs_vec).unwrap();
        let mut rng = rand::thread_rng();
        for shot in shots_array.iter_mut().take(shots as usize) {
            *shot = dist.sample(&mut rng) as u64;
        }
    }
}
