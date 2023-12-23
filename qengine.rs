type Real = f64;
type Complex = num_complex::Complex<Real>;

trait QInterface {
    fn set_qubit_count(&mut self, qb: usize);
    fn get_amplitude(&self, index: usize) -> Complex;
    fn set_amplitude(&mut self, index: usize, amplitude: Complex);
    fn apply_m(&mut self, q_power: usize, result: bool, nrm: Complex);
    fn apply_2x2(
        &mut self,
        offset1: usize,
        offset2: usize,
        mtrx: &[Complex],
        bit_count: usize,
        q_powers_sorted: &[usize],
        do_calc_norm: bool,
        norm_thresh: Real,
    );
    fn apply_controlled_2x2(&mut self, controls: &[usize], target: usize, mtrx: &[Complex]);
    fn apply_anti_controlled_2x2(&mut self, controls: &[usize], target: usize, mtrx: &[Complex]);
    fn apply_controlled_2x2_mask(
        &mut self,
        controls: &[usize],
        target_mask: usize,
        mtrx: &[Complex],
        bit_count: usize,
        q_powers_sorted: &[usize],
        do_calc_norm: bool,
        norm_thresh: Real,
    );
    fn apply_anti_controlled_2x2_mask(
        &mut self,
        controls: &[usize],
        target_mask: usize,
        mtrx: &[Complex],
        bit_count: usize,
        q_powers_sorted: &[usize],
        do_calc_norm: bool,
        norm_thresh: Real,
    );
}

struct QEngine {
    qubit_count: usize,
    use_host_ram: bool,
    running_norm: Real,
    max_q_power_ocl: usize,
}

impl QEngine {
    fn new(
        qubit_count: usize,
        rgp: Option<Arc<dyn Rng>>,
        do_norm: bool,
        random_global_phase: bool,
        use_host_mem: bool,
        use_hardware_rng: bool,
        norm_thresh: Real,
    ) -> Self {
        if qubit_count > std::mem::size_of::<usize>() * 8 {
            panic!("Cannot instantiate a register with greater capacity than native types on emulating system.");
        }
        Self {
            qubit_count,
            use_host_ram: use_host_mem,
            running_norm: 1.0,
            max_q_power_ocl: 2usize.pow(qubit_count as u32),
        }
    }
}

impl QInterface for QEngine {
    fn set_qubit_count(&mut self, qb: usize) {
        self.qubit_count = qb;
        self.max_q_power_ocl = 2usize.pow(qb as u32);
    }

    fn get_amplitude(&self, index: usize) -> Complex {
        unimplemented!()
    }

    fn set_amplitude(&mut self, index: usize, amplitude: Complex) {
        unimplemented!()
    }

    fn apply_m(&mut self, q_power: usize, result: bool, nrm: Complex) {
        unimplemented!()
    }

    fn apply_2x2(
        &mut self,
        offset1: usize,
        offset2: usize,
        mtrx: &[Complex],
        bit_count: usize,
        q_powers_sorted: &[usize],
        do_calc_norm: bool,
        norm_thresh: Real,
    ) {
        unimplemented!()
    }

    fn apply_controlled_2x2(&mut self, controls: &[usize], target: usize, mtrx: &[Complex]) {
        unimplemented!()
    }

    fn apply_anti_controlled_2x2(&mut self, controls: &[usize], target: usize, mtrx: &[Complex]) {
        unimplemented!()
    }

    fn apply_controlled_2x2_mask(
        &mut self,
        controls: &[usize],
        target_mask: usize,
        mtrx: &[Complex],
        bit_count: usize,
        q_powers_sorted: &[usize],
        do_calc_norm: bool,
        norm_thresh: Real,
    ) {
        unimplemented!()
    }

    fn apply_anti_controlled_2x2_mask(
        &mut self,
        controls: &[usize],
        target_mask: usize,
        mtrx: &[Complex],
        bit_count: usize,
        q_powers_sorted: &[usize],
        do_calc_norm: bool,
        norm_thresh: Real,
    ) {
        unimplemented!()
    }
}

fn main() {
    unimplemented!()
}


