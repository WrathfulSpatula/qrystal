use std::sync::Mutex;

pub trait QBdtNodeInterface {
    fn select_bit(perm: u64, bit: u32) -> usize {
        (perm >> bit) as usize & 1
    }

    fn par_for_qbdt(end: u64, fn: BdtFunc) {
        // implementation here
    }

    fn push_state_vector(&self, mtrx: &[complex], b0: &mut dyn QBdtNodeInterface, b1: &mut dyn QBdtNodeInterface, depth: u32, par_depth: u32) {
        unimplemented!("QBdtNodeInterface::push_state_vector() not implemented! (You probably set QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    fn insert_at_depth(&mut self, b: &mut dyn QBdtNodeInterface, depth: u32, size: &u32, par_depth: u32) {
        unimplemented!("QBdtNodeInterface::insert_at_depth() not implemented! (You probably set QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    fn remove_separable_at_depth(&mut self, depth: u32, size: &u32, par_depth: u32) -> Option<Box<dyn QBdtNodeInterface>> {
        unimplemented!("QBdtNodeInterface::remove_separable_at_depth() not implemented! (You probably set QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    fn set_zero(&mut self) {
        self.scale = Complex::new(0.0, 0.0);
        if let Some(b0) = self.branches[0].as_mut() {
            let _lock = b0.mtx.lock().unwrap();
            self.branches[0] = None;
        }
        if let Some(b1) = self.branches[1].as_mut() {
            let _lock = b1.mtx.lock().unwrap();
            self.branches[1] = None;
        }
    }

    fn is_equal(&self, r: &dyn QBdtNodeInterface) -> bool {
        // implementation here
    }

    fn is_equal_under(&self, r: &dyn QBdtNodeInterface) -> bool {
        // implementation here
    }

    fn is_equal_branch(&self, r: &dyn QBdtNodeInterface, b: bool) -> bool {
        // implementation here
    }

    fn shallow_clone(&self) -> Box<dyn QBdtNodeInterface> {
        unimplemented!("QBdtNodeInterface::shallow_clone() not implemented! (You probably set QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    fn pop_state_vector(&mut self, depth: u32, par_depth: u32) {
        unimplemented!("QBdtNodeInterface::pop_state_vector() not implemented! (You probably set QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    fn branch(&mut self, depth: u32, par_depth: u32) {
        unimplemented!("QBdtNodeInterface::branch() not implemented! (You probably set QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    fn prune(&mut self, depth: u32, par_depth: u32) {
        unimplemented!("QBdtNodeInterface::prune() not implemented! (You probably set QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    fn normalize(&mut self, depth: u32) {
        unimplemented!("QBdtNodeInterface::normalize() not implemented! (You probably set QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    fn apply_2x2(&mut self, mtrx: &[complex], depth: u32) {
        unimplemented!("QBdtNodeInterface::apply_2x2() not implemented! (You probably set QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    fn push_special(&mut self, mtrx: &[complex], b1: &mut dyn QBdtNodeInterface) {
        unimplemented!("QBdtNodeInterface::push_special() not implemented! (You probably called push_state_vector() past terminal depth.)");
    }
}

pub struct QBdtNodeInterfaceImpl {
    scale: Complex,
    branches: [Option<Box<dyn QBdtNodeInterface>>; 2],
    mtx: Mutex<()>,
}

impl QBdtNodeInterfaceImpl {
    pub fn new(scale: Complex) -> Self {
        Self {
            scale,
            branches: [None, None],
            mtx: Mutex::new(()),
        }
    }
}

impl QBdtNodeInterface for QBdtNodeInterfaceImpl {
    // implement trait methods here
}

impl PartialEq for dyn QBdtNodeInterface {
    fn eq(&self, other: &Self) -> bool {
        // implementation here
    }
}

impl Eq for dyn QBdtNodeInterface {}

mod qrack {
    use super::*;

    pub fn operator_eq(lhs: &dyn QBdtNodeInterface, rhs: &dyn QBdtNodeInterface) -> bool {
        // implementation here
    }

    pub fn operator_ne(lhs: &dyn QBdtNodeInterface, rhs: &dyn QBdtNodeInterface) -> bool {
        // implementation here
    }
}


