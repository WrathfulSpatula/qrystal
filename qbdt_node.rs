use std::rc::Rc;
use std::cell::RefCell;

pub trait QBdtNodeInterface {}

pub struct QBdtNode {
    scale: complex,
    branches: Vec<Rc<RefCell<dyn QBdtNodeInterface>>>,
}

impl QBdtNode {
    pub fn new() -> Self {
        Self {
            scale: complex::default(),
            branches: Vec::new(),
        }
    }

    pub fn new_with_scale(scl: complex) -> Self {
        Self {
            scale: scl,
            branches: Vec::new(),
        }
    }

    pub fn new_with_scale_and_branches(scl: complex, b: Vec<Rc<RefCell<dyn QBdtNodeInterface>>>) -> Self {
        Self {
            scale: scl,
            branches: b,
        }
    }

    pub fn shallow_clone(&self) -> Rc<RefCell<dyn QBdtNodeInterface>> {
        Rc::new(RefCell::new(QBdtNode {
            scale: self.scale,
            branches: self.branches.clone(),
        }))
    }

    pub fn insert_at_depth(&mut self, b: Rc<RefCell<dyn QBdtNodeInterface>>, depth: bitLenInt, size: bitLenInt, parDepth: bitLenInt = 1) {
        // Implementation here
    }

    pub fn pop_state_vector(&mut self, depth: bitLenInt = 1, parDepth: bitLenInt = 1) {
        // Implementation here
    }

    pub fn branch(&mut self, depth: bitLenInt = 1, parDepth: bitLenInt = 1) {
        // Implementation here
    }

    pub fn prune(&mut self, depth: bitLenInt = 1, parDepth: bitLenInt = 1) {
        // Implementation here
    }

    pub fn normalize(&mut self, depth: bitLenInt = 1) {
        // Implementation here
    }

    pub fn apply_2x2(&mut self, mtrx: complex2, depth: bitLenInt) {
        // Implementation here
    }
}


