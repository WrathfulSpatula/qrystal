use std::rc::Rc;

pub trait QParity {
    fn m_parity(&self, mask: u64) -> bool {
        self.force_m_parity(mask, false, false)
    }
    
    fn uniform_parity_rz(&self, mask: u64, angle: f64) {
        self.c_uniform_parity_rz(vec![], mask, angle);
    }
    
    fn prob_parity(&self, mask: u64) -> f64;
    
    fn force_m_parity(&self, mask: u64, result: bool, do_force: bool) -> bool;
    
    fn c_uniform_parity_rz(&self, controls: Vec<u32>, mask: u64, angle: f64);
}
