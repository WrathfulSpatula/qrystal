use std::rc::Rc;

pub trait QAlu {
    fn m(&self, qubit_index: i32) -> bool;
    fn x(&self, qubit_index: i32);
    fn phase_flip_if_less(&self, greater_perm: i64, start: i32, length: i32);
    fn c_phase_flip_if_less(&self, greater_perm: i64, start: i32, length: i32, flag_index: i32);
    fn inc(&self, to_add: i64, start: i32, length: i32);
    fn dec(&self, to_sub: i64, start: i32, length: i32);
    fn c_inc(&self, to_add: i64, start: i32, length: i32, controls: &[i32]);
    fn c_dec(&self, to_sub: i64, start: i32, length: i32, controls: &[i32]);
    fn inc_c(&self, to_add: i64, start: i32, length: i32, carry_index: i32);
    fn dec_c(&self, to_sub: i64, start: i32, length: i32, carry_index: i32);
    fn inc_dec_c(&self, to_mod: i64, start: i32, length: i32, carry_index: i32);
    fn inc_s(&self, to_add: i64, start: i32, length: i32, overflow_index: i32);
    fn dec_s(&self, to_add: i64, start: i32, length: i32, overflow_index: i32);
    fn inc_sc(&self, to_add: i64, start: i32, length: i32, overflow_index: i32, carry_index: i32);
    fn inc_sc(&self, to_add: i64, start: i32, length: i32, carry_index: i32);
    fn dec_sc(&self, to_sub: i64, start: i32, length: i32, overflow_index: i32, carry_index: i32);
    fn dec_sc(&self, to_sub: i64, start: i32, length: i32, carry_index: i32);
    fn inc_dec_sc(&self, to_mod: i64, start: i32, length: i32, carry_index: i32);
    fn inc_dec_sc(&self, to_mod: i64, start: i32, length: i32, overflow_index: i32, carry_index: i32);
    fn mul(&self, to_mul: i64, start: i32, carry_start: i32, length: i32);
    fn div(&self, to_div: i64, start: i32, carry_start: i32, length: i32);
    fn mul_mod_n_out(&self, to_mul: i64, mod_n: i64, in_start: i32, out_start: i32, length: i32);
    fn imul_mod_n_out(&self, to_mul: i64, mod_n: i64, in_start: i32, out_start: i32, length: i32);
    fn pow_mod_n_out(&self, base: i64, mod_n: i64, in_start: i32, out_start: i32, length: i32);
    fn cmul(&self, to_mul: i64, start: i32, carry_start: i32, length: i32, controls: &[i32]);
    fn cdiv(&self, to_div: i64, start: i32, carry_start: i32, length: i32, controls: &[i32]);
    fn cmul_mod_n_out(&self, to_mul: i64, mod_n: i64, in_start: i32, out_start: i32, length: i32, controls: &[i32]);
    fn cimul_mod_n_out(&self, to_mul: i64, mod_n: i64, in_start: i32, out_start: i32, length: i32, controls: &[i32]);
    fn cpow_mod_n_out(&self, base: i64, mod_n: i64, in_start: i32, out_start: i32, length: i32, controls: &[i32]);
    fn inc_bcd(&self, to_add: i64, start: i32, length: i32);
    fn dec_bcd(&self, to_sub: i64, start: i32, length: i32);
    fn inc_dec_bcd_c(&self, to_mod: i64, start: i32, length: i32, carry_index: i32);
    fn indexed_lda(&self, index_start: i32, index_length: i32, value_start: i32, value_length: i32, values: &[u8], reset_value: bool) -> i64;
    fn indexed_adc(&self, index_start: i32, index_length: i32, value_start: i32, value_length: i32, carry_index: i32, values: &[u8]) -> i64;
    fn indexed_sbc(&self, index_start: i32, index_length: i32, value_start: i32, value_length: i32, carry_index: i32, values: &[u8]) -> i64;
    fn hash(&self, start: i32, length: i32, values: &[u8]);
    fn inc_bcdc(&self, to_add: i64, start: i32, length: i32, carry_index: i32);
    fn dec_bcdc(&self, to_sub: i64, start: i32, length: i32, carry_index: i32);
}

pub type QAluPtr = Rc<dyn QAlu>;


