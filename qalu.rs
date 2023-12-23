use std::rc::Rc;

pub trait QAlu {
    fn m(&self, qubit_index: i32) -> bool;
    fn x(&self, qubit_index: i32);
    fn phase_flip_if_less(&self, greater_perm: i64, start: i32, length: i32);
    fn c_phase_flip_if_less(&self, greater_perm: i64, start: i32, length: i32, flag_index: i32);
    fn inc(&self, to_add: i64, start: i32, length: i32);
    fn dec(&self, to_sub: i64, start: i32, length: i32);
    fn c_inc(&self, to_add: i64, start: i32, length: i32, controls: &Vec<i32>);
    fn c_dec(&self, to_sub: i64, start: i32, length: i32, controls: &Vec<i32>);
    fn inc_dec(&self, to_mod: i64, start: i32, length: i32, carry_index: i32);
    fn dec_dec(&self, to_mod: i64, start: i32, length: i32, carry_index: i32);
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
    fn cmul(&self, to_mul: i64, start: i32, carry_start: i32, length: i32, controls: &Vec<i32>);
    fn cdiv(&self, to_div: i64, start: i32, carry_start: i32, length: i32, controls: &Vec<i32>);
    fn cmul_mod_n_out(&self, to_mul: i64, mod_n: i64, in_start: i32, out_start: i32, length: i32, controls: &Vec<i32>);
    fn cimul_mod_n_out(&self, to_mul: i64, mod_n: i64, in_start: i32, out_start: i32, length: i32, controls: &Vec<i32>);
    fn cpow_mod_n_out(&self, base: i64, mod_n: i64, in_start: i32, out_start: i32, length: i32, controls: &Vec<i32>);
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

pub struct QAluImpl {}

impl QAlu for QAluImpl {
    fn m(&self, qubit_index: i32) -> bool {
        unimplemented!()
    }
    fn x(&self, qubit_index: i32) {
        unimplemented!()
    }
    fn phase_flip_if_less(&self, greater_perm: i64, start: i32, length: i32) {
        unimplemented!()
    }
    fn c_phase_flip_if_less(&self, greater_perm: i64, start: i32, length: i32, flag_index: i32) {
        unimplemented!()
    }
    fn inc(&self, to_add: i64, start: i32, length: i32) {
        unimplemented!()
    }
    fn dec(&self, to_sub: i64, start: i32, length: i32) {
        let inv_to_sub = 2_i64.pow(length as u32) - to_sub;
        self.inc(inv_to_sub, start, length);
    }
    fn c_inc(&self, to_add: i64, start: i32, length: i32, controls: &Vec<i32>) {
        unimplemented!()
    }
    fn c_dec(&self, to_sub: i64, start: i32, length: i32, controls: &Vec<i32>) {
        let inv_to_sub = 2_i64.pow(length as u32) - to_sub;
        self.c_inc(inv_to_sub, start, length, controls);
    }
    fn inc_dec(&self, to_mod: i64, start: i32, length: i32, carry_index: i32) {
        unimplemented!()
    }
    fn dec_dec(&self, to_mod: i64, start: i32, length: i32, carry_index: i32) {
        unimplemented!()
    }
    fn inc_s(&self, to_add: i64, start: i32, length: i32, overflow_index: i32) {
        unimplemented!()
    }
    fn dec_s(&self, to_add: i64, start: i32, length: i32, overflow_index: i32) {
        unimplemented!()
    }
    fn inc_sc(&self, to_add: i64, start: i32, length: i32, overflow_index: i32, carry_index: i32) {
        unimplemented!()
    }
    fn inc_sc(&self, to_add: i64, start: i32, length: i32, carry_index: i32) {
        unimplemented!()
    }
    fn dec_sc(&self, to_sub: i64, start: i32, length: i32, overflow_index: i32, carry_index: i32) {
        unimplemented!()
    }
    fn dec_sc(&self, to_sub: i64, start: i32, length: i32, carry_index: i32) {
        unimplemented!()
    }
    fn inc_dec_sc(&self, to_mod: i64, start: i32, length: i32, carry_index: i32) {
        unimplemented!()
    }
    fn inc_dec_sc(&self, to_mod: i64, start: i32, length: i32, overflow_index: i32, carry_index: i32) {
        unimplemented!()
    }
    fn mul(&self, to_mul: i64, start: i32, carry_start: i32, length: i32) {
        unimplemented!()
    }
    fn div(&self, to_div: i64, start: i32, carry_start: i32, length: i32) {
        unimplemented!()
    }
    fn mul_mod_n_out(&self, to_mul: i64, mod_n: i64, in_start: i32, out_start: i32, length: i32) {
        unimplemented!()
    }
    fn imul_mod_n_out(&self, to_mul: i64, mod_n: i64, in_start: i32, out_start: i32, length: i32) {
        unimplemented!()
    }
    fn pow_mod_n_out(&self, base: i64, mod_n: i64, in_start: i32, out_start: i32, length: i32) {
        unimplemented!()
    }
    fn cmul(&self, to_mul: i64, start: i32, carry_start: i32, length: i32, controls: &Vec<i32>) {
        unimplemented!()
    }
    fn cdiv(&self, to_div: i64, start: i32, carry_start: i32, length: i32, controls: &Vec<i32>) {
        unimplemented!()
    }
    fn cmul_mod_n_out(&self, to_mul: i64, mod_n: i64, in_start: i32, out_start: i32, length: i32, controls: &Vec<i32>) {
        unimplemented!()
    }
    fn cimul_mod_n_out(&self, to_mul: i64, mod_n: i64, in_start: i32, out_start: i32, length: i32, controls: &Vec<i32>) {
        unimplemented!()
    }
    fn cpow_mod_n_out(&self, base: i64, mod_n: i64, in_start: i32, out_start: i32, length: i32, controls: &Vec<i32>) {
        unimplemented!()
    }
    fn inc_bcd(&self, to_add: i64, start: i32, length: i32) {
        unimplemented!()
    }
    fn dec_bcd(&self, to_sub: i64, start: i32, length: i32) {
        let inv_to_sub = 10_i64.pow(length as u32 / 4) - to_sub;
        self.inc_bcd(inv_to_sub, start, length);
    }
    fn inc_dec_bcd_c(&self, to_mod: i64, start: i32, length: i32, carry_index: i32) {
        unimplemented!()
    }
    fn indexed_lda(&self, index_start: i32, index_length: i32, value_start: i32, value_length: i32, values: &[u8], reset_value: bool) -> i64 {
        unimplemented!()
    }
    fn indexed_adc(&self, index_start: i32, index_length: i32, value_start: i32, value_length: i32, carry_index: i32, values: &[u8]) -> i64 {
        unimplemented!()
    }
    fn indexed_sbc(&self, index_start: i32, index_length: i32, value_start: i32, value_length: i32, carry_index: i32, values: &[u8]) -> i64 {
        unimplemented!()
    }
    fn hash(&self, start: i32, length: i32, values: &[u8]) {
        unimplemented!()
    }
    fn inc_bcdc(&self, to_add: i64, start: i32, length: i32, carry_index: i32) {
        unimplemented!()
    }
    fn dec_bcdc(&self, to_sub: i64, start: i32, length: i32, carry_index: i32) {
        unimplemented!()
    }
}

