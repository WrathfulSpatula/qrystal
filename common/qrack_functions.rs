use std::collections::HashSet;

fn bi_not_ip(left: &mut u32) {
    *left = !(*left);
}

fn bi_and_ip(left: &mut u32, right: u32) {
    *left &= right;
}

fn bi_or_ip(left: &mut u32, right: u32) {
    *left |= right;
}

fn bi_xor_ip(left: &mut u32, right: u32) {
    *left ^= right;
}

fn bi_to_double(in_: u32) -> f64 {
    in_ as f64
}

fn bi_increment(p_big_int: &mut u32, value: u32) {
    *p_big_int += value;
}

fn bi_decrement(p_big_int: &mut u32, value: u32) {
    *p_big_int -= value;
}

fn bi_lshift_ip(left: &mut u32, right: u32) {
    *left <<= right;
}

fn bi_rshift_ip(left: &mut u32, right: u32) {
    *left >>= right;
}

fn bi_and_1(left: u32) -> i32 {
    (left & 1) as i32
}

fn bi_compare(left: u32, right: u32) -> i32 {
    if left > right {
        return 1;
    }
    if left < right {
        return -1;
    }
    0
}

fn bi_compare_0(left: u32) -> i32 {
    if left != 0 {
        return 1;
    }
    0
}

fn bi_compare_1(left: u32) -> i32 {
    if left > 1 {
        return 1;
    }
    if left < 1 {
        return -1;
    }
    0
}

fn bi_add_ip(left: &mut u32, right: u32) {
    *left += right;
}

fn bi_sub_ip(left: &mut u32, right: u32) {
    *left -= right;
}

fn bi_div_mod(left: u32, right: u32, quotient: &mut u32, rmndr: &mut u32) {
    if quotient != 0 {
        *quotient = left / right;
    }
    if rmndr != 0 {
        *rmndr = left % right;
    }
}

fn bi_div_mod_small(left: u32, right: u32, quotient: &mut u32, rmndr: &mut u32) {
    if quotient != 0 {
        *quotient = left / right;
    }
    if rmndr != 0 {
        *rmndr = left % right;
    }
}

fn bi_not_ip_64(left: &mut u64) {
    *left = !(*left);
}

fn bi_and_ip_64(left: &mut u64, right: u64) {
    *left &= right;
}

fn bi_or_ip_64(left: &mut u64, right: u64) {
    *left |= right;
}

fn bi_xor_ip_64(left: &mut u64, right: u64) {
    *left ^= right;
}

fn bi_to_double_64(in_: u64) -> f64 {
    in_ as f64
}

fn bi_increment_64(p_big_int: &mut u64, value: u64) {
    *p_big_int += value;
}

fn bi_decrement_64(p_big_int: &mut u64, value: u64) {
    *p_big_int -= value;
}

fn bi_lshift_ip_64(left: &mut u64, right: u64) {
    *left <<= right;
}

fn bi_rshift_ip_64(left: &mut u64, right: u64) {
    *left >>= right;
}

fn bi_and_1_64(left: u64) -> i32 {
    (left & 1) as i32
}

fn bi_compare_64(left: u64, right: u64) -> i32 {
    if left > right {
        return 1;
    }
    if left < right {
        return -1;
    }
    0
}

fn bi_compare_0_64(left: u64) -> i32 {
    if left != 0 {
        return 1;
    }
    0
}

fn bi_compare_1_64(left: u64) -> i32 {
    if left > 1 {
        return 1;
    }
    if left < 1 {
        return -1;
    }
    0
}

fn bi_add_ip_64(left: &mut u64, right: u64) {
    *left += right;
}

fn bi_sub_ip_64(left: &mut u64, right: u64) {
    *left -= right;
}

fn bi_div_mod_64(left: u64, right: u64, quotient: &mut u64, rmndr: &mut u64) {
    if quotient != 0 {
        *quotient = left / right;
    }
    if rmndr != 0 {
        *rmndr = left % right;
    }
}

fn bi_div_mod_small_64(left: u64, right: u64, quotient: &mut u64, rmndr: &mut u64) {
    if quotient != 0 {
        *quotient = left / right;
    }
    if rmndr != 0 {
        *rmndr = left % right;
    }
}

fn log2_ocl(n: u32) -> i32 {
    (32 - n.leading_zeros() - 1) as i32
}

fn bi_log2(n: u64) -> i32 {
    (64 - n.leading_zeros() - 1) as i32
}

fn log2(n: u32) -> i32 {
    bi_log2(n)
}

fn pow2(p: i32) -> u32 {
    1 << p
}

fn pow2_ocl(p: i32) -> u32 {
    1 << p
}

fn pow2_mask(p: i32) -> u32 {
    (1 << p) - 1
}

fn pow2_mask_ocl(p: i32) -> u32 {
    (1 << p) - 1
}

fn bit_slice(bit: i32, source: u32) -> u32 {
    (1 << bit) & source
}

fn bit_slice_ocl(bit: i32, source: u32) -> u32 {
    (1 << bit) & source
}

fn bit_reg_mask(start: i32, length: i32) -> u32 {
    let mut to_ret = 1 << length;
    to_ret -= 1;
    to_ret <<= start;
    to_ret
}

fn bit_reg_mask_ocl(start: i32, length: i32) -> u32 {
    ((1 << length) - 1) << start
}

fn is_power_of_two(x: u32) -> bool {
    let y = x;
    let mut y = y - 1;
    y &= x;
    x != 0 && y == 0
}

fn is_power_of_two_ocl(x: u32) -> bool {
    x != 0 && (x & (x - 1)) == 0
}

fn is_bad_bit_range(start: i32, length: i32, qubit_count: i32) -> bool {
    (start + length) > qubit_count || (start + length) < start
}

fn is_bad_perm_range(start: u32, length: u32, max_q_power_ocl: u32) -> bool {
    (start + length) > max_q_power_ocl || (start + length) < start
}

fn throw_if_qb_id_array_is_bad(controls: &[i32], qubit_count: i32, message: &str) {
    let mut dupes = HashSet::new();
    for &control in controls {
        if control >= qubit_count {
            panic!(message);
        }
        if !dupes.insert(control) {
            panic!(message.to_string() + " (Found duplicate qubit indices!)");
        }
    }
}

fn cl_alloc(uchar_count: usize) -> *mut u8 {
    let mut buffer = Vec::with_capacity(uchar_count);
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

fn cl_free(to_free: *mut u8) {
    unsafe {
        Vec::from_raw_parts(to_free, 0, 0);
    }
}

fn mul2x2(left: &[complex], right: &[complex], out: &mut [complex]) {
    out[0] = complex {
        re: left[0].re * right[0].re - left[0].im * right[0].im,
        im: left[0].re * right[0].im + left[0].im * right[0].re,
    };
    out[1] = complex {
        re: left[1].re * right[0].re - left[1].im * right[0].im,
        im: left[1].re * right[0].im + left[1].im * right[0].re,
    };
    out[2] = complex {
        re: left[0].re * right[1].re - left[0].im * right[1].im,
        im: left[0].re * right[1].im + left[0].im * right[1].re,
    };
    out[3] = complex {
        re: left[1].re * right[1].re - left[1].im * right[1].im,
        im: left[1].re * right[1].im + left[1].im * right[1].re,
    };
}

fn exp2x2(matrix2x2: &[complex], out_matrix2x2: &mut [complex]) {
    let a = matrix2x2[0].re;
    let b = matrix2x2[0].im;
    let c = matrix2x2[1].re;
    let d = matrix2x2[1].im;

    let exp_a = a.exp();
    let cos_b = b.cos();
    let sin_b = b.sin();

    out_matrix2x2[0] = complex {
        re: exp_a * cos_b,
        im: exp_a * sin_b,
    };
    out_matrix2x2[1] = complex {
        re: c * cos_b - d * sin_b,
        im: c * sin_b + d * cos_b,
    };
    out_matrix2x2[2] = complex {
        re: a * cos_b - b * sin_b,
        im: a * sin_b + b * cos_b,
    };
    out_matrix2x2[3] = complex {
        re: exp_a * cos_b,
        im: exp_a * -sin_b,
    };
}

fn log2x2(matrix2x2: &[complex], out_matrix2x2: &mut [complex]) {
    let a = matrix2x2[0].re;
    let b = matrix2x2[0].im;
    let c = matrix2x2[1].re;
    let d = matrix2x2[1].im;

    let abs_a = a.abs();
    let abs_b = b.abs();
    let abs_c = c.abs();
    let abs_d = d.abs();

    let log_abs_a = abs_a.ln();
    let log_abs_b = abs_b.ln();
    let log_abs_c = abs_c.ln();
    let log_abs_d = abs_d.ln();

    let arg_a = a.atan2(b);
    let arg_c = c.atan2(d);

    out_matrix2x2[0] = complex {
        re: log_abs_a + arg_a,
        im: arg_a,
    };
    out_matrix2x2[1] = complex {
        re: log_abs_c + arg_c,
        im: arg_c,
    };
    out_matrix2x2[2] = complex {
        re: log_abs_a + arg_a,
        im: arg_a + std::f64::consts::PI,
    };
    out_matrix2x2[3] = complex {
        re: log_abs_c + arg_c,
        im: arg_c + std::f64::consts::PI,
    };
}

fn inv2x2(matrix2x2: &[complex], out_matrix2x2: &mut [complex]) {
    let a = matrix2x2[0].re;
    let b = matrix2x2[0].im;
    let c = matrix2x2[1].re;
    let d = matrix2x2[1].im;

    let det = a * d - b * c;

    out_matrix2x2[0] = complex {
        re: d / det,
        im: -b / det,
    };
    out_matrix2x2[1] = complex {
        re: -c / det,
        im: a / det,
    };
    out_matrix2x2[2] = complex {
        re: -b / det,
        im: a / det,
    };
    out_matrix2x2[3] = complex {
        re: d / det,
        im: -c / det,
    };
}

fn is_overflow_add(in_out_int: u32, in_int: u32, sign_mask: u32, length_power: u32) -> bool {
    let sum = in_out_int.wrapping_add(in_int);
    (sum ^ in_out_int) & (sum ^ in_int) & sign_mask != 0
}

fn is_overflow_sub(in_out_int: u32, in_int: u32, sign_mask: u32, length_power: u32) -> bool {
    let diff = in_out_int.wrapping_sub(in_int);
    (diff ^ in_out_int) & !(diff ^ in_int) & sign_mask != 0
}

fn push_apart_bits(perm: u32, skip_powers: &[u32]) -> u32 {
    let mut result = 0;
    for &skip_power in skip_powers {
        let skip_mask = (1 << skip_power) - 1;
        let skip_bits = perm & skip_mask;
        let perm_bits = perm & !skip_mask;
        result <<= skip_power;
        result |= skip_bits;
        result <<= 1;
        result |= perm_bits;
    }
    result
}

fn int_pow(base: u32, power: u32) -> u32 {
    base.pow(power)
}

fn int_pow_ocl(base: u32, power: u32) -> u32 {
    base.pow(power)
}

const QRACK_QBDT_SEPARABILITY_THRESHOLD: f64 = std::f64::EPSILON;

fn main() {
    // Example usage
    let controls = vec![0, 1, 2];
    let qubit_count = 3;
    let message = "Invalid qubit indices";
    throw_if_qb_id_array_is_bad(&controls, qubit_count, message);
}


