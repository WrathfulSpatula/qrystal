use std::cmp::Ordering;
use std::ops::{Add, Sub, BitAnd, BitOr, BitXor, Not, Shl, Shr};

const BIG_INTEGER_WORD_BITS: u32 = 64;
const BIG_INTEGER_WORD_POWER: u32 = 6;
type BIG_INTEGER_WORD = u64;
type BIG_INTEGER_HALF_WORD = u32;
const BIG_INTEGER_HALF_WORD_POW: u64 = 0x100000000;
const BIG_INTEGER_HALF_WORD_MASK: u64 = 0xFFFFFFFF;
const BIG_INTEGER_HALF_WORD_MASK_NOT: u64 = 0xFFFFFFFF00000000;

const BIG_INTEGER_BITS: usize = 1 << QBCAPPOW;
const BIG_INTEGER_WORD_SIZE: i64 = (BIG_INTEGER_BITS / BIG_INTEGER_WORD_BITS) as i64;

const BIG_INTEGER_HALF_WORD_BITS: u32 = BIG_INTEGER_WORD_BITS >> 1;
const BIG_INTEGER_HALF_WORD_SIZE: i32 = (BIG_INTEGER_WORD_SIZE << 1) as i32;
const BIG_INTEGER_MAX_WORD_INDEX: usize = (BIG_INTEGER_WORD_SIZE - 1) as usize;

#[derive(Clone)]
struct BigInteger {
    bits: [BIG_INTEGER_WORD; BIG_INTEGER_WORD_SIZE as usize],
}

impl BigInteger {
    fn new() -> Self {
        BigInteger {
            bits: [0; BIG_INTEGER_WORD_SIZE as usize],
        }
    }

    fn from_val(val: BIG_INTEGER_WORD) -> Self {
        let mut bits = [0; BIG_INTEGER_WORD_SIZE as usize];
        bits[0] = val;
        BigInteger { bits }
    }

    fn set_0(&mut self) {
        for i in 0..BIG_INTEGER_WORD_SIZE as usize {
            self.bits[i] = 0;
        }
    }

    fn copy(&self) -> Self {
        let mut result = BigInteger::new();
        for i in 0..BIG_INTEGER_WORD_SIZE as usize {
            result.bits[i] = self.bits[i];
        }
        result
    }

    fn copy_ip(&self, out: &mut BigInteger) {
        for i in 0..BIG_INTEGER_WORD_SIZE as usize {
            out.bits[i] = self.bits[i];
        }
    }

    fn compare(&self, other: &BigInteger) -> Ordering {
        for i in (0..=BIG_INTEGER_MAX_WORD_INDEX).rev() {
            if self.bits[i] > other.bits[i] {
                return Ordering::Greater;
            }
            if self.bits[i] < other.bits[i] {
                return Ordering::Less;
            }
        }
        Ordering::Equal
    }

    fn compare_0(&self) -> Ordering {
        for i in 0..BIG_INTEGER_WORD_SIZE as usize {
            if self.bits[i] != 0 {
                return Ordering::Greater;
            }
        }
        Ordering::Equal
    }

    fn compare_1(&self) -> Ordering {
        for i in (0..BIG_INTEGER_MAX_WORD_INDEX).rev() {
            if self.bits[i] != 0 {
                return Ordering::Greater;
            }
        }
        if self.bits[0] > 1 {
            return Ordering::Greater;
        }
        if self.bits[0] < 1 {
            return Ordering::Less;
        }
        Ordering::Equal
    }

    fn add(&self, other: &BigInteger) -> BigInteger {
        let mut result = BigInteger::new();
        result.bits[0] = 0;
        for i in 0..BIG_INTEGER_MAX_WORD_INDEX {
            result.bits[i] += self.bits[i] + other.bits[i];
            result.bits[i + 1] = if result.bits[i] < self.bits[i] { 1 } else { 0 };
        }
        result.bits[BIG_INTEGER_MAX_WORD_INDEX] += other.bits[BIG_INTEGER_MAX_WORD_INDEX];
        result
    }

    fn add_ip(&mut self, other: &BigInteger) {
        for i in 0..BIG_INTEGER_MAX_WORD_INDEX {
            let temp = self.bits[i];
            self.bits[i] += other.bits[i];
            let mut j = i;
            while j < BIG_INTEGER_MAX_WORD_INDEX && self.bits[j] < temp {
                temp = self.bits[j + 1];
                self.bits[j + 1] += 1;
                j += 1;
            }
        }
        self.bits[BIG_INTEGER_MAX_WORD_INDEX] += other.bits[BIG_INTEGER_MAX_WORD_INDEX];
    }

    fn sub(&self, other: &BigInteger) -> BigInteger {
        let mut result = BigInteger::new();
        result.bits[0] = 0;
        for i in 0..BIG_INTEGER_MAX_WORD_INDEX {
            result.bits[i] += self.bits[i] - other.bits[i];
            result.bits[i + 1] = if result.bits[i] > self.bits[i] { -1 } else { 0 };
        }
        result.bits[BIG_INTEGER_MAX_WORD_INDEX] -= other.bits[BIG_INTEGER_MAX_WORD_INDEX];
        result
    }

    fn sub_ip(&mut self, other: &BigInteger) {
        for i in 0..BIG_INTEGER_MAX_WORD_INDEX {
            let temp = self.bits[i];
            self.bits[i] -= other.bits[i];
            let mut j = i;
            while j < BIG_INTEGER_MAX_WORD_INDEX && self.bits[j] > temp {
                temp = self.bits[j + 1];
                self.bits[j + 1] -= 1;
                j += 1;
            }
        }
        self.bits[BIG_INTEGER_MAX_WORD_INDEX] -= other.bits[BIG_INTEGER_MAX_WORD_INDEX];
    }

    fn increment(&mut self, value: BIG_INTEGER_WORD) {
        let mut temp = self.bits[0];
        self.bits[0] += value;
        if temp <= self.bits[0] {
            return;
        }
        for i in 1..BIG_INTEGER_WORD_SIZE as usize {
            temp = self.bits[i];
            self.bits[i] += 1;
            if temp <= self.bits[i] {
                break;
            }
        }
    }

    fn decrement(&mut self, value: BIG_INTEGER_WORD) {
        let mut temp = self.bits[0];
        self.bits[0] -= value;
        if temp >= self.bits[0] {
            return;
        }
        for i in 0..BIG_INTEGER_WORD_SIZE as usize {
            temp = self.bits[i];
            self.bits[i] -= 1;
            if temp >= self.bits[i] {
                break;
            }
        }
    }

    fn load(a: &[BIG_INTEGER_WORD]) -> BigInteger {
        let mut result = BigInteger::new();
        for i in 0..BIG_INTEGER_WORD_SIZE as usize {
            result.bits[i] = a[i];
        }
        result
    }

    fn lshift_word(&self, right_mult: BIG_INTEGER_WORD) -> BigInteger {
        if right_mult == 0 {
            return self.clone();
        }
        let mut result = BigInteger::new();
        for i in right_mult as usize..BIG_INTEGER_WORD_SIZE as usize {
            result.bits[i] = self.bits[i - right_mult as usize];
        }
        result
    }

    fn lshift_word_ip(&mut self, right_mult: BIG_INTEGER_WORD) {
        let right_mult = right_mult & 63;
        if right_mult == 0 {
            return;
        }
        for i in right_mult as usize..BIG_INTEGER_WORD_SIZE as usize {
            self.bits[i] = self.bits[i - right_mult as usize];
        }
        for i in 0..right_mult as usize {
            self.bits[i] = 0;
        }
    }

    fn rshift_word(&self, right_mult: BIG_INTEGER_WORD) -> BigInteger {
        if right_mult == 0 {
            return self.clone();
        }
        let mut result = BigInteger::new();
        for i in right_mult as usize..BIG_INTEGER_WORD_SIZE as usize {
            result.bits[i - right_mult as usize] = self.bits[i];
        }
        result
    }

    fn rshift_word_ip(&mut self, right_mult: BIG_INTEGER_WORD) {
        if right_mult == 0 {
            return;
        }
        for i in right_mult as usize..BIG_INTEGER_WORD_SIZE as usize {
            self.bits[i - right_mult as usize] = self.bits[i];
        }
        for i in 0..right_mult as usize {
            self.bits[BIG_INTEGER_MAX_WORD_INDEX - i] = 0;
        }
    }

    fn lshift(&self, right: BIG_INTEGER_WORD) -> BigInteger {
        let r_shift64 = right >> BIG_INTEGER_WORD_POWER;
        let r_mod = right - (r_shift64 << BIG_INTEGER_WORD_POWER);
        let mut result = self.lshift_word(r_shift64);
        if r_mod == 0 {
            return result;
        }
        let r_mod_comp = BIG_INTEGER_WORD_BITS - r_mod;
        let mut carry = 0;
        for i in 0..BIG_INTEGER_WORD_SIZE as usize {
            let right = result.bits[i];
            result.bits[i] = carry | (right << r_mod);
            carry = right >> r_mod_comp;
        }
        result
    }

    fn lshift_ip(&mut self, right: BIG_INTEGER_WORD) {
        let r_shift64 = right >> BIG_INTEGER_WORD_POWER;
        let r_mod = right - (r_shift64 << BIG_INTEGER_WORD_POWER);
        self.lshift_word_ip(r_shift64);
        if r_mod == 0 {
            return;
        }
        let r_mod_comp = BIG_INTEGER_WORD_BITS - r_mod;
        let mut carry = 0;
        for i in 0..BIG_INTEGER_WORD_SIZE as usize {
            let right = self.bits[i];
            self.bits[i] = carry | (right << r_mod);
            carry = right >> r_mod_comp;
        }
    }

    fn rshift(&self, right: BIG_INTEGER_WORD) -> BigInteger {
        let r_shift64 = right >> BIG_INTEGER_WORD_POWER;
        let r_mod = right - (r_shift64 << BIG_INTEGER_WORD_POWER);
        let mut result = self.rshift_word(r_shift64);
        if r_mod == 0 {
            return result;
        }
        let r_mod_comp = BIG_INTEGER_WORD_BITS - r_mod;
        let mut carry = 0;
        for i in (0..=BIG_INTEGER_MAX_WORD_INDEX).rev() {
            let right = result.bits[i];
            result.bits[i] = carry | (right >> r_mod);
            carry = right << r_mod_comp;
        }
        result
    }

    fn rshift_ip(&mut self, right: BIG_INTEGER_WORD) {
        let r_shift64 = right >> BIG_INTEGER_WORD_POWER;
        let r_mod = right - (r_shift64 << BIG_INTEGER_WORD_POWER);
        self.rshift_word_ip(r_shift64);
        if r_mod == 0 {
            return;
        }
        let r_mod_comp = BIG_INTEGER_WORD_BITS - r_mod;
        let mut carry = 0;
        for i in (0..=BIG_INTEGER_MAX_WORD_INDEX).rev() {
            let right = self.bits[i];
            self.bits[i] = carry | (right >> r_mod);
            carry = right << r_mod_comp;
        }
    }

    fn log2(&self) -> i32 {
        let mut pw = 0;
        let mut p = self.clone() >> 1;
        while p.compare_0() != Ordering::Equal {
            p.rshift_ip(1);
            pw += 1;
        }
        pw
    }

    fn and_1(&self) -> BIG_INTEGER_WORD {
        self.bits[0] & 1
    }

    fn and(&self, other: &BigInteger) -> BigInteger {
        let mut result = BigInteger::new();
        for i in 0..BIG_INTEGER_WORD_SIZE as usize {
            result.bits[i] = self.bits[i] & other.bits[i];
        }
        result
    }

    fn and_ip(&mut self, other: &BigInteger) {
        for i in 0..BIG_INTEGER_WORD_SIZE as usize {
            self.bits[i] &= other.bits[i];
        }
    }

    fn or(&self, other: &BigInteger) -> BigInteger {
        let mut result = BigInteger::new();
        for i in 0..BIG_INTEGER_WORD_SIZE as usize {
            result.bits[i] = self.bits[i] | other.bits[i];
        }
        result
    }

    fn or_ip(&mut self, other: &BigInteger) {
        for i in 0..BIG_INTEGER_WORD_SIZE as usize {
            self.bits[i] |= other.bits[i];
        }
    }

    fn xor(&self, other: &BigInteger) -> BigInteger {
        let mut result = BigInteger::new();
        for i in 0..BIG_INTEGER_WORD_SIZE as usize {
            result.bits[i] = self.bits[i] ^ other.bits[i];
        }
        result
    }

    fn xor_ip(&mut self, other: &BigInteger) {
        for i in 0..BIG_INTEGER_WORD_SIZE as usize {
            self.bits[i] ^= other.bits[i];
        }
    }

    fn not(&self) -> BigInteger {
        let mut result = BigInteger::new();
        for i in 0..BIG_INTEGER_WORD_SIZE as usize {
            result.bits[i] = !self.bits[i];
        }
        result
    }

    fn not_ip(&mut self) {
        for i in 0..BIG_INTEGER_WORD_SIZE as usize {
            self.bits[i] = !self.bits[i];
        }
    }

    fn to_double(&self) -> f64 {
        let mut to_ret = 0.0;
        for i in 0..BIG_INTEGER_WORD_SIZE as usize {
            if self.bits[i] != 0 {
                to_ret += self.bits[i] as f64 * (2.0 as f64).powi((BIG_INTEGER_WORD_BITS * i) as i32);
            }
        }
        to_ret
    }

    fn bi_div_mod_small(&self, right: u32, quotient: &mut BigInteger, rmndr: &mut u32) {
        let mut carry: u64 = 0;
        if let Some(quotient) = quotient {
            bi_set_0(quotient);
            for i in (0..BIG_INTEGER_HALF_WORD_SIZE).rev() {
                let i2 = i >> 1;
                carry <<= BIG_INTEGER_HALF_WORD_BITS;
                if i & 1 != 0 {
                    carry |= self.bits[i2] >> BIG_INTEGER_HALF_WORD_BITS;
                    quotient.bits[i2] |= (carry / right) << BIG_INTEGER_HALF_WORD_BITS;
                } else {
                    carry |= self.bits[i2] & BIG_INTEGER_HALF_WORD_MASK;
                    quotient.bits[i2] |= carry / right;
                }
                carry %= right as u64;
            }
        } else {
            for i in (0..BIG_INTEGER_HALF_WORD_SIZE).rev() {
                let i2 = i >> 1;
                carry <<= BIG_INTEGER_HALF_WORD_BITS;
                if i & 1 != 0 {
                    carry |= self.bits[i2] >> BIG_INTEGER_HALF_WORD_BITS;
                } else {
                    carry |= self.bits[i2] & BIG_INTEGER_HALF_WORD_MASK;
                }
                carry %= right as u64;
            }
        }
        if let Some(rmndr) = rmndr {
            *rmndr = carry as u32;
        }
    }

    fn bi_div_mod(&self, right: &BigInteger, quotient: &mut BigInteger, rmndr: &mut BigInteger) {
        let lr_compare = bi_compare(self, right);
        if lr_compare < 0 {
            if let Some(quotient) = quotient {
                bi_set_0(quotient);
            }
            if let Some(rmndr) = rmndr {
                bi_copy_ip(self, rmndr);
            }
            return;
        }
        if lr_compare == 0 {
            if let Some(quotient) = quotient {
                bi_set_0(quotient);
                quotient.bits[0] = 1;
            }
            if let Some(rmndr) = rmndr {
                bi_set_0(rmndr);
            }
            return;
        }
        if right.bits[0] < BIG_INTEGER_HALF_WORD_POW {
            let mut word_size = 1;
            for i in 1..BIG_INTEGER_WORD_SIZE {
                if right.bits[i] != 0 {
                    break;
                }
                word_size += 1;
            }
            if word_size >= BIG_INTEGER_WORD_SIZE {
                if let Some(rmndr) = rmndr {
                    let t = bi_div_mod_small(self, right.bits[0] as u32, quotient, &mut t);
                    rmndr.bits[0] = t;
                    for i in 1..BIG_INTEGER_WORD_SIZE {
                        rmndr.bits[i] = 0;
                    }
                } else {
                    bi_div_mod_small(self, right.bits[0] as u32, quotient, 0);
                }
                return;
            }
        }
        let bi1 = 1;
        let right_log2 = bi_log2(right);
        let right_test = bi1 << right_log2;
        if bi_compare(right, right_test) < 0 {
            right_log2 += 1;
        }
        let mut rem = BigInteger::new();
        bi_copy_ip(self, &mut rem);
        if let Some(quotient) = quotient {
            bi_set_0(quotient);
            while bi_compare(&rem, right) >= 0 {
                let log_diff = bi_log2(&rem) - right_log2;
                if log_diff > 0 {
                    let part_mul = right << log_diff;
                    let part_quo = bi1 << log_diff;
                    bi_sub_ip(&mut rem, &part_mul);
                    bi_add_ip(quotient, &part_quo);
                } else {
                    bi_sub_ip(&mut rem, right);
                    bi_increment(quotient, 1);
                }
            }
        } else {
            while bi_compare(&rem, right) >= 0 {
                let log_diff = bi_log2(&rem) - right_log2;
                if log_diff > 0 {
                    let part_mul = right << log_diff;
                    bi_sub_ip(&mut rem, &part_mul);
                } else {
                    bi_sub_ip(&mut rem, right);
                }
            }
        }
        if let Some(rmndr) = rmndr {
            *rmndr = rem;
        }
    }
}

impl PartialEq for BigInteger {
    fn eq(&self, other: &Self) -> bool {
        self.compare(other) == Ordering::Equal
    }
}

impl PartialOrd for BigInteger {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.compare(other))
    }
}

impl Add for BigInteger {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.add(&other)
    }
}

impl Sub for BigInteger {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.sub(&other)
    }
}

impl BitAnd for BigInteger {
    type Output = Self;

    fn bitand(self, other: Self) -> Self {
        self.and(&other)
    }
}

impl BitOr for BigInteger {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        self.or(&other)
    }
}

impl BitXor for BigInteger {
    type Output = Self;

    fn bitxor(self, other: Self) -> Self {
        self.xor(&other)
    }
}

impl Not for BigInteger {
    type Output = Self;

    fn not(self) -> Self {
        self.not()
    }
}

impl Shl<BIG_INTEGER_WORD> for BigInteger {
    type Output = Self;

    fn shl(self, right: BIG_INTEGER_WORD) -> Self {
        self.lshift(right)
    }
}

impl Shr<BIG_INTEGER_WORD> for BigInteger {
    type Output = Self;

    fn shr(self, right: BIG_INTEGER_WORD) -> Self {
        self.rshift(right)
    }
}

fn main() {
    let _ = BigInteger::new();
}


