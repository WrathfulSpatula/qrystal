#[cfg(target_os = "windows")]
use std::arch::x86_64::*;

#[cfg(not(target_os = "windows"))]
use std::arch::x86::*;

use std::complex;

pub struct Complex2 {
    c2: __m128,
}

impl Complex2 {
    pub fn new() -> Self {
        Complex2 { c2: unsafe { _mm_setzero_ps() } }
    }

    pub fn from_cmplx(cm1: complex::Complex<f32>, cm2: complex::Complex<f32>) -> Self {
        let c2 = unsafe { _mm_set_ps(cm2.imag(), cm2.real(), cm1.imag(), cm1.real()) };
        Complex2 { c2 }
    }

    pub fn from_floats(r1: f32, i1: f32, r2: f32, i2: f32) -> Self {
        let c2 = unsafe { _mm_set_ps(i2, r2, i1, r1) };
        Complex2 { c2 }
    }

    pub fn c(&self, i: usize) -> complex::Complex<f32> {
        complex::Complex::new(self.f[i << 1], self.f[(i << 1) + 1])
    }

    pub fn add(&self, other: &Complex2) -> Complex2 {
        let result = unsafe { _mm_add_ps(self.c2, other.c2) };
        Complex2 { c2: result }
    }

    pub fn add_assign(&mut self, other: &Complex2) {
        self.c2 = unsafe { _mm_add_ps(self.c2, other.c2) };
    }

    pub fn sub(&self, other: &Complex2) -> Complex2 {
        let result = unsafe { _mm_sub_ps(self.c2, other.c2) };
        Complex2 { c2: result }
    }

    pub fn sub_assign(&mut self, other: &Complex2) {
        self.c2 = unsafe { _mm_sub_ps(self.c2, other.c2) };
    }

    pub fn mul(&self, other: &Complex2) -> Complex2 {
        let o_val2 = other.c2;
        let result = unsafe {
            _mm_add_ps(
                _mm_mul_ps(_mm_shuffle_ps(self.c2, self.c2, 177), _mm_xor_ps(SIGNMASK, _mm_shuffle_ps(o_val2, o_val2, 245))),
                _mm_mul_ps(self.c2, _mm_shuffle_ps(o_val2, o_val2, 160)),
            )
        };
        Complex2 { c2: result }
    }

    pub fn mul_assign(&mut self, other: &Complex2) {
        let o_val2 = other.c2;
        self.c2 = unsafe {
            _mm_add_ps(
                _mm_mul_ps(_mm_shuffle_ps(self.c2, self.c2, 177), _mm_xor_ps(SIGNMASK, _mm_shuffle_ps(o_val2, o_val2, 245))),
                _mm_mul_ps(self.c2, _mm_shuffle_ps(o_val2, o_val2, 160)),
            )
        };
    }

    pub fn mul_scalar(&self, rhs: f32) -> Complex2 {
        let result = unsafe { _mm_mul_ps(self.c2, _mm_set1_ps(rhs)) };
        Complex2 { c2: result }
    }

    pub fn neg(&self) -> Complex2 {
        let result = unsafe { _mm_mul_ps(_mm_set1_ps(-1.0), self.c2) };
        Complex2 { c2: result }
    }

    pub fn mul_assign_scalar(&mut self, other: f32) {
        self.c2 = unsafe { _mm_mul_ps(self.c2, _mm_set1_ps(other)) };
    }
}

pub fn mtrx_col_shuff(mtrx_col: &Complex2) -> Complex2 {
    let result = unsafe { _mm_shuffle_ps(mtrx_col.c2, mtrx_col.c2, 177) };
    Complex2 { c2: result }
}

pub fn matrix_mul(mtrx_col1: &Complex2, mtrx_col2: &Complex2, mtrx_col1_shuff: &Complex2, mtrx_col2_shuff: &Complex2, qubit: &Complex2) -> Complex2 {
    let col1 = mtrx_col1.c2;
    let col2 = mtrx_col2.c2;
    let dupe_lo = unsafe { _mm_shuffle_ps(qubit.c2, qubit.c2, 68) };
    let dupe_hi = unsafe { _mm_shuffle_ps(qubit.c2, qubit.c2, 238) };
    let result = unsafe {
        _mm_add_ps(
            _mm_add_ps(
                _mm_mul_ps(mtrx_col1_shuff.c2, _mm_xor_ps(SIGNMASK, _mm_shuffle_ps(dupe_lo, dupe_lo, 245))),
                _mm_mul_ps(col1, _mm_shuffle_ps(dupe_lo, dupe_lo, 160)),
            ),
            _mm_add_ps(
                _mm_mul_ps(mtrx_col2_shuff.c2, _mm_xor_ps(SIGNMASK, _mm_shuffle_ps(dupe_hi, dupe_hi, 245))),
                _mm_mul_ps(col2, _mm_shuffle_ps(dupe_hi, dupe_hi, 160))),
        )
    };
    Complex2 { c2: result }
}

pub fn matrix_mul_scalar(nrm: f32, mtrx_col1: &Complex2, mtrx_col2: &Complex2, mtrx_col1_shuff: &Complex2, mtrx_col2_shuff: &Complex2, qubit: &Complex2) -> Complex2 {
    matrix_mul(mtrx_col1, mtrx_col2, mtrx_col1_shuff, mtrx_col2_shuff, qubit).mul_scalar(nrm)
}

pub fn mul_scalar(lhs: f32, rhs: &Complex2) -> Complex2 {
    let result = unsafe { _mm_mul_ps(_mm_set1_ps(lhs), rhs.c2) };
    Complex2 { c2: result }
}

pub fn norm(c: &Complex2) -> f32 {
    let n = unsafe { _mm_mul_ps(c.c2, c.c2) };
    let shuf = unsafe { _mm_shuffle_ps(n, n, _MM_SHUFFLE(2, 3, 0, 1)) };
    let sums = unsafe { _mm_add_ps(n, shuf) };
    let shuf = unsafe { _mm_movehl_ps(shuf, sums) };
    let result = unsafe { _mm_add_ss(sums, shuf) };
    unsafe { _mm_cvtss_f32(result) }
}


