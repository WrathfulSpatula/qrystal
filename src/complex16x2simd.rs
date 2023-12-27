#[cfg(target_os = "windows")]
use std::arch::x86_64::*;
#[cfg(not(target_os = "windows"))]
use std::arch::x86_64::{__m256d, _mm256_set_pd, _mm256_add_pd, _mm256_sub_pd, _mm256_mul_pd, _mm256_shuffle_pd, _mm256_permute2f128_pd, _mm256_xor_pd};

use std::complex;

namespace Qrack {
    static const SIGNMASK: __m256d = _mm256_set_pd(-0.0, -0.0, -0.0, -0.0);

    union complex2 {
        c2: __m256d,
        f: [f64; 4],
    }

    impl complex2 {
        fn new() -> Self {
            complex2 { c2: _mm256_set_pd(0.0, 0.0, 0.0, 0.0) }
        }

        fn from_cm2(cm2: __m256d) -> Self {
            complex2 { c2: cm2 }
        }

        fn from_complex(cm1: std::complex<f64>, cm2: std::complex<f64>) -> Self {
            complex2 { c2: _mm256_set_pd(cm2.imag(), cm2.real(), cm1.imag(), cm1.real()) }
        }

        fn from_values(r1: f64, i1: f64, r2: f64, i2: f64) -> Self {
            complex2 { c2: _mm256_set_pd(i2, r2, i1, r1) }
        }

        fn c(&self, i: usize) -> std::complex<f64> {
            std::complex(self.f[i << 1], self.f[(i << 1) + 1])
        }

        fn add(&self, other: &Self) -> Self {
            complex2 { c2: _mm256_add_pd(self.c2, other.c2) }
        }

        fn add_assign(&mut self, other: &Self) -> Self {
            self.c2 = _mm256_add_pd(self.c2, other.c2);
            self.c2
        }

        fn sub(&self, other: &Self) -> Self {
            complex2 { c2: _mm256_sub_pd(self.c2, other.c2) }
        }

        fn sub_assign(&mut self, other: &Self) -> Self {
            self.c2 = _mm256_sub_pd(self.c2, other.c2);
            self.c2
        }

        fn mul(&self, other: &Self) -> Self {
            complex2 {
                c2: _mm256_add_pd(
                    _mm256_mul_pd(
                        _mm256_shuffle_pd(self.c2, self.c2, 5),
                        _mm256_shuffle_pd(_mm256_xor_pd(SIGNMASK, other.c2), other.c2, 15),
                    ),
                    _mm256_mul_pd(self.c2, _mm256_shuffle_pd(other.c2, other.c2, 0)),
                ),
            }
        }

        fn mul_assign(&mut self, other: &Self) -> Self {
            self.c2 = _mm256_add_pd(
                _mm256_mul_pd(
                    _mm256_shuffle_pd(self.c2, self.c2, 5),
                    _mm256_shuffle_pd(_mm256_xor_pd(SIGNMASK, other.c2), other.c2, 15),
                ),
                _mm256_mul_pd(self.c2, _mm256_shuffle_pd(other.c2, other.c2, 0)),
            );
            self.c2
        }

        fn mul_scalar(&self, rhs: f64) -> Self {
            complex2 { c2: _mm256_mul_pd(self.c2, _mm256_set1_pd(rhs)) }
        }

        fn neg(&self) -> Self {
            complex2 { c2: _mm256_mul_pd(_mm256_set1_pd(-1.0), self.c2) }
        }

        fn mul_assign_scalar(&mut self, other: f64) -> Self {
            self.c2 = _mm256_mul_pd(self.c2, _mm256_set1_pd(other));
            self.c2
        }
    }

    fn mtrx_col_shuff(mtrx_col: &complex2) -> complex2 {
        complex2 { c2: _mm256_shuffle_pd(mtrx_col.c2, mtrx_col.c2, 5) }
    }

    fn matrix_mul(mtrx_col1: &complex2, mtrx_col2: &complex2, mtrx_col1_shuff: &complex2, mtrx_col2_shuff: &complex2, qubit: &complex2) -> complex2 {
        let col1 = mtrx_col1.c2;
        let col2 = mtrx_col2.c2;
        let dupe_lo = _mm256_permute2f128_pd(qubit.c2, qubit.c2, 0);
        let dupe_hi = _mm256_permute2f128_pd(qubit.c2, qubit.c2, 17);
        complex2 {
            c2: _mm256_add_pd(
                _mm256_add_pd(
                    _mm256_mul_pd(
                        mtrx_col1_shuff.c2,
                        _mm256_shuffle_pd(_mm256_xor_pd(SIGNMASK, dupe_lo), dupe_lo, 15),
                    ),
                    _mm256_mul_pd(col1, _mm256_shuffle_pd(dupe_lo, dupe_lo, 0)),
                ),
                _mm256_add_pd(
                    _mm256_mul_pd(
                        mtrx_col2_shuff.c2,
                        _mm256_shuffle_pd(_mm256_xor_pd(SIGNMASK, dupe_hi), dupe_hi, 15),
                    ),
                    _mm256_mul_pd(col2, _mm256_shuffle_pd(dupe_hi, dupe_hi, 0)),
                ),
            ),
        }
    }

    fn matrix_mul_scalar(nrm: f32, mtrx_col1: &complex2, mtrx_col2: &complex2, mtrx_col1_shuff: &complex2, mtrx_col2_shuff: &complex2, qubit: &complex2) -> complex2 {
        matrix_mul(mtrx_col1, mtrx_col2, mtrx_col1_shuff, mtrx_col2_shuff, qubit).mul_scalar(nrm)
    }

    fn mul_scalar(lhs: f64, rhs: &complex2) -> complex2 {
        complex2 { c2: _mm256_mul_pd(_mm256_set1_pd(lhs), rhs.c2) }
    }

    fn norm(c: &complex2) -> f64 {
        let cu = complex2 { c2: _mm256_mul_pd(c.c2, c.c2) };
        cu.f[0] + cu.f[1] + cu.f[2] + cu.f[3]
    }
}


