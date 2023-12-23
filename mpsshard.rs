use std::cmp::PartialEq;
use std::ops::{Add, Sub, Mul, Div};
use std::f64::consts::PI;

#[derive(Clone, Copy)]
struct Complex {
    real: f64,
    imag: f64,
}

impl Complex {
    fn new(real: f64, imag: f64) -> Self {
        Complex { real, imag }
    }

    fn norm(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
    }

    fn abs(&self) -> f64 {
        self.norm().sqrt()
    }
}

impl PartialEq for Complex {
    fn eq(&self, other: &Self) -> bool {
        self.real == other.real && self.imag == other.imag
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Complex::new(self.real + other.real, self.imag + other.imag)
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Complex::new(self.real - other.real, self.imag - other.imag)
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Complex::new(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )
    }
}

impl Div for Complex {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let denom = other.norm();
        Complex::new(
            (self.real * other.real + self.imag * other.imag) / denom,
            (self.imag * other.real - self.real * other.imag) / denom,
        )
    }
}

const ONE_CMPLX: Complex = Complex { real: 1.0, imag: 0.0 };
const ZERO_CMPLX: Complex = Complex { real: 0.0, imag: 0.0 };
const ZERO_R1: f64 = 0.0;
const FP_NORM_EPSILON: f64 = 1e-6;
const SQRT1_2_R1: f64 = 1.0 / 2.0_f64.sqrt();

struct MpsShard {
    gate: [Complex; 4],
}

impl MpsShard {
    fn new() -> Self {
        MpsShard {
            gate: [ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ONE_CMPLX],
        }
    }

    fn from_array(g: &[Complex; 4]) -> Self {
        MpsShard { gate: *g }
    }

    fn clone(&self) -> MpsShard {
        MpsShard::from_array(&self.gate)
    }

    fn compose(&mut self, g: &[Complex; 4]) {
        let mut o = self.gate;
        self.gate = [
            g[0] * o[0] + g[1] * o[2],
            g[0] * o[1] + g[1] * o[3],
            g[2] * o[0] + g[3] * o[2],
            g[2] * o[1] + g[3] * o[3],
        ];

        if self.gate[1].norm() <= FP_NORM_EPSILON && self.gate[2].norm() <= FP_NORM_EPSILON {
            self.gate[1] = ZERO_CMPLX;
            self.gate[2] = ZERO_CMPLX;
            self.gate[0] = self.gate[0] / self.gate[0].abs();
            self.gate[3] = self.gate[3] / self.gate[3].abs();
        }

        if self.gate[0].norm() <= FP_NORM_EPSILON && self.gate[3].norm() <= FP_NORM_EPSILON {
            self.gate[0] = ZERO_CMPLX;
            self.gate[3] = ZERO_CMPLX;
            self.gate[1] = self.gate[1] / self.gate[1].abs();
            self.gate[2] = self.gate[2] / self.gate[2].abs();
        }
    }

    fn is_phase(&self) -> bool {
        self.gate[1].norm() <= FP_NORM_EPSILON && self.gate[2].norm() <= FP_NORM_EPSILON
    }

    fn is_invert(&self) -> bool {
        self.gate[0].norm() <= FP_NORM_EPSILON && self.gate[3].norm() <= FP_NORM_EPSILON
    }

    fn is_h_phase(&self) -> bool {
        (self.gate[0] - self.gate[1]).norm() <= FP_NORM_EPSILON
            && (self.gate[2] + self.gate[3]).norm() <= FP_NORM_EPSILON
    }

    fn is_h_invert(&self) -> bool {
        (self.gate[0] + self.gate[1]).norm() <= FP_NORM_EPSILON
            && (self.gate[2] - self.gate[3]).norm() <= FP_NORM_EPSILON
    }

    fn is_identity(&self) -> bool {
        self.is_phase() && (self.gate[0] - self.gate[3]).norm() <= FP_NORM_EPSILON
    }

    fn is_x(&self, rand_global_phase: bool) -> bool {
        self.is_invert()
            && (self.gate[1] - self.gate[2]).norm() <= FP_NORM_EPSILON
            && (rand_global_phase || (ONE_CMPLX - self.gate[1]).norm() <= FP_NORM_EPSILON)
    }

    fn is_y(&self, rand_global_phase: bool) -> bool {
        self.is_invert()
            && (self.gate[1] + self.gate[2]).norm() <= FP_NORM_EPSILON
            && (rand_global_phase || (ONE_CMPLX + self.gate[1]).norm() <= FP_NORM_EPSILON)
    }

    fn is_z(&self, rand_global_phase: bool) -> bool {
        self.is_phase()
            && (self.gate[0] + self.gate[3]).norm() <= FP_NORM_EPSILON
            && (rand_global_phase || (ONE_CMPLX - self.gate[0]).norm() <= FP_NORM_EPSILON)
    }

    fn is_h(&self) -> bool {
        (SQRT1_2_R1 - self.gate[0]).norm() <= FP_NORM_EPSILON
            && (SQRT1_2_R1 - self.gate[1]).norm() <= FP_NORM_EPSILON
            && (SQRT1_2_R1 - self.gate[2]).norm() <= FP_NORM_EPSILON
            && (SQRT1_2_R1 + self.gate[3]).norm() <= FP_NORM_EPSILON
    }
}


