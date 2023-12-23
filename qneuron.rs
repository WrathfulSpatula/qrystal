use std::cmp::max;
use std::f64::consts::{PI, E};
use std::erf;

enum QNeuronActivationFn {
    Sigmoid = 0,
    ReLU = 1,
    GeLU = 2,
    Generalized_Logistic = 3,
    Leaky_ReLU = 4,
}

struct QNeuron {
    input_power: u64,
    output_index: usize,
    activation_fn: QNeuronActivationFn,
    alpha: f64,
    tolerance: f64,
    input_indices: Vec<usize>,
    angles: Vec<f64>,
    q_reg: QInterfacePtr,
}

impl QNeuron {
    fn new(reg: QInterfacePtr, input_indices: Vec<usize>, output_index: usize, activation_fn: QNeuronActivationFn, alpha: f64, tol: f64) -> Self {
        let input_power = 2u64.pow(input_indices.len() as u32);
        let angles = vec![0.0; input_power as usize];
        QNeuron {
            input_power,
            output_index,
            activation_fn,
            alpha,
            tolerance: tol,
            input_indices,
            angles,
            q_reg: reg,
        }
    }

    fn set_alpha(&mut self, a: f64) {
        self.alpha = a;
    }

    fn get_alpha(&self) -> f64 {
        self.alpha
    }

    fn set_activation_fn(&mut self, f: QNeuronActivationFn) {
        self.activation_fn = f;
    }

    fn get_activation_fn(&self) -> QNeuronActivationFn {
        self.activation_fn
    }

    fn set_angles(&mut self, n_angles: &[f64]) {
        self.angles.copy_from_slice(n_angles);
    }

    fn get_angles(&self, o_angles: &mut [f64]) {
        o_angles.copy_from_slice(&self.angles);
    }

    fn get_input_count(&self) -> usize {
        self.input_indices.len()
    }

    fn get_input_power(&self) -> u64 {
        self.input_power
    }

    fn predict(&mut self, expected: bool, reset_init: bool) -> f64 {
        if reset_init {
            self.q_reg.set_bit(self.output_index, false);
            self.q_reg.ry(PI / 2.0, self.output_index);
        }
        if self.input_indices.is_empty() {
            match self.activation_fn {
                QNeuronActivationFn::ReLU => self.q_reg.ry(apply_relu(self.angles[0]), self.output_index),
                QNeuronActivationFn::GeLU => self.q_reg.ry(apply_gelu(self.angles[0]), self.output_index),
                QNeuronActivationFn::Generalized_Logistic => self.q_reg.ry(apply_alpha(self.angles[0], self.alpha), self.output_index),
                QNeuronActivationFn::Leaky_ReLU => self.q_reg.ry(apply_leaky_relu(self.angles[0], self.alpha), self.output_index),
                QNeuronActivationFn::Sigmoid => self.q_reg.ry(self.angles[0], self.output_index),
            }
        } else if self.activation_fn == QNeuronActivationFn::Sigmoid {
            self.q_reg.uniformly_controlled_ry(&self.input_indices, self.output_index, &self.angles);
        } else {
            let mut n_angles = vec![0.0; self.input_power as usize];
            match self.activation_fn {
                QNeuronActivationFn::ReLU => {
                    for i in 0..self.input_power {
                        n_angles[i as usize] = apply_relu(self.angles[i as usize]);
                    }
                }
                QNeuronActivationFn::GeLU => {
                    for i in 0..self.input_power {
                        n_angles[i as usize] = apply_gelu(self.angles[i as usize]);
                    }
                }
                QNeuronActivationFn::Generalized_Logistic => {
                    for i in 0..self.input_power {
                        n_angles[i as usize] = apply_alpha(self.angles[i as usize], self.alpha);
                    }
                }
                QNeuronActivationFn::Leaky_ReLU => {
                    for i in 0..self.input_power {
                        n_angles[i as usize] = apply_leaky_relu(self.angles[i as usize], self.alpha);
                    }
                }
                QNeuronActivationFn::Sigmoid => {}
            }
            self.q_reg.uniformly_controlled_ry(&self.input_indices, self.output_index, &n_angles);
        }
        let prob = self.q_reg.prob(self.output_index);
        if !expected {
            1.0 - prob
        } else {
            prob
        }
    }

    fn unpredict(&mut self, expected: bool) -> f64 {
        if self.input_indices.is_empty() {
            match self.activation_fn {
                QNeuronActivationFn::ReLU => self.q_reg.ry(neg_apply_relu(self.angles[0]), self.output_index),
                QNeuronActivationFn::GeLU => self.q_reg.ry(neg_apply_gelu(self.angles[0]), self.output_index),
                QNeuronActivationFn::Generalized_Logistic => self.q_reg.ry(-apply_alpha(self.angles[0], self.alpha), self.output_index),
                QNeuronActivationFn::Leaky_ReLU => self.q_reg.ry(-apply_leaky_relu(self.angles[0], self.alpha), self.output_index),
                QNeuronActivationFn::Sigmoid => self.q_reg.ry(-self.angles[0], self.output_index),
            }
        } else {
            let mut n_angles = vec![0.0; self.input_power as usize];
            match self.activation_fn {
                QNeuronActivationFn::ReLU => {
                    for i in 0..self.input_power {
                        n_angles[i as usize] = neg_apply_relu(self.angles[i as usize]);
                    }
                    self.q_reg.uniformly_controlled_ry(&self.input_indices, self.output_index, &n_angles);
                }
                QNeuronActivationFn::GeLU => {
                    for i in 0..self.input_power {
                        n_angles[i as usize] = neg_apply_gelu(self.angles[i as usize]);
                    }
                    self.q_reg.uniformly_controlled_ry(&self.input_indices, self.output_index, &n_angles);
                }
                QNeuronActivationFn::Generalized_Logistic => {
                    for i in 0..self.input_power {
                        n_angles[i as usize] = -apply_alpha(self.angles[i as usize], self.alpha);
                    }
                    self.q_reg.uniformly_controlled_ry(&self.input_indices, self.output_index, &n_angles);
                }
                QNeuronActivationFn::Leaky_ReLU => {
                    for i in 0..self.input_power {
                        n_angles[i as usize] = -apply_leaky_relu(self.angles[i as usize], self.alpha);
                    }
                    self.q_reg.uniformly_controlled_ry(&self.input_indices, self.output_index, &n_angles);
                }
                QNeuronActivationFn::Sigmoid => {
                    for i in 0..self.input_power {
                        n_angles[i as usize] = -self.angles[i as usize];
                    }
                    self.q_reg.uniformly_controlled_ry(&self.input_indices, self.output_index, &n_angles);
                }
            }
        }
        let prob = self.q_reg.prob(self.output_index);
        if !expected {
            1.0 - prob
        } else {
            prob
        }
    }

    fn learn_cycle(&mut self, expected: bool) -> f64 {
        let result = self.predict(expected, false);
        self.unpredict(expected);
        result
    }

    fn learn(&mut self, eta: f64, expected: bool, reset_init: bool) {
        let start_prob = self.predict(expected, reset_init);
        self.unpredict(expected);
        if 1.0 - start_prob <= self.tolerance {
            return;
        }
        for perm in 0..self.input_power {
            let start_prob = self.learn_internal(expected, eta, perm, start_prob);
            if start_prob < 0.0 {
                break;
            }
        }
    }

    fn learn_permutation(&mut self, eta: f64, expected: bool, reset_init: bool) {
        let start_prob = self.predict(expected, reset_init);
        self.unpredict(expected);
        if 1.0 - start_prob <= self.tolerance {
            return;
        }
        let mut perm = 0;
        for (i, &input_index) in self.input_indices.iter().enumerate() {
            if self.q_reg.m(input_index) {
                perm |= 1 << i;
            }
        }
        self.learn_internal(expected, eta, perm, start_prob);
    }

    fn learn_internal(&mut self, expected: bool, eta: f64, perm: u64, start_prob: f64) -> f64 {
        let orig_angle = self.angles[perm as usize];
        let angle = &mut self.angles[perm as usize];

        *angle += eta * PI;
        let plus_prob = self.learn_cycle(expected);
        if 1.0 - plus_prob <= self.tolerance {
            *angle = clamp_angle(*angle);
            return -1.0;
        }

        *angle = orig_angle - eta * PI;
        let minus_prob = self.learn_cycle(expected);
        if 1.0 - minus_prob <= self.tolerance {
            *angle = clamp_angle(*angle);
            return -1.0;
        }

        if start_prob >= plus_prob && start_prob >= minus_prob {
            *angle = orig_angle;
            start_prob
        } else if plus_prob > minus_prob {
            *angle = orig_angle + eta * PI;
            plus_prob
        } else {
            minus_prob
        }
    }
}

fn apply_relu(angle: f64) -> f64 {
    max(0.0, angle)
}

fn neg_apply_relu(angle: f64) -> f64 {
    -max(0.0, angle)
}

fn apply_gelu(angle: f64) -> f64 {
    angle * (1.0 + erf(angle * (2.0 / std::f64::consts::SQRT_PI)))
}

fn neg_apply_gelu(angle: f64) -> f64 {
    -angle * (1.0 + erf(angle * (2.0 / std::f64::consts::SQRT_PI)))
}

fn apply_alpha(angle: f64, alpha: f64) -> f64 {
    let mut to_ret = 0.0;
    let mut angle = angle;
    if angle > PI {
        angle -= PI;
        to_ret = PI;
    } else if angle <= -PI {
        angle += PI;
        to_ret = -PI;
    }
    to_ret + (angle.abs() / PI).powf(alpha) * (PI / 2.0) * angle.signum()
}

fn apply_leaky_relu(angle: f64, alpha: f64) -> f64 {
    max(alpha * angle, angle)
}

fn clamp_angle(angle: f64) -> f64 {
    let mut angle = angle % (4.0 * PI);
    if angle <= -2.0 * PI {
        angle += 4.0 * PI;
    } else if angle > 2.0 * PI {
        angle -= 4.0 * PI;
    }
    angle
}


