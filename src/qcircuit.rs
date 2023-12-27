use std::collections::{HashMap, HashSet, LinkedList};
use std::cmp::Ordering;
use std::ops::{BitAnd, BitOr, BitXor};
use std::fmt::{Display, Formatter, Result};
use std::io::{Read, Write};
use std::mem::drop;
use std::ptr::null_mut;
use std::slice::from_raw_parts_mut;
use std::sync::Arc;
use std::rc::Rc;

const FP_NORM_EPSILON: f64 = 1e-9;

#[derive(Clone, Copy, Debug)]
struct Complex {
    real: f64,
    imag: f64,
}

impl Complex {
    fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }
}

impl PartialEq for Complex {
    fn eq(&self, other: &Self) -> bool {
        (self.real - other.real).abs() <= FP_NORM_EPSILON && (self.imag - other.imag).abs() <= FP_NORM_EPSILON
    }
}

impl BitAnd for Complex {
    type Output = Self;

    fn bitand(self, other: Self) -> Self {
        Self {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }
}

impl BitOr for Complex {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
}

impl BitXor for Complex {
    type Output = Self;

    fn bitxor(self, other: Self) -> Self {
        Self {
            real: self.real - other.real,
            imag: self.imag - other.imag,
        }
    }
}

impl Display for Complex {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        if (self.real - 1.0).abs() <= FP_NORM_EPSILON && self.imag.abs() <= FP_NORM_EPSILON {
            write!(f, "1")
        } else if (self.real + 1.0).abs() <= FP_NORM_EPSILON && self.imag.abs() <= FP_NORM_EPSILON {
            write!(f, "-1")
        } else if self.real.abs() <= FP_NORM_EPSILON && (self.imag - 1.0).abs() <= FP_NORM_EPSILON {
            write!(f, "i")
        } else if self.real.abs() <= FP_NORM_EPSILON && (self.imag + 1.0).abs() <= FP_NORM_EPSILON {
            write!(f, "-i")
        } else if self.real.abs() <= FP_NORM_EPSILON {
            write!(f, "{}i", self.imag)
        } else if self.imag.abs() <= FP_NORM_EPSILON {
            write!(f, "{}", self.real)
        } else {
            write!(f, "{} + {}i", self.real, self.imag)
        }
    }
}

#[derive(Clone, Debug)]
struct QCircuitGate {
    target: usize,
    payloads: HashMap<usize, Arc<[Complex]>>,
    controls: HashSet<usize>,
}

impl QCircuitGate {
    fn new() -> Self {
        Self {
            target: 0,
            payloads: HashMap::new(),
            controls: HashSet::new(),
        }
    }

    fn with_qubits(q1: usize, q2: usize) -> Self {
        Self {
            target: q1,
            payloads: HashMap::new(),
            controls: [q2].iter().cloned().collect(),
        }
    }

    fn with_matrix(target: usize, matrix: &[Complex]) -> Self {
        let mut payloads = HashMap::new();
        payloads.insert(0, Arc::from(matrix.to_owned()));
        Self {
            target,
            payloads,
            controls: HashSet::new(),
        }
    }

    fn with_matrix_and_controls(target: usize, matrix: &[Complex], controls: &HashSet<usize>, perm: usize) -> Self {
        let mut payloads = HashMap::new();
        payloads.insert(perm, Arc::from(matrix.to_owned()));
        Self {
            target,
            payloads,
            controls: controls.clone(),
        }
    }

    fn with_payloads_and_controls(
        target: usize,
        payloads: &HashMap<usize, Arc<[Complex]>>,
        controls: &HashSet<usize>,
    ) -> Self {
        let mut new_payloads = HashMap::new();
        for (key, value) in payloads {
            new_payloads.insert(*key, Arc::from(value.to_owned()));
        }
        Self {
            target,
            payloads: new_payloads,
            controls: controls.clone(),
        }
    }

    fn clone(&self) -> Self {
        Self {
            target: self.target,
            payloads: self.payloads.clone(),
            controls: self.controls.clone(),
        }
    }

    fn can_combine(&self, other: &Self, clifford: bool) -> bool {
        if self.target != other.target {
            return false;
        }
        if self.controls.is_empty() && other.controls.is_empty() {
            return true;
        }
        if clifford {
            let mc = self.is_clifford();
            let oc = other.is_clifford();
            if mc != oc {
                return false;
            }
            if mc {
                return self.controls.is_empty()
                    || other.controls.is_empty()
                    || *(self.controls.iter().next().unwrap()) == *(other.controls.iter().next().unwrap());
            }
        }
        self.controls.is_subset(&other.controls) || other.controls.is_subset(&self.controls)
    }

    fn clear(&mut self) {
        self.controls.clear();
        self.payloads.clear();
        self.payloads.insert(0, Arc::from([Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)]));
    }

    fn add_control(&mut self, c: usize) {
        if !self.controls.contains(&c) {
            self.controls.insert(c);
            let cpos = self.controls.iter().position(|&x| x == c).unwrap();
            let mid_pow = 1 << cpos;
            let low_mask = mid_pow - 1;
            let high_mask = !low_mask;
            let mut n_payloads = HashMap::new();
            for (key, value) in &self.payloads {
                let n_key = (key & low_mask) | ((key & high_mask) << 1);
                n_payloads.insert(n_key, value.clone());
                let mut np = vec![Complex::new(0.0, 0.0); 4];
                np.copy_from_slice(&value);
                n_key |= mid_pow;
                n_payloads.insert(n_key, Arc::from(np));
            }
            self.payloads = n_payloads;
        }
    }

    fn can_remove_control(&self, c: usize) -> bool {
        let cpos = self.controls.iter().position(|&x| x == c).unwrap();
        let mid_pow = 1 << cpos;
        for (key, value) in &self.payloads {
            let n_key = !mid_pow & key;
            if n_key.cmp(key) == Ordering::Equal {
                if !self.payloads.contains_key(&(n_key | mid_pow)) {
                    return false;
                }
            } else {
                if !self.payloads.contains_key(&n_key) {
                    return false;
                }
            }
            let l = &self.payloads[&n_key];
            let h = &self.payloads[&(n_key | mid_pow)];
            if (l[0] - h[0]).norm() <= FP_NORM_EPSILON
                && (l[1] - h[1]).norm() <= FP_NORM_EPSILON
                && (l[2] - h[2]).norm() <= FP_NORM_EPSILON
                && (l[3] - h[3]).norm() <= FP_NORM_EPSILON
            {
                continue;
            }
            return false;
        }
        true
    }

    fn remove_control(&mut self, c: usize) {
        let cpos = self.controls.iter().position(|&x| x == c).unwrap();
        let mid_pow = 1 << cpos;
        let low_mask = mid_pow - 1;
        let high_mask = !(low_mask | mid_pow);
        let mut n_payloads = HashMap::new();
        for (key, value) in &self.payloads {
            n_payloads.insert((key & low_mask) | ((key & high_mask) >> 1), value.clone());
        }
        self.payloads = n_payloads;
        self.controls.remove(&c);
    }

    fn try_remove_control(&mut self, c: usize) -> bool {
        if !self.can_remove_control(c) {
            return false;
        }
        self.remove_control(c);
        true
    }

    fn combine(&mut self, other: &Self) {
        let ctrls_to_test: HashSet<_> = self.controls.intersection(&other.controls).cloned().collect();
        if self.controls.len() < other.controls.len() {
            for oc in &other.controls {
                self.add_control(*oc);
            }
        } else if self.controls.len() > other.controls.len() {
            for c in &self.controls {
                other.add_control(*c);
            }
        }
        for (key, value) in &other.payloads {
            if !self.payloads.contains_key(key) {
                self.payloads.insert(*key, value.clone());
                continue;
            }
            let p = self.payloads.get_mut(key).unwrap();
            let mut out = vec![Complex::new(0.0, 0.0); 4];
            for i in 0..4 {
                out[i] = value[i] & p[i];
            }
            if (out[1] - Complex::new(0.0, 0.0)).norm() <= FP_NORM_EPSILON
                && (out[2] - Complex::new(0.0, 0.0)).norm() <= FP_NORM_EPSILON
                && (Complex::new(1.0, 0.0) - out[0]).norm() <= FP_NORM_EPSILON
                && (Complex::new(1.0, 0.0) - out[3]).norm() <= FP_NORM_EPSILON
            {
                self.payloads.remove(key);
                continue;
            }
            p.copy_from_slice(&out);
        }
        if self.payloads.is_empty() {
            self.clear();
            return;
        }
        for c in &ctrls_to_test {
            self.try_remove_control(*c);
        }
    }

    fn try_combine(&mut self, other: &Self, clifford: bool) -> bool {
        if !self.can_combine(other, clifford) {
            return false;
        }
        self.combine(other);
        true
    }

    fn is_identity(&self) -> bool {
        if !self.controls.is_empty() {
            return false;
        }
        if self.payloads.len() != 1 {
            return false;
        }
        let p = self.payloads.get(&0).unwrap();
        (p[1] - Complex::new(0.0, 0.0)).norm() <= FP_NORM_EPSILON
            && (p[2] - Complex::new(0.0, 0.0)).norm() <= FP_NORM_EPSILON
            && (Complex::new(1.0, 0.0) - p[0]).norm() <= FP_NORM_EPSILON
            && (Complex::new(1.0, 0.0) - p[3]).norm() <= FP_NORM_EPSILON
    }

    fn is_phase(&self) -> bool {
        for (_, value) in &self.payloads {
            if value[1].norm() > FP_NORM_EPSILON || value[2].norm() > FP_NORM_EPSILON {
                return false;
            }
        }
        true
    }

    fn is_invert(&self) -> bool {
        for (_, value) in &self.payloads {
            if value[0].norm() > FP_NORM_EPSILON || value[3].norm() > FP_NORM_EPSILON {
                return false;
            }
        }
        true
    }

    fn is_phase_invert(&self) -> bool {
        for (_, value) in &self.payloads {
            if (value[0].norm() > FP_NORM_EPSILON || value[3].norm() > FP_NORM_EPSILON)
                && (value[1].norm() > FP_NORM_EPSILON || value[2].norm() > FP_NORM_EPSILON)
            {
                return false;
            }
        }
        true
    }

    fn is_cnot(&self) -> bool {
        if self.controls.len() != 1 || self.payloads.len() != 1 || !self.payloads.contains_key(&1) {
            return false;
        }
        let p = self.payloads.get(&1).unwrap();
        (p[0].norm() <= FP_NORM_EPSILON)
            && (p[3].norm() <= FP_NORM_EPSILON)
            && (Complex::new(1.0, 0.0) - p[1]).norm() <= FP_NORM_EPSILON
            && (Complex::new(1.0, 0.0) - p[2]).norm() <= FP_NORM_EPSILON
    }

    fn is_clifford(&self) -> bool {
        if self.payloads.is_empty() {
            return true;
        }
        if self.controls.len() > 1 {
            return false;
        }
        if self.controls.is_empty() {
            return self.is_clifford_phase_invert(self.payloads.get(&0).unwrap());
        }
        for (_, value) in &self.payloads {
            if (value[1].norm() <= FP_NORM_EPSILON && value[2].norm() <= FP_NORM_EPSILON)
                || (value[0].norm() <= FP_NORM_EPSILON && value[3].norm() <= FP_NORM_EPSILON)
            {
                if !self.is_clifford_phase_invert(value) {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }

    fn can_pass(&self, other: &Self) -> bool {
        if let Some(c) = other.controls.get(&self.target) {
            if self.controls.contains(other.target) {
                return self.is_phase() && other.is_phase();
            }
            if self.is_phase() {
                return true;
            }
            if !self.is_phase_invert()
                || !other.controls.is_subset(&self.controls)
                || !other.controls.is_subset(&self.controls)
            {
                return false;
            }
            let mut opf_pows = Vec::with_capacity(self.controls.len());
            for ctrl in &self.controls {
                opf_pows.push(1 << other.controls.iter().position(|&x| x == *ctrl).unwrap());
            }
            let p = 1 << other.controls.iter().position(|&x| x == *c).unwrap();
            let mut n_payloads = HashMap::new();
            for (key, value) in &other.payloads {
                let mut pf = 0;
                for i in 0..opf_pows.len() {
                    if (key & opf_pows[i]) != 0 {
                        pf |= 1 << i;
                    }
                }
                if (key & p) != 0 {
                    if let Some(poi) = self.payloads.get(&pf) {
                        if poi[0].norm() > FP_NORM_EPSILON {
                            n_payloads.insert(*key, value.clone());
                        } else {
                            n_payloads.insert(key ^ p, value.clone());
                        }
                    } else {
                        n_payloads.insert(*key, value.clone());
                    }
                } else {
                    n_payloads.insert(*key, value.clone());
                }
            }
            other.payloads = n_payloads;
            return true;
        }
        if self.controls.contains(&other.target) {
            return other.is_phase();
        }
        self.target != other.target || (self.is_phase() && other.is_phase())
    }

    fn make_uniformly_controlled_payload(&self) -> Vec<Complex> {
        let max_q_power = 1 << self.controls.len();
        let mut to_ret = vec![Complex::new(0.0, 0.0); max_q_power << 2];
        let identity = [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];
        for i in 0..max_q_power {
            let mtrx = &mut to_ret[(i << 2)..((i + 1) << 2)];
            if let Some(p) = self.payloads.get(&i) {
                mtrx.copy_from_slice(p);
            } else {
                mtrx.copy_from_slice(&identity);
            }
        }
        to_ret
    }

    fn get_controls_vector(&self) -> Vec<usize> {
        self.controls.iter().cloned().collect()
    }

    fn post_select_control(&mut self, c: usize, eigen: bool) {
        if self.controls.contains(&c) {
            let cpos = self.controls.iter().position(|&x| x == c).unwrap();
            let mid_pow = 1 << cpos;
            let low_mask = mid_pow - 1;
            let high_mask = !(low_mask | mid_pow);
            let qubit_pow = 1 << cpos;
            let eigen_pow = if eigen { qubit_pow } else { 0 };
            let mut n_payloads = HashMap::new();
            for (key, value) in &self.payloads {
                if (key & qubit_pow).cmp(&eigen_pow) != Ordering::Equal {
                    continue;
                }
                n_payloads.insert((key & low_mask) | ((key & high_mask) >> 1), value.clone());
            }
            self.payloads = n_payloads;
            self.controls.remove(&c);
        }
    }
}

impl Display for QCircuitGate {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "QCircuitGate {{ target: {}, payloads: {:?}, controls: {:?} }}", self.target, self.payloads, self.controls)
    }
}

impl Read for QCircuitGate {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let mut target_buf = [0u8; 8];
        let mut payloads_len_buf = [0u8; 8];
        let mut controls_len_buf = [0u8; 8];
        let mut controls_buf = vec![0u8; self.controls.len() * 8];
        let mut payloads_buf = vec![0u8; self.payloads.len() * 8 * 4];
        let mut total_bytes_read = 0usize;
        let mut bytes_read = self.target.read(&mut target_buf)?;
        total_bytes_read += bytes_read;
        if bytes_read < 8 {
            return Ok(total_bytes_read);
        }
        self.target = usize::from_ne_bytes(target_buf);
        bytes_read = self.payloads.len().read(&mut payloads_len_buf)?;
        total_bytes_read += bytes_read;
        if bytes_read < 8 {
            return Ok(total_bytes_read);
        }
        let payloads_len = usize::from_ne_bytes(payloads_len_buf);
        bytes_read = self.controls.len().read(&mut controls_len_buf)?;
        total_bytes_read += bytes_read;
        if bytes_read < 8 {
            return Ok(total_bytes_read);
        }
        let controls_len = usize::from_ne_bytes(controls_len_buf);
        bytes_read = self.controls.read(&mut controls_buf)?;
        total_bytes_read += bytes_read;
        if bytes_read < self.controls.len() * 8 {
            return Ok(total_bytes_read);
        }
        bytes_read = self.payloads.read(&mut payloads_buf)?;
        total_bytes_read += bytes_read;
        if bytes_read < self.payloads.len() * 8 * 4 {
            return Ok(total_bytes_read);
        }
        let mut controls = HashSet::new();
        for i in 0..controls_len {
            let control = usize::from_ne_bytes(controls_buf[(i * 8)..((i + 1) * 8)].try_into().unwrap());
            controls.insert(control);
        }
        let mut payloads = HashMap::new();
        for i in 0..payloads_len {
            let key = usize::from_ne_bytes(payloads_buf[(i * 8 * 4)..((i + 1) * 8 * 4)].try_into().unwrap());
            let value = Arc::from(payloads_buf[((i * 8 * 4) + 8)..((i + 1) * 8 * 4)].try_into().unwrap());
            payloads.insert(key, value);
        }
        self.controls = controls;
        self.payloads = payloads;
        Ok(total_bytes_read)
    }
}

impl Write for QCircuitGate {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut target_buf = self.target.to_ne_bytes();
        let mut payloads_len_buf = (self.payloads.len() as usize).to_ne_bytes();
        let mut controls_len_buf = (self.controls.len() as usize).to_ne_bytes();
        let mut controls_buf = vec![0u8; self.controls.len() * 8];
        let mut payloads_buf = vec![0u8; self.payloads.len() * 8 * 4];
        let mut total_bytes_written = 0usize;
        let mut bytes_written = self.target.write(&target_buf)?;
        total_bytes_written += bytes_written;
        if bytes_written < 8 {
            return Ok(total_bytes_written);
        }
        bytes_written = self.payloads.len().write(&mut payloads_len_buf)?;
        total_bytes_written += bytes_written;
        if bytes_written < 8 {
            return Ok(total_bytes_written);
        }
        bytes_written = self.controls.len().write(&mut controls_len_buf)?;
        total_bytes_written += bytes_written;
        if bytes_written < 8 {
            return Ok(total_bytes_written);
        }
        let mut i = 0;
        for control in &self.controls {
            controls_buf[(i * 8)..((i + 1) * 8)].copy_from_slice(&control.to_ne_bytes());
            i += 1;
        }
        bytes_written = self.controls.write(&controls_buf)?;
        total_bytes_written += bytes_written;
        if bytes_written < self.controls.len() * 8 {
            return Ok(total_bytes_written);
        }
        i = 0;
        for (key, value) in &self.payloads {
            payloads_buf[(i * 8 * 4)..((i + 1) * 8 * 4)].copy_from_slice(&key.to_ne_bytes());
            payloads_buf[((i * 8 * 4) + 8)..((i + 1) * 8 * 4)].copy_from_slice(&value);
            i += 1;
        }
        bytes_written = self.payloads.write(&payloads_buf)?;
        total_bytes_written += bytes_written;
        if bytes_written < self.payloads.len() * 8 * 4 {
            return Ok(total_bytes_written);
        }
        Ok(total_bytes_written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl PartialEq for QCircuitGate {
    fn eq(&self, other: &Self) -> bool {
        self.target == other.target && self.payloads == other.payloads && self.controls == other.controls
    }
}

impl Eq for QCircuitGate {}

impl PartialOrd for QCircuitGate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QCircuitGate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.target.cmp(&other.target)
    }
}

impl std::hash::Hash for QCircuitGate {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.target.hash(state);
        self.payloads.hash(state);
        self.controls.hash(state);
    }
}

impl std::fmt::Debug for QCircuitGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QCircuitGate {{ target: {}, payloads: {:?}, controls: {:?} }}", self.target, self.payloads, self.controls)
    }
}

impl std::fmt::Display for QCircuitGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QCircuitGate {{ target: {}, payloads: {:?}, controls: {:?} }}", self.target, self.payloads, self.controls)
    }
}

struct QCircuitGate {
    target: bitLenInt,
    payloads: Vec<complex>,
    controls: Vec<bitLenInt>,
    eigen: bitCapInt,
}

impl QCircuitGate {
    fn new(target: bitLenInt, payloads: Vec<complex>, controls: Vec<bitLenInt>, eigen: bitCapInt) -> Self {
        QCircuitGate {
            target,
            payloads,
            controls,
            eigen,
        }
    }

    fn clone(&self) -> Self {
        QCircuitGate {
            target: self.target,
            payloads: self.payloads.clone(),
            controls: self.controls.clone(),
            eigen: self.eigen,
        }
    }

    fn post_select_control(&mut self, qubit: bitLenInt, eigen: bool) {
        if eigen {
            self.controls.push(qubit);
        } else {
            if let Some(index) = self.controls.iter().position(|&x| x == qubit) {
                self.controls.remove(index);
            }
        }
    }

    fn is_phase(&self) -> bool {
        self.payloads.len() == 1 && self.payloads[0] == ONE_CMPLX
    }
}

struct QCircuit {
    is_collapsed: bool,
    is_near_clifford: bool,
    qubit_count: bitLenInt,
    gates: LinkedList<Rc<QCircuitGate>>,
}

impl QCircuit {
    fn new(collapse: bool, clifford: bool) -> Self {
        QCircuit {
            is_collapsed: collapse,
            is_near_clifford: clifford,
            qubit_count: 0,
            gates: LinkedList::new(),
        }
    }

    fn with_qubit_count(qb_count: bitLenInt, g: Vec<Rc<QCircuitGate>>, collapse: bool, clifford: bool) -> Self {
        let mut gates = LinkedList::new();
        for gate in g {
            gates.push_back(gate.clone());
        }
        QCircuit {
            is_collapsed: collapse,
            is_near_clifford: clifford,
            qubit_count: qb_count,
            gates,
        }
    }

    fn clone(&self) -> Rc<Self> {
        Rc::new(QCircuit {
            is_collapsed: self.is_collapsed,
            is_near_clifford: self.is_near_clifford,
            qubit_count: self.qubit_count,
            gates: self.gates.clone(),
        })
    }

    fn inverse(&self) -> Rc<Self> {
        let clone = self.clone();
        for gate in clone.gates.iter_mut() {
            for p in gate.payloads.iter_mut() {
                let m = p.as_ptr();
                let inv = [conj(m[0]), conj(m[2]), conj(m[1]), conj(m[3])];
                p.copy_from_slice(&inv);
            }
        }
        clone.gates.reverse();
        clone
    }

    fn get_qubit_count(&self) -> bitLenInt {
        self.qubit_count
    }

    fn set_qubit_count(&mut self, n: bitLenInt) {
        self.qubit_count = n;
    }

    fn get_gate_list(&self) -> LinkedList<Rc<QCircuitGate>> {
        self.gates.clone()
    }

    fn set_gate_list(&mut self, gl: LinkedList<Rc<QCircuitGate>>) {
        self.gates = gl;
    }

    fn swap(&mut self, q1: bitLenInt, q2: bitLenInt) {
        if q1 == q2 {
            return;
        }

        if q1 > q2 {
            std::mem::swap(&mut q1, &mut q2);
        }
        let m = [ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX];
        let s1 = vec![q1];
        let s2 = vec![q2];
        self.append_gate(Rc::new(QCircuitGate::new(q1, m, s2, ONE_BCI)));
        self.append_gate(Rc::new(QCircuitGate::new(q2, m, s1, ONE_BCI)));
        self.append_gate(Rc::new(QCircuitGate::new(q1, m, s2, ONE_BCI)));
    }

    fn append(&mut self, circuit: Rc<Self>) {
        if circuit.qubit_count > self.qubit_count {
            self.qubit_count = circuit.qubit_count;
        }
        self.gates.extend(circuit.gates.iter().cloned());
    }

    fn combine(&mut self, circuit: Rc<Self>) {
        if circuit.qubit_count > self.qubit_count {
            self.qubit_count = circuit.qubit_count;
        }
        for g in circuit.gates.iter() {
            self.append_gate(g.clone());
        }
    }

    fn append_gate(&mut self, n_gate: Rc<QCircuitGate>) {
        self.gates.push_back(n_gate);
    }

    fn run(&self, qsim: QInterfacePtr) {
        // TODO: Implement this function
    }

    fn is_non_phase_target(&self, qubit: bitLenInt) -> bool {
        for gate in self.gates.iter() {
            if gate.target == qubit && !gate.is_phase() {
                return true;
            }
        }
        false
    }

    fn delete_phase_target(&mut self, qubit: bitLenInt, eigen: bool) {
        let mut n_gates = LinkedList::new();
        self.gates.reverse();
        for gate in self.gates.iter() {
            if gate.target == qubit {
                continue;
            }
            let n_gate = gate.clone();
            n_gate.post_select_control(qubit, eigen);
            n_gates.push_front(n_gate);
        }
        self.gates = n_gates;
    }

    fn past_light_cone(&self, qubits: &mut std::collections::HashSet<bitLenInt>) -> Rc<Self> {
        let mut n_gates = LinkedList::new();
        self.gates.reverse();
        for gate in self.gates.iter() {
            if !qubits.contains(&gate.target) {
                let mut is_non_causal = true;
                for c in gate.controls.iter() {
                    if qubits.contains(c) {
                        is_non_causal = false;
                        break;
                    }
                }
                if is_non_causal {
                    continue;
                }
            }
            n_gates.push_front(gate.clone());
            qubits.insert(gate.target);
            for c in gate.controls.iter() {
                qubits.insert(*c);
            }
        }
        self.gates.reverse();
        Rc::new(QCircuit {
            qubit_count: self.qubit_count,
            gates: n_gates,
            ..*self
        })
    }

    #[cfg(ENABLE_ALU)]
    fn inc(&mut self, to_add: bitCapInt, start: bitLenInt, length: bitLenInt) {
        // TODO: Implement this function
    }
}
