use std::default::Default;
use std::rc::Rc;
use std::vec::Vec;
use std::boxed::Box;
use std::shared::Shared;

struct _QrackTimeEvolveOpHeader {
    target: u32,
    controlLen: u32,
    controls: [u32; 32],
}

struct HamiltonianOp {
    targetBit: u32,
    anti: bool,
    uniform: bool,
    matrix: BitOp,
    controls: Vec<u32>,
    toggles: Vec<bool>,
}

impl Default for HamiltonianOp {
    fn default() -> Self {
        HamiltonianOp {
            targetBit: 0,
            anti: false,
            uniform: false,
            matrix: BitOp::default(),
            controls: Vec::new(),
            toggles: Vec::new(),
        }
    }
}

impl HamiltonianOp {
    fn new(target: u32, mtrx: BitOp) -> Self {
        HamiltonianOp {
            targetBit: target,
            anti: false,
            uniform: false,
            matrix: mtrx,
            controls: Vec::new(),
            toggles: Vec::new(),
        }
    }

    fn new_with_ctrls(target: u32, mtrx: BitOp, ctrls: Vec<u32>, anti_ctrlled: bool, ctrl_toggles: Vec<bool>) -> Self {
        HamiltonianOp {
            targetBit: target,
            anti: anti_ctrlled,
            uniform: false,
            matrix: mtrx,
            controls: ctrls,
            toggles: ctrl_toggles,
        }
    }
}

struct UniformHamiltonianOp {
    uniform: bool,
}

impl UniformHamiltonianOp {
    fn new(ctrls: Vec<u32>, target: u32, mtrx: BitOp) -> Self {
        UniformHamiltonianOp {
            uniform: true,
        }
    }

    fn new_with_teoh(teoh: &_QrackTimeEvolveOpHeader, mtrx: *mut f64) -> Self {
        let target_bit = teoh.target as u32;
        let control_len = teoh.controlLen as u32;
        let controls = teoh.controls.iter().take(control_len as usize).cloned().collect::<Vec<u32>>();
        let uniform = true;
        let mtrx_term_count = (1 << control_len) * 4;
        let mut m = BitOp::new(vec![0.0; mtrx_term_count as usize]);
        for i in 0..mtrx_term_count {
            m[i] = Complex::new(mtrx[i as usize * 2], mtrx[i as usize * 2 + 1]);
        }
        UniformHamiltonianOp {
            uniform: true,
        }
    }
}

type HamiltonianOpPtr = Rc<HamiltonianOp>;
type Hamiltonian = Vec<HamiltonianOpPtr>;


