use std::sync::Arc;
use std::collections::HashMap;

struct QUnitCliffordAmp {
    amp: Complex,
    stabilizer: QUnitCliffordPtr,
}

impl QUnitCliffordAmp {
    fn new(amp: Complex, stabilizer: QUnitCliffordPtr) -> Self {
        Self {
            amp,
            stabilizer,
        }
    }
}

struct QStabilizerHybrid {
    useHostRam: bool,
    doNormalize: bool,
    isSparse: bool,
    useTGadget: bool,
    isRoundingFlushed: bool,
    thresholdQubits: bitLenInt,
    ancillaCount: bitLenInt,
    deadAncillaCount: bitLenInt,
    maxEngineQubitCount: bitLenInt,
    maxAncillaCount: bitLenInt,
    maxStateMapCacheQubitCount: bitLenInt,
    separabilityThreshold: real1_f,
    devID: i64,
    phaseFactor: Complex,
    engine: QInterfacePtr,
    stabilizer: QUnitCliffordPtr,
    deviceIDs: Vec<i64>,
    engineTypes: Vec<QInterfaceEngine>,
    cloneEngineTypes: Vec<QInterfaceEngine>,
    shards: Vec<MpsShardPtr>,
    stateMapCache: HashMap<bitCapInt, Complex>,
}

type QStabilizerHybridPtr = Arc<QStabilizerHybrid>;

impl QStabilizerHybrid {
    fn new(eng: Vec<QInterfaceEngine>, qBitCount: bitLenInt, initState: bitCapInt, rgp: qrack_rand_gen_ptr, phaseFac: Complex, doNorm: bool, randomGlobalPhase: bool, useHostMem: bool, deviceId: i64, useHardwareRNG: bool, useSparseStateVec: bool, norm_thresh: real1_f, devList: Vec<i64>, qubitThreshold: bitLenInt, separation_thresh: real1_f) -> Self {
        Self {
            useHostRam: useHostMem,
            doNormalize: doNorm,
            isSparse: useSparseStateVec,
            useTGadget: false,
            isRoundingFlushed: false,
            thresholdQubits: qubitThreshold,
            ancillaCount: 0,
            deadAncillaCount: 0,
            maxEngineQubitCount: 0,
            maxAncillaCount: 0,
            maxStateMapCacheQubitCount: 0,
            separabilityThreshold: separation_thresh,
            devID: deviceId,
            phaseFactor: phaseFac,
            engine: QInterfacePtr::new(),
            stabilizer: QUnitCliffordPtr::new(),
            deviceIDs: devList,
            engineTypes: eng.clone(),
            cloneEngineTypes: eng.clone(),
            shards: Vec::new(),
            stateMapCache: HashMap::new(),
        }
    }

    fn make_stabilizer(&self, perm: bitCapInt) -> QUnitCliffordPtr {
        QUnitCliffordPtr::new()
    }

    fn make_engine(&self, perm: bitCapInt) -> QInterfacePtr {
        QInterfacePtr::new()
    }

    fn make_engine_with_qb_count(&self, perm: bitCapInt, qbCount: bitLenInt) -> QInterfacePtr {
        QInterfacePtr::new()
    }

    fn invert_buffer(&self, qubit: bitLenInt) {
        
    }

    fn flush_h(&self, qubit: bitLenInt) {
        
    }

    fn flush_if_blocked(&self, control: bitLenInt, target: bitLenInt, isPhase: bool) {
        
    }

    fn collapse_separable_shard(&self, qubit: bitLenInt) -> bool {
        false
    }

    fn trim_controls(&self, lControls: &Vec<bitLenInt>, output: &mut Vec<bitLenInt>, anti: bool) -> bool {
        false
    }

    fn cache_eigenstate(&self, target: bitLenInt) {
        
    }

    fn flush_buffers(&self) {
        
    }

    fn dump_buffers(&self) {
        
    }

    fn either_is_buffered(&self, logical: bool) -> bool {
        false
    }

    fn is_buffered(&self) -> bool {
        self.either_is_buffered(false)
    }

    fn is_logical_buffered(&self) -> bool {
        self.either_is_buffered(true)
    }

    fn either_is_prob_buffered(&self, logical: bool) -> bool {
        false
    }

    fn is_prob_buffered(&self) -> bool {
        self.either_is_prob_buffered(false)
    }

    fn is_logical_prob_buffered(&self) -> bool {
        self.either_is_prob_buffered(true)
    }

    fn get_qubit_reduced_density_matrix(&self, qubit: bitLenInt) -> Vec<Complex> {
        Vec::new()
    }

    fn check_shots<F>(&self, shots: u32, m: bitCapInt, partProb: real1_f, qPowers: &Vec<bitCapInt>, rng: &mut Vec<real1_f>, fn: F) where F: Fn(bitCapInt, u32) {
        
    }

    fn generate_shot_probs(&self, shots: u32) -> Vec<real1_f> {
        Vec::new()
    }

    fn fractional_rz_angle_with_flush(&self, i: bitLenInt, angle: real1_f, isGateSuppressed: bool) -> real1_f {
        0.0
    }

    fn flush_clifford_from_buffers(&self) {
        
    }

    fn combine_ancillae(&self) {
        
    }

    fn rdm_clone_helper(&self) -> QStabilizerHybridPtr {
        QStabilizerHybridPtr::new()
    }

    fn rdm_clone_flush(&self, threshold: real1_f) {
        
    }

    fn expectation_factorized<F>(&self, isFloat: bool, bits: &Vec<bitLenInt>, perms: &Vec<bitCapInt>, weights: &Vec<real1_f>, offset: bitCapInt, roundRz: bool, fn: F) where F: Fn() -> real1_f {
        
    }

    fn clear_ancilla(&self, i: bitLenInt) {
        
    }

    fn approx_compare_helper(&self, toCompare: QStabilizerHybridPtr, isDiscreteBool: bool, error_tol: real1_f) -> real1_f {
        0.0
    }

    fn i_swap_helper(&self, qubit1: bitLenInt, qubit2: bitLenInt, inverse: bool) {
        
    }

    fn get_amplitude_or_prob(&self, perm: bitCapInt, isProb: bool) -> Complex {
        Complex::new(0.0, 0.0)
    }
}



