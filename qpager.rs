use std::sync::Arc;
use std::vec::Vec;

pub struct QPager {
    useGpuThreshold: bool,
    isSparse: bool,
    useTGadget: bool,
    maxPageSetting: i64,
    maxPageQubits: i64,
    thresholdQubitsPerPage: i64,
    baseQubitsPerPage: i64,
    maxQubits: i64,
    devID: i64,
    rootEngine: QInterfaceEngine,
    basePageMaxQPower: i64,
    basePageCount: i64,
    phaseFactor: Complex,
    devicesHostPointer: Vec<bool>,
    deviceIDs: Vec<i64>,
    engines: Vec<QInterfaceEngine>,
    qPages: Vec<QEnginePtr>,
}

type QPagerPtr = Arc<QPager>;

impl QPager {
    fn new(
        eng: Vec<QInterfaceEngine>,
        qBitCount: i64,
        initState: i64,
        rgp: qrack_rand_gen_ptr,
        phaseFac: Complex,
        doNorm: bool,
        ignored: bool,
        useHostMem: bool,
        deviceId: i64,
        useHardwareRNG: bool,
        useSparseStateVec: bool,
        norm_thresh: f32,
        devList: Vec<i64>,
        qubitThreshold: i64,
        separation_thresh: f32,
    ) -> QPagerPtr {
        // ...
    }

    fn set_concurrency(&mut self, threadsPerEngine: u32) {
        QInterface::set_concurrency(self, threadsPerEngine);
        for i in 0..self.qPages.len() {
            self.qPages[i].set_concurrency(threadsPerEngine);
        }
    }

    fn set_t_injection(&mut self, useGadget: bool) {
        self.useTGadget = useGadget;
        for i in 0..self.qPages.len() {
            self.qPages[i].set_t_injection(useTGadget);
        }
    }

    fn get_t_injection(&self) -> bool {
        self.useTGadget
    }

    fn is_opencl(&self) -> bool {
        self.qPages[0].is_opencl()
    }

    fn release_engine(&mut self) -> QEnginePtr {
        self.combine_engines();
        self.qPages[0].clone()
    }

    fn lock_engine(&mut self, eng: QEnginePtr) {
        self.qPages.resize(1);
        self.qPages[0] = eng;
        eng.set_device(self.deviceIDs[0]);
    }

    fn zero_amplitudes(&mut self) {
        for i in 0..self.qPages.len() {
            self.qPages[i].zero_amplitudes();
        }
    }

    fn copy_state_vec(&mut self, src: QEnginePtr) {
        self.copy_state_vec(src.clone());
    }

    fn copy_state_vec(&mut self, src: QPagerPtr) {
        let qpp = self.qubits_per_page();
        src.combine_engines(qpp);
        src.separate_engines(qpp, true);
        for i in 0..self.qPages.len() {
            self.qPages[i].copy_state_vec(src.qPages[i].clone());
        }
    }

    fn is_zero_amplitude(&self) -> bool {
        for i in 0..self.qPages.len() {
            if !self.qPages[i].is_zero_amplitude() {
                return false;
            }
        }
        true
    }

    fn get_amplitude_page(&self, pagePtr: *mut Complex, offset: i64, length: i64) {
        self.get_set_amplitude_page(pagePtr, std::ptr::null(), offset, length);
    }

    fn set_amplitude_page(&self, pagePtr: *const Complex, offset: i64, length: i64) {
        self.get_set_amplitude_page(std::ptr::null_mut(), pagePtr, offset, length);
    }

    fn set_amplitude_page(&self, pageEnginePtr: QEnginePtr, srcOffset: i64, dstOffset: i64, length: i64) {
        self.set_amplitude_page(pageEnginePtr.clone(), srcOffset, dstOffset, length);
    }

    fn set_amplitude_page(&self, pageEnginePtr: QPagerPtr, srcOffset: i64, dstOffset: i64, length: i64) {
        self.combine_engines();
        pageEnginePtr.combine_engines();
        self.qPages[0].set_amplitude_page(pageEnginePtr.qPages[0].clone(), srcOffset, dstOffset, length);
    }

    fn shuffle_buffers(&mut self, engine: QEnginePtr) {
        self.shuffle_buffers(engine.clone());
    }

    fn shuffle_buffers(&mut self, engine: QPagerPtr) {
        let qpp = self.qubits_per_page();
        let tcqpp = engine.qubits_per_page();
        engine.separate_engines(qpp, true);
        self.separate_engines(tcqpp, true);
        if self.qPages.len() == 1 {
            self.qPages[0].shuffle_buffers(engine.qPages[0].clone());
            return;
        }
        let offset = self.qPages.len() >> 1;
        for i in 0..offset {
            std::mem::swap(&mut self.qPages[offset + i], &mut engine.qPages[i]);
        }
    }

    fn clone_empty(&self) -> QEnginePtr {
        // ...
    }

    fn queue_set_do_normalize(&mut self, doNorm: bool) {
        self.finish();
        self.doNormalize = doNorm;
    }

    fn queue_set_running_norm(&mut self, runningNrm: f32) {
        self.finish();
        self.runningNorm = runningNrm;
    }

    fn prob_reg(&mut self, start: i64, length: i64, permutation: i64) -> f32 {
        self.combine_engines();
        self.qPages[0].prob_reg(start, length, permutation)
    }

    fn apply_m(&mut self, regMask: i64, result: i64, nrm: Complex) {
        self.combine_engines();
        self.qPages[0].apply_m(regMask, result, nrm);
    }

    fn get_expectation(&mut self, valueStart: i64, valueLength: i64) -> f32 {
        self.combine_engines();
        self.qPages[0].get_expectation(valueStart, valueLength)
    }

    fn apply_2x2(
        &mut self,
        offset1: i64,
        offset2: i64,
        mtrx: *const Complex,
        bitCount: i64,
        qPowersSorted: *const i64,
        doCalcNorm: bool,
        norm_thresh: f32,
    ) {
        self.combine_engines();
        self.qPages[0].apply_2x2(offset1, offset2, mtrx, bitCount, qPowersSorted, doCalcNorm, norm_thresh);
    }

    fn get_running_norm(&self) -> f32 {
        let mut toRet = 0.0;
        for i in 0..self.qPages.len() {
            toRet += self.qPages[i].get_running_norm();
        }
        toRet
    }

    fn first_nonzero_phase(&self) -> f32 {
        for i in 0..self.qPages.len() {
            if !self.qPages[i].is_zero_amplitude() {
                return self.qPages[i].first_nonzero_phase();
            }
        }
        0.0
    }

    fn set_quantum_state(&self, inputState: *const Complex) {
        // ...
    }

    fn get_quantum_state(&self, outputState: *mut Complex) {
        // ...
    }

    fn get_probs(&self, outputProbs: *mut f32) {
        // ...
    }

    fn get_amplitude(&self, perm: i64) -> Complex {
        let (p, a) = bi_div_mod(perm, self.page_max_q_power());
        self.qPages[p].get_amplitude(a)
    }

    fn set_amplitude(&self, perm: i64, amp: Complex) {
        let (p, a) = bi_div_mod(perm, self.page_max_q_power());
        self.qPages[p].set_amplitude(a, amp);
    }

    fn prob_all(&self, perm: i64) -> f32 {
        let (p, a) = bi_div_mod(perm, self.page_max_q_power());
        self.qPages[p].prob_all(a)
    }

    fn set_permutation(&self, perm: i64, phaseFac: Complex) {
        let (p, a) = bi_div_mod(perm, self.page_max_q_power());
        self.qPages[p].set_permutation(a, phaseFac);
    }

    fn compose(&mut self, toCopy: QPagerPtr) -> i64 {
        self.compose_either(toCopy, false)
    }

    fn compose(&mut self, toCopy: QInterfacePtr) -> i64 {
        self.compose(toCopy.clone())
    }

    fn compose_no_clone(&mut self, toCopy: QPagerPtr) -> i64 {
        self.compose_either(toCopy, true)
    }

    fn compose_no_clone(&mut self, toCopy: QInterfacePtr) -> i64 {
        self.compose_no_clone(toCopy.clone())
    }

    fn compose_either(&mut self, toCopy: QPagerPtr, willDestroy: bool) -> i64 {
        // ...
    }

    fn decompose(&mut self, start: i64, dest: QPagerPtr) {
        self.decompose(start, dest.clone())
    }

    fn decompose(&mut self, start: i64, dest: QInterfacePtr) {
        self.decompose(start, dest.clone())
    }

    fn decompose(&mut self, start: i64, dest: QPagerPtr) {
        // ...
    }

    fn decompose(&mut self, start: i64, length: i64) -> QInterfacePtr {
        // ...
    }

    fn dispose(&mut self, start: i64, length: i64) {
        // ...
    }

    fn dispose(&mut self, start: i64, length: i64, disposedPerm: i64) {
        // ...
    }

    fn allocate(&mut self, start: i64, length: i64) -> i64 {
        // ...
    }

    fn mtrx(&mut self, mtrx: *const Complex, target: i64) {
        // ...
    }

    fn phase(&mut self, topLeft: Complex, bottomRight: Complex, qubitIndex: i64) {
        self.apply_single_either(false, topLeft, bottomRight, qubitIndex);
    }

    fn invert(&mut self, topRight: Complex, bottomLeft: Complex, qubitIndex: i64) {
        self.apply_single_either(true, topRight, bottomLeft, qubitIndex);
    }

    fn mcmtrx(
        &mut self,
        controlPerm: i64,
        controls: &[i64],
        target: i64,
        mtrx: *const Complex,
        isSqiCtrl: bool,
        isIntraCtrled: bool,
    ) {
        // ...
    }

    fn semi_meta_controlled(
        &mut self,
        controlPerm: i64,
        controls: &[i64],
        target: i64,
        fn: Qubit1Fn,
    ) {
        // ...
    }

    fn meta_swap(&mut self, qubit1: i64, qubit2: i64, isIPhaseFac: bool, isInverse: bool) {
        // ...
    }

    fn combine_and_op<F>(&mut self, fn: F, bits: &[i64])
    where
        F: Fn(&mut QEngine, i64),
    {
        // ...
    }

    fn combine_and_op_controlled<F>(&mut self, fn: F, bits: &[i64], controls: &[i64])
    where
        F: Fn(&mut QEngine, i64),
    {
        // ...
    }

    fn apply_single_either(&mut self, isInvert: bool, top: Complex, bottom: Complex, target: i64) {
        // ...
    }

    fn apply_either_controlled_single_bit(
        &mut self,
        controlPerm: i64,
        controls: &[i64],
        target: i64,
        mtrx: *const Complex,
    ) {
        // ...
    }

    fn either_i_swap(&mut self, qubit1: i64, qubit2: i64, isInverse: bool) {
        // ...
    }

    fn get_set_amplitude_page(
        &self,
        pagePtr: *mut Complex,
        cPagePtr: *const Complex,
        offset: i64,
        length: i64,
    ) {
        // ...
    }

    fn is_finished(&self) -> bool {
        for i in 0..self.qPages.len() {
            if !self.qPages[i].is_finished() {
                return false;
            }
        }
        true
    }

    fn dump(&self) {
        for i in 0..self.qPages.len() {
            self.qPages[i].dump();
        }
    }

    fn clone(&self) -> QInterfacePtr {
        // ...
    }

    fn set_device(&mut self, dID: i64) {
        self.deviceIDs.clear();
        self.deviceIDs.push(dID);
        for i in 0..self.qPages.len() {
            self.qPages[i].set_device(dID);
        }
        // ...
    }

    fn get_device(&self) -> i64 {
        self.qPages[0].get_device()
    }

    fn get_max_size(&self) -> i64 {
        self.qPages[0].get_max_size()
    }

    fn sum_sqr_diff(&self, toCompare: QPagerPtr) -> f32 {
        // ...
    }

    fn sum_sqr_diff(&self, toCompare: QPagerPtr) -> f32 {
        // ...
    }
}


