use std::sync::Arc;
use std::cell::RefCell;
use std::rc::Rc;

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

pub type QPagerPtr = Arc<RefCell<QPager>>;

impl QPager {
    pub fn new(eng: Vec<QInterfaceEngine>, qBitCount: i64, initState: i64, rgp: qrack_rand_gen_ptr, phaseFac: Complex, doNorm: bool, ignored: bool, useHostMem: bool, deviceId: i64, useHardwareRNG: bool, useSparseStateVec: bool, norm_thresh: real1_f, devList: Vec<i64>, qubitThreshold: i64, separation_thresh: real1_f) -> QPagerPtr {
        let qPager = QPager {
            useGpuThreshold: false,
            isSparse: useSparseStateVec,
            useTGadget: true,
            maxPageSetting: -1,
            maxPageQubits: -1,
            thresholdQubitsPerPage: qubitThreshold,
            baseQubitsPerPage: -1,
            maxQubits: qBitCount,
            devID: deviceId,
            rootEngine: QInterfaceEngine::default(),
            basePageMaxQPower: -1,
            basePageCount: -1,
            phaseFactor: phaseFac,
            devicesHostPointer: vec![],
            deviceIDs: devList,
            engines: eng,
            qPages: vec![],
        };
        let qPagerPtr = Arc::new(RefCell::new(qPager));
        qPagerPtr.borrow_mut().init();
        if qBitCount == 0 {
            return qPagerPtr;
        }
        let initState = initState & (qPagerPtr.borrow().maxQPower + 1);
        let initStateOcl = initState as i64;
        let mut pagePerm = 0;
        for i in 0..qPagerPtr.borrow().basePageCount {
            let isPermInPage = initState >= pagePerm;
            pagePerm += qPagerPtr.borrow().basePageMaxQPower;
            let isPermInPage = isPermInPage && initStateOcl < pagePerm;
            if isPermInPage {
                let mut qPage = qPagerPtr.borrow_mut().make_engine(qPagerPtr.borrow().baseQubitsPerPage, i);
                qPage.set_permutation(initStateOcl - (pagePerm - qPagerPtr.borrow().basePageMaxQPower));
                qPagerPtr.borrow_mut().qPages.push(qPage);
            } else {
                let qPage = qPagerPtr.borrow_mut().make_engine(qPagerPtr.borrow().baseQubitsPerPage, i);
                qPagerPtr.borrow_mut().qPages.push(qPage);
            }
        }
        qPagerPtr
    }

    pub fn set_qubit_count(&mut self, qb: i64) {
        QInterface::set_qubit_count(qb);
        self.baseQubitsPerPage = if self.qubitCount < self.thresholdQubitsPerPage { self.qubitCount } else { self.thresholdQubitsPerPage };
        self.basePageCount = pow2Ocl(self.qubitCount - self.baseQubitsPerPage);
        self.basePageMaxQPower = pow2Ocl(self.baseQubitsPerPage);
    }

    pub fn page_max_q_power(&self) -> i64 {
        let toRet = 0;
        bi_div_mod_small(self.maxQPower, self.qPages.len(), &toRet, NULL);
        toRet as i64
    }

    pub fn paged_qubit_count(&self) -> i64 {
        log2Ocl(self.qPages.len()) as i64
    }

    pub fn qubits_per_page(&self) -> i64 {
        log2Ocl(self.page_max_q_power()) as i64
    }

    pub fn get_page_device(&self, page: i64) -> i64 {
        self.deviceIDs[page % self.deviceIDs.len()]
    }

    pub fn get_page_host_pointer(&self, page: i64) -> bool {
        self.devicesHostPointer[page % self.devicesHostPointer.len()]
    }
    pub fn combine_engines(&mut self) {
        self.combine_engines(self.qubitCount);
    }

    pub fn separate_engines(&mut self) {
        self.separate_engines(self.baseQubitsPerPage, false);
    }

    pub fn set_concurrency(&mut self, threadsPerEngine: u32) {
        QInterface::set_concurrency(threadsPerEngine);
        for i in 0..self.qPages.len() {
            self.qPages[i].set_concurrency(threadsPerEngine);
        }
    }

    pub fn set_t_injection(&mut self, useGadget: bool) {
        self.useTGadget = useGadget;
        for i in 0..self.qPages.len() {
            self.qPages[i].set_t_injection(useGadget);
        }
    }

    pub fn get_t_injection(&self) -> bool {
        self.useTGadget
    }

    pub fn is_open_cl(&self) -> bool {
        self.qPages[0].is_open_cl()
    }

    pub fn release_engine(&mut self) -> QEnginePtr {
        self.combine_engines();
        self.qPages[0].clone()
    }

    pub fn lock_engine(&mut self, eng: QEnginePtr) {
        self.qPages.resize(1);
        self.qPages[0] = eng;
        self.qPages[0].set_device(self.deviceIDs[0]);
    }

    pub fn zero_amplitudes(&mut self) {
        for i in 0..self.qPages.len() {
            self.qPages[i].zero_amplitudes();
        }
    }

    pub fn copy_state_vec(&mut self, src: QEnginePtr) {
        let qpp = self.qubits_per_page();
        src.borrow_mut().combine_engines(qpp);
        src.borrow_mut().separate_engines(qpp, true);
        for i in 0..self.qPages.len() {
            self.qPages[i].copy_state_vec(src.borrow().qPages[i]);
        }
    }

    pub fn is_zero_amplitude(&self) -> bool {
        for i in 0..self.qPages.len() {
            if !self.qPages[i].is_zero_amplitude() {
                return false;
            }
        }
        true
    }

    pub fn get_amplitude_page(&self, pagePtr: *mut Complex, offset: i64, length: i64) {
        self.get_set_amplitude_page(pagePtr, NULL, offset, length);
    }

    pub fn set_amplitude_page(&self, pagePtr: *const Complex, offset: i64, length: i64) {
        self.get_set_amplitude_page(NULL, pagePtr, offset, length);
    }

    pub fn set_amplitude_page(&self, pageEnginePtr: QEnginePtr, srcOffset: i64, dstOffset: i64, length: i64) {
        self.set_amplitude_page(pageEnginePtr.borrow().qPages[0], srcOffset, dstOffset, length);
    }

    pub fn set_amplitude_page(&self, pageEnginePtr: QPagerPtr, srcOffset: i64, dstOffset: i64, length: i64) {
        self.combine_engines();
        pageEnginePtr.borrow_mut().combine_engines();
        self.qPages[0].set_amplitude_page(pageEnginePtr.borrow().qPages[0], srcOffset, dstOffset, length);
    }

    pub fn shuffle_buffers(&self, engine: QEnginePtr) {
        self.shuffle_buffers(engine.borrow().qPages[0]);
    }

    pub fn shuffle_buffers(&self, engine: QPagerPtr) {
        let qpp = self.qubits_per_page();
        let tcqpp = engine.borrow().qubits_per_page();
        engine.borrow_mut().separate_engines(qpp, true);
        self.separate_engines(tcqpp, true);
        if self.qPages.len() == 1 {
            self.qPages[0].shuffle_buffers(engine.borrow().qPages[0]);
            return;
        }
        let offset = self.qPages.len() >> 1;
        for i in 0..offset {
            self.qPages[offset + i].swap(engine.borrow().qPages[i]);
        }
    }

    pub fn queue_set_do_normalize(&mut self, doNorm: bool) {
        self.finish();
        self.doNormalize = doNorm;
    }

    pub fn queue_set_running_norm(&mut self, runningNrm: real1_f) {
        self.finish();
        self.runningNorm = runningNrm;
    }

    pub fn prob_reg(&mut self, start: i64, length: i64, permutation: i64) -> real1_f {
        self.combine_engines();
        self.qPages[0].prob_reg(start, length, permutation)
    }

    pub fn apply_m(&mut self, regMask: i64, result: i64, nrm: Complex) {
        self.combine_engines();
        self.qPages[0].apply_m(regMask, result, nrm);
    }

    pub fn get_expectation(&mut self, valueStart: i64, valueLength: i64) -> real1_f {
        self.combine_engines();
        self.qPages[0].get_expectation(valueStart, valueLength)
    }

    pub fn apply_2x2(&mut self, offset1: i64, offset2: i64, mtrx: *const Complex, bitCount: i64, qPowersSorted: *const i64, doCalcNorm: bool, norm_thresh: real1_f) {
        self.combine_engines();
        self.qPages[0].apply_2x2(offset1, offset2, mtrx, bitCount, qPowersSorted, doCalcNorm, norm_thresh);
    }

    pub fn get_running_norm(&self) -> real1_f {
        let mut toRet = 0.0;
        for i in 0..self.qPages.len() {
            toRet += self.qPages[i].get_running_norm();
        }
        toRet
    }

    pub fn first_nonzero_phase(&self) -> real1_f {
        for i in 0..self.qPages.len() {
            if !self.qPages[i].is_zero_amplitude() {
                return self.qPages[i].first_nonzero_phase();
            }
        }
        0.0
    }

    pub fn get_amplitude(&self, perm: i64) -> Complex {
        let (p, a) = bi_div_mod(perm, self.page_max_q_power(), &p, &a);
        self.qPages[p as usize].get_amplitude(a)
    }

    pub fn set_amplitude(&self, perm: i64, amp: Complex) {
        let (p, a) = bi_div_mod(perm, self.page_max_q_power(), &p, &a);
        self.qPages[p as usize].set_amplitude(a, amp);
    }

    pub fn prob_all(&self, perm: i64) -> real1_f {
        let (p, a) = bi_div_mod(perm, self.page_max_q_power(), &p, &a);
        self.qPages[p as usize].prob_all(a)
    }

    pub fn compose(&mut self, toCopy: QPagerPtr) -> i64 {
        self.compose_either(toCopy, false)
    }

    pub fn compose(&mut self, toCopy: QInterfacePtr) -> i64 {
        self.compose(toCopy.borrow().qPages[0])
    }

    pub fn compose_no_clone(&mut self, toCopy: QPagerPtr) -> i64 {
        self.compose_either(toCopy, true)
    }

    pub fn compose_no_clone(&mut self, toCopy: QInterfacePtr) -> i64 {
        self.compose_no_clone(toCopy.borrow().qPages[0])
    }

    pub fn decompose(&mut self, start: i64, dest: QPagerPtr) {
        self.decompose(start, dest.borrow().qPages[0]);
    }

    pub fn decompose(&mut self, start: i64, dest: QInterfacePtr) {
        self.decompose(start, dest.borrow().qPages[0]);
    }
    pub fn dispose(&mut self, start: i64, length: i64) {
        self.dispose(start, length, 0);
    }

    pub fn phase(&mut self, topLeft: Complex, bottomRight: Complex, qubitIndex: i64) {
        self.apply_single_either(false, topLeft, bottomRight, qubitIndex);
    }

    pub fn invert(&mut self, topRight: Complex, bottomLeft: Complex, qubitIndex: i64) {
        self.apply_single_either(true, topRight, bottomLeft, qubitIndex);
    }

    pub fn mcmtrx(&mut self, controls: Vec<i64>, mtrx: *const Complex, target: i64) {
        let p = pow2(controls.len());
        bi_decrement(&p, 1);
        self.apply_either_controlled_single_bit(p, controls, target, mtrx);
    }

    pub fn macmtrx(&mut self, controls: Vec<i64>, mtrx: *const Complex, target: i64) {
        self.apply_either_controlled_single_bit(0, controls, target, mtrx);
    }

    pub fn z_mask(&mut self, mask: i64) {
        self.phase_parity(PI_R1, mask);
    }

    pub fn phase_parity(&mut self, radians: real1_f, mask: i64) {
        if bi_compare_0(mask) == 0 {
            return;
        }
        self.combine_engines();
        self.qPages[0].phase_parity(radians, mask);
    }

    pub fn force_m_reg(&mut self, start: i64, length: i64, result: i64, doForce: bool, doApply: bool) -> i64 {
        QInterface::force_m_reg(start, length, result, doForce, doApply)
    }

    pub fn iswap(&mut self, qubit1: i64, qubit2: i64) {
        self.either_iswap(qubit1, qubit2, false);
    }

    pub fn iiswap(&mut self, qubit1: i64, qubit2: i64) {
        self.either_iswap(qubit1, qubit2, true);
    }

    pub fn prob_parity(&mut self, mask: i64) -> real1_f {
        if bi_compare_0(mask) == 0 {
            return 0.0;
        }
        self.combine_engines();
        self.qPages[0].prob_parity(mask)
    }

    pub fn force_m_parity(&mut self, mask: i64, result: bool, doForce: bool) -> bool {
        if bi_compare_0(mask) == 0 {
            return 0.0;
        }
        self.combine_engines();
        self.qPages[0].force_m_parity(mask, result, doForce)
    }

    pub fn finish(&mut self) {
        for i in 0..self.qPages.len() {
            self.qPages[i].finish();
        }
    }

    pub fn is_finished(&self) -> bool {
        for i in 0..self.qPages.len() {
            if !self.qPages[i].is_finished() {
                return false;
            }
        }
        true
    }

    pub fn dump(&self) {
        for i in 0..self.qPages.len() {
            self.qPages[i].dump();
        }
    }

    pub fn set_device(&mut self, dID: i64) {
        self.deviceIDs.clear();
        self.deviceIDs.push(dID);
        for i in 0..self.qPages.len() {
            self.qPages[i].set_device(dID);
        }
        if self.rootEngine != QINTERFACE_CPU {
            self.maxPageQubits = log2Ocl(OCLEngine::Instance().GetDeviceContextPtr(devID).GetMaxAlloc() / std::mem::size_of::<Complex>());
            self.maxPageQubits = if self.maxPageSetting < self.maxPageQubits { self.maxPageSetting } else { 1 };
        }
        if !self.useGpuThreshold {
            return;
        }
        self.thresholdQubitsPerPage = self.maxPageQubits;
    }

    pub fn get_device(&self) -> i64 {
        self.qPages[0].get_device()
    }

    pub fn get_max_size(&self) -> i64 {
        self.qPages[0].get_max_size()
    }

    pub fn sum_sqr_diff(&self, toCompare: QPagerPtr) -> real1_f {
        self.sum_sqr_diff(toCompare.borrow().qPages[0])
    }

    pub fn sum_sqr_diff(&self, toCompare: QInterfacePtr) -> real1_f {
        self.sum_sqr_diff(toCompare.borrow().qPages[0])
    }

fn init(&mut self) {
    if self.engines.is_empty() {
        #[cfg(any(ENABLE_OPENCL, ENABLE_CUDA))]
        {
            if QRACK_GPU_SINGLETON.GetDeviceCount() != 0 {
                self.engines.push(QRACK_GPU_ENGINE);
            } else {
                self.engines.push(QINTERFACE_CPU);
            }
        }
        #[cfg(not(any(ENABLE_OPENCL, ENABLE_CUDA)))]
        {
            self.engines.push(QINTERFACE_CPU);
        }
    }
    if self.engines[0] == QINTERFACE_HYBRID || self.engines[0] == QRACK_GPU_ENGINE {
        #[cfg(any(ENABLE_OPENCL, ENABLE_CUDA))]
        {
            if QRACK_GPU_SINGLETON.GetDeviceCount() == 0 {
                self.engines[0] = QINTERFACE_CPU;
            }
        }
        #[cfg(not(any(ENABLE_OPENCL, ENABLE_CUDA)))]
        {
            self.engines[0] = QINTERFACE_CPU;
        }
    }
    #[cfg(ENABLE_ENV_VARS)]
    {
        if let Some(max_page_qb) = std::env::var("QRACK_MAX_PAGE_QB").ok().map(|s| s.parse::<bitLenInt>().unwrap()) {
            self.maxPageSetting = max_page_qb;
        }
    }
    let mut engine_level = 0;
    let mut root_engine = self.engines[0];
    while engine_level < self.engines.len() && (root_engine != QINTERFACE_CPU) && (root_engine != QRACK_GPU_ENGINE) && (root_engine != QINTERFACE_HYBRID) {
        engine_level += 1;
        root_engine = self.engines[engine_level];
    }
    #[cfg(any(ENABLE_OPENCL, ENABLE_CUDA))]
    {
        if root_engine != QINTERFACE_CPU {
            let max_page_qubits = log2Ocl(QRACK_GPU_SINGLETON.GetDeviceContextPtr(devID).GetMaxAlloc() / std::mem::size_of::<complex>());
            self.maxPageQubits = if self.maxPageSetting < max_page_qubits { self.maxPageSetting } else { 1 };
        }
        if root_engine != QINTERFACE_CPU && root_engine != QRACK_GPU_ENGINE {
            root_engine = QINTERFACE_HYBRID;
        }
        if !self.thresholdQubitsPerPage && (root_engine == QRACK_GPU_ENGINE || root_engine == QINTERFACE_HYBRID) {
            self.useGpuThreshold = true;
            self.thresholdQubitsPerPage = self.maxPageQubits;
        }
    }
    if !self.thresholdQubitsPerPage {
        self.useGpuThreshold = false;
        #[cfg(ENABLE_ENV_VARS)]
        {
            let pStridePow = std::env::var("QRACK_PSTRIDEPOW").ok().map(|s| s.parse::<bitLenInt>().unwrap()).unwrap_or(PSTRIDEPOW);
            #[cfg(ENABLE_PTHREAD)]
            {
                let numCores = GetConcurrencyLevel();
                self.thresholdQubitsPerPage = pStridePow + if numCores == 1 { 1 } else { log2Ocl(numCores - 1) + 1 };
            }
            #[cfg(not(ENABLE_PTHREAD))]
            {
                self.thresholdQubitsPerPage = pStridePow + 1;
            }
        }
    }
    self.set_qubit_count(self.qubitCount);
    self.maxQubits = std::mem::size_of::<bitCapIntOcl>() * bitsInByte;
    #[cfg(ENABLE_ENV_VARS)]
    {
        if let Some(max_paging_qb) = std::env::var("QRACK_MAX_PAGING_QB").ok().map(|s| s.parse::<bitLenInt>().unwrap()) {
            self.maxQubits = max_paging_qb;
        }
    }
    if self.qubitCount > self.maxQubits {
        panic!("Cannot instantiate a QPager with greater capacity than environment variable QRACK_MAX_PAGING_QB.");
    }
    #[cfg(ENABLE_ENV_VARS)]
    {
        if let Some(qpager_devices) = std::env::var("QRACK_QPAGER_DEVICES").ok() {
            let dev_list_str = qpager_devices;
            self.deviceIDs.clear();
            if dev_list_str != "" {
                let mut dev_list_str_stream = dev_list_str.split(",");
                let re = regex::Regex::new("[.]").unwrap();
                while let Some(term) = dev_list_str_stream.next() {
                    let tokens: Vec<&str> = re.split(term).collect();
                    if tokens.len() == 1 {
                        let device_id = term.parse::<i32>().unwrap();
                        if device_id == -2 {
                            self.deviceIDs.push(devID as i32);
                        } else if device_id == -1 {
                            #[cfg(any(ENABLE_OPENCL, ENABLE_CUDA))]
                            {
                                self.deviceIDs.push(QRACK_GPU_SINGLETON.GetDefaultDeviceID() as i32);
                            }
                            #[cfg(not(any(ENABLE_OPENCL, ENABLE_CUDA)))]
                            {
                                self.deviceIDs.push(0);
                            }
                        } else {
                            self.deviceIDs.push(device_id);
                        }
                        continue;
                    }
                    let max_i = tokens[0].parse::<u32>().unwrap();
                    let mut ids = Vec::with_capacity(tokens.len() - 1);
                    for i in 1..tokens.len() {
                        let device_id = tokens[i].parse::<i32>().unwrap();
                        if device_id == -2 {
                            ids.push(devID as i32);
                        } else if device_id == -1 {
                            #[cfg(any(ENABLE_OPENCL, ENABLE_CUDA))]
                            {
                                ids.push(QRACK_GPU_SINGLETON.GetDefaultDeviceID() as i32);
                            }
                            #[cfg(not(any(ENABLE_OPENCL, ENABLE_CUDA)))]
                            {
                                ids.push(0);
                            }
                        } else {
                            ids.push(device_id);
                        }
                    }
                    for _ in 0..max_i {
                        for j in 0..ids.len() {
                            self.deviceIDs.push(ids[j]);
                        }
                    }
                }
            }
        }
        if let Some(qpager_devices_host_pointer) = std::env::var("QRACK_QPAGER_DEVICES_HOST_POINTER").ok() {
            let dev_list_str = qpager_devices_host_pointer;
            if dev_list_str != "" {
                let mut dev_list_str_stream = dev_list_str.split(",");
                let re = regex::Regex::new("[.]").unwrap();
                while let Some(term) = dev_list_str_stream.next() {
                    let tokens: Vec<&str> = re.split(term).collect();
                    if tokens.len() == 1 {
                        self.devicesHostPointer.push(term.parse::<bool>().unwrap());
                        continue;
                    }
                    let max_i = tokens[0].parse::<u32>().unwrap();
                    let mut hps = Vec::with_capacity(tokens.len() - 1);
                    for i in 1..tokens.len() {
                        hps.push(tokens[i].parse::<bool>().unwrap());
                    }
                    for _ in 0..max_i {
                        for j in 0..hps.len() {
                            self.devicesHostPointer.push(hps[j]);
                        }
                    }
                }
            } else {
                self.devicesHostPointer.push(self.useHostRam);
            }
        }
    }
    if self.deviceIDs.is_empty() {
        self.deviceIDs.push(devID);
    }
}

fn make_engine(&self, length: bitLenInt, pageId: bitCapIntOcl) -> QEnginePtr {
    let toRet = CreateQuantumInterface(
        &self.engines,
        0,
        ZERO_BCI,
        rand_generator,
        phaseFactor,
        false,
        false,
        GetPageHostPointer(pageId),
        GetPageDevice(pageId),
        useRDRAND,
        isSparse,
        amplitudeFloor,
    );
    toRet.SetQubitCount(length);
    toRet.SetConcurrency(GetConcurrencyLevel());
    toRet.SetTInjection(useTGadget);
    toRet
}


pub fn get_set_amplitude_page(&self, pagePtr: &mut [complex], cPagePtr: Option<&[complex]>, offset: bitCapIntOcl, length: bitCapIntOcl) {
    let pageLength = pageMaxQPower();
    let mut perm = 0;
    for i in 0..self.qPages.len() {
        if perm + length < offset {
            continue;
        }
        if perm >= offset + length {
            break;
        }
        let partOffset = if perm < offset { offset - perm } else { 0 };
        let partLength = if length < pageLength { length } else { pageLength };
        if let Some(cPagePtr) = cPagePtr {
            self.qPages[i].SetAmplitudePage(cPagePtr, partOffset, partLength);
        } else {
            self.qPages[i].GetAmplitudePage(pagePtr, partOffset, partLength);
        }
        perm += pageLength;
    }
}

fn combine_engines(&mut self, bit: bitLenInt) {
    if bit > self.qubitCount {
        bit = self.qubitCount;
    }
    if bit <= self.qubitsPerPage() {
        return;
    }
    let groupCount = pow2Ocl(self.qubitCount - bit);
    let groupSize = self.qPages.len() / groupCount;
    let pagePower = pageMaxQPower();
    let mut nQPages = Vec::with_capacity(groupCount * groupSize);
    for i in 0..groupCount {
        let engine = self.make_engine(bit, i);
        nQPages.push(engine);
        for j in 0..groupSize {
            let page = j + i * groupSize;
            engine.SetAmplitudePage(self.qPages[page], 0, j * pagePower, pagePower);
            self.qPages[page] = None;
        }
    }
    self.qPages = nQPages;
}

fn separate_engines(&mut self, thresholdBits: bitLenInt, noBaseFloor: bool) {
    if !noBaseFloor && thresholdBits < self.baseQubitsPerPage {
        thresholdBits = self.baseQubitsPerPage;
    }
    if thresholdBits >= self.qubitsPerPage() {
        return;
    }
    let pagesPer = pow2Ocl(self.qubitCount - thresholdBits) / self.qPages.len();
    let pageMaxQPower = pow2Ocl(thresholdBits);
    let mut nQPages = Vec::with_capacity(self.qPages.len() * pagesPer);
    for i in 0..self.qPages.len() {
        for j in 0..pagesPer {
            nQPages.push(self.make_engine(thresholdBits, j + i * pagesPer));
            nQPages.last_mut().unwrap().SetAmplitudePage(self.qPages[i], j * pageMaxQPower, 0, pageMaxQPower);
        }
        self.qPages[i] = None;
    }
    self.qPages = nQPages;
}

fn single_bit_gate<Qubit1Fn>(&mut self, target: bitLenInt, fn: Qubit1Fn, isSqiCtrl: bool, isAnti: bool)
where
    Qubit1Fn: Fn(&mut QEnginePtr, bitLenInt),
{
    let qpp = self.qubitsPerPage();
    if self.doNormalize {
        let mut runningNorm = ZERO_R1;
        for i in 0..self.qPages.len() {
            self.qPages[i].Finish();
            runningNorm += self.qPages[i].GetRunningNorm();
        }
        for i in 0..self.qPages.len() {
            self.qPages[i].QueueSetRunningNorm(runningNorm);
            self.qPages[i].QueueSetDoNormalize(true);
        }
    }
    if target < qpp {
        for i in 0..self.qPages.len() {
            let engine = &mut self.qPages[i];
            fn(engine, target);
            if self.doNormalize {
                engine.QueueSetDoNormalize(false);
            }
        }
        return;
    }
    let sqi = qpp - 1;
    let target = target - qpp;
    let targetPow = pow2Ocl(target);
    let targetMask = targetPow - 1;
    let maxLcv = self.qPages.len() >> 1;
    #[cfg(ENABLE_PTHREAD)]
    let numCores = GetConcurrencyLevel();
    for i in 0..maxLcv {
        let j = i & targetMask | (i ^ (i & targetMask)) << 1;
        let engine1 = &mut self.qPages[j];
        let engine2 = &mut self.qPages[j | targetPow];
        let doNrm = self.doNormalize;
        #[cfg(ENABLE_PTHREAD)]
        {
            let iF = i % numCores;
            if i != iF {
                futures[iF].get();
            }
            futures[iF] = std::thread::spawn(move || {
        }
        engine1.ShuffleBuffers(engine2);
        if !isSqiCtrl || isAnti {
            fn(engine1, sqi);
        }
        if !isSqiCtrl || !isAnti {
            fn(engine2, sqi);
        }
        engine1.ShuffleBuffers(engine2);
        if doNrm {
            engine1.QueueSetDoNormalize(false);
            engine2.QueueSetDoNormalize(false);
        }
        #[cfg(ENABLE_PTHREAD)]
        });
    }
    #[cfg(ENABLE_PTHREAD)]
    for i in 0..futures.len() {
        futures[i].join().unwrap();
    }
}

fn meta_controlled<Qubit1Fn>(
    &mut self,
    controlPerm: bitCapInt,
    controls: &[bitLenInt],
    target: bitLenInt,
    fn: Qubit1Fn,
    mtrx: Option<&[complex]>,
    isSqiCtrl: bool,
    isIntraCtrled: bool,
) where
    Qubit1Fn: Fn(&mut QEnginePtr, bitLenInt),
{
    let qpp = self.qubitsPerPage();
    let sqi = qpp - 1;
    let target = target - qpp;
    let mut sortedMasks = Vec::with_capacity(1 + controls.len());
    let targetPow = pow2Ocl(target);
    sortedMasks.push(targetPow - 1);
    let mut controlMask = 0;
    for i in 0..controls.len() {
        sortedMasks.push(pow2Ocl(controls[i] - qpp) - 1);
        if (controlPerm >> i) & 1 != 0 {
            controlMask |= sortedMasks[i];
        }
    }
    sortedMasks.sort();
    let isSpecial;
    let isInvert;
    let top;
    let bottom;
    if !isSqiCtrl && !isIntraCtrled && IS_NORM_0(mtrx.unwrap()[1]) && IS_NORM_0(mtrx.unwrap()[2]) {
        isSpecial = true;
        isInvert = false;
        top = mtrx.unwrap()[0];
        bottom = mtrx.unwrap()[3];
    } else if !isSqiCtrl && !isIntraCtrled && IS_NORM_0(mtrx.unwrap()[0]) && IS_NORM_0(mtrx.unwrap()[3]) {
        isSpecial = true;
        isInvert = true;
        top = mtrx.unwrap()[1];
        bottom = mtrx.unwrap()[2];
    } else {
        isSpecial = false;
        isInvert = false;
        top = ZERO_CMPLX;
        bottom = ZERO_CMPLX;
    }
    let maxLcv = self.qPages.len() >> sortedMasks.len();
    #[cfg(ENABLE_PTHREAD)]
    let numCores = GetConcurrencyLevel();
    for i in 0..maxLcv {
        let mut jHi = i;
        let mut j = 0;
        for k in 0..sortedMasks.len() {
            let jLo = jHi & sortedMasks[k];
            jHi = (jHi ^ jLo) << 1;
            j |= jLo;
        }
        j |= jHi | controlMask;
        if isInvert {
            self.qPages.swap(j, j | targetPow);
        }
        let engine1 = &mut self.qPages[j];
        let engine2 = &mut self.qPages[j | targetPow];
        if isSpecial {
            if !IS_NORM_0(ONE_CMPLX - top) {
                engine1.Phase(top, top, 0);
            }
            if !IS_NORM_0(ONE_CMPLX - bottom) {
                engine2.Phase(bottom, bottom, 0);
            }
            continue;
        }
        let isAnti = (controlPerm >> controls.len()) & 1 == 0;
        #[cfg(ENABLE_PTHREAD)]
        {
            let iF = i % numCores;
            if i != iF {
                futures[iF].get();
            }
            futures[iF] = std::thread::spawn(move || {
        }
        engine1.ShuffleBuffers(engine2);
        if !isSqiCtrl || isAnti {
            fn(engine1, sqi);
        }
        if !isSqiCtrl || !isAnti {
            fn(engine2, sqi);
        }
        engine1.ShuffleBuffers(engine2);
        #[cfg(ENABLE_PTHREAD)]
        });
    }
    if isSpecial {
        return;
    }
    #[cfg(ENABLE_PTHREAD)]
    for i in 0..futures.len() {
        futures[i].join().unwrap();
    }
}

fn semi_meta_controlled<Qubit1Fn>(&mut self, controlPerm: bitCapInt, controls: &[bitLenInt], target: bitLenInt, fn: Qubit1Fn)
where
    Qubit1Fn: Fn(&mut QEnginePtr, bitLenInt),
{
    let qpp = self.qubitsPerPage();
    let mut sortedMasks = Vec::with_capacity(controls.len());
    let mut controlMask = 0;
    for i in 0..controls.len() {
        sortedMasks.push(pow2Ocl(controls[i] - qpp) - 1);
        if (controlPerm >> i) & 1 != 0 {
            controlMask |= sortedMasks[i];
        }
    }
    sortedMasks.sort();
    let maxLcv = self.qPages.len() >> sortedMasks.len();
    for i in 0..maxLcv {
        let mut jHi = i;
        let mut j = 0;
        for k in 0..sortedMasks.len() {
            let jLo = jHi & sortedMasks[k];
            jHi = (jHi ^ jLo) << 1;
            j |= jLo;
        }
        j |= jHi | controlMask;
        fn(&mut self.qPages[j], target);
    }
}

fn combine_and_op<F>(&mut self, mut fn: F, bits: &[bitLenInt])
where
    F: FnMut(&mut QEnginePtr),
{
    let highestBit = bits.iter().max().copied().unwrap_or(0);
    self.combine_engines(highestBit + 1);
    for i in 0..self.qPages.len() {
        fn(&mut self.qPages[i]);
    }
}

fn combine_and_op_controlled<F>(&mut self, mut fn: F, bits: &[bitLenInt], controls: &[bitLenInt])
where
    F: FnMut(&mut QEnginePtr),
{
    let mut bits = bits.to_vec();
    bits.extend_from_slice(controls);
    self.combine_and_op(fn, &bits);
}

fn compose_either(&mut self, toCopy: &mut QPager, willDestroy: bool) -> bitLenInt {
    let toRet = self.qubitCount;
    if toCopy.qubitCount == 0 {
        return toRet;
    }
    let nQubitCount = self.qubitCount + toCopy.qubitCount;
    if nQubitCount > self.maxQubits {
        panic!("Cannot instantiate a QPager with greater capacity than environment variable QRACK_MAX_PAGING_QB.");
    }
    if nQubitCount <= self.thresholdQubitsPerPage {
        self.combine_engines();
        toCopy.combine_engines();
        self.set_qubit_count(nQubitCount);
        return self.qPages[0].Compose(toCopy.qPages[0]);
    }
    if self.qubitCount < toCopy.qubitCount {
        let mut toCopyClone = if willDestroy { toCopy } else { toCopy.clone() };
        toCopyClone.compose(self, 0);
        self.qPages = toCopyClone.qPages;
        self.set_qubit_count(nQubitCount);
        return toRet;
    }
    let qpp = self.qubitsPerPage();
    let nPagePow = pow2Ocl(self.thresholdQubitsPerPage);
    let pmqp = self.pageMaxQPower();
    let oPagePow = toCopy.pageMaxQPower();
    let tcqpp = toCopy.qubitsPerPage();
    let maxI = toCopy.maxQPowerOcl - 1;
    let mut oOffset = oPagePow;
    let mut nQPages = Vec::with_capacity((maxI + 1) * self.qPages.len());
    for i in 0..maxI {
        if willDestroy && i == oOffset {
            oOffset -= oPagePow;
            toCopy.qPages[oOffset >> tcqpp] = None;
            oOffset += oPagePow << 1;
        }
        let amp = toCopy.GetAmplitude(i);
        if IS_NORM_0(amp) {
            for j in 0..self.qPages.len() {
                let page = i * self.qPages.len() + j;
                nQPages.push(self.make_engine(qpp, (pmqp * page) / nPagePow));
            }
            continue;
        }
        for j in 0..self.qPages.len() {
            let page = i * self.qPages.len() + j;
            nQPages.push(self.make_engine(qpp, (pmqp * page) / nPagePow));
            if !self.qPages[j].IsZeroAmplitude() {
                nQPages[page].SetAmplitudePage(self.qPages[j], 0, 0, nQPages[page].GetMaxQPower() as bitCapIntOcl);
                nQPages[page].Phase(amp, amp, 0);
            }
        }
    }
    let amp = toCopy.GetAmplitude(maxI);
    if willDestroy {
        toCopy.qPages[toCopy.qPages.len() - 1] = None;
    }
    if IS_NORM_0(amp) {
        for j in 0..self.qPages.len() {
            let page = maxI * self.qPages.len() + j;
            nQPages.push(self.make_engine(qpp, (pmqp * page) / nPagePow));
            self.qPages[j] = None;
        }
    } else {
        for j in 0..self.qPages.len() {
            let page = maxI * self.qPages.len() + j;
            nQPages.push(self.make_engine(qpp, (pmqp * page) / nPagePow));
            if !self.qPages[j].IsZeroAmplitude() {
                nQPages[page].SetAmplitudePage(self.qPages[j], 0, 0, nQPages[page].GetMaxQPower() as bitCapIntOcl);
                nQPages[page].Phase(amp, amp, 0);
            }
            self.qPages[j] = None;
        }
    }
    self.qPages = nQPages;
    self.set_qubit_count(nQubitCount);
    self.combine_engines(self.thresholdQubitsPerPage);
    self.separate_engines();
    toRet
}

    pub fn decompose(&self, start: bitLenInt, length: bitLenInt) -> QPagerPtr {
        let dest = Rc::new(QPager::new(
            self.engines.clone(),
            self.qubitCount,
            ZERO_BCI,
            self.rand_generator.clone(),
            ONE_CMPLX,
            self.doNormalize,
            self.randGlobalPhase,
            false,
            0,
            if self.hardware_rand_generator.is_none() {
                false
            } else {
                true
            },
            self.isSparse,
            self.amplitudeFloor,
            self.deviceIDs.clone(),
            self.thresholdQubitsPerPage,
        ));
        self.decompose_internal(start, Rc::clone(&dest));
        dest
    }

    fn decompose_internal(&self, start: bitLenInt, dest: QPagerPtr) {
        let length = dest.get_qubit_count();
        self.combine_engines(length + 1);
        if (start + length) > self.qubits_per_page() {
            self.ror(start, 0, self.qubitCount);
            self.decompose_internal(0, dest);
            self.rol(start, 0, self.qubitCount);
            return;
        }
        dest.combine_engines();
        let mut isDecomposed = false;
        for i in 0..self.qPages.len() {
            if !isDecomposed && !self.qPages[i].is_zero_amplitude() {
                self.qPages[i].decompose(start, dest.qPages[0]);
                dest.qPages[0].update_running_norm();
                dest.qPages[0].normalize_state();
                isDecomposed = true;
            } else {
                self.qPages[i].dispose(start, length);
            }
        }
        self.set_qubit_count(self.qubitCount - length);
        self.combine_engines(self.thresholdQubitsPerPage);
        self.separate_engines();
    }

    pub fn dispose(&self, start: bitLenInt, length: bitLenInt) {
        self.combine_engines(length + 1);
        if (start + length) > self.qubits_per_page() {
            self.ror(start, 0, self.qubitCount);
            self.dispose(0, length);
            self.rol(start, 0, self.qubitCount);
            return;
        }
        for i in 0..self.qPages.len() {
            self.qPages[i].dispose(start, length);
        }
        self.set_qubit_count(self.qubitCount - length);
        self.combine_engines(self.thresholdQubitsPerPage);
        self.separate_engines();
    }

    pub fn dispose_with_perm(&self, start: bitLenInt, length: bitLenInt, disposedPerm: bitCapInt) {
        self.combine_engines(length + 1);
        if (start + length) > self.qubits_per_page() {
            self.ror(start, 0, self.qubitCount);
            self.dispose_with_perm(0, length, disposedPerm);
            self.rol(start, 0, self.qubitCount);
            return;
        }
        for i in 0..self.qPages.len() {
            self.qPages[i].dispose_with_perm(start, length, disposedPerm);
        }
        self.set_qubit_count(self.qubitCount - length);
        self.combine_engines(self.thresholdQubitsPerPage);
        self.separate_engines();
    }

    pub fn allocate(&self, start: bitLenInt, length: bitLenInt) -> bitLenInt {
        if length == 0 {
            return start;
        }
        let nQubits = Rc::new(QPager::new(
            self.engines.clone(),
            length,
            ZERO_BCI,
            self.rand_generator.clone(),
            ONE_CMPLX,
            self.doNormalize,
            self.randGlobalPhase,
            false,
            0,
            if self.hardware_rand_generator.is_none() {
                false
            } else {
                true
            },
            self.isSparse,
            self.amplitudeFloor,
            self.deviceIDs.clone(),
            self.thresholdQubitsPerPage,
        ));
        self.compose(nQubits, start)
    }

pub fn set_quantum_state(&mut self, input_state: &[Complex]) {
    let page_power = self.page_max_q_power() as usize;
    let mut page_perm = 0;
    #[cfg(feature = "enable_pthread")]
    {
        let num_cores = self.get_concurrency_level();
        let f_count = if self.q_pages.len() < num_cores {
            self.q_pages.len()
        } else {
            num_cores
        };
        let mut futures = Vec::with_capacity(f_count);
        for i in 0..self.q_pages.len() {
            let engine = &mut self.q_pages[i];
            let do_norm = self.do_normalize;
            #[cfg(feature = "enable_pthread")]
            {
                let i_f = i % f_count;
                if i != i_f {
                    futures[i_f].take().unwrap().wait().unwrap();
                }
                let future = std::thread::spawn(move || {
                    engine.set_quantum_state(&input_state[page_perm..]);
                    if do_norm {
                        engine.update_running_norm();
                    }
                });
                futures[i_f] = Some(future);
            }
            page_perm += page_power;
        }
        for future in futures {
            future.unwrap().wait().unwrap();
        }
    }
}

pub fn get_quantum_state(&self, output_state: &mut [Complex]) {
    let page_power = self.page_max_q_power() as usize;
    let mut page_perm = 0;
    #[cfg(feature = "enable_pthread")]
    {
        let num_cores = self.get_concurrency_level();
        let f_count = if self.q_pages.len() < num_cores {
            self.q_pages.len()
        } else {
            num_cores
        };
        let mut futures = Vec::with_capacity(f_count);
        for i in 0..self.q_pages.len() {
            let engine = &self.q_pages[i];
            #[cfg(feature = "enable_pthread")]
            {
                let i_f = i % f_count;
                if i != i_f {
                    futures[i_f].take().unwrap().wait().unwrap();
                }
                let future = std::thread::spawn(move || {
                    engine.get_quantum_state(&mut output_state[page_perm..]);
                });
                futures[i_f] = Some(future);
            }
            page_perm += page_power;
        }
        for future in futures {
            future.unwrap().wait().unwrap();
        }
    }
}

pub fn get_probs(&self, output_probs: &mut [Real1]) {
    let page_power = self.page_max_q_power() as usize;
    let mut page_perm = 0;
    #[cfg(feature = "enable_pthread")]
    {
        let num_cores = self.get_concurrency_level();
        let f_count = if self.q_pages.len() < num_cores {
            self.q_pages.len()
        } else {
            num_cores
        };
        let mut futures = Vec::with_capacity(f_count);
        for i in 0..self.q_pages.len() {
            let engine = &self.q_pages[i];
            #[cfg(feature = "enable_pthread")]
            {
                let i_f = i % f_count;
                if i != i_f {
                    futures[i_f].take().unwrap().wait().unwrap();
                }
                let future = std::thread::spawn(move || {
                    engine.get_probs(&mut output_probs[page_perm..]);
                });
                futures[i_f] = Some(future);
            }
            page_perm += page_power;
        }
        for future in futures {
            future.unwrap().wait().unwrap();
        }
    }
}

pub fn set_permutation(&mut self, perm: bit_cap_int, phase_fac: Complex) {
    let page_power = self.page_max_q_power() as usize;
    let perm_ocl = perm & (self.max_q_power_ocl - 1);
    let mut page_perm = 0;
    for i in 0..self.q_pages.len() {
        let is_perm_in_page = (perm_ocl >= page_perm);
        page_perm += page_power;
        let is_perm_in_page = is_perm_in_page && (perm_ocl < page_perm);
        if is_perm_in_page {
            self.q_pages[i].set_permutation(perm_ocl - (page_perm - page_power), phase_fac);
            continue;
        }
        self.q_pages[i].zero_amplitudes();
    }
}

pub fn mtrx(&mut self, mtrx: &[Complex], target: bit_len_int) {
    if mtrx[1].norm() == 0.0 && mtrx[2].norm() == 0.0 {
        self.phase(mtrx[0], mtrx[3], target);
        return;
    } else if mtrx[0].norm() == 0.0 && mtrx[3].norm() == 0.0 {
        self.invert(mtrx[1], mtrx[2], target);
        return;
    }
    self.single_bit_gate(target, |engine, l_target| {
        engine.mtrx(mtrx, l_target);
    });
}

fn apply_single_either(&mut self, is_invert: bool, top: Complex, bottom: Complex, target: bit_len_int) {
    let qpp = self.qubits_per_page();
    if target < qpp {
        if is_invert {
            self.single_bit_gate(target, |engine, l_target| {
                engine.invert(top, bottom, l_target);
            });
        } else {
            self.single_bit_gate(target, |engine, l_target| {
                engine.phase(top, bottom, l_target);
            });
        }
        return;
    }
    if self.rand_global_phase {
        let bottom = bottom / top;
        let top = Complex::new(1.0, 0.0);
    }
    let target = target - qpp;
    let target_pow = pow2_ocl(target);
    let q_mask = target_pow - 1;
    let max_lcv = self.q_pages.len() >> 1;
    for i in 0..max_lcv {
        let j = i & q_mask;
        let j = j | ((i ^ j) << 1);
        if is_invert {
            self.q_pages.swap(j, j | target_pow);
        }
        if (1.0 - top.norm()).abs() > 0.0 {
            self.q_pages[j].phase(top, top, 0);
        }
        if (1.0 - bottom.norm()).abs() > 0.0 {
            self.q_pages[j | target_pow].phase(bottom, bottom, 0);
        }
    }
}

fn apply_either_controlled_single_bit(
    &mut self,
    control_perm: bit_cap_int,
    controls: &[bit_len_int],
    target: bit_len_int,
    mtrx: &[Complex],
) {
    if controls.is_empty() {
        self.mtrx(mtrx, target);
        return;
    }
    let qpp = self.qubits_per_page();
    let mut meta_controls = Vec::new();
    let mut intra_controls = Vec::new();
    let mut is_sqi_ctrl = false;
    let mut sqi_index = 0;
    let intra_ctrl_perm = controls.iter().enumerate().fold(0, |acc, (i, &control)| {
        if target >= qpp && control == (qpp - 1) {
            is_sqi_ctrl = true;
            sqi_index = i;
        } else if control < qpp {
            acc | (((control_perm >> i) & 1) << intra_controls.len())
        } else {
            meta_controls.push(control);
            acc
        }
    });
    let meta_ctrl_perm = controls.iter().enumerate().fold(0, |acc, (i, &control)| {
        if control >= qpp {
            acc | (((control_perm >> i) & 1) << meta_controls.len())
        } else {
            acc
        }
    });
    let is_anti = ((control_perm >> sqi_index) & 1) == 0;
    if is_sqi_ctrl && !is_anti {
        meta_controls.push(qpp - 1);
    }
    let sg = |engine: &mut QEngine, l_target: bit_len_int| {
        engine.uc_mtrx(&intra_controls, mtrx, l_target, intra_ctrl_perm);
    };
    if meta_controls.is_empty() {
        self.single_bit_gate(target, sg, is_sqi_ctrl, is_anti);
    } else if target < qpp {
        self.semi_meta_controlled(meta_ctrl_perm, &meta_controls, target, sg);
    } else {
        self.meta_controlled(meta_ctrl_perm, &meta_controls, target, sg, mtrx, is_sqi_ctrl, intra_controls.len());
    }
}

pub fn uniform_parity_rz(&mut self, mask: bit_cap_int, angle: real1_f) {
    self.combine_and_op(|engine| {
        engine.uniform_parity_rz(mask, angle);
    }, &[log2(mask)]);
}

pub fn c_uniform_parity_rz(&mut self, controls: &[bit_len_int], mask: bit_cap_int, angle: real1_f) {
    self.combine_and_op_controlled(|engine| {
        engine.c_uniform_parity_rz(controls, mask, angle);
    }, &[log2(mask)], controls);
}

pub fn x_mask(&mut self, mask: bit_cap_int) {
    let page_mask = self.page_max_q_power() - 1;
    let intra_mask = mask & page_mask;
    let mut inter_mask = mask ^ intra_mask;
    while inter_mask != 0 {
        let v = inter_mask & (inter_mask - 1);
        let bit = log2_ocl(inter_mask ^ v);
        inter_mask = v;
        self.x(bit);
    }
    for engine in &mut self.q_pages {
        engine.x_mask(intra_mask);
    }
}

pub fn phase_parity(&mut self, radians: real1_f, mask: bit_cap_int) {
    let parity_start_size = 4 * std::mem::size_of::<bit_cap_int>();
    let page_mask = self.page_max_q_power() - 1;
    let intra_mask = mask & page_mask;
    let inter_mask = (mask ^ intra_mask) >> self.qubits_per_page();
    let phase_fac = Complex::from_polar(&1.0, &(radians / 2.0));
    let i_phase_fac = Complex::new(1.0, 0.0) / phase_fac;
    for engine in &mut self.q_pages {
        let v = inter_mask & engine.index();
        let mut v = (0..parity_start_size).fold(v, |v, parity_size| {
            v ^ (v >> parity_size)
        });
        v &= 1;
        if intra_mask != 0 {
            engine.phase_parity(if v != 0 { -radians } else { radians }, intra_mask);
        } else if v != 0 {
            engine.phase(phase_fac, phase_fac, 0);
        } else {
            engine.phase(i_phase_fac, i_phase_fac, 0);
        }
    }
}

pub fn force_m(&mut self, qubit: bit_len_int, result: bool, do_force: bool, do_apply: bool) -> bool {
    if self.q_pages.len() == 1 {
        return self.q_pages[0].force_m(qubit, result, do_force, do_apply);
    }
    let one_chance = self.prob(qubit);
    let result = if !do_force {
        if one_chance >= 1.0 {
            true
        } else if one_chance <= 0.0 {
            false
        } else {
            let prob = rand::random::<real1_f>();
            prob <= one_chance
        }
    } else {
        result
    };
    let nrmlzr = if result {
        one_chance
    } else {
        1.0 - one_chance
    };
    if nrmlzr <= 0.0 {
        panic!("QPager::ForceM() forced a measurement result with 0 probability");
    }
    if !do_apply || (1.0 - nrmlzr) <= 0.0 {
        return result;
    }
    let nrm = self.get_nonunitary_phase() / (nrmlzr.sqrt() as real1);
    let qpp = self.qubits_per_page();
    if qubit < qpp {
        let q_power = pow2_ocl(qubit);
        for engine in &mut self.q_pages {
            engine.apply_m(q_power, result, nrm);
        }
    } else {
        let meta_qubit = qubit - qpp;
        let q_power = pow2_ocl(meta_qubit);
        for engine in &mut self.q_pages {
            if (engine.index() & q_power == 0) == !result {
                engine.phase(nrm, nrm, 0);
                engine.update_running_norm();
            } else {
                engine.zero_amplitudes();
            }
        }
    }
    result
}

#[cfg(ENABLE_ALU)]
pub fn incdecsc(&mut self, to_add: bit_cap_int, start: bit_len_int, length: bit_len_int, overflow_index: bit_len_int, carry_index: bit_len_int) {
    self.combine_and_op(|engine| engine.incdecsc(to_add, start, length, overflow_index, carry_index), vec![start + length - 1, overflow_index, carry_index]);
}

#[cfg(ENABLE_ALU)]
pub fn incdecsc(&mut self, to_add: bit_cap_int, start: bit_len_int, length: bit_len_int, carry_index: bit_len_int) {
    self.combine_and_op(|engine| engine.incdecsc(to_add, start, length, carry_index), vec![start + length - 1, carry_index]);
}

#[cfg(all(ENABLE_ALU, ENABLE_BCD))]
pub fn incbcd(&mut self, to_add: bit_cap_int, start: bit_len_int, length: bit_len_int) {
    self.combine_and_op(|engine| engine.incbcd(to_add, start, length), vec![start + length - 1]);
}

#[cfg(all(ENABLE_ALU, ENABLE_BCD))]
pub fn incdec_bcdc(&mut self, to_add: bit_cap_int, start: bit_len_int, length: bit_len_int, carry_index: bit_len_int) {
    self.combine_and_op(|engine| engine.incdec_bcdc(to_add, start, length, carry_index), vec![start + length - 1, carry_index]);
}

pub fn mul(&mut self, to_mul: bit_cap_int, in_out_start: bit_len_int, carry_start: bit_len_int, length: bit_len_int) {
    self.combine_and_op(|engine| engine.mul(to_mul, in_out_start, carry_start, length), vec![in_out_start + length - 1, carry_start + length - 1]);
}

pub fn div(&mut self, to_div: bit_cap_int, in_out_start: bit_len_int, carry_start: bit_len_int, length: bit_len_int) {
    self.combine_and_op(|engine| engine.div(to_div, in_out_start, carry_start, length), vec![in_out_start + length - 1, carry_start + length - 1]);
}

pub fn mul_mod_n_out(&mut self, to_mul: bit_cap_int, mod_n: bit_cap_int, in_start: bit_len_int, out_start: bit_len_int, length: bit_len_int) {
    self.combine_and_op(|engine| engine.mul_mod_n_out(to_mul, mod_n, in_start, out_start, length), vec![in_start + length - 1, out_start + length - 1]);
}

pub fn imul_mod_n_out(&mut self, to_mul: bit_cap_int, mod_n: bit_cap_int, in_start: bit_len_int, out_start: bit_len_int, length: bit_len_int) {
    self.combine_and_op(|engine| engine.imul_mod_n_out(to_mul, mod_n, in_start, out_start, length), vec![in_start + length - 1, out_start + length - 1]);
}

pub fn pow_mod_n_out(&mut self, base: bit_cap_int, mod_n: bit_cap_int, in_start: bit_len_int, out_start: bit_len_int, length: bit_len_int) {
    self.combine_and_op(|engine| engine.pow_mod_n_out(base, mod_n, in_start, out_start, length), vec![in_start + length - 1, out_start + length - 1]);
}

pub fn cmul(&mut self, to_mul: bit_cap_int, in_out_start: bit_len_int, carry_start: bit_len_int, length: bit_len_int, controls: &Vec<bit_len_int>) {
    if controls.is_empty() {
        self.mul(to_mul, in_out_start, carry_start, length);
        return;
    }
    self.combine_and_op_controlled(|engine| engine.cmul(to_mul, in_out_start, carry_start, length, controls), vec![in_out_start + length - 1, carry_start + length - 1], controls);
}

pub fn cdiv(&mut self, to_div: bit_cap_int, in_out_start: bit_len_int, carry_start: bit_len_int, length: bit_len_int, controls: &Vec<bit_len_int>) {
    if controls.is_empty() {
        self.div(to_div, in_out_start, carry_start, length);
        return;
    }
    self.combine_and_op_controlled(|engine| engine.cdiv(to_div, in_out_start, carry_start, length, controls), vec![in_out_start + length - 1, carry_start + length - 1], controls);
}

pub fn cmul_mod_n_out(&mut self, to_mul: bit_cap_int, mod_n: bit_cap_int, in_start: bit_len_int, out_start: bit_len_int, length: bit_len_int, controls: &Vec<bit_len_int>) {
    if controls.is_empty() {
        self.mul_mod_n_out(to_mul, mod_n, in_start, out_start, length);
        return;
    }
    self.combine_and_op_controlled(|engine| engine.cmul_mod_n_out(to_mul, mod_n, in_start, out_start, length, controls), vec![in_start + length - 1, out_start + length - 1], controls);
}

pub fn cimul_mod_n_out(&mut self, to_mul: bit_cap_int, mod_n: bit_cap_int, in_start: bit_len_int, out_start: bit_len_int, length: bit_len_int, controls: &Vec<bit_len_int>) {
    if controls.is_empty() {
        self.imul_mod_n_out(to_mul, mod_n, in_start, out_start, length);
        return;
    }
    self.combine_and_op_controlled(|engine| engine.cimul_mod_n_out(to_mul, mod_n, in_start, out_start, length, controls), vec![in_start + length - 1, out_start + length - 1], controls);
}

pub fn cpow_mod_n_out(&mut self, base: bit_cap_int, mod_n: bit_cap_int, in_start: bit_len_int, out_start: bit_len_int, length: bit_len_int, controls: &Vec<bit_len_int>) {
    if controls.is_empty() {
        self.pow_mod_n_out(base, mod_n, in_start, out_start, length);
        return;
    }
    self.combine_and_op_controlled(|engine| engine.cpow_mod_n_out(base, mod_n, in_start, out_start, length, controls), vec![in_start + length - 1, out_start + length - 1], controls);
}

pub fn indexed_lda(&mut self, index_start: bit_len_int, index_length: bit_len_int, value_start: bit_len_int, value_length: bit_len_int, values: &[u8], ignored: bool) -> bit_cap_int {
    self.combine_engines();
    self.q_pages[0].indexed_lda(index_start, index_length, value_start, value_length, values, ignored)
}

pub fn indexed_adc(&mut self, index_start: bit_len_int, index_length: bit_len_int, value_start: bit_len_int, value_length: bit_len_int, carry_index: bit_len_int, values: &[u8]) -> bit_cap_int {
    self.combine_engines();
    self.q_pages[0].indexed_adc(index_start, index_length, value_start, value_length, carry_index, values)
}

pub fn indexed_sbc(&mut self, index_start: bit_len_int, index_length: bit_len_int, value_start: bit_len_int, value_length: bit_len_int, carry_index: bit_len_int, values: &[u8]) -> bit_cap_int {
    self.combine_engines();
    self.q_pages[0].indexed_sbc(index_start, index_length, value_start, value_length, carry_index, values)
}

pub fn hash(&mut self, start: bit_len_int, length: bit_len_int, values: &[u8]) {
    self.combine_engines();
    self.q_pages[0].hash(start, length, values)
}

pub fn c_phase_flip_if_less(&mut self, greater_perm: bit_cap_int, start: bit_len_int, length: bit_len_int, flag_index: bit_len_int) {
    self.combine_engines();
    self.q_pages[0].c_phase_flip_if_less(greater_perm, start, length, flag_index)
}

pub fn phase_flip_if_less(&mut self, greater_perm: bit_cap_int, start: bit_len_int, length: bit_len_int) {
    self.combine_and_op(|engine| engine.phase_flip_if_less(greater_perm, start, length), vec![start + length - 1])
}

fn meta_swap(&mut self, qubit1: bit_len_int, qubit2: bit_len_int, is_i_phase_fac: bool, is_inverse: bool) {
    let qpp = self.qubits_per_page();
    let mut qubit1 = qubit1 - qpp;
    let mut qubit2 = qubit2 - qpp;
    if qubit2 < qubit1 {
        std::mem::swap(&mut qubit1, &mut qubit2);
    }
    let qubit1_pow = pow2_ocl(qubit1);
    let qubit1_mask = qubit1_pow - 1;
    let qubit2_pow = pow2_ocl(qubit2);
    let qubit2_mask = qubit2_pow - 1;
    let max_lcv = self.q_pages.len() >> 2;
    for i in 0..max_lcv {
        let j = i & qubit1_mask;
        let j_hi = (i ^ j) << 1;
        let j_lo = j_hi & qubit2_mask;
        let j = j | j_lo | ((j_hi ^ j_lo) << 1);
        std::mem::swap(&mut self.q_pages[j | qubit1_pow], &mut self.q_pages[j | qubit2_pow]);
        if !is_i_phase_fac {
            continue;
        }
        if is_inverse {
            self.q_pages[j | qubit1_pow].phase(-I_CMPLX, -I_CMPLX, 0);
            self.q_pages[j | qubit2_pow].phase(-I_CMPLX, -I_CMPLX, 0);
        } else {
            self.q_pages[j | qubit1_pow].phase(I_CMPLX, I_CMPLX, 0);
            self.q_pages[j | qubit2_pow].phase(I_CMPLX, I_CMPLX, 0);
        }
    }
}

pub fn swap(&mut self, qubit1: bit_len_int, qubit2: bit_len_int) {
    if qubit1 == qubit2 {
        return;
    }
    let is_qubit1_meta = qubit1 >= base_qubits_per_page;
    let is_qubit2_meta = qubit2 >= base_qubits_per_page;
    if is_qubit1_meta && is_qubit2_meta {
        self.separate_engines();
        self.meta_swap(qubit1, qubit2, false, false);
        return;
    }
    if is_qubit1_meta || is_qubit2_meta {
        self.separate_engines();
        self.swap(qubit1, qubit2);
        return;
    }
    for i in 0..self.q_pages.len() {
        self.q_pages[i].swap(qubit1, qubit2);
    }
}

fn either_i_swap(&mut self, qubit1: bit_len_int, qubit2: bit_len_int, is_inverse: bool) {
    if qubit1 == qubit2 {
        return;
    }
    let is_qubit1_meta = qubit1 >= base_qubits_per_page;
    let is_qubit2_meta = qubit2 >= base_qubits_per_page;
    if is_qubit1_meta && is_qubit2_meta {
        self.separate_engines();
        self.meta_swap(qubit1, qubit2, true, is_inverse);
        return;
    }
    if is_qubit1_meta || is_qubit2_meta {
        self.separate_engines();
        self.swap(qubit1, qubit2);
    }
    if is_qubit1_meta {
        let qubit1 = qubit1 - base_qubits_per_page;
        let phase_fac = if is_inverse { -I_CMPLX } else { I_CMPLX };
        for i in 0..self.q_pages.len() {
            if (i >> qubit1) & 1 != 0 {
                self.q_pages[i].phase(phase_fac, ONE_CMPLX, qubit2);
            } else {
                self.q_pages[i].phase(ONE_CMPLX, phase_fac, qubit2);
            }
        }
        return;
    }
    if is_qubit2_meta {
        let qubit2 = qubit2 - base_qubits_per_page;
        let phase_fac = if is_inverse { -I_CMPLX } else { I_CMPLX };
        for i in 0..self.q_pages.len() {
            if (i >> qubit2) & 1 != 0 {
                self.q_pages[i].phase(phase_fac, ONE_CMPLX, qubit1);
            } else {
                self.q_pages[i].phase(ONE_CMPLX, phase_fac, qubit1);
            }
        }
        return;
    }
    if is_inverse {
        for i in 0..self.q_pages.len() {
            self.q_pages[i].iiswap(qubit1, qubit2);
        }
    } else {
        for i in 0..self.q_pages.len() {
            self.q_pages[i].iswap(qubit1, qubit2);
        }
    }
}

pub fn f_sim(&mut self, theta: real1_f, phi: real1_f, qubit1: bit_len_int, qubit2: bit_len_int) {
    if qubit1 == qubit2 {
        return;
    }
    let sin_theta = theta.sin();
    if sin_theta * sin_theta <= FP_NORM_EPSILON {
        self.mc_phase(vec![qubit1], ONE_CMPLX, (phi * I_CMPLX).exp(), qubit2);
        return;
    }
    let exp_i_phi = (phi * I_CMPLX).exp();
    let sin_theta_diff_neg = 1.0 + sin_theta;
    if sin_theta_diff_neg * sin_theta_diff_neg <= FP_NORM_EPSILON {
        self.iswap(qubit1, qubit2);
        self.mc_phase(vec![qubit1], ONE_CMPLX, exp_i_phi, qubit2);
        return;
    }
    let sin_theta_diff_pos = 1.0 - sin_theta;
    if sin_theta_diff_pos * sin_theta_diff_pos <= FP_NORM_EPSILON {
        self.iiswap(qubit1, qubit2);
        self.mc_phase(vec![qubit1], ONE_CMPLX, exp_i_phi, qubit2);
        return;
    }
    self.combine_and_op(|engine| engine.f_sim(theta, phi, qubit1, qubit2), vec![qubit1, qubit2]);
}

pub fn prob(&mut self, qubit: bit_len_int) -> real1_f {
    if self.q_pages.len() == 1 {
        return self.q_pages[0].prob(qubit);
    }
    let qpp = self.qubits_per_page();
    let mut one_chance = 0.0;
    if qubit < qpp {
        for i in 0..self.q_pages.len() {
            one_chance += self.q_pages[i].prob(qubit);
        }
    } else {
        let q_power = pow2_ocl(qubit - qpp);
        let q_mask = q_power - 1;
        let f_size = self.q_pages.len() >> 1;
        for i in 0..f_size {
            let j = i & q_mask;
            let j_hi = (i ^ j) << 1;
            let j_lo = j_hi & q_mask;
            let j = j | j_lo | ((j_hi ^ j_lo) << 1);
            self.q_pages[j | q_power].update_running_norm();
            one_chance += self.q_pages[j | q_power].get_running_norm();
        }
    }
    clamp_prob(one_chance)
}

pub fn prob_mask(&mut self, mask: bit_cap_int, permutation: bit_cap_int) -> real1_f {
    self.combine_engines(log2(mask) + 1);
    let mut mask_chance = 0.0;
    for i in 0..self.q_pages.len() {
        mask_chance += self.q_pages[i].prob_mask(mask, permutation);
    }
    clamp_prob(mask_chance)
}

pub fn expectation_bits_all(&mut self, bits: &[bit_len_int], offset: bit_cap_int) -> real1_f {
    if bits.len() != self.qubit_count {
        return self.expectation_bits_all(bits, offset);
    }
    for i in 0..bits.len() {
        if bits[i] != i as bit_len_int {
            return self.expectation_bits_all(bits, offset);
        }
    }
    let page_power = self.page_max_q_power() as bit_cap_int;
    let mut expectation = 0.0;
    let mut page_perm = 0;
    for i in 0..self.q_pages.len() {
        expectation += self.q_pages[i].expectation_bits_all(bits, page_perm + offset);
        page_perm += page_power;
    }
    expectation
}

pub fn update_running_norm(&mut self, norm_thresh: real1_f) {
    for i in 0..self.q_pages.len() {
        self.q_pages[i].update_running_norm(norm_thresh);
    }
}

pub fn normalize_state(&mut self, nrm: real1_f, norm_thresh: real1_f, phase_arg: real1_f) {
    let nmlzr = if nrm == REAL1_DEFAULT_ARG {
        let mut nmlzr = 0.0;
        for i in 0..self.q_pages.len() {
            nmlzr += self.q_pages[i].get_running_norm();
        }
        nmlzr
    } else {
        nrm
    };
    for i in 0..self.q_pages.len() {
        self.q_pages[i].normalize_state(nmlzr, norm_thresh, phase_arg);
    }
}

pub fn clone_q_interface(&self) -> Box<dyn QInterface> {
    let mut clone = QPager::new(
        self.engines,
        self.qubit_count,
        ZERO_BCI,
        self.rand_generator,
        ONE_CMPLX,
        self.do_normalize,
        self.rand_global_phase,
        false,
        0,
        self.hardware_rand_generator.is_some(),
        self.is_sparse,
        self.amplitude_floor as real1_f,
        self.device_ids.clone(),
        self.threshold_qubits_per_page,
    );
    for i in 0..self.q_pages.len() {
        clone.q_pages[i] = self.q_pages[i].clone_q_interface();
    }
    Box::new(clone)
}

pub fn clone_empty_q_engine(&self) -> Box<dyn QEngine> {
    let mut clone = QPager::new(
        self.engines,
        self.qubit_count,
        ZERO_BCI,
        self.rand_generator,
        ONE_CMPLX,
        self.do_normalize,
        self.rand_global_phase,
        false,
        0,
        self.hardware_rand_generator.is_some(),
        self.is_sparse,
        self.amplitude_floor as real1_f,
        self.device_ids.clone(),
        self.threshold_qubits_per_page,
    );
    for i in 0..self.q_pages.len() {
        clone.q_pages[i] = self.q_pages[i].clone_empty_q_engine();
    }
    Box::new(clone)
}

pub fn sum_sqr_diff(&mut self, to_compare: &mut dyn QPager) -> real1_f {
    if self as *mut QPager == to_compare as *mut QPager {
        return ZERO_R1_F;
    }
    if self.qubit_count != to_compare.qubit_count {
        return ONE_R1_F;
    }
    self.separate_engines(to_compare.qubits_per_page());
    to_compare.separate_engines(self.qubits_per_page());
    self.combine_engines(to_compare.qubits_per_page());
    to_compare.combine_engines(self.qubits_per_page());
    let mut to_ret = ZERO_R1_F;
    for i in 0..self.q_pages.len() {
        to_ret += self.q_pages[i].sum_sqr_diff(to_compare.q_pages[i].as_mut());
    }
    to_ret
}

}
