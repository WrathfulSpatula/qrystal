use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::sync::Weak;

struct clBufferWrapper {
    // implementation details
}

struct OCLAPI {
    // implementation details
}

struct QEngineOCL {
    didInit: bool,
    usingHostRam: bool,
    unlockHostMem: bool,
    callbackError: cl_int,
    nrmGroupCount: usize,
    nrmGroupSize: usize,
    totalOclAllocSize: usize,
    deviceID: i64,
    lockSyncFlags: cl_map_flags,
    permutationAmp: complex,
    stateVec: Arc<RwLock<Option<Vec<complex>>>>,
    queue_mutex: Mutex<()>,
    queue: clCommandQueueWrapper,
    context: clContextWrapper,
    stateBuffer: Option<BufferPtr>,
    nrmBuffer: Option<BufferPtr>,
    device_context: Option<DeviceContextPtr>,
    wait_refs: Vec<EventVecPtr>,
    wait_queue_items: Vec<QueueItem>,
    poolItems: Vec<PoolItemPtr>,
    nrmArray: Option<Box<[real1]>>,
}

struct QueueItem {
    api_call: OCLAPI,
    workItemCount: usize,
    localGroupSize: usize,
    deallocSize: usize,
    buffers: Vec<BufferPtr>,
    localBuffSize: usize,
    isSetDoNorm: bool,
    isSetRunningNorm: bool,
    doNorm: bool,
    runningNorm: real1,
}

struct PoolItem {
    cmplxBuffer: BufferPtr,
    realBuffer: BufferPtr,
    ulongBuffer: BufferPtr,
    probArray: Option<Arc<RwLock<Option<Vec<real1>>>>>,
    angleArray: Option<Arc<RwLock<Option<Vec<real1>>>>>,
}

impl QEngineOCL {
    fn checkCallbackError(&self) {
        if self.callbackError == CL_SUCCESS {
            return;
        }
        self.wait_queue_items.clear();
        self.wait_refs.clear();
        panic!("Failed to enqueue kernel, error code: {}", self.callbackError);
    }

    fn tryOcl<F>(&self, message: &str, oclCall: F)
    where
        F: Fn() -> cl_int,
    {
        self.checkCallbackError();
        if oclCall() == CL_SUCCESS {
            return;
        }

        clFinish();
        if oclCall() == CL_SUCCESS {
            return;
        }

        clFinish(true);
        let error = oclCall();
        if error == CL_SUCCESS {
            return;
        }
        self.wait_queue_items.clear();
        self.wait_refs.clear();

        panic!("{} , error code: {}", message, error);
    }

    fn addAlloc(&mut self, size: usize) {
        let currentAlloc = OCLEngine::Instance().AddToActiveAllocSize(self.deviceID, size);
        if let Some(device_context) = &self.device_context {
            if currentAlloc > device_context.GetGlobalAllocLimit() {
                OCLEngine::Instance().SubtractFromActiveAllocSize(self.deviceID, size);
                panic!("VRAM limits exceeded in QEngineOCL::AddAlloc()");
            }
        }
        self.totalOclAllocSize += size;
    }

    fn subtractAlloc(&mut self, size: usize) {
        OCLEngine::Instance().SubtractFromActiveAllocSize(self.deviceID, size);
        self.totalOclAllocSize -= size;
    }

    fn makeBuffer(&self, flags: cl_mem_flags, size: usize, host_ptr: *mut c_void) -> BufferPtr {
        self.checkCallbackError();
        let error;
        let toRet = std::make_shared<clBufferWrapper>(self.context, flags, size, host_ptr, &error);
        if error == CL_SUCCESS {
            return toRet;
        }

        clFinish();
        let toRet = std::make_shared<clBufferWrapper>(self.context, flags, size, host_ptr, &error);
        if error == CL_SUCCESS {
            return toRet;
        }

        clFinish(true);
        let toRet = std::make_shared<clBufferWrapper>(self.context, flags, size, host_ptr, &error);
        if error != CL_SUCCESS {
            if error == CL_MEM_OBJECT_ALLOCATION_FAILURE {
                panic!("CL_MEM_OBJECT_ALLOCATION_FAILURE in QEngineOCL::MakeBuffer()");
            }
            if error == CL_OUT_OF_HOST_MEMORY {
                panic!("CL_OUT_OF_HOST_MEMORY in QEngineOCL::MakeBuffer()");
            }
            if error == CL_INVALID_BUFFER_SIZE {
                panic!("CL_INVALID_BUFFER_SIZE in QEngineOCL::MakeBuffer()");
            }
            panic!("OpenCL error code on buffer allocation attempt: {}", error);
        }
        toRet
    }

    fn getExpectation(&self, valueStart: bitLenInt, valueLength: bitLenInt) -> real1 {
        // implementation details
    }

    fn allocStateVec(&self, elemCount: bitCapIntOcl, doForceAlloc: bool) -> Arc<RwLock<Option<Vec<complex>>>> {
        // implementation details
    }

    fn freeStateVec(&mut self) {
        self.stateVec = Arc::new(RwLock::new(None));
    }

    fn resetStateBuffer(&mut self, nStateBuffer: BufferPtr) {
        self.stateBuffer = Some(nStateBuffer);
    }

    fn makeStateVecBuffer(&self, nStateVec: Arc<RwLock<Option<Vec<complex>>>>) -> BufferPtr {
        // implementation details
    }

    fn reinitBuffer(&mut self) {
        // implementation details
    }

    fn compose(&mut self, apiCall: OCLAPI, bciArgs: &[bitCapIntOcl; BCI_ARG_LEN], toCopy: QEngineOCLPtr) {
        // implementation details
    }

    fn initOCL(&mut self, devID: i64) {
        // implementation details
    }

    fn getFreePoolItem(&mut self) -> PoolItemPtr {
        // implementation details
    }

    fn parSum(&self, toSum: &[real1], maxI: bitCapIntOcl) -> real1 {
        // implementation details
    }

    fn lockSync(&self, flags: cl_map_flags) {
        // implementation details
    }

    fn unlockSync(&self) {
        // implementation details
    }

    fn clFinish(&self, doHard: bool) {
        // implementation details
    }

    fn clDump(&self) {
        // implementation details
    }

    fn fixWorkItemCount(&self, maxI: usize, wic: usize) -> usize {
        // implementation details
    }

    fn fixGroupSize(&self, wic: usize, gs: usize) -> usize {
        // implementation details
    }

    fn decomposeDispose(&mut self, start: bitLenInt, length: bitLenInt, dest: QEngineOCLPtr) {
        // implementation details
    }

    fn apply2x2(
        &mut self,
        offset1: bitCapIntOcl,
        offset2: bitCapIntOcl,
        mtrx: &[complex],
        bitCount: bitLenInt,
        qPowersSorted: &[bitCapIntOcl],
        doCalcNorm: bool,
        special: SPECIAL_2X2,
        norm_thresh: real1,
    ) {
        // implementation details
    }

    fn bitMask(&mut self, mask: bitCapIntOcl, api_call: OCLAPI, phase: real1) {
        // implementation details
    }

    fn applyMx(&mut self, api_call: OCLAPI, bciArgs: &[bitCapIntOcl; BCI_ARG_LEN], nrm: complex) {
        // implementation details
    }

    fn probx(&mut self, api_call: OCLAPI, bciArgs: &[bitCapIntOcl; BCI_ARG_LEN]) -> real1 {
        // implementation details
    }

    fn arithmeticCall(
        &mut self,
        api_call: OCLAPI,
        bciArgs: &[bitCapIntOcl; BCI_ARG_LEN],
        values: Option<&[u8]>,
        valuesLength: bitCapIntOcl,
    ) {
        // implementation details
    }

    fn carithmeticCall(
        &mut self,
        api_call: OCLAPI,
        bciArgs: &[bitCapIntOcl; BCI_ARG_LEN],
        controlPowers: &[bitCapIntOcl],
        controlLen: bitLenInt,
        values: Option<&[u8]>,
        valuesLength: bitCapIntOcl,
    ) {
        // implementation details
    }

    fn rox(&mut self, api_call: OCLAPI, shift: bitLenInt, start: bitLenInt, length: bitLenInt) {
        // implementation details
    }

    fn clearBuffer(&mut self, buff: BufferPtr, offset: bitCapIntOcl, size: bitCapIntOcl) {
        // implementation details
    }
}


