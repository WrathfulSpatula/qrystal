#[cfg(feature = "opencl")]
use common::oclengine::OclEngine;
#[cfg(feature = "cuda")]
use common::cudaengine::CudaEngine;
use qengine_opencl::QEngineOpenCL;
use qengine_cuda::QEngineCuda;
use qunit::QUnit;
use std::cmp::Ordering;

pub struct QEngineInfo {
    unit: QInterfacePtr,
    device_index: usize,
}

impl QEngineInfo {
    pub fn new(unit: QInterfacePtr, device_index: usize) -> Self {
        Self {
            unit,
            device_index,
        }
    }
}

impl PartialOrd for QEngineInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let v = self.unit.get_max_q_power().cmp(&other.unit.get_max_q_power());
        if v == Ordering::Equal {
            Some(other.device_index.cmp(&self.device_index))
        } else {
            Some(v)
        }
    }
}

impl PartialEq for QEngineInfo {
    fn eq(&self, other: &Self) -> bool {
        self.unit.get_max_q_power() == other.unit.get_max_q_power()
            && self.device_index == other.device_index
    }
}

impl Eq for QEngineInfo {}

pub struct DeviceInfo {
    id: usize,
    max_size: bitCapIntOcl,
}

impl PartialOrd for DeviceInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.max_size.partial_cmp(&other.max_size)
    }
}

impl PartialEq for DeviceInfo {
    fn eq(&self, other: &Self) -> bool {
        self.max_size == other.max_size
    }
}

impl Eq for DeviceInfo {}

pub struct QUnitMulti {
    is_redistributing: bool,
    is_q_engine_ocl: bool,
    default_device_id: usize,
    device_list: Vec<DeviceInfo>,
    device_qb_list: Vec<bitLenInt>,
}

impl QUnitMulti {
    pub fn new(
        eng: Vec<QInterfaceEngine>,
        q_bit_count: bitLenInt,
        init_state: bitCapInt,
        rgp: qrack_rand_gen_ptr,
        phase_fac: complex,
        do_norm: bool,
        random_global_phase: bool,
        use_host_mem: bool,
        device_id: i64,
        use_hardware_rng: bool,
        use_sparse_state_vec: bool,
        norm_thresh: real1_f,
        dev_list: Vec<i64>,
        qubit_threshold: bitLenInt,
        separation_thresh: real1_f,
    ) -> Self {
        Self {
            is_redistributing: false,
            is_q_engine_ocl: false,
            default_device_id: 0,
            device_list: Vec::new(),
            device_qb_list: Vec::new(),
        }
    }

    pub fn clone(&self) -> QInterfacePtr {
        for i in 0..self.qubit_count {
            self.revert_basis2_qb(i);
        }
        let copy_ptr = QUnitMulti::new(
            self.engines.clone(),
            self.qubit_count,
            ZERO_BCI,
            self.rand_generator,
            self.phase_factor,
            self.do_normalize,
            self.rand_global_phase,
            self.use_host_ram,
            self.default_device_id,
            self.use_rdrand,
            self.is_sparse,
            self.amplitude_floor as real1_f,
            self.device_ids.clone(),
            self.threshold_qubits,
            self.separability_threshold,
        );
        copy_ptr.set_reactive_separate(self.is_reactive_separate);
        self.clone_body(copy_ptr)
    }

    fn get_q_infos(&self) -> Vec<QEngineInfo> {
        unimplemented!()
    }

    fn separate_bit(&self, value: bool, qubit: bitLenInt) -> bool {
        let to_ret = self.separate_bit(value, qubit);
        self.redistribute_q_engines();
        to_ret
    }

    fn detach(&self, start: bitLenInt, length: bitLenInt, dest: QUnitPtr) {
        self.detach(start, length, dest);
    }

    fn detach_multi(&self, start: bitLenInt, length: bitLenInt, dest: QUnitMultiPtr) {
        if length == 0 {
            return;
        }
        self.detach(start, length, dest);
        self.redistribute_q_engines();
    }

    fn entangle_in_current_basis(
        &self,
        first: std::slice::Iter<bitLenInt>,
        last: std::slice::Iter<bitLenInt>,
    ) -> QInterfacePtr {
        let to_ret = self.entangle_in_current_basis(first, last);
        self.redistribute_q_engines();
        to_ret
    }

    fn redistribute_q_engines(&self) {
        unimplemented!()
    }
}


