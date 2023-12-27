use std::vec::Vec;
use std::rc::Rc;

enum QInterfaceEngine {
    QINTERFACE_CPU,
    QINTERFACE_STABILIZER,
    QINTERFACE_QUNIT_CLIFFORD,
    QINTERFACE_BDT,
    QINTERFACE_BDT_HYBRID,
    QINTERFACE_QPAGER,
    QINTERFACE_STABILIZER_HYBRID,
    QINTERFACE_QUNIT,
    QINTERFACE_TENSOR_NETWORK,
    QINTERFACE_OPENCL,
    QINTERFACE_CUDA,
    QINTERFACE_HYBRID,
    QINTERFACE_QUNIT_MULTI,
}

struct QInterfacePtr {
    // Define your struct here
}

impl QInterfacePtr {
    // Define your methods here
}

fn create_quantum_interface<Ts>(engine1: QInterfaceEngine, engine2: QInterfaceEngine, engine3: QInterfaceEngine, args: Ts) -> QInterfacePtr {
    let engine = engine1;
    let engines: Vec<QInterfaceEngine> = vec![engine2, engine3];
    match engine {
        QInterfaceEngine::QINTERFACE_CPU => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_STABILIZER => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_QUNIT_CLIFFORD => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_BDT => QInterfacePtr::new(engines, args),
        QInterfaceEngine::QINTERFACE_BDT_HYBRID => QInterfacePtr::new(engines, args),
        QInterfaceEngine::QINTERFACE_QPAGER => QInterfacePtr::new(engines, args),
        QInterfaceEngine::QINTERFACE_STABILIZER_HYBRID => QInterfacePtr::new(engines, args),
        QInterfaceEngine::QINTERFACE_QUNIT => QInterfacePtr::new(engines, args),
        QInterfaceEngine::QINTERFACE_TENSOR_NETWORK => QInterfacePtr::new(engines, args),
        QInterfaceEngine::QINTERFACE_OPENCL => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_CUDA => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_HYBRID => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_QUNIT_MULTI => QInterfacePtr::new(engines, args),
    }
}

fn create_quantum_interface<Ts>(engine1: QInterfaceEngine, engine2: QInterfaceEngine, args: Ts) -> QInterfacePtr {
    let engine = engine1;
    let engines: Vec<QInterfaceEngine> = vec![engine2];
    match engine {
        QInterfaceEngine::QINTERFACE_CPU => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_STABILIZER => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_QUNIT_CLIFFORD => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_BDT => QInterfacePtr::new(engines, args),
        QInterfaceEngine::QINTERFACE_BDT_HYBRID => QInterfacePtr::new(engines, args),
        QInterfaceEngine::QINTERFACE_QPAGER => QInterfacePtr::new(engines, args),
        QInterfaceEngine::QINTERFACE_STABILIZER_HYBRID => QInterfacePtr::new(engines, args),
        QInterfaceEngine::QINTERFACE_QUNIT => QInterfacePtr::new(engines, args),
        QInterfaceEngine::QINTERFACE_TENSOR_NETWORK => QInterfacePtr::new(engines, args),
        QInterfaceEngine::QINTERFACE_OPENCL => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_CUDA => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_HYBRID => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_QUNIT_MULTI => QInterfacePtr::new(engines, args),
    }
}

fn create_quantum_interface<Ts>(engine: QInterfaceEngine, args: Ts) -> QInterfacePtr {
    match engine {
        QInterfaceEngine::QINTERFACE_CPU => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_STABILIZER => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_QUNIT_CLIFFORD => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_BDT => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_BDT_HYBRID => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_QPAGER => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_STABILIZER_HYBRID => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_QUNIT => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_TENSOR_NETWORK => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_OPENCL => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_CUDA => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_HYBRID => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_QUNIT_MULTI => QInterfacePtr::new(args),
    }
}

fn create_quantum_interface<Ts>(engines: Vec<QInterfaceEngine>, args: Ts) -> QInterfacePtr {
    let engine = engines[0];
    let mut engines = engines;
    engines.remove(0);
    match engine {
        QInterfaceEngine::QINTERFACE_CPU => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_STABILIZER => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_QUNIT_CLIFFORD => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_BDT => {
            if engines.len() > 0 {
                QInterfacePtr::new(engines, args)
            } else {
                QInterfacePtr::new(args)
            }
        },
        QInterfaceEngine::QINTERFACE_BDT_HYBRID => {
            if engines.len() > 0 {
                QInterfacePtr::new(engines, args)
            } else {
                QInterfacePtr::new(args)
            }
        },
        QInterfaceEngine::QINTERFACE_QPAGER => {
            if engines.len() > 0 {
                QInterfacePtr::new(engines, args)
            } else {
                QInterfacePtr::new(args)
            }
        },
        QInterfaceEngine::QINTERFACE_STABILIZER_HYBRID => {
            if engines.len() > 0 {
                QInterfacePtr::new(engines, args)
            } else {
                QInterfacePtr::new(args)
            }
        },
        QInterfaceEngine::QINTERFACE_QUNIT => {
            if engines.len() > 0 {
                QInterfacePtr::new(engines, args)
            } else {
                QInterfacePtr::new(args)
            }
        },
        QInterfaceEngine::QINTERFACE_TENSOR_NETWORK => {
            if engines.len() > 0 {
                QInterfacePtr::new(engines, args)
            } else {
                QInterfacePtr::new(args)
            }
        },
        QInterfaceEngine::QINTERFACE_OPENCL => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_CUDA => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_HYBRID => QInterfacePtr::new(args),
        QInterfaceEngine::QINTERFACE_QUNIT_MULTI => {
            if engines.len() > 0 {
                QInterfacePtr::new(engines, args)
            } else {
                QInterfacePtr::new(args)
            }
        },
    }
}

fn create_arranged_layers<Ts>(md: bool, sd: bool, sh: bool, bdt: bool, pg: bool, tn: bool, hy: bool, oc: bool, args: Ts) -> QInterfacePtr {
    let is_ocl_multi = oc && md;
    let mut simulator_type: Vec<QInterfaceEngine> = Vec::new();
    if !hy {
        #[cfg(ENABLE_OPENCL)]
        {
            simulator_type.push(if oc { QInterfaceEngine::QINTERFACE_OPENCL } else { QInterfaceEngine::QINTERFACE_CPU });
        }
        #[cfg(ENABLE_CUDA)]
        {
            simulator_type.push(if oc { QInterfaceEngine::QINTERFACE_CUDA } else { QInterfaceEngine::QINTERFACE_CPU });
        }
    }
    if pg && !simulator_type.is_empty() {
        simulator_type.push(QInterfaceEngine::QINTERFACE_QPAGER);
    }
    #[cfg(ENABLE_QBDT)]
    {
        if bdt {
            simulator_type.push(QInterfaceEngine::QINTERFACE_BDT_HYBRID);
        }
    }
    if sh && (!sd || !simulator_type.is_empty()) {
        simulator_type.push(QInterfaceEngine::QINTERFACE_STABILIZER_HYBRID);
    }
    if sd {
        simulator_type.push(if is_ocl_multi { QInterfaceEngine::QINTERFACE_QUNIT_MULTI } else { QInterfaceEngine::QINTERFACE_QUNIT });
    }
    if tn {
        simulator_type.push(QInterfaceEngine::QINTERFACE_TENSOR_NETWORK);
    }
    simulator_type.reverse();
    if simulator_type.is_empty() {
        #[cfg(ENABLE_OPENCL)]
        {
            simulator_type.push(if hy && oc { QInterfaceEngine::QINTERFACE_HYBRID } else { QInterfaceEngine::QINTERFACE_OPENCL });
        }
        #[cfg(not(ENABLE_OPENCL))]
        {
            simulator_type.push(if hy && oc { QInterfaceEngine::QINTERFACE_HYBRID } else { QInterfaceEngine::QINTERFACE_CUDA });
        }
    }
    create_quantum_interface(simulator_type, args)
}

#[cfg(ENABLE_OPENCL)]
const DEVICE_COUNT: usize = OCLEngine::Instance().GetDeviceCount();
#[cfg(ENABLE_CUDA)]
const DEVICE_COUNT: usize = CUDAEngine::Instance().GetDeviceCount();


