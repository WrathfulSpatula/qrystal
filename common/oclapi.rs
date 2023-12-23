pub mod config {
    // config.h contents here
}

pub mod qrack {
    pub enum OCLAPI {
        OCL_API_UNKNOWN = 0,
        OCL_API_APPLY2X2,
        OCL_API_APPLY2X2_SINGLE,
        OCL_API_APPLY2X2_NORM_SINGLE,
        OCL_API_APPLY2X2_DOUBLE,
        OCL_API_APPLY2X2_WIDE,
        OCL_API_APPLY2X2_SINGLE_WIDE,
        OCL_API_APPLY2X2_NORM_SINGLE_WIDE,
        OCL_API_APPLY2X2_DOUBLE_WIDE,
        OCL_API_PHASE_SINGLE,
        OCL_API_PHASE_SINGLE_WIDE,
        OCL_API_INVERT_SINGLE,
        OCL_API_INVERT_SINGLE_WIDE,
        OCL_API_UNIFORMLYCONTROLLED,
        OCL_API_UNIFORMPARITYRZ,
        OCL_API_UNIFORMPARITYRZ_NORM,
        OCL_API_CUNIFORMPARITYRZ,
        OCL_API_COMPOSE,
        OCL_API_COMPOSE_WIDE,
        OCL_API_COMPOSE_MID,
        OCL_API_DECOMPOSEPROB,
        OCL_API_DECOMPOSEAMP,
        OCL_API_DISPOSEPROB,
        OCL_API_DISPOSE,
        OCL_API_PROB,
        OCL_API_CPROB,
        OCL_API_PROBREG,
        OCL_API_PROBREGALL,
        OCL_API_PROBMASK,
        OCL_API_PROBMASKALL,
        OCL_API_PROBPARITY,
        OCL_API_FORCEMPARITY,
        OCL_API_EXPPERM,
        OCL_API_X_SINGLE,
        OCL_API_X_SINGLE_WIDE,
        OCL_API_X_MASK,
        OCL_API_Z_SINGLE,
        OCL_API_Z_SINGLE_WIDE,
        OCL_API_PHASE_PARITY,
        OCL_API_ROL,
        OCL_API_APPROXCOMPARE,
        OCL_API_NORMALIZE,
        OCL_API_NORMALIZE_WIDE,
        OCL_API_UPDATENORM,
        OCL_API_APPLYM,
        OCL_API_APPLYMREG,
        OCL_API_CLEARBUFFER,
        OCL_API_SHUFFLEBUFFERS,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_INC,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_CINC,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_INCDECC,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_INCS,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_INCDECSC_1,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_INCDECSC_2,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_MUL,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_DIV,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_MULMODN_OUT,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_IMULMODN_OUT,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_POWMODN_OUT,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_CMUL,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_CDIV,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_CMULMODN_OUT,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_CIMULMODN_OUT,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_CPOWMODN_OUT,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_FULLADD,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_IFULLADD,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_INDEXEDLDA,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_INDEXEDADC,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_INDEXEDSBC,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_HASH,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_CPHASEFLIPIFLESS,
        #[cfg(feature = "ENABLE_ALU")]
        OCL_API_PHASEFLIPIFLESS,
        #[cfg(all(feature = "ENABLE_ALU", feature = "ENABLE_BCD"))]
        OCL_API_INCBCD,
        #[cfg(all(feature = "ENABLE_ALU", feature = "ENABLE_BCD"))]
        OCL_API_INCDECBCDC,
    }
}


