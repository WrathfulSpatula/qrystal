// TODO: Determine if this is even necessary in Rust!
use std::alloc::AllocError;

pub mod qrack {
    pub enum SPECIAL_2X2 {
        NONE = 0,
        PAULIX,
        PAULIZ,
        INVERT,
        PHASE,
    }

    pub struct BadAlloc {
        message: String,
    }

    impl BadAlloc {
        pub fn new(message: String) -> Self {
            BadAlloc { message }
        }
    }

    impl std::fmt::Debug for BadAlloc {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.message)
        }
    }

    impl std::fmt::Display for BadAlloc {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.message)
        }
    }

    impl std::error::Error for BadAlloc {}
}


