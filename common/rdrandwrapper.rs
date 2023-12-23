use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::env;
use std::error::Error;
use std::thread;
use std::time::Duration;

fn read_directory_file_names(path: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let mut result = Vec::new();
    let entries = std::fs::read_dir(path)?;
    for entry in entries {
        let entry = entry?;
        let file_name = entry.file_name().into_string()?;
        if file_name != "." && file_name != ".." {
            result.push(format!("{}/{}", path, file_name));
        }
    }
    result.sort();
    Ok(result)
}

fn get_default_random_number_file_path() -> String {
    if let Some(rng_path) = env::var("QRACK_RNG_PATH").ok() {
        let mut to_ret = rng_path;
        if !to_ret.ends_with('/') && !to_ret.ends_with('\\') {
            #[cfg(windows)]
            {
                to_ret += "\\";
            }
            #[cfg(not(windows))]
            {
                to_ret += "/";
            }
        }
        return to_ret;
    }
    #[cfg(windows)]
    {
        return format!(
            "{}{}",
            env::var("HOMEDRIVE").unwrap_or_default(),
            env::var("HOMEPATH").unwrap_or_default()
        ) + "\\.qrack\\rng\\";
    }
    #[cfg(not(windows))]
    {
        return format!(
            "{}/.qrack/rng/",
            env::var("HOME").unwrap_or_default()
        );
    }
}

struct RandFile {
    file_offset: usize,
    data_file: Option<File>,
}

impl RandFile {
    fn new() -> Result<Self, Box<dyn Error>> {
        let mut rand_file = RandFile {
            file_offset: 0,
            data_file: None,
        };
        rand_file.read_next_rand_data_file()?;
        Ok(rand_file)
    }

    fn read_next_rand_data_file(&mut self) -> Result<(), Box<dyn Error>> {
        if let Some(data_file) = &self.data_file {
            data_file.close()?;
        }
        let path = get_default_random_number_file_path();
        let file_names = read_directory_file_names(&path)?;
        if file_names.len() <= self.file_offset {
            return Err("Out of RNG files!".into());
        }
        while let Err(_) = File::open(&file_names[self.file_offset]) {
            thread::sleep(Duration::from_millis(10));
        }
        self.file_offset += 1;
        Ok(())
    }

    fn next_raw(&mut self) -> Result<u32, Box<dyn Error>> {
        let mut buffer = [0u8; 4];
        let mut f_size = 0;
        let mut v = 0;
        while f_size < 1 {
            f_size = self.data_file.as_ref().unwrap().read(&mut buffer)?;
            if f_size < 1 {
                self.read_next_rand_data_file()?;
            }
        }
        v = u32::from_le_bytes(buffer);
        Ok(v)
    }
}

struct RdRandom {
    rand_file: Option<RandFile>,
}

impl RdRandom {
    fn new() -> Result<Self, Box<dyn Error>> {
        let rand_file = if cfg!(ENABLE_RNDFILE) && !cfg!(ENABLE_DEVRAND) {
            Some(RandFile::new()?)
        } else {
            None
        };
        Ok(RdRandom { rand_file })
    }

    fn get_rd_rand(&mut self, pv: &mut u32) -> bool {
        #[cfg(all(not(ENABLE_RDRAND), not(ENABLE_DEVRAND)))]
        {
            return false;
        }
        #[cfg(all(ENABLE_RDRAND, not(ENABLE_DEVRAND)))]
        {
            const MAX_RDRAND_TRIES: usize = 10;
            for _ in 0..MAX_RDRAND_TRIES {
                if let Ok(v) = rdrand32_step() {
                    *pv = v;
                    return true;
                }
            }
        }
        #[cfg(all(ENABLE_RDRAND, ENABLE_DEVRAND))]
        {
            const MAX_RDRAND_TRIES: usize = 10;
            for _ in 0..MAX_RDRAND_TRIES {
                if let Ok(v) = rdrand32_step() {
                    *pv = v;
                    return true;
                }
            }
            if let Ok(v) = getrandom::getrandom(pv) {
                return true;
            }
        }
        false
    }

    fn supports_rdrand(&mut self) -> bool {
        #[cfg(not(ENABLE_RDRAND))]
        {
            return false;
        }
        #[cfg(ENABLE_RDRAND)]
        {
            #[cfg(target_arch = "x86")]
            {
                let flag_rdrand = 1 << 30;
                let mut ex = [0u32; 4];
                unsafe {
                    cpuid::cpuid_count(1, 0, &mut ex);
                }
                return (ex[2] & flag_rdrand) == flag_rdrand;
            }
            #[cfg(target_arch = "x86_64")]
            {
                let flag_rdrand = 1 << 30;
                let mut eax = 0;
                let mut ebx = 0;
                let mut ecx = 0;
                let mut edx = 0;
                unsafe {
                    cpuid::cpuid(1, &mut eax, &mut ebx, &mut ecx, &mut edx);
                }
                return (ecx & flag_rdrand) == flag_rdrand;
            }
        }
    }

    fn next_raw(&mut self) -> Result<u32, Box<dyn Error>> {
        if let Some(rand_file) = &mut self.rand_file {
            return rand_file.next_raw();
        }
        let mut v = 0;
        if !self.get_rd_rand(&mut v) {
            return Err("Random number generator failed up to retry limit.".into());
        }
        Ok(v)
    }

    fn next(&mut self) -> Result<f32, Box<dyn Error>> {
        let v = self.next_raw()?;
        let mut res = 0.0;
        let mut part = 1.0;
        for i in 0..32 {
            part /= 2.0;
            if (v >> i) & 1 == 1 {
                res += part;
            }
        }
        #[cfg(FPPOW > 5)]
        {
            let v = self.next_raw()?;
            for i in 0..32 {
                part /= 2.0;
                if (v >> i) & 1 == 1 {
                    res += part;
                }
            }
        }
        Ok(res)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut rd_random = RdRandom::new()?;
    let _ = rd_random.next()?;
    Ok(())
}


