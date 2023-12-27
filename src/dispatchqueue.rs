use std::sync::{Arc, Mutex, Condvar};
use std::thread;
use std::collections::VecDeque;

type DispatchFn = Box<dyn FnOnce() + Send>;

pub struct DispatchQueue {
    lock: Arc<(Mutex<bool>, Condvar)>,
    q: Arc<Mutex<VecDeque<DispatchFn>>>,
    quit: Arc<(Mutex<bool>, Condvar)>,
    is_finished: Arc<(Mutex<bool>, Condvar)>,
    is_started: Arc<(Mutex<bool>, Condvar)>,
    thread: Option<thread::JoinHandle<()>>,
}

impl DispatchQueue {
    pub fn new() -> Self {
        DispatchQueue {
            lock: Arc::new((Mutex::new(false), Condvar::new())),
            q: Arc::new(Mutex::new(VecDeque::new())),
            quit: Arc::new((Mutex::new(false), Condvar::new())),
            is_finished: Arc::new((Mutex::new(true), Condvar::new())),
            is_started: Arc::new((Mutex::new(false), Condvar::new())),
            thread: None,
        }
    }

    pub fn dispatch(&self, op: DispatchFn) {
        let (lock, cvar) = &*self.lock;
        let mut q = self.q.lock().unwrap();
        let mut is_finished = self.is_finished.0.lock().unwrap();
        let mut is_started = self.is_started.0.lock().unwrap();

        if *is_finished {
            *is_finished = false;
            *is_started = true;
            let q_clone = Arc::clone(&self.q);
            let lock_clone = Arc::clone(&self.lock);
            let quit_clone = Arc::clone(&self.quit);
            let is_finished_clone = Arc::clone(&self.is_finished);
            let is_started_clone = Arc::clone(&self.is_started);
            self.thread = Some(thread::spawn(move || {
                dispatch_thread_handler(q_clone, lock_clone, quit_clone, is_finished_clone, is_started_clone);
            }));
        }

        q.push_back(op);
        drop(q);
        cvar.notify_one();
    }

    pub fn finish(&self) {
        let (quit_lock, quit_cvar) = &*self.quit;
        let (is_finished_lock, is_finished_cvar) = &*self.is_finished;
        let is_started_lock = self.is_started.0.lock().unwrap();

        if *is_started_lock {
            quit_lock.lock().unwrap();
            is_finished_cvar.wait_while(is_finished_lock, |is_finished| !*is_finished).unwrap();
        }
    }

    pub fn dump(&self) {
        let (quit_lock, quit_cvar) = &*self.quit;
        let (is_finished_lock, is_finished_cvar) = &*self.is_finished;
        let is_started_lock = self.is_started.0.lock().unwrap();

        if *is_started_lock {
            let mut q = self.q.lock().unwrap();
            q.clear();
            *is_finished_lock = true;
            drop(q);
            is_finished_cvar.notify_all();
        }
    }
}

impl Drop for DispatchQueue {
    fn drop(&mut self) {
        let (is_started_lock, is_started_cvar) = &*self.is_started;
        let (quit_lock, quit_cvar) = &*self.quit;

        if *is_started_lock.lock().unwrap() {
            let mut q = self.q.lock().unwrap();
            q.clear();
            *quit_lock.lock().unwrap() = true;
            drop(q);
            self.lock.1.notify_all();
            self.thread.take().unwrap().join().unwrap();
            *self.is_finished.0.lock().unwrap() = true;
            self.is_finished.1.notify_all();
        }
    }
}

fn dispatch_thread_handler(
    q: Arc<Mutex<VecDeque<DispatchFn>>>,
    lock: Arc<(Mutex<bool>, Condvar)>,
    quit: Arc<(Mutex<bool>, Condvar)>,
    is_finished: Arc<(Mutex<bool>, Condvar)>,
    is_started: Arc<(Mutex<bool>, Condvar)>,
) {
    let (quit_lock, quit_cvar) = &*quit;
    let (is_finished_lock, is_finished_cvar) = &*is_finished;

    loop {
        let op = {
            let mut q = q.lock().unwrap();
            while q.is_empty() {
                q = lock.1.wait(q).unwrap();
            }
            q.pop_front().unwrap()
        };

        if *quit_lock.lock().unwrap() {
            continue;
        }

        op();

        let mut q = q.lock().unwrap();
        if q.is_empty() {
            *is_finished_lock = true;
            drop(q);
            is_finished_cvar.notify_all();
        }
    }
}


