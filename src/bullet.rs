use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

#[derive(Clone, Debug, Default)]
pub struct AutoShoot(Arc<AtomicU32>);

impl AutoShoot {
    pub fn new(n: u32) -> Self {
        Self(Arc::new(AtomicU32::new(n)))
    }

    #[inline(always)]
    pub fn set(&self, n: u32) {
        self.0.store(n, Ordering::Release);
    }

    #[inline(always)]
    pub fn get(&self) -> u32 {
        self.0.load(Ordering::Acquire)
    }

    #[inline(always)]
    pub fn disable(&self) -> bool {
        self.get() == 0
    }

    #[inline(always)]
    pub fn enable(&self) -> bool {
        !self.disable()
    }

    #[inline(always)]
    pub fn sub(&self, value: u32) {
        let remain = self.get();
        if remain >= value {
            self.0.fetch_sub(value, Ordering::AcqRel);
        } else if remain > 0 {
            self.set(0);
        }
    }

    pub fn sub_1(&self) {
        self.sub(1);
    }
}
