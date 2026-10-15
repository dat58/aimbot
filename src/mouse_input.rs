//! The hand's own movement, read back off the firmware.
//!
//! The static planner continues a movement rather than making one, so it wants
//! the reports that came before the handoff. MAKCU V4 has no command that
//! streams mouse motion — `km.axis` and `km.mouse` are V3.x only, and
//! `km.stream(mouse)` carries buttons but states outright that motion is not
//! emitted — so the nearest thing is `km.getpos`, polled and differenced.
//!
//! That is a dead-reckoned estimate on the firmware's virtual screen, clamped
//! to it, at whatever rate the link sustains rather than the 1 kHz the planner
//! reads. The difference is made explicit here: a poll covers several
//! milliseconds, and its displacement is spread across the bins it spans
//! rather than pretended to be one report.

use crate::mouse::MouseVirtual;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// Bins are 1 ms, which is the grid both planners read.
const BIN: Duration = Duration::from_millis(1);

/// A ring of 1 ms bins, newest last, in the planner's x-right/y-up counts.
#[derive(Debug, Clone)]
pub struct Bins {
    ring: Vec<[f32; 2]>,
}

impl Bins {
    fn new(capacity: usize) -> Self {
        Self {
            ring: vec![[0.; 2]; capacity],
        }
    }

    fn push(&mut self, bin: [f32; 2]) {
        self.ring.rotate_left(1);
        let last = self.ring.len() - 1;
        self.ring[last] = bin;
    }

    /// The most recent `count` bins, oldest first.
    pub fn recent(&self, count: usize) -> &[[f32; 2]] {
        let from = self.ring.len().saturating_sub(count);
        &self.ring[from..]
    }
}

/// Take as much of `owed` out of `observed` as this sample can account for,
/// and return what is left.
///
/// The firmware spreads one `km.move` over several reports, so subtracting a
/// whole injection the moment it is sent invents a spike in one direction and
/// an equal one back as the position catches up — measured as 480 counts of
/// phantom path for a 120-count injection whose net came out at zero. Carrying
/// the debt instead means each sample gives back only what it actually moved.
pub fn absorb(observed: i64, owed: &mut i64) -> i64 {
    let take = if *owed > 0 {
        observed.clamp(0, *owed)
    } else {
        observed.clamp(*owed, 0)
    };
    *owed -= take;
    observed - take
}

/// Spread one polled displacement across the bins it covers.
///
/// A poll that took 7 ms is seven bins of real time, not one report: handing
/// the planner a single 7-count spike would be a velocity the hand never had.
/// Whole counts go first and the remainder lands on the newest bin, so the sum
/// is exactly what was observed.
pub fn spread(delta: (i32, i32), bins: usize) -> Vec<[f32; 2]> {
    let bins = bins.max(1);
    let mut out = vec![[0f32; 2]; bins];
    for (axis, total) in [delta.0, delta.1].into_iter().enumerate() {
        let step = total / bins as i32;
        for bin in out.iter_mut() {
            bin[axis] = step as f32;
        }
        out[bins - 1][axis] += (total - step * bins as i32) as f32;
    }
    out
}

/// Polls `km.getpos` in the background and keeps the last `capacity` bins.
pub struct CursorTrack {
    output: triple_buffer::Output<Bins>,
    running: Arc<AtomicBool>,
    capacity: usize,
}

impl CursorTrack {
    /// `None` when the firmware will not answer `km.getpos`, which is the
    /// honest answer on anything older than V4.026.
    pub fn start(mouse: Arc<MouseVirtual>, hz: u32, capacity: usize) -> Option<Self> {
        let first = match mouse.getpos() {
            Ok(position) => position,
            Err(e) => {
                tracing::warn!("[Cursor] cannot read the pointer back: {e}");
                return None;
            }
        };
        tracing::info!("[Cursor] tracking from {first:?} at {hz} Hz");

        let (mut input, output) = triple_buffer::triple_buffer(&Bins::new(capacity));
        let running = Arc::new(AtomicBool::new(true));
        let keep = running.clone();
        let period = Duration::from_micros(1_000_000 / hz.clamp(1, 1000) as u64);

        thread::spawn(move || {
            let mut bins = Bins::new(capacity);
            let mut last = first;
            let mut owed = (0i64, 0i64);
            let mut at = Instant::now();
            while keep.load(Ordering::Acquire) {
                thread::sleep(period);
                let Ok(now) = mouse.getpos() else { continue };
                let elapsed = at.elapsed();
                at = Instant::now();
                // What this process injected is not the hand. Measured on the
                // device: `km.getpos` moves one unit per count we send, so
                // what is left after taking ours out is the physical movement.
                let ours = mouse.take_injected();
                owed = (owed.0 + ours.0, owed.1 + ours.1);
                let moved = (
                    absorb((now.0 - last.0) as i64, &mut owed.0),
                    absorb((now.1 - last.1) as i64, &mut owed.1),
                );
                last = now;
                // y comes back screen-down; the planner reads y-up.
                let delta = (moved.0 as i32, -moved.1 as i32);
                let covered = (elapsed.as_micros() / BIN.as_micros()).max(1) as usize;
                for bin in spread(delta, covered) {
                    bins.push(bin);
                }
                input.write(bins.clone());
            }
        });

        Some(Self {
            output,
            running,
            capacity,
        })
    }

    /// The most recent bins, oldest first.
    pub fn recent(&mut self, count: usize) -> Vec<[f32; 2]> {
        self.output.read().recent(count.min(self.capacity)).to_vec()
    }
}

impl Drop for CursorTrack {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The firmware applies an injection over several reports, so the debt has
    /// to be paid off as the position catches up rather than all at once.
    #[test]
    fn an_injection_is_cancelled_as_it_actually_lands() {
        let mut owed = 120;
        // Nothing has moved yet: the whole debt stays outstanding.
        assert_eq!(absorb(0, &mut owed), 0);
        assert_eq!(owed, 120);
        // It lands over four samples and none of them read as the hand.
        for chunk in [30, 40, 20, 30] {
            assert_eq!(absorb(chunk, &mut owed), 0);
        }
        assert_eq!(owed, 0);
    }

    #[test]
    fn a_hand_moving_through_an_injection_still_gets_through() {
        let mut owed = 50;
        // 70 observed while 50 is owed: 20 of it is the hand.
        assert_eq!(absorb(70, &mut owed), 20);
        assert_eq!(owed, 0);
    }

    /// A debt one way must not swallow movement the other way.
    #[test]
    fn an_opposite_move_is_not_absorbed() {
        let mut owed = 120;
        assert_eq!(absorb(-30, &mut owed), -30);
        assert_eq!(owed, 120);

        let mut owed = -120;
        assert_eq!(absorb(-30, &mut owed), 0);
        assert_eq!(owed, -90);
        assert_eq!(absorb(40, &mut owed), 40);
    }

    #[test]
    fn nothing_owed_passes_straight_through() {
        let mut owed = 0;
        assert_eq!(absorb(17, &mut owed), 17);
        assert_eq!(absorb(-17, &mut owed), -17);
        assert_eq!(owed, 0);
    }

    /// The sum has to survive: the planner reads a displacement, and losing a
    /// count to rounding moves the handoff.
    #[test]
    fn a_polled_displacement_keeps_its_total() {
        for delta in [(7, -3), (1, 0), (-13, 29), (0, 0), (100, -100)] {
            for bins in [1usize, 2, 7, 10] {
                let out = spread(delta, bins);
                assert_eq!(out.len(), bins);
                let total: (f32, f32) = out
                    .iter()
                    .fold((0., 0.), |sum, bin| (sum.0 + bin[0], sum.1 + bin[1]));
                assert_eq!(
                    (total.0 as i32, total.1 as i32),
                    delta,
                    "{delta:?} over {bins} bins"
                );
            }
        }
    }

    /// A poll covering several milliseconds is several bins of real time, not
    /// one spike the hand never produced.
    #[test]
    fn a_slow_poll_is_spread_rather_than_spiked() {
        let out = spread((10, 0), 10);
        assert!(out.iter().all(|bin| bin[0] == 1.));
    }

    #[test]
    fn the_ring_keeps_the_newest_bins_last() {
        let mut bins = Bins::new(4);
        for k in 1..=6 {
            bins.push([k as f32, 0.]);
        }
        assert_eq!(bins.recent(4), &[[3., 0.], [4., 0.], [5., 0.], [6., 0.]]);
        assert_eq!(bins.recent(2), &[[5., 0.], [6., 0.]]);
        // Asking for more than it holds gives everything it has.
        assert_eq!(bins.recent(99).len(), 4);
    }
}
