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
use std::time::Duration;

/// One `km.getpos` round trip measures about a millisecond end to end, which
/// is already the grid the planners read, so the gap between polls only has to
/// be short enough not to add to it.
const POLL_GAP: Duration = Duration::from_micros(200);

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

/// Polls `km.getpos` in the background and keeps the last `capacity` bins.
pub struct CursorTrack {
    output: triple_buffer::Output<Bins>,
    running: Arc<AtomicBool>,
    capacity: usize,
}

impl CursorTrack {
    /// `None` when the firmware will not answer `km.getpos`, which is the
    /// honest answer on anything older than V4.026.
    pub fn start(mouse: Arc<MouseVirtual>, capacity: usize) -> Option<Self> {
        let first = match mouse.getpos() {
            Ok(position) => position,
            Err(e) => {
                tracing::warn!("[Cursor] cannot read the pointer back: {e}");
                return None;
            }
        };
        tracing::info!("[Cursor] tracking from {first:?}");

        let (mut input, output) = triple_buffer::triple_buffer(&Bins::new(capacity));
        let running = Arc::new(AtomicBool::new(true));
        let keep = running.clone();

        thread::spawn(move || {
            let mut bins = Bins::new(capacity);
            let mut last = first;
            let mut owed = (0i64, 0i64);
            while keep.load(Ordering::Acquire) {
                thread::sleep(POLL_GAP);
                let Ok(now) = mouse.getpos() else { continue };
                // What this process injected is not the hand. `km.getpos`
                // moves one unit per count we send, measured on the device.
                let ours = mouse.take_injected();
                owed = (owed.0 + ours.0, owed.1 + ours.1);
                bins.push([
                    absorb((now.0 - last.0) as i64, &mut owed.0) as f32,
                    // y comes back screen-down; the planner reads y-up.
                    -absorb((now.1 - last.1) as i64, &mut owed.1) as f32,
                ]);
                last = now;
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
