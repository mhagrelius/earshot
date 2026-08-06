//! The microphone.
//!
//! `pw-record` is spawned and its raw samples are read off a pipe by the GLib
//! main loop. No audio library and no thread of our own: PipeWire is what these
//! apps run on, `pw-record` ships with it, and `gio::Subprocess` already knows
//! how to read a pipe without blocking anything.
//!
//! Samples come back as 16-bit little-endian mono at 16 kHz, which is what the
//! speech models want, so PipeWire does the resampling and this does none. They
//! are handed on as `f32` in −1.0..1.0, the form the models take.

use gio::prelude::*;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

/// What the models expect. Not configurable, because nothing else is correct.
pub const SAMPLE_RATE: u32 = 16_000;

/// How much audio one read asks for, in milliseconds.
///
/// Short enough that a level meter looks live and an endpointer reacts, long
/// enough that the main loop is not woken constantly.
///
/// **A read almost never returns this much.** Measured against `pw-record`: 92
/// reads in three seconds, not one of them a full block, most a little over
/// half. Anything keeping a clock has to use the length of the audio it actually
/// got — assuming this constant per read ran one endpointer's clock at nearly
/// twice real time, which made its patience expire while somebody was still
/// talking.
pub const BLOCK_MS: u32 = 40;

const READ_BYTES: usize = (SAMPLE_RATE as usize * BLOCK_MS as usize / 1000) * 2;

type OnChunk = Rc<dyn Fn(&[f32])>;

/// Told why the microphone stopped, when it stops on its own.
type OnEnd = Box<dyn FnOnce(String)>;

/// A refusal to start recording, in terms the user can act on.
#[derive(Debug)]
pub enum StartError {
    /// `pw-record` is not installed.
    ToolMissing,
    Spawn(glib::Error),
}

impl std::fmt::Display for StartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartError::ToolMissing => write!(
                f,
                "pw-record is not installed. It comes with PipeWire, in the pipewire-bin package \
                 on Debian and Ubuntu and in pipewire-utils on Fedora."
            ),
            StartError::Spawn(error) => write!(f, "The microphone could not be opened: {error}"),
        }
    }
}

impl std::error::Error for StartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StartError::ToolMissing => None,
            StartError::Spawn(error) => Some(error),
        }
    }
}

/// A running capture.
pub struct Recorder {
    process: gio::Subprocess,
    /// Everything heard since the last [`Self::take`], for a pass over the whole
    /// utterance at the end.
    samples: Rc<RefCell<Vec<f32>>>,
    stopped: Rc<Cell<bool>>,
}

impl Recorder {
    /// Start recording.
    ///
    /// `source` is a PipeWire node name, or empty for the system default.
    /// `on_chunk` runs on the main loop with each new block of samples.
    ///
    /// `on_end` runs if the microphone stops of its own accord — the pipe
    /// closing because `pw-record` exited, or a read failing. **Something has to
    /// be told.** An app whose state advances only when audio arrives will sit
    /// there forever otherwise, and `pw-record` given a source that no longer
    /// exists exits immediately, with stderr silenced and nothing to see. It is
    /// not called for a deliberate [`Self::finish`].
    pub fn start(
        source: &str,
        on_chunk: impl Fn(&[f32]) + 'static,
        on_end: impl FnOnce(String) + 'static,
    ) -> Result<Self, StartError> {
        if glib::find_program_in_path("pw-record").is_none() {
            return Err(StartError::ToolMissing);
        }

        let mut argv: Vec<&str> = vec![
            "pw-record",
            "--rate",
            "16000",
            "--channels",
            "1",
            "--format",
            "s16",
        ];
        let source = source.trim();
        if !source.is_empty() {
            argv.push("--target");
            argv.push(source);
        }
        // `-` is stdout, which is where we read from.
        argv.push("-");

        let process = gio::Subprocess::newv(
            &argv.iter().map(std::ffi::OsStr::new).collect::<Vec<_>>(),
            gio::SubprocessFlags::STDOUT_PIPE | gio::SubprocessFlags::STDERR_SILENCE,
        )
        .map_err(StartError::Spawn)?;

        let samples = Rc::new(RefCell::new(Vec::new()));
        let stopped = Rc::new(Cell::new(false));

        if let Some(stdout) = process.stdout_pipe() {
            read_loop(
                stdout,
                samples.clone(),
                stopped.clone(),
                Rc::new(on_chunk),
                Box::new(on_end),
            );
        }

        Ok(Self {
            process,
            samples,
            stopped,
        })
    }

    /// Take everything heard so far and keep recording.
    ///
    /// For an app that holds the microphone open across a whole exchange —
    /// closing it is what makes an assistant impossible to interrupt — this is
    /// how one utterance is lifted out of a stream that does not stop.
    pub fn take(&self) -> Vec<f32> {
        std::mem::take(&mut *self.samples.borrow_mut())
    }

    /// Throw away everything but the last `keep` samples.
    ///
    /// Used at an interruption: what came before it is the assistant's own voice
    /// coming back off the speakers, and what is kept is the moment the person
    /// started talking — enough that their first word survives being the thing
    /// that triggered the interruption.
    pub fn keep_last(&self, keep: usize) {
        let mut samples = self.samples.borrow_mut();
        if samples.len() > keep {
            let excess = samples.len() - keep;
            samples.drain(..excess);
        }
    }

    /// How much audio is held, as a duration.
    pub fn duration(&self) -> std::time::Duration {
        let count = self.samples.borrow().len();
        std::time::Duration::from_secs_f64(count as f64 / SAMPLE_RATE as f64)
    }

    /// Stop recording and take everything heard. Does not call `on_end`.
    pub fn finish(self) -> Vec<f32> {
        self.stopped.set(true);
        self.process.force_exit();
        std::mem::take(&mut *self.samples.borrow_mut())
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        self.stopped.set(true);
        self.process.force_exit();
    }
}

/// Read the pipe until it ends, one block at a time.
///
/// Each read schedules the next, so there is exactly one outstanding read and no
/// way to overlap two into the same buffer.
fn read_loop(
    stream: gio::InputStream,
    samples: Rc<RefCell<Vec<f32>>>,
    stopped: Rc<Cell<bool>>,
    on_chunk: OnChunk,
    on_end: OnEnd,
) {
    glib::spawn_future_local(async move {
        // Why the loop stopped. `None` means it was stopped deliberately and
        // nobody needs telling.
        let ended: Option<String> = loop {
            if stopped.get() {
                break None;
            }
            let buffer = vec![0u8; READ_BYTES];
            let (buffer, read) = match stream.read_future(buffer, glib::Priority::DEFAULT).await {
                Ok(result) => result,
                Err(error) => break Some(format!("The microphone stopped: {}", error.1)),
            };
            if read == 0 {
                break Some(
                    "The microphone closed on its own. Check that the input device still exists."
                        .to_string(),
                );
            }
            if stopped.get() {
                break None;
            }

            // An odd byte count would split a sample across two reads. Reading a
            // whole number of frames each time keeps that from happening.
            let usable = read - (read % 2);
            let block: Vec<f32> = buffer[..usable]
                .chunks_exact(2)
                .map(|pair| i16::from_le_bytes([pair[0], pair[1]]) as f32 / 32_768.0)
                .collect();

            samples.borrow_mut().extend_from_slice(&block);
            on_chunk(&block);
        };

        if let Some(reason) = ended {
            if !stopped.get() {
                on_end(reason);
            }
        }
    });
}

/// Loudness of a block, as a 0.0..1.0 figure.
///
/// Root mean square rather than peak: a peak meter on speech spends its time
/// pinned by consonants and tells nobody whether they are being heard.
///
/// The exponent then bends the scale. Speech sits around an RMS of 0.05 to 0.2,
/// which on a linear meter is a bar that never leaves the left-hand tenth;
/// raising it to a fractional power lifts that range into the middle while still
/// leaving somewhere for a shout to go.
///
/// **Every threshold either caller has is calibrated against this curve.**
/// Changing the exponent changes what counts as speech in both apps, and the
/// numbers it invalidates are in their own source, not here.
pub fn level(block: &[f32]) -> f64 {
    if block.is_empty() {
        return 0.0;
    }
    let sum: f64 = block.iter().map(|s| (*s as f64) * (*s as f64)).sum();
    let rms = (sum / block.len() as f64).sqrt();
    rms.clamp(0.0, 1.0).powf(0.4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_reads_as_no_level() {
        assert_eq!(level(&[0.0; 128]), 0.0);
        assert_eq!(level(&[]), 0.0);
    }

    #[test]
    fn a_full_scale_tone_pins_the_meter() {
        let square: Vec<f32> = (0..128)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        assert_eq!(level(&square), 1.0);
    }

    #[test]
    fn louder_input_reads_higher() {
        let quiet: Vec<f32> = (0..128)
            .map(|i| if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();
        let loud: Vec<f32> = (0..128)
            .map(|i| if i % 2 == 0 { 0.3 } else { -0.3 })
            .collect();
        assert!(level(&quiet) < level(&loud));
        assert!(level(&loud) < 1.0, "speech must leave headroom above it");
    }

    #[test]
    fn ordinary_speech_lands_in_the_middle_of_the_bar() {
        // The meter exists to tell the user the microphone is picking them up.
        // A bar that sits near zero while they talk normally fails at that, and
        // so does one that is already pinned.
        let speech: Vec<f32> = (0..128)
            .map(|i| if i % 2 == 0 { 0.1 } else { -0.1 })
            .collect();
        let reading = level(&speech);
        assert!((0.25..0.75).contains(&reading), "read {reading}");
    }

    #[test]
    fn the_meter_never_leaves_its_range() {
        let over: Vec<f32> = vec![9.0; 64];
        assert!((0.0..=1.0).contains(&level(&over)));
    }

    #[test]
    fn a_read_is_a_whole_number_of_frames() {
        // An odd read size would split a 16-bit sample across two reads.
        assert_eq!(READ_BYTES % 2, 0);
        assert_eq!(
            READ_BYTES,
            (SAMPLE_RATE as usize * BLOCK_MS as usize / 1000) * 2
        );
    }
}
