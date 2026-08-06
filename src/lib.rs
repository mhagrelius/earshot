//! Hearing, for a GNOME application.
//!
//! A microphone read from the GLib main loop, and two local speech models on a
//! worker thread behind it. Extracted from Scribe and Familiar, which had grown
//! near-identical copies of both: the same `pw-record` invocation, the same
//! loudness curve, the same channel to the same two models. A bug fixed in one
//! copy stayed broken in the other, which is what this crate exists to stop.
//!
//! ## What is here and what is not
//!
//! The line is **the boundary, not the policy**. Opening a microphone, turning
//! bytes into samples, and getting words out of a model are the same job in
//! every app. Deciding when somebody stopped talking, where model files live,
//! what to do with the words, and what to say when it goes wrong are not, and
//! they stay with the caller:
//!
//! - [`Recorder`] hands over blocks of samples and says when it stopped. It has
//!   no opinion about when an utterance ended — Scribe ends one on a keypress
//!   and Familiar ends one on silence, and neither belongs here.
//! - [`Speech`] transcribes what it is given. **It does not know where models
//!   live**: `Speech::new` takes a resolver the caller supplies, called per job
//!   so a model downloaded while the app runs is picked up without a restart.
//!   Scribe owns its model directory and Familiar reads Scribe's copy, and that
//!   difference is exactly the kind of thing a shared crate should not decide.
//! - [`level`] is the loudness curve, and the thresholds calibrated against it
//!   stay with the caller, because they are a property of a room and a
//!   microphone rather than of this code.
//!
//! ## Two things not to change casually
//!
//! **The sample format is not a preference.** 16-bit mono at 16 kHz is what the
//! models take, so PipeWire is asked to resample and nothing here does.
//!
//! **[`STREAM_CHUNK`] is not a free parameter.** The streaming encoder is
//! cache-aware and was trained on 560 ms; a different chunk is not the model
//! anybody measured. It also emits *nothing* until it has a whole one, so the
//! remainder at the end of an utterance has to be padded and sent or the last
//! words are simply never decoded. Both apps got that wrong independently.

mod microphone;
mod models;
mod speech;

pub use microphone::{level, Recorder, StartError, BLOCK_MS, SAMPLE_RATE};
pub use models::{has_encoder, is_complete, Model};
pub use speech::{Speech, SpeechError, STREAM_CHUNK};

/// The models themselves, for a diagnostic that needs to drive one directly.
///
/// [`Speech`] is the way to use them: it owns the thread and answers on the main
/// loop. This is here so a tool measuring the models' own behaviour — whether a
/// part-chunk really is discarded, say — can do it outside a main loop without a
/// second version pin on `parakeet-rs` in each caller's `Cargo.toml`.
pub use parakeet_rs;
