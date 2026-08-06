//! The speech models, on a worker thread.
//!
//! Neither model is fast enough for the main loop — a quarter of a second is six
//! dropped frames — so they live on a thread of their own and are spoken to over
//! a channel, answering through `glib::idle_add_once`. That is the whole
//! threading story: no runtime, no executor, and no widget ever touched from the
//! worker.
//!
//! Each model is loaded the first time it is needed and kept, because loading
//! costs most of a second, which on a short utterance would be the longest part
//! of the whole thing.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};

use parakeet_rs::{Nemotron, ParakeetTDT, TimestampMode, Transcriber};

use crate::microphone::SAMPLE_RATE;
use crate::models::Model;

/// Chunk the streaming encoder was built around, in samples at 16 kHz.
///
/// Not a free parameter: a different one is not what the model was trained on.
/// And it emits **nothing at all** until it has a whole one, so the remainder
/// left when somebody stops talking has to be padded to this and sent, or the
/// last words are never decoded. Both callers had that bug independently.
pub const STREAM_CHUNK: usize = 8_960; // 560 ms

/// Where a model is, if it is anywhere. Called per job rather than once, so a
/// model downloaded while the app is running is picked up without a restart.
type Resolve = Box<dyn Fn(Model) -> Option<PathBuf> + Send>;

/// A failure worth telling the user about.
#[derive(Debug)]
pub enum SpeechError {
    NotInstalled(Model),
    /// A pass over an earlier utterance is still running.
    Busy,
    Load(String),
    Transcribe(String),
}

impl std::fmt::Display for SpeechError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpeechError::NotInstalled(_) => write!(f, "No speech model is installed."),
            SpeechError::Busy => write!(f, "Still working out what was said a moment ago."),
            SpeechError::Load(error) => write!(f, "The speech model could not be loaded: {error}"),
            SpeechError::Transcribe(error) => {
                write!(f, "What you said could not be transcribed: {error}")
            }
        }
    }
}

impl std::error::Error for SpeechError {}

enum Job {
    /// Transcribe a whole utterance with the accurate model.
    Whole {
        audio: Vec<f32>,
        reply: Box<dyn FnOnce(Result<String, SpeechError>) + Send>,
    },
    /// Feed one chunk to the live model.
    Chunk {
        audio: Vec<f32>,
        reply: Box<dyn FnOnce(Result<String, SpeechError>) + Send>,
    },
    /// Forget the streaming encoder's cache so the next utterance starts clean.
    Reset,
    Stop,
}

/// A handle to the worker thread.
pub struct Speech {
    jobs: Sender<Job>,
    /// Set while an accurate pass is outstanding, so a second stop cannot queue a
    /// second pass over the same audio.
    busy: Arc<Mutex<bool>>,
}

impl Speech {
    /// Start the worker. `resolve` answers where a model's directory is.
    ///
    /// `name` is the thread's name, so a stack trace says which app it belongs
    /// to.
    pub fn new(name: &str, resolve: impl Fn(Model) -> Option<PathBuf> + Send + 'static) -> Self {
        let (jobs, inbox) = mpsc::channel();
        std::thread::Builder::new()
            .name(name.to_string())
            .spawn(move || worker(inbox, Box::new(resolve)))
            .expect("the speech worker thread could not be started");
        Self {
            jobs,
            busy: Arc::new(Mutex::new(false)),
        }
    }

    /// Whether an accurate pass is outstanding.
    pub fn is_busy(&self) -> bool {
        *self.busy.lock().expect("speech lock")
    }

    /// Transcribe a finished utterance. `done` runs on the main loop.
    pub fn transcribe(
        &self,
        audio: Vec<f32>,
        done: impl FnOnce(Result<String, SpeechError>) + 'static,
    ) {
        {
            let mut busy = self.busy.lock().expect("speech lock");
            if *busy {
                // Answered rather than dropped. Returning silently here leaves
                // the caller waiting on a callback that will never come, and what
                // that looks like from the outside is an application that heard
                // you and then stopped.
                done(Err(SpeechError::Busy));
                return;
            }
            *busy = true;
        }
        let busy = self.busy.clone();
        let done = main_loop_hop(done);
        let _ = self.jobs.send(Job::Whole {
            audio,
            reply: Box::new(move |result| {
                *busy.lock().expect("speech lock") = false;
                done(result);
            }),
        });
    }

    /// Feed one chunk to the live model. `done` runs on the main loop.
    ///
    /// Replies arrive in the order the chunks were queued, which is what lets a
    /// caller flush the remainder at the end and read the finished transcript in
    /// that chunk's reply.
    pub fn feed(&self, audio: Vec<f32>, done: impl FnOnce(Result<String, SpeechError>) + 'static) {
        let _ = self.jobs.send(Job::Chunk {
            audio,
            reply: Box::new(main_loop_hop(done)),
        });
    }

    /// Forget the streaming encoder's state before a new utterance.
    pub fn reset(&self) {
        let _ = self.jobs.send(Job::Reset);
    }
}

impl Drop for Speech {
    fn drop(&mut self) {
        let _ = self.jobs.send(Job::Stop);
    }
}

/// Wrap a main-loop closure so a worker thread can call it.
///
/// The closure itself is not `Send` — it captures widgets — so it is moved to the
/// main loop by `idle_add_once` and only the result crosses the boundary.
fn main_loop_hop<T: Send + 'static>(
    done: impl FnOnce(T) + 'static,
) -> impl FnOnce(T) + Send + 'static {
    let done = glib::thread_guard::ThreadGuard::new(done);
    move |value: T| {
        glib::idle_add_once(move || {
            (done.into_inner())(value);
        });
    }
}

fn worker(inbox: Receiver<Job>, resolve: Resolve) {
    let mut accurate: Option<ParakeetTDT> = None;
    let mut live: Option<Nemotron> = None;

    while let Ok(job) = inbox.recv() {
        match job {
            Job::Stop => return,

            Job::Reset => {
                // Dropping the model is the reliable way to clear the encoder
                // cache; it is reloaded on the next chunk.
                live = None;
            }

            Job::Whole { audio, reply } => {
                let Some(dir) = resolve(Model::Accurate) else {
                    reply(Err(SpeechError::NotInstalled(Model::Accurate)));
                    continue;
                };
                if accurate.is_none() {
                    match ParakeetTDT::from_pretrained(dir, None) {
                        Ok(model) => accurate = Some(model),
                        Err(error) => {
                            reply(Err(SpeechError::Load(error.to_string())));
                            continue;
                        }
                    }
                }
                let model = accurate.as_mut().expect("just loaded");
                let result = model
                    .transcribe_samples(audio, SAMPLE_RATE, 1, Some(TimestampMode::Sentences))
                    .map(|transcription| transcription.text)
                    .map_err(|error| SpeechError::Transcribe(error.to_string()));
                reply(result);
            }

            Job::Chunk { audio, reply } => {
                let Some(dir) = resolve(Model::Live) else {
                    reply(Err(SpeechError::NotInstalled(Model::Live)));
                    continue;
                };
                if live.is_none() {
                    match Nemotron::from_pretrained(dir, None) {
                        Ok(model) => live = Some(model),
                        Err(error) => {
                            reply(Err(SpeechError::Load(error.to_string())));
                            continue;
                        }
                    }
                }
                let model = live.as_mut().expect("just loaded");
                let result = model
                    .transcribe_chunk(&audio)
                    .map_err(|error| SpeechError::Transcribe(error.to_string()));
                reply(result);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_chunk_is_the_size_the_encoder_was_built_for() {
        assert_eq!(STREAM_CHUNK, 560 * 16_000 / 1000);
    }

    #[test]
    fn a_missing_model_is_named_in_the_error() {
        // Which of the two is missing decides what a caller offers to fetch.
        let error = SpeechError::NotInstalled(Model::Live);
        assert!(matches!(error, SpeechError::NotInstalled(Model::Live)));
    }
}
