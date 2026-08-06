//! What a model directory has to contain.
//!
//! The file layout is shared because it is the *models'* layout, not an app's
//! choice: both callers download the same two directories from the same place.
//! **Where those directories live is not here** — Scribe owns its copy and
//! Familiar reads Scribe's if it finds one, and a crate under both of them is
//! the wrong place to settle that.

use std::path::Path;

/// Which of the two models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    /// Parakeet TDT 0.6B v3. Transcribes a whole utterance at once, with
    /// punctuation, at roughly 48x real time on a CPU.
    Accurate,
    /// Nemotron's cache-aware streaming encoder. Takes fixed chunks and emits
    /// words as they are said, for showing somebody they are being heard.
    Live,
}

impl Model {
    /// The directory name this model is distributed under.
    pub fn folder(self) -> &'static str {
        match self {
            Model::Accurate => "parakeet-tdt-0.6b-v3-int8",
            Model::Live => "nemotron-streaming-en-0.6b",
        }
    }
}

/// Whether a directory holds a usable copy of `model`.
///
/// An encoder on its own is what an interrupted download leaves, and loading
/// that fails deep inside the ONNX Runtime with nothing a user could act on.
pub fn is_complete(dir: &Path, model: Model) -> bool {
    let required: &[&str] = match model {
        Model::Accurate => &["vocab.txt", "nemo128.onnx"],
        Model::Live => &["tokenizer.model"],
    };
    dir.is_dir() && required.iter().all(|name| dir.join(name).exists()) && has_encoder(dir)
}

/// Whether a directory holds an encoder under any of the names it ships as.
///
/// The int8 and float builds of the same model name it differently, and both are
/// valid downloads.
pub fn has_encoder(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries.filter_map(Result::ok).any(|entry| {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        name.starts_with("encoder") && name.ends_with(".onnx")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_directory_is_not_a_model() {
        let dir = tempfile::tempdir().expect("temp dir");
        assert!(!is_complete(dir.path(), Model::Accurate));
        assert!(!is_complete(dir.path(), Model::Live));
    }

    #[test]
    fn an_encoder_is_found_under_any_of_its_names() {
        for name in [
            "encoder-model.int8.onnx",
            "encoder.onnx",
            "encoder-model.onnx",
        ] {
            let dir = tempfile::tempdir().expect("temp dir");
            std::fs::write(dir.path().join(name), b"").expect("write");
            assert!(has_encoder(dir.path()), "{name} should count as an encoder");
        }
    }

    #[test]
    fn a_decoder_alone_is_not_an_encoder() {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(dir.path().join("decoder_joint-model.onnx"), b"").expect("write");
        assert!(!has_encoder(dir.path()));
    }

    #[test]
    fn a_half_downloaded_model_does_not_count_as_installed() {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(dir.path().join("encoder.onnx"), b"").expect("write");
        assert!(!is_complete(dir.path(), Model::Accurate));
    }

    #[test]
    fn the_two_models_live_in_directories_of_their_own() {
        assert_ne!(Model::Accurate.folder(), Model::Live.folder());
    }
}
