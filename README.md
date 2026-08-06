# earshot

Microphone capture and local speech recognition for GNOME applications, driven
from the GLib main loop.

Extracted from [Scribe](https://github.com/mhagrelius/scribe) and Familiar, which
had grown near-identical copies of both halves: the same `pw-record` invocation,
the same loudness curve, the same channel to the same two models. The copies
drifted, and a bug fixed in one stayed broken in the other — the streaming
encoder's final part-chunk was being discarded in both, and was fixed in Scribe
weeks before anyone noticed the same hole in Familiar. That is what this exists to
stop.

## What it does

```rust
// The microphone. Blocks arrive on the main loop; `on_end` fires if it stops
// on its own, which is the difference between a failure and a silent hang.
let recorder = earshot::Recorder::start(
    "",                                     // a PipeWire node name, or default
    |block| meter.set(earshot::level(block)),
    |reason| app.report(&reason),
)?;

// The models, on one worker thread. Where they live is the caller's business.
let speech = earshot::Speech::new("myapp-speech", |model| my_lookup(model));
speech.transcribe(recorder.finish(), |result| match result { .. });
```

- **`Recorder`** spawns `pw-record` and reads its pipe with `gio::Subprocess`.
  No audio library and no thread: PipeWire is what these apps run on, and it
  resamples to the 16-bit mono 16 kHz the models want so nothing here has to.
- **`Speech`** runs Parakeet TDT (accurate, whole utterances, punctuated) and
  Nemotron's cache-aware streaming encoder (words as they are said) on a worker
  thread, answering through `glib::idle_add_once`. Each model is loaded on first
  use and kept, because loading costs most of a second.
- **`level`** is the loudness curve: RMS, then raised to 0.4 so ordinary speech
  lands in the middle of a meter rather than in its left-hand tenth.

## Where the line is

The boundary, not the policy. Opening a microphone and getting words out of a
model are the same job in every app. These are not, and they stay with the
caller:

| | Why it is not here |
|---|---|
| When an utterance ended | Scribe ends one on a keypress, Familiar on silence |
| Where model files live | Scribe owns its copy; Familiar reads Scribe's if it finds one, so `Speech::new` takes a resolver called per job |
| Gate and threshold levels | A property of a room and a microphone, not of this code |
| What to tell the user | `SpeechError::NotInstalled` names the model; only the app knows where it looks or whether it can fetch it |

## Two things not to change casually

**The sample format is not a preference.** 16-bit mono at 16 kHz is what the
models take.

**`STREAM_CHUNK` is not a free parameter.** The streaming encoder is cache-aware
and was trained on 560 ms. It also emits *nothing* until it has a whole one, so
the remainder left when somebody stops talking must be padded to a full chunk and
sent, or the last words are never decoded.

## Building

`parakeet-rs` statically links an ONNX Runtime that `ort-sys` downloads at build
time, so the first build on a machine needs the network and about 100 MB of
`~/.cache/ort.pyke.io`. `glib` and `gio` are pinned to what `gtk4-rs` 0.11
resolves to — a second `glib` in the tree would make this crate's
`gio::InputStream` a different type from the consuming app's.

```sh
cargo test
```

Licensed GPL-3.0-or-later, as both callers are.
