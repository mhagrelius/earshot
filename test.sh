#!/usr/bin/env bash
# The gate: fmt, clippy with warnings as errors, tests. No display needed —
# nothing here draws anything.
set -euo pipefail
export RUST_BACKTRACE=1

echo "==> cargo fmt"
cargo fmt --check

echo "==> cargo clippy"
cargo clippy --all-targets --all-features -- -D warnings

echo "==> cargo test"
cargo test --all-targets

echo "All checks passed."
