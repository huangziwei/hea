#!/usr/bin/env bash
# Rebuild the hea._rs extension (crate hea-rs) from rust/ and drop the compiled
# module into the source tree (hea/_rs.*.so), where the editable install finds it.
#
# build-backend is `maturin` now, so `uv sync` / `uv pip install -e .` build the
# extension on their own and this script is only needed to refresh it in place.
# Prefer it over `maturin develop`, which can leave a stale .so behind.
#
# Sandbox note: cargo cannot write ~/.cargo here, so CARGO_HOME is redirected
# into the workdir. Network is needed on the first build (crate downloads).
set -euo pipefail
cd "$(dirname "$0")/.."

export CARGO_HOME="${CARGO_HOME:-$PWD/.cargo-home}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/target}"

.venv/bin/maturin build --release --interpreter .venv/bin/python -o target/wheels

whl="$(ls -t target/wheels/hea-*.whl | head -1)"
rm -rf target/wheel_extract
.venv/bin/python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall('target/wheel_extract')" "$whl"
cp target/wheel_extract/hea/_rs*.so hea/
echo "✓ installed $(ls hea/_rs*.so) from $(basename "$whl")"
