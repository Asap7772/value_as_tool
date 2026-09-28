#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
artifact_dir="$repo_root/artifacts"
destination="${1:-$artifact_dir/qwen38_27b_harness_v4}"
expected_sha256="60157023b28e470d0e6fb48e34059ddf5c485a53e52a21080291b06f66e3efbb"

parts=("$artifact_dir"/qwen38_27b_harness_v4.tar.zst.part-*)
if [[ ${#parts[@]} -ne 8 || ! -f "${parts[0]}" ]]; then
  echo "Expected eight qwen38_27b_harness_v4 archive parts in $artifact_dir" >&2
  exit 1
fi

actual_sha256="$(cat -- "${parts[@]}" | sha256sum | awk '{print $1}')"
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "Artifact checksum mismatch: expected $expected_sha256, got $actual_sha256" >&2
  exit 1
fi

if [[ -e "$destination" || -L "$destination" ]]; then
  echo "Refusing to overwrite existing destination: $destination" >&2
  exit 1
fi

mkdir -p -- "$destination"
cat -- "${parts[@]}" | tar --zstd -xf - -C "$destination"

echo "Restored Qwen 27B artifacts to $destination"
