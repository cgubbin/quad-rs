#!/usr/bin/env bash
set -euo pipefail

cargo set-version --bump "$1"

VERSION=$(sed -n 's/^version *= *"\(.*\)"/\1/p' Cargo.toml)

echo "Releasing v$VERSION"

git add Cargo.toml README.md Cargo.lock
git commit -m "chore(release): v$VERSION"
git tag "v$VERSION"
git push
git push --tags
cargo publish
