#!/usr/bin/env bash
# Point git at the tracked hooks directory. Run once per clone.
set -e
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath .githooks
echo "hooks enabled: $(git config core.hooksPath)"
