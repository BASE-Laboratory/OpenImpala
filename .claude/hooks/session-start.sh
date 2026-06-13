#!/bin/bash
# SessionStart hook: attribute commits made in Claude Code (especially the
# ephemeral "on the web" containers, which clone the repo fresh and have no
# git identity configured) to James Le Houx's GitHub account.
#
# Uses GitHub's noreply email, which reliably maps to the account for commit
# attribution. Scoped to this repo (--local) and idempotent.
set -euo pipefail

git config --local user.name "James Le Houx"
git config --local user.email "37665786+jameslehoux@users.noreply.github.com"

echo "git identity set: James Le Houx <37665786+jameslehoux@users.noreply.github.com>"
