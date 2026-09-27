#!/usr/bin/env bash
# Push the ER pipeline code to GitHub (HTTPS through the gh login -- no SSH key needed).
#
#   bash push_repo.sh                 # normal: commit current code on top of existing history
#   FRESH=1 bash push_repo.sh         # clean one-commit history (use if old commits hold data/big files)
#   VISIBILITY=public bash push_repo.sh   # default is private (competition still running)
#
# Never pushes: the dataset, candidate parquets, model outputs, submissions, virtualenvs.
set -euo pipefail

REPO="${REPO:-FLagbusted/ness-ent}"
VISIBILITY="${VISIBILITY:-private}"
BRANCH=main
MAX_MB=95                                   # GitHub hard limit is 100 MB per file

say() { printf '\n== %s\n' "$*"; }

# ---------------------------------------------------------------- auth: gh -> git over HTTPS
say "checking gh login"
gh auth status >/dev/null 2>&1 || { echo "not logged in: run  gh auth login  (choose HTTPS)"; exit 1; }
gh auth setup-git                            # git uses gh's token for https://github.com

if ! gh repo view "$REPO" >/dev/null 2>&1; then
  say "creating $REPO ($VISIBILITY)"
  gh repo create "$REPO" --"$VISIBILITY" --description "Business entity resolution (S1 -> S2/S3) pipeline"
fi

[ -d .git ] || git init -q
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/$REPO.git"

# ---------------------------------------------------------------- what may never be committed
cat > .gitignore <<'EOF'
# competition data and everything derived from it
dataset/
*.parquet
*.tsv
*.npz
output*/
loco_*/
eda_out/
utils/
# model artifacts (rebuild with train_matcher.py; large)
matcher_*/
m*_val_scores.npz
# python / env
__pycache__/
*.pyc
.venv/
venv/
*.log
*.png
*.jpg
EOF

# challenge's own README is kept, renamed, so ours can be README.md
if [ -f README.md ] && [ -f README_pipeline.md ]; then
  [ -f CHALLENGE_README.md ] || git mv -f README.md CHALLENGE_README.md 2>/dev/null || mv README.md CHALLENGE_README.md
  mv README_pipeline.md README.md
fi

# ---------------------------------------------------------------- stage
if [ "${FRESH:-0}" = "1" ]; then
  say "FRESH=1: new history with a single commit"
  git checkout -q --orphan fresh_tmp
  git rm -rq --cached . >/dev/null 2>&1 || true
fi
git add -A
# drop anything that slipped past .gitignore but is too big or is data
git diff --cached --name-only -z | while IFS= read -r -d '' f; do
  [ -f "$f" ] || continue
  sz=$(( $(stat -c %s "$f") / 1048576 ))
  if [ "$sz" -ge "$MAX_MB" ]; then echo "  unstaging $f (${sz} MB)"; git reset -q -- "$f"; fi
done

say "files to be committed"
git diff --cached --stat | tail -40

# ---------------------------------------------------------------- history safety (normal mode)
if [ "${FRESH:-0}" != "1" ] && git rev-parse -q --verify HEAD >/dev/null; then
  bad=$(git log --all --name-only --format= | grep -E '^dataset/|\.parquet$|\.tsv$|^matcher_|^output' | sort -u | head -5 || true)
  big=$(git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
        | awk -v m=$((MAX_MB*1048576)) '$1=="blob" && $2>=m {print $3}' | head -5)
  if [ -n "$bad$big" ]; then
    echo "!! older commits contain data or >${MAX_MB}MB files:"; echo "$bad"; echo "$big"
    echo "   rerun with:  FRESH=1 bash push_repo.sh"; exit 1
  fi
fi

git -c user.name="${GIT_NAME:-$(git config user.name || echo FLagbusted)}" \
    -c user.email="${GIT_EMAIL:-$(git config user.email || echo FLagbusted@users.noreply.github.com)}" \
    commit -qm "${MSG:-ER pipeline: blocking v7, features v2 tokenization, stage-2 LightGBM, decision tuning}" \
  || echo "(nothing new to commit)"

if [ "${FRESH:-0}" = "1" ]; then
  git branch -D "$BRANCH" 2>/dev/null || true
  git branch -m "$BRANCH"
  git push -u --force origin "$BRANCH"
else
  git branch -M "$BRANCH"
  git push -u origin "$BRANCH"
fi
say "done: https://github.com/$REPO"
