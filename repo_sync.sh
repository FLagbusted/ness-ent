#!/usr/bin/env bash
# Put the solution code in a git repo and push it, so teammates can clone / pull.
# Re-run it any time: it commits whatever changed and pushes again.
#
#   first time : bash repo_sync.sh git@github.com:<user>/<repo>.git      (or an https URL)
#                bash repo_sync.sh --gh <repo-name>   (creates a PRIVATE repo with the gh CLI)
#   later      : bash repo_sync.sh                    (commit + push changes)
#   options    : --models  also commit matcher_v9/ and matcher_v10/ (model txt + meta; files >95MB skipped)
#                -m "msg"  commit message
#
# Never commits the dataset, parquet/tsv files, outputs or the venv (see .gitignore below):
# the competition data must not be published.
set -euo pipefail
cd "$(dirname "$0")"

REMOTE="" ; GH_NAME="" ; MODELS=0 ; MSG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --gh) GH_NAME="$2"; shift 2 ;;
    --models) MODELS=1; shift ;;
    -m) MSG="$2"; shift 2 ;;
    *) REMOTE="$1"; shift ;;
  esac
done

cat > .gitignore <<'EOF'
# data / large artifacts -- never pushed
dataset/
*.parquet
*.tsv
*.npz
*.zip
*.tar*
output*/
loco_*/
eda_out/
matcher_*/
scores*/
# env / cache
.venv/
venv/
__pycache__/
*.pyc
*.log
EOF

if [ ! -d .git ]; then
  git init -q
  git checkout -q -b main 2>/dev/null || true
  echo "initialized git repo in $(pwd)"
fi
git config user.name  >/dev/null || git config user.name  "$(whoami)"
git config user.email >/dev/null || git config user.email "$(whoami)@$(hostname)"

# code, configs, docs
git add .gitignore
for f in *.py *.sh *.md requirements.txt translit_dict.json ranker_*.json decision*.json dec_*.json; do
  [ -f "$f" ] && git add "$f"
done
[ -d utils ] && git add utils

if [ "$MODELS" = 1 ]; then
  for d in matcher_v9 matcher_v10; do
    [ -d "$d" ] || continue
    for f in "$d"/*.txt "$d"/*.json; do
      [ -f "$f" ] || continue
      sz=$(stat -c %s "$f")
      if [ "$sz" -gt 95000000 ]; then echo "SKIP $f ($((sz/1000000))MB > GitHub 100MB limit)"; continue; fi
      git add -f "$f"
    done
  done
fi

if git diff --cached --quiet; then
  echo "nothing new to commit"
else
  git commit -q -m "${MSG:-update $(date '+%Y-%m-%d %H:%M')}"
  echo "committed: $(git log -1 --oneline)"
  git show --stat --oneline HEAD | tail -n +2 | tail -25
fi

if [ -n "$GH_NAME" ]; then
  command -v gh >/dev/null || { echo "gh CLI not installed -- create the repo on github.com and pass its URL instead"; exit 1; }
  if ! git remote get-url origin >/dev/null 2>&1; then
    gh repo create "$GH_NAME" --private --source . --remote origin
  fi
elif [ -n "$REMOTE" ]; then
  if git remote get-url origin >/dev/null 2>&1; then git remote set-url origin "$REMOTE"; else git remote add origin "$REMOTE"; fi
fi

if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin main
  echo "pushed to $(git remote get-url origin)"
  echo "teammates: git clone $(git remote get-url origin)   |   later: git pull"
else
  echo "no remote yet: run again with a repo URL or --gh <name>"
fi
