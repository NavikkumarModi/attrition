#!/usr/bin/env bash
# Regenerate the terminal-recording GIFs embedded in the README's
# "Watch it run" section, from a live run of examples/10_live_race.py --
# not hand-edited, same regenerate-from-live-source philosophy as
# make_readme_images.py and paper/make_figures.py.
#
# Requires two external tools, neither a Python dependency of this package:
#   asciinema  https://asciinema.org           (brew install asciinema)
#   agg        https://github.com/asciinema/agg (brew install agg)
#
# Run from the repo root:  bash docs/make_casts.sh
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p docs/casts docs/images/casts

# scenario:speed pairs -- speed is seconds/round, tuned per scenario so the
# recording is watchable (fast for the 8-round synthetic one, faster still
# for the 40-round real-data ones so the GIF doesn't run long).
PAIRS=(
  "adaptive-therapy:0.35"
  "antibiotic-stewardship-real:0.06"
  "fisheries-commons-real:0.3"
  "exploit-catalog-real:0.05"
)

for pair in "${PAIRS[@]}"; do
  scenario="${pair%%:*}"
  speed="${pair##*:}"
  echo "recording ${scenario}..."
  asciinema rec --command "python examples/10_live_race.py ${scenario} --speed ${speed}" \
    --overwrite "docs/casts/${scenario}.cast"
  echo "converting ${scenario}..."
  agg "docs/casts/${scenario}.cast" "docs/images/casts/${scenario}.gif"
done

echo "done -- docs/casts/*.cast and docs/images/casts/*.gif regenerated"
