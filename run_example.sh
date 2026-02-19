#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/analyze_hotspot_visual.py" \
  --video_id HOTSPOT_TEST_1 \
  --csv "$SCRIPT_DIR/data/Hotspot Test 1 - Annotator 1.csv" "$SCRIPT_DIR/data/Hotspot Test 1 - Annotator 2.csv" "$SCRIPT_DIR/data/Hotspot Test 1 - Annotator 3.csv" \
  --annotators ANNOTATOR_1 ANNOTATOR_2 ANNOTATOR_3 \
  --video_duration 30.2 \
  --output_dir "$SCRIPT_DIR"
