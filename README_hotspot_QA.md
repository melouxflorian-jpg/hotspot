# CLERC - Inter-Annotator QA Timeline and Hotspots

This tool checks **inter-annotator agreement** for one video and generates an **industrial-style QA timeline**:

- one horizontal row per annotator
- colored bars for each segment
- green vertical bands for agreement zones
- red vertical bands for disagreement hotspots
- reliability score + per-annotator timing deltas

Main script:

`./analyze_hotspot_visual.py`

---

## 1. Expected input formats

For one video, provide **N CSV files** (N >= 2), one per annotator.

### Format A (standard)

```csv
segment_id,start,end,gloss
1,0.35,0.92,YOUR
2,0.95,1.10,DAD
3,1.40,2.00,SMALL
```

Required logical columns:

- `start` (seconds, float)
- `end` (seconds, float)
- `gloss` (string)

### Format B (ELAN-wide export)

Also supported. The loader auto-detects the wide format with rows like:

- `Category`
- `start`
- `End`
- `Labels`

Only the `Gloss` tier is extracted for analysis.

---

## 2. What the script computes

For one video:

1. Loads all annotator files.
2. Builds a time grid (`dt`, default `0.05s`).
3. Computes agreement/disagreement over time.
4. Detects disagreement hotspots.
5. Computes reliability score and industrial gate.
6. Computes per-annotator timing deltas vs consensus median.
7. Renders `timeline_<video_id>.png`.

### Consensus and deltas

For each comparable segment index:

- `consensus_start = median(starts across annotators)`
- `consensus_end = median(ends across annotators)`
- `consensus_dur = consensus_end - consensus_start`

Per annotator:

- `delta_start = |start_annotator - consensus_start|`
- `delta_end = |end_annotator - consensus_end|`
- `delta_dur = |duration_annotator - consensus_dur|`

Aggregated metrics include:

- `avg_delta_start_s`, `min_delta_start_s`, `max_delta_start_s`
- `avg_delta_end_s`, `min_delta_end_s`, `max_delta_end_s`
- `avg_delta_dur_s`, `min_delta_dur_s`, `max_delta_dur_s`

Figure footer displays these in **milliseconds**.

---

## 3. Installation

```bash
cd /path/to/handoff_patrick_hotspot_qa
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If needed, minimum deps:

```bash
pip install pandas numpy matplotlib
```

---

## 4. Run

### Example (3 annotators, Hotspot Test 1 files)

```bash
python3 ./analyze_hotspot_visual.py \
  --video_id HOTSPOT_TEST_1 \
  --csv "./data/Hotspot Test 1 - Annotator 1.csv" "./data/Hotspot Test 1 - Annotator 2.csv" "./data/Hotspot Test 1 - Annotator 3.csv" \
  --annotators ANNOTATOR_1 ANNOTATOR_2 ANNOTATOR_3 \
  --video_duration 30.2 \
  --output_dir .
```

### CLI options (current script)

- `--video_id` required
- `--csv` required (one or more files)
- `--annotators` required (same order as `--csv`)
- `--dt` optional (default `0.05`)
- `--min_overlap` optional (default `2`)
- `--min_hotspot_duration` optional (default `0.10`)
- `--video_duration` optional: force full timeline extent from start to end
- `--timeline_start` optional (default `0.0`), used with `--video_duration`
- `--output_dir` optional (default `.`)

---

## 5. Figure interpretation

- Green bands: agreement zones.
- Red bands: hotspots (disagreement).
- If no hotspots: explicit green message indicates perfect agreement over comparable time.
- Footer includes:
  - global stats (coverage/agreement/disagreement/hotspots/score)
  - consensus explanation
  - per-annotator deltas in ms

Typical per-annotator line:

`FLORIAN: n=3, dur=1.20 s | avg Δstart=33 ms (20-50) | avg Δend=67 ms (40-90) | avg Δdur=33 ms (15-50)`

---

## 6. Intro message for Patrick

Subject: New inter-annotator QA tool (timeline + hotspots)

Hi Patrick,

I'm sending you a small QA utility we've been iterating on:
`analyze_hotspot_visual.py` + industrial timeline output.

Goal: for a given video and N annotators, it computes inter-annotator agreement, detects disagreement hotspots, and generates a CLERC-style QA timeline (one row per annotator, green agreement zones, red hotspots, score + timing deltas).

This README explains:

- expected CSV formats
- dependency setup
- run commands
- metric interpretation

Feel free to refactor/harden it. Once validated, we can use it as the standard CLERC overlap QA view.

Florian
