# MoRAL-Score: A Novel Multi-Dimensional Metric for VLM Driving Reasoning

## Motivation

Existing VLM benchmarks (BLEU, action-match accuracy) fail to capture **reasoning quality** —
a model can get the right answer for the wrong reason, or produce excellent reasoning with
a slightly wrong number. MoRAL-Score decomposes evaluation into **independently measurable dimensions**
that together characterize how well a VLM reasons about driving scenes.

---

## MoRAL-Score Components

```
MoRAL-Score = weighted combination of 5 sub-scores:

  S₁: GSA  (Grounded Spatial Accuracy)     — 25%
  S₂: RSQ  (Reasoning Structure Quality)   — 20%
  S₃: ACT  (Action Decision Accuracy)      — 20%
  S₄: SEN  (Sensor Grounding Fidelity)     — 20%
  S₅: NOV  (Novel Detection Capability)    — 15%
```

### Why This Decomposition Is Novel

| Metric | Measures | Existing? |
|--------|----------|-----------|
| **GSA** | Numerical accuracy of distance/TTC/velocity claims | **No** — BLEU doesn't check numbers |
| **RSQ** | Whether reasoning follows structured chain-of-thought | **No** — just checks final answer |
| **ACT** | Correctness of driving action decision | Partially (action-match exists) |
| **SEN** | Whether model references real sensor observations | **No** — hallucination detection for driving |
| **NOV** | Model detects objects/risks missed by GT annotations | **No** — novel contribution |

---

## S₁: GSA — Grounded Spatial Accuracy (0–100)

**Per question type, extract the predicted number and compare to GT.**

### Scoring Thresholds

| Question Type | GT Field | Exact Match | Good (±threshold) | Partial | Score |
|--------------|----------|:---:|:---:|:---:|:---:|
| `spatial` | `distance_m` | ±10% | ±20% | ±50% | 100/75/40/0 |
| `safety` | `ttc_s` | ±15% | ±30% | ±50% | 100/75/40/0 |
| `velocity` | `velocity_ms` | ±15% | ±30% | ±50% | 100/75/40/0 |
| `physics` | `nearest_ahead_m` | ±10% | ±20% | ±50% | 100/75/40/0 |
| `counterfactual` | `t_impact_s` | ±20% | ±40% | ±60% | 100/75/40/0 |
| `trajectory` | `objects_entering_path` | exact | ±1 | ±2 | 100/75/40/0 |
| `near_miss` | `near_miss_count` | exact | ±1 | ±2 | 100/75/40/0 |
| `sensor_limit` | `gt_estimated_count` | exact | ±1 | ±2 | 100/75/40/0 |
| `ethical` | `min_dilemma_ttc_s` | ±20% | ±40% | — | 100/75/0 |
| `multi_conflict` | `top_risk_score` | ±15% | ±30% | ±50% | 100/75/40/0 |
| `planning` | `min_ttc_s` | ±15% | ±30% | ±50% | 100/75/40/0 |
| `occlusion` | — (qualitative) | — | — | — | LLM judge only |
| `gap` | — (qualitative) | — | — | — | LLM judge only |
| `zone` | — (qualitative) | — | — | — | LLM judge only |

**Why these thresholds:**
- Distance ±20%: At 10m, 8-12m is acceptable (1 car length). At 50m, 40-60m is reasonable.
- TTC ±30%: At 2s TTC, 1.4-2.6s captures the danger level correctly.
- Velocity ±30%: At 10m/s, 7-13m/s distinguishes "moving" from "stationary".
- Count metrics: exact or ±1 because these are small integers (0-5 typically).

### GSA Formula
```
GSA = (Σ per-sample GSA scores) / (N × 100) × 100

Per sample:
  if |predicted - gt| / |gt| ≤ threshold_exact → 100
  if |predicted - gt| / |gt| ≤ threshold_good  → 75
  if |predicted - gt| / |gt| ≤ threshold_partial → 40
  else → 0
  
For qualitative questions (occlusion/gap/zone): use LLM judge score × 20
```

---

## S₂: RSQ — Reasoning Structure Quality (0–100)

**Measures whether the model follows structured chain-of-thought reasoning.**

| Component | Points | How to Score |
|-----------|:---:|-------------|
| `[BEV]` section present and relevant | 25 | Programmatic: regex check |
| `[CAM]` section present and relevant | 25 | Programmatic: regex check |
| `[GT]` section present with numbers | 25 | Programmatic: regex + number extraction |
| `[DECISION]` section with clear action | 25 | Programmatic: regex + action word check |

**Bonus (up to +10):**
- `[GT]` section references specific distances → +5
- `[DECISION]` references TTC explicitly → +5

**Programmatic implementation:**
```python
def rsq_score(pred_text):
    score = 0
    if re.search(r'\[BEV\]', pred_text): score += 25
    if re.search(r'\[CAM\]', pred_text): score += 25
    if re.search(r'\[GT\]', pred_text):
        score += 25
        # Bonus: GT section has numbers
        gt_section = re.split(r'\[DECISION\]', re.split(r'\[GT\]', pred_text)[-1])[0]
        if re.search(r'\d+\.?\d*\s*m', gt_section): score += 5
    if re.search(r'\[DECISION\]', pred_text):
        score += 25
        if re.search(r'TTC|time.to.collision', pred_text, re.I): score += 5
    return min(100, score)
```

---

## S₃: ACT — Action Decision Accuracy (0–100)

| Match Level | Score | Definition |
|-------------|:---:|-----------|
| Exact match | 100 | `pred_action == gt_action` (e.g., EMERGENCY_BRAKE = EMERGENCY_BRAKE) |
| Family match | 75 | Same family (BRAKE ≈ EMERGENCY_BRAKE ≈ HARD_BRAKE) |
| Direction match | 40 | Same direction (any brake action vs any accelerate) |
| Wrong | 0 | Different action family |
| No GT action | null | Question doesn't have a GT action — skip |

**Action families:**
```
BRAKE: {BRAKE, EMERGENCY_BRAKE, HARD_BRAKE, STOP}
MAINTAIN: {MAINTAIN, CONTINUE, CRUISE}
ACCELERATE: {ACCELERATE, SPEED_UP}
YIELD: {YIELD, WAIT, SLOW_DOWN}
EVADE: {SWERVE, LANE_CHANGE, MERGE}
```

---

## S₄: SEN — Sensor Grounding Fidelity (0–100)

**Measures whether the model's visual claims match reality.** Scored by LLM judge.

The LLM judge evaluates:
1. Does the `[BEV]` section describe objects actually visible in the BEV map?
2. Does the `[CAM]` section describe what the camera actually shows?
3. Are invented/hallucinated objects penalized?
4. Are spatial references (range rings, quadrants, colors) correct?

**LLM judge prompt addition:**
```
Rate SENSOR_GROUNDING 1-5 and we normalize to 0-100:
  5 = 100: All visual claims verified, no hallucination
  4 = 80: Mostly accurate, minor visual errors
  3 = 60: Mixed correct and fabricated
  2 = 40: Mostly fabricated visual descriptions  
  1 = 20: Pure hallucination
```

---

## S₅: NOV — Novel Detection Capability (0–100)

**This is the unique thesis contribution metric.**

**Concept:** Check if the model mentions objects, risks, or spatial details
that are NOT in the GT annotations but ARE plausible given the scene.

### How to Score NOV

**Step 1: Extract mentioned objects from model output**
```python
# Parse prediction text for object mentions
OBJECT_CLASSES = {'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle',
                  'pedestrian', 'traffic_cone', 'barrier', 'construction_vehicle'}

def extract_mentioned_objects(pred_text):
    """Extract all unique object mentions with approximate positions."""
    mentions = []
    for cls in OBJECT_CLASSES:
        # Find all mentions of this class with nearby distance
        pattern = rf'{cls}.*?(\d+\.?\d*)\s*m'
        for m in re.finditer(pattern, pred_text, re.I):
            mentions.append({
                'class': cls,
                'distance_m': float(m.group(1)),
                'text_span': m.group(0)[:80]
            })
    return mentions
```

**Step 2: Compare against GT detections**
```python
def find_novel_detections(mentioned_objects, gt_detections, match_radius=5.0):
    """Find objects mentioned by model but not in GT."""
    novel = []
    for mention in mentioned_objects:
        matched = False
        for gt in gt_detections:
            if (mention['class'] == gt['class'] and
                abs(mention['distance_m'] - gt['distance_m']) < match_radius):
                matched = True
                break
        if not matched:
            novel.append(mention)
    return novel
```

**Step 3: Score**
- Novel mentions that are plausible (LLM judge confirms): +20 per novel detection (capped at 100)
- Novel mentions that are hallucinations: -10 per hallucination
- No novel detections: 50 (neutral — not penalized, not rewarded)

### Why NOV Matters for Your Thesis

If your fine-tuned model detects a pedestrian at 15m that the GT annotator missed
(e.g., partially occluded, at scene boundary), that's **stronger than GT** — your model
with BEV+radar reasoning found something a human reviewer didn't.

This directly supports your thesis claim that multi-modal reasoning chains improve
safety-critical perception.

---

## Combined MoRAL-Score

```
MoRAL-Score = 0.25 × GSA + 0.20 × RSQ + 0.20 × ACT + 0.20 × SEN + 0.15 × NOV

Range: 0–100
```

### Interpretation Scale

| Range | Label | Meaning |
|-------|-------|---------|
| 80–100 | Excellent | Near-human spatial reasoning |
| 60–79 | Good | Reliable for advisory systems |
| 40–59 | Fair | Useful with supervision |
| 20–39 | Poor | Major deficiencies |
| 0–19 | Failing | Not usable |

### Per-Question-Type Expected Ranges

| qtype | What Makes It Hard | Expected Score Range |
|-------|-------------------|:---:|
| `spatial` | Must extract exact distance | 20–70 |
| `safety` | Must identify correct threat + TTC | 15–60 |
| `velocity` | Must distinguish moving vs static | 20–65 |
| `physics` | Must compute stopping distance | 10–50 |
| `planning` | Must choose correct action + reason | 15–55 |
| `occlusion` | Qualitative, subjective | 30–70 |
| `gap` | Binary safe/unsafe judgment | 30–75 |
| `zone` | Must map x,y to 8 directional zones | 20–65 |
| `trajectory` | Must predict future positions | 10–45 |
| `counterfactual` | Must project "what if" scenarios | 10–40 |
| `near_miss` | Must count events | 20–60 |
| `multi_conflict` | Must rank threats | 15–50 |
| `sensor_limit` | Must identify sensor blind spots | 15–55 |
| `ethical` | Must reason about dilemmas | 20–60 |

---

## Human-in-the-Loop (HIL) Evaluation

### Sample Size
- **30 samples minimum** for statistical significance per condition
- **Stratified sampling:** 2-3 samples per question type × 14 types = 28-42 samples per run
- Compare: 1 zero-shot run vs 1 fine-tuned run = **~60-84 total HIL evaluations**

### HIL Protocol

**Step 1:** Sample selection
```python
# Stratified sampling: 2 per qtype, balanced across scenes
import random
def sample_for_hil(results, n_per_qtype=2):
    by_qtype = defaultdict(list)
    for r in results:
        by_qtype[r['qtype']].append(r)
    selected = []
    for qtype, records in by_qtype.items():
        selected.extend(random.sample(records, min(n_per_qtype, len(records))))
    return selected
```

**Step 2:** Blind presentation (human doesn't know which model produced the output)

**Step 3:** Human rates each dimension 1-5

**Step 4:** Compute inter-rater agreement (if >1 human) using Cohen's κ

### HIL Evaluation Sheet Format

For each sample, the human evaluator fills in:

```
Scene: scene-0061  |  Question Type: spatial  |  Sample ID: 037

QUESTION:
"What is the closest object directly in the vehicle's forward path
 and how far away is it?"

MODEL OUTPUT (anonymized):
"The closest object is a truck at approximately 17 meters ahead..."

GROUND TRUTH:
  Distance: 16.84m | Object: truck | TTC: 1.85s

Rate each dimension (1-5, or N/A):

[ ] Spatial Accuracy:     ___  (Are distances/positions correct?)
[ ] Reasoning Quality:    ___  (Is reasoning structured and logical?)
[ ] Action Correctness:   ___  (Is the recommended action safe?)
[ ] Sensor Grounding:     ___  (Does it reference real BEV/camera content?)
[ ] Physical Plausibility:___  (Are physics claims reasonable?)
[ ] Novel Observations:   ___  (Does it notice anything GT missed?)

Free text notes: ________________________________________

Does this answer identify objects NOT in the ground truth?
[ ] Yes — describe: ____________________________________
[ ] No
```

---

## Condition Thresholds Summary

| Condition | Label | Sensor Input | What It Tests |
|-----------|-------|-------------|---------------|
| A | Camera only | CAM_FRONT | Baseline — no spatial augmentation |
| B | LiDAR BEV | BEV (GT boxes) + CAM | Does BEV geometry help reasoning? |
| D | LiDAR+Radar BEV | BEV (GT boxes + radar) + CAM | Does velocity info help? |
| Clean BEV variants | | LiDAR/Radar point cloud only | Realistic sensor input |

**Evaluation thresholds for each condition:**
- Compare across conditions using **same model** to isolate sensor effect
- Compare across models using **same condition** to isolate fine-tuning effect
- The threshold for "improvement" is **≥5 MoRAL-Score points** between conditions
