# Evaluation protocol

Pre-registered 2026-10-30, before any model is trained or threshold is tuned.
Anything changed after the test split has been scored goes in the amendments
section at the bottom with a date and a reason. Numbers from before an amendment
are not deleted.

## What is being decided

A driver applicant submits a selfie and a photo of a government ID (ID card or
passport). The system returns one of:

- **approve**: identity accepted, no human involved
- **reject**: refused, no human involved
- **review**: sent to a human reviewer

The point of the project is to send fewer cases to review without loosening the
security constraints below.

## Data and splits

| Component | Data | Split unit |
|---|---|---|
| Face verification | Meta Casual Conversations v1 | subject |
| Document localization, type, OCR | MIDV-2020 | document instance |
| Selfie PAD | CelebA-Spoof | subject (official split) |
| PAD, cross-device | self-captured, consenting participants | participant |

- Splits are train / dev / test and subject-disjoint. Split files (subject IDs)
  and the seed that made them are committed under `data/splits/` before any
  model is fit on that data. Proportions are recorded alongside the split files.
- Dev is used for every choice: thresholds, calibration, rules-baseline tuning,
  model selection.
- Test is scored once per frozen model version. Every test run is logged in
  MLflow, including runs that were thrown away.
- ID-side face images come from a CC frame rendered onto MIDV-2020-style
  templates. Selfies come from *other* videos of the same subject.
- Impostor pairs are demographically matched (same gender label, same age band,
  same Fitzpatrick group) unless stated otherwise.

## Metrics

Score convention: higher means more likely genuine or bona fide; accept when
score >= threshold.

- **Face verification:** FAR, FRR, EER, FRR @ FAR = 0.1%.
- **PAD** (ISO/IEC 30107-3): APCER per attack species, APCER = worst species,
  BPCER, BPCER @ APCER = 1%. ACER is reported only for comparison with
  published CelebA-Spoof results.
- **Documents:** exact-match accuracy per field (name, date of birth, document
  number, expiry) after normalising case and whitespace; MRZ checksum pass rate;
  localization IoU.
- **Routing:** per-class outcome rates (below) and the overall review rate.
- **Latency:** P95 from the server receiving the POST to the decision being
  written, under the load test, models warm, at least 1,000 requests. The GPU
  model is recorded with the result.

## Security constraints

Fixed now, before tuning. Every routing policy, ours and the baseline, has to
meet all three on dev, and is then scored on test.

| Constraint | Limit |
|---|---|
| Impostor auto-approved | <= 0.1% of impostor attempts |
| Spoof auto-approved | <= 1% of spoof attempts |
| Genuine auto-rejected | <= 1% of genuine attempts |

These are per-class rates, so they do not depend on the traffic mix. On test we
report the point estimate and the 95% CI. A constraint counts as met on test if
the point estimate is within the limit. When the upper CI bound is above the
limit, the results table says so.

Sample size: with zero errors in n attempts the 95% upper bound is about 3/n,
so showing <= 0.1% needs at least 3,000 impostor attempts. Below that, the
impostor result is reported as inconclusive rather than as passing.

## Routing comparison (primary result)

- **Baseline:** hand-written rules over the raw component scores (match,
  liveness, quality, document checks), with thresholds tuned on dev under the
  same constraints.
- **Ours:** calibrated component scores feed a risk model with two thresholds
  (approve below t1, reject above t2, review in between), tuned on dev under the
  same constraints.
- **Primary outcome:** review rate under the reference traffic mix, ours vs
  baseline, as a relative reduction with a 95% CI.

Reference traffic mix:

| Class | Share |
|---|---|
| clean genuine | 88% |
| poor-quality genuine | 7% |
| impostor | 3% |
| spoof | 2% |

"Poor-quality genuine" means genuine attempts with seeded perturbations (see
Robustness) at severity 3 to 5.

Sensitivity sweep: impostor share in {1, 3, 5, 10}%, spoof share in {1, 2, 5}%,
poor-quality share in {3, 7, 15}%, with clean genuine making up the rest.
Thresholds are not re-tuned per mix.

## Uncertainty

- 95% percentile bootstrap, 2,000 resamples, seed 0.
- The resampling unit is the subject, not the image or pair. An impostor pair is
  assigned to its selfie subject.
- Every reported rate comes with its CI and its n.

## Robustness

Seeded perturbations at the fixed operating threshold, with no re-tuning:
exposure shift, motion blur, Gaussian blur, JPEG quality, downscaling, sensor
noise. Five severities each, applied to the selfie, the ID, or both. Casual
Conversations dark and bright lighting labels are also reported as is.

## Subgroups

- Groups: age band (18-30, 31-45, 46+), gender label, Fitzpatrick (I-II, III-IV,
  V-VI), lighting.
- One global threshold. No per-group thresholds.
- FAR within a group uses impostor pairs drawn from inside that group.
- A group with fewer than 100 subjects is listed as insufficient data, with no
  rate.
- A group is flagged if its FRR is above 1.25x the overall FRR and its CI
  excludes the overall value. Flagged groups go in Known limitations in the
  README.

## PAD protocol

- CelebA-Spoof official intra-dataset test.
- Leave-one-attack-type-out: train without one species, test on it.
- Self-captured set (2-3 phones, print and replay attacks), reported separately
  as cross-device. It is small, so it shows the trend and is not a headline
  number.
- Deepfakes and injection attacks are out of scope and are not claimed.

## Reporting rules

- Every number in `results/` and in README tables is generated by `make report`
  from MLflow run IDs. Nothing is typed in by hand.
- Each run logs its config, git SHA (clean tree required) and data manifest hash.

## Amendments

None yet.
