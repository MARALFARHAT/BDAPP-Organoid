# BDAPP culture-age prediction extension

Maral Farhatnejad · 23012856 · 27 October 2026

This extension adds a **separate age-prediction benchmark** using published
electrophysiology features from organoid-derived dissociated cultures. The target
is **days post dissociation (DPD)**. It is not intact-organoid biological maturation,
age since original organoid creation, or remaining functional lifespan. Each
source recording lasts five minutes; this is not the original one-hour task.

## Actual results

| Evaluation | Model | Mean absolute error, DPD | Pooled R² |
|---|---|---:|---:|
| Whole plate/batch groups held out | Training median | 8.11 | -0.078 |
| Whole plate/batch groups held out | Firing-rate Ridge | 7.97 | 0.052 |
| Whole plate/batch groups held out | Six-feature Ridge | 7.10 | 0.225 |
| Whole plate/batch groups held out | Six-feature forest | **6.60** | **0.302** |
| Unseen donor cell line, linked plates excluded from training | Training median | 8.11 | -0.078 |
| Unseen donor cell line, linked plates excluded from training | Firing-rate Ridge | 7.96 | 0.052 |
| Unseen donor cell line, linked plates excluded from training | Six-feature Ridge | 7.29 | 0.189 |
| Unseen donor cell line, linked plates excluded from training | Six-feature forest | **6.73** | **0.282** |

The primary forest reduces pooled MAE by about **18.7%** relative to the
training-median baseline. This is a modest predictive signal, not precise or
widely validated biological-age estimation. Positive R² is not a percentage of
prediction accuracy. All four models and all fold results are included.

The forest has substantial bias at the age extremes: average primary-test
predictions are about 15.90 DPD for actual 6-DPD recordings and 25.25 DPD for
actual 37-DPD recordings. Their average absolute errors are about 9.90 and
11.75 days, respectively. The overall 6.60-day MAE should not hide this limitation.
See `reference_results/age_specific_errors.csv`.

Restricting to untreated recordings gives 840 rows over 6, 15, 22, and 29 DPD.
On that different population, the forest MAE is 6.00 DPD and pooled R² is 0.254,
versus training-median MAE 7.41 DPD. These errors are not directly comparable
to the broader primary population, which includes 37-DPD vehicle controls.

## Source and labels

Source dataset: Pavlinek et al., King's College London,
**Electrophysiological development and functional plasticity in dissociated human
cerebral organoids across multiple cell lines — Well average data and metadata**.

- Dataset: https://doi.org/10.18742/30394168
- Metadata API: https://api.figshare.com/v2/articles/30394168
- Paper: https://doi.org/10.1016/j.crmeth.2026.101371
- Public paper text: https://pmc.ncbi.nlm.nih.gov/articles/PMC13107059/

The included source files were downloaded from the dataset's published file URLs
and matched against its supplied MD5 checksums. SHA256 snapshots and the public
source record are included. The source dataset lists CC0. The paper provides
the experimental context; cite both paper and dataset when presenting this work.

The source development-with-NAP table contains **920 recording/well rows, 216
repeated wells, six donor cell lines, 15 cell-line/batch combinations, and nine
physical plates**. Sampled ages are 6, 15, 22, 29, and 37 DPD. Multiple organoids
were pooled and dissociated before plating; rows and wells are not individual
intact organoids or independent differentiations.

All 920 record identifiers match `metadata.csv` one to one. The cell-line, batch,
and age fields agree between the two files. Plate number and DPD also agree with
the source recording identifiers. Metadata distinguishes untreated recordings
from vehicle controls; no active drug-treatment rows enter this benchmark.

The authors' `data/README.txt` contains RTF markup despite its .txt extension.
It is preserved unchanged. Our analysis uses the source definition of DPD rather
than adding an assumed 45-day organoid age.

## Reproduce locally

The model script, small source tables, frozen analysis plan, and already computed
results are all included. In your existing environment, from this folder:

```powershell
conda activate fs437-local
python .\age_model_stage.py
```

The default creates a fresh timestamped folder beneath `results`, prints all
pooled scores, and writes `Age_extension_results.zip`. `--output <new folder>`
selects a different fresh directory. Existing result directories are not
overwritten. The reference results were generated here; local library/platform
differences may produce small changes, particularly for forests. Each run records
the actual versions, source hash, plan hash, and validation status.

Dependencies: numpy, pandas, matplotlib, and scikit-learn, as already used in
the existing project. No additional large recordings are needed for this
published-feature benchmark. `requirements-reference.txt` documents the exact
reference runtime.

## Features and validation

Predictors are restricted to six published activity columns:

- Mean Firing Rate (Hz)
- Number of Bursts
- Area Under Normalized Cross-Correlation
- Number of Network Bursts
- NDmean
- sigEdgesMean

The script uses the last two source column names without assigning an unverified
new definition or unit. Firing rate and the two burst counts use log(1+x).
Age, filename, well ID, plate, cell line, and batch are excluded from predictors.
These are author-computed well summaries; the extension does not independently
re-extract them from raw signals or equate them with FinalSpark feature definitions.

The analysis plan was written and hashed before fitting these models. Ridge alpha
is 10. The forest has 200 trees, maximum depth 4, minimum leaf size 5, seed 42,
and one worker. Median imputation, missingness indicators, and scaling are fitted
inside each training fold. There is no search over hyperparameters, clipping,
selected-subset performance claim, or replacement of the prior FinalSpark results.

The primary protocol holds out each of eight connected plate/batch groups.
Plates sharing a cell-line/batch pair are linked; plates 41 and 42 stay together.
This excludes repeated wells, shared physical plates, and shared differentiation
batches across training and testing. It is more conservative than randomly
splitting the 920 repeated recording rows.

The secondary protocol holds out each of six donor cell lines. It also removes
all associated plate/batch groups from training, including other cell lines sharing
those plates. Evaluation includes only the held-out line. This is still
within-study cross-validation, not a prospective external experiment.

The untreated-only sensitivity protocol repeats grouped evaluation after excluding
vehicle controls. All protocols save every split, prediction, fold score, and
pooled score. Means across folds and pooled metrics are labeled separately. Checks
verify group separation, complete out-of-fold coverage, training-median predictions,
source hashes, labels, and reconciliation of saved metrics with saved predictions.

## Interpretation for the BDAPP project

The prior FinalSpark analysis showed poor age/endpoint transfer between two
aggregate experiments. This new dataset supports a narrower, explicitly labeled
culture-time task with a modest positive result under grouped evaluation.
Together they show why source labels and study design matter.

We can report: **Published neural-activity features predict days after dissociation
better than a training-median baseline in this dataset, with limited accuracy.**

We cannot report: reliable intact-organoid biological age, individual lifespan,
health diagnosis, a one-hour prediction system, or successful transfer to FinalSpark.
The five sampled ages, six donor lines, batch dependence, fewer late control wells,
source feature processing, and substantial early/late prediction bias limit the
claim. Remaining functional lifespan still requires appropriate longitudinal
follow-up and endpoint labels.
