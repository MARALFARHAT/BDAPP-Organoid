# BDAPP: neural activity, culture age and recorded endpoints

Maral Farhatnejad, student 23012856, Sejong University. Professor: Mugahed A. Al-antari.

The project builds a Python pipeline to process large neural-event tables, extract activity features, compare regression models, and audit saved predictions. The final-project goal is to investigate age estimation from short neural recordings and the possible value of environmental measurements. Remaining functional lifespan requires individual follow-up and independently defined endpoint labels.

## Start here

| Material | Location |
| --- | --- |
| Supported local analysis and instructions | [BDAPP_complete_code](BDAPP_complete_code/README.md) |
| Separate labeled culture-time benchmark | [age_extension](BDAPP_complete_code/age_extension/README.md) |
| Ten JPEG slides and spoken video script | [presentation](presentation/README.md) |
| 22 scientific figures, source tables, SVGs and PDF | [figure guide](presentation/figures/FIGURE_GUIDE.txt) |
| Measured Windows endpoint tolerance scores | [results CSV](presentation/results/finalspark_endpoint_tolerance_windows.csv) |

## Reproduce the analysis

Use the existing `fs437-local` environment, or install the requirements in a fresh environment. The small derived input archives are included. Large raw recordings are not required for these commands.

```powershell
conda activate fs437-local
cd BDAPP_complete_code
python .\run_project.py --check-only
python .\run_project.py
cd age_extension
python .\age_model_stage.py
```

Each model run creates a new timestamped results directory. Source readers for the original large recordings are documented in the complete-code README. The original notebook and earlier stage folders remain as historical material.

## Results and scope

- **FinalSpark:** train on 509 FS369 hourly observations and test on 115 FS437 observations. The activity forest predicts time until the recorded experiment endpoint with Windows MAE 2.822576 days, compared with the training-median MAE of 7.398430 days. This is 61.8% lower average absolute error. All primary transfer R² values remain negative.
- **Explicit tolerance score:** 84/115 Windows activity-forest endpoint predictions have absolute error at most two days (73.0%); 95/115 are within three days (82.6%). These are repeated hourly observations from one experiment, not independent organoids, and the tolerance scores do not establish lifespan accuracy or performance on future experiments.
- **Separate culture-time benchmark:** published five-minute summaries from organoid-derived dissociated cultures predict days post dissociation. The grouped forest has pooled MAE 6.596074 days and R² 0.301625, an 18.7% MAE reduction against the training-median baseline. This is not intact-organoid biological age.
- **DANDI and Zenodo:** separate descriptive analyses, not pooled into the prediction models.

Slide 8 uses the measured 73.0% within-two-days score. An earlier 87% scenario is retained only as a clearly labeled illustrative alternative, not a measured result. The 87.183763 value in the six-hour model output is MAE in days, not percent accuracy.

## Data provenance

FinalSpark source events: FS369 94,154,693; FS437 2,643,769. DANDI 002020 contributes 129 sessions from seven rat cortical cultures. Zenodo 22755004 contributes 45 human organoid recordings. The separate KCL benchmark contains 920 recording rows from 216 repeated wells, six donor cell lines and 15 line/batch combinations.

KCL dataset: https://doi.org/10.18742/30394168. Paper: https://doi.org/10.1016/j.crmeth.2026.101371. Other source and audit records are retained in the code and figure folders. Upstream material retains its source terms; this repository does not assign a new blanket license to third-party data.

Checks verify input hashes, labels, split bookkeeping and metric calculations. They do not establish biological validity. No recorded presentation video has been supplied yet.
