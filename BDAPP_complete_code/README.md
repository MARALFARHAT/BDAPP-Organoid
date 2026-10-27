# BDAPP: neural activity analysis and exploratory prediction

Prepared for Maral Farhatnejad · Student ID 23012856  
Big Data Analysis and Prediction Project · Sejong University  
Professor: Mugahed A. Al-antari · Code checkpoint: 27 October 2026

The implemented project processes neural event data, extracts activity features,
compares regression baselines, and audits the results. The source readers and
derived-data model pipeline are complete for this checkpoint. Reliable biological
age and remaining individual lifespan prediction have **not** been established.

## Run the complete derived-data analysis

Extract this folder into `F:\BDAPP-Organoid`, then run:

```powershell
conda activate fs437-local
cd F:\BDAPP-Organoid\BDAPP_complete_code
python .\run_project.py
```

The four derived input archives are included. This command fits the six-hour
baselines and one-hour models, checks their predictions and splits, and combines
the descriptive DANDI and Zenodo results. It does not rescan the large recordings.
It writes a fresh `results\run_<UTC timestamp>` directory containing:

- `PROJECT_RESULTS.txt`: readable results and interpretation.
- `project_checks.json`: data and evaluation consistency checks.
- `dataset_inventory.csv`: dataset units and event counts.
- `primary_hourly_results.csv`: all primary model comparisons.
- `six_hour_baselines` and `hourly_models`: full scores, predictions, figures,
  exclusions, feature shifts, and ablations.
- `external_descriptive_results`: the user's actual DANDI and Zenodo result ZIPs.
- `run_provenance.json`: software versions, source hashes, and input hashes.
- `BDAPP_project_results.zip`: the combined results for that run.

An existing output directory is never overwritten. `--output <new folder>` selects
a different destination. Paths in Python commands can contain spaces when quoted.

The included `reference_results` were recomputed here from the user's uploaded
derived inputs. They are not a copy of the user's local model run. Each model
stage records its software versions. Small forest differences between platforms
or dependency versions are possible; use the newly generated local results for
the local run. Input SHA256s identify the exact included tables.

To audit the reference results without training:

```powershell
python .\run_project.py --check-only
```

## What the data supports

| Track | Processed data | Role |
|---|---|---|
| FinalSpark FS369 | 94,154,693 events; 509 event-containing hours in a 510-hour grid | Source experiment for exploratory models |
| FinalSpark FS437 | 2,643,769 events; 115 event-containing hours in a 138-hour grid | Transfer evaluation after prior inspection |
| DANDI 002020 | 129 recording sessions; 7 rat cortical cultures; 130,035,509 detections | Separate descriptive analysis |
| Zenodo 22755004 | 45 individual-organoid recordings; 16 experiment prefixes; 856,988 detections | Separate descriptive analysis |

Hours and recording sessions are repeated observations, not independent biological
replicates. Missing event records remain unknown observation, not physiological
silence. DANDI and Zenodo are not pooled with the FinalSpark models.

The FinalSpark labels are calendar time since the reported batch creation date and
time to an activity-defined recorded experiment endpoint. The source metadata does
not provide independently verified deaths or electrode-to-individual mappings.
All primary one-hour transfer models have negative R². Some improve MAE over the
training-median baseline, but reliable cross-experiment prediction is not shown.

## Code map

| File | Purpose |
|---|---|
| `run_project.py` | One command to reproduce and check the included derived-data analysis |
| `src/project_checks.py` | Check hashes, record totals, labels, predictor lists, splits, and prediction metrics |
| `src/hourly_stage.py` | Read FinalSpark event exports in chunks and compute one-hour features |
| `src/hourly_models.py` | Compare one-hour models, temporal holdout, stimulation-log sensitivity, and feature ablations |
| `src/model_stage.py` | Reproduce the earlier six-hour baselines and recording-clock diagnostic |
| `src/run_finalspark.py` | General earlier FinalSpark aggregate reader and six-hour overview |
| `src/dandi_stage.py` | Read the actual custom NWB detected-spike tables in chunks |
| `src/zenodo_stage.py` | Read individual-organoid MATLAB v7.3 files, including files inside their ZIP |

`data` contains the unchanged user-uploaded derived archives. `provenance` holds
their hashes and earlier public-source checks. `archive/BDAPP-Organoid_original.zip`
preserves the initial uploaded project, including its earlier Colab notebook.
That notebook and the archive are historical material; neither is executed by
`run_project.py`. The current run commands above are the supported workflow.

## Optional raw-data extraction

Extraction already completed on the user's PC. It is not needed to run this
checkpoint. If source files change, use these commands to create **new** extracted
outputs and review the source audit before replacing the frozen input tables.
The large recordings are not included in this package.

```powershell
# Both FinalSpark experiments: one-hour extraction, using the original PC paths
python .\src\hourly_stage.py

# Override the two package locations when needed
python .\src\hourly_stage.py --fs369 "F:\FinalSpark\fs369data\fs369_package.hdf5" --fs437 "F:\FinalSpark\fs437\fs437_export\fs437_package.hdf5"

# DANDI: nested folders are searched recursively
python .\src\dandi_stage.py --root "F:\DANDI\002020-0.261004.1410"

# Zenodo: reads loose individual MAT files or Indiv_organoids.zip
python .\src\zenodo_stage.py --root "F:\DANDI\22755004"

# Earlier six-hour aggregate reader (each source can be read separately)
python .\src\run_finalspark.py --dataset FS369
python .\src\run_finalspark.py --dataset FS437
```

All readers support output-location arguments; run an individual script with
`--help`. The one-hour extractor temporarily partitions events by hour on disk,
needs about 2.35 GiB of working disk for the larger experiment, and applies a
5-million-events-per-hour memory guard. Inputs are opened read-only. Only
temporary partitions generated in that run are deleted after successful extraction.
No raw waveform dataset is opened.

The frozen six-hour archive includes the earlier FS437 aggregates and earlier
FS369 extraction. The optional general aggregate reader is supplied for source
inspection; its output is not automatically substituted for those historical inputs.
New input snapshots need their own manifest and scientific review.

## Dependencies and validation

The existing `fs437-local` environment has already run the individual stages.
For a fresh environment, `requirements.txt` lists the package ranges. The exact
reference runtime versions are in `requirements-reference.txt` and each result's
provenance. Model fitting does not need PyNWB: the actual NWB HDF5 tables are read
directly with h5py by the separate reader.

The full derived-data run was executed successfully before delivery. Its checks
independently recompute MAE, RMSE, and R² from all saved predictions, reconcile
hourly/six-hour event totals, check metadata-derived targets, check the hourly
train/test split and time embargo, and retain all 58 hourly and 28 six-hour score
rows. The Zenodo upload matches the independently tested public-file hashes and
features. These checks establish computational consistency, not biological validity.

Read `docs/METHODS_AND_LIMITATIONS.md` for the analysis definitions and scope.
`docs/EXPLAIN_THE_CODE.txt` provides a short spoken explanation of the implementation.
