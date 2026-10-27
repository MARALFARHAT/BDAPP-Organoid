# Methods and interpretation

## Research question and implemented scope

The initial research question was whether a short neural recording could predict
organoid age and remaining functional lifespan, and whether environmental
measurements improve prediction. The available exports support an exploratory
experiment-level analysis and descriptive external datasets. Individual lifespan
prediction remains unsupported by the verified labels and recording metadata.

This document describes the current whole-project scope. The six-hour outputs
also contain stage-specific historical status messages, such as no one-hour model
or external data being present in that particular six-hour input ZIP. The current
project includes later hourly models and separate DANDI/Zenodo descriptive tracks;
the external tracks still do not validate lifespan or age prediction.

## FinalSpark extraction

FS369 and FS437 use pandas HDF5 event tables. The source fields are
`time_of_event`, `electrode`, and `max_voltage_uv`. The richer extractor processes
100,000 rows at a time, rejects invalid time/channel identifiers into the audit,
partitions valid rows by UTC clock hour, and sorts time within each channel and
hour. Each saved feature window is `[window_start, window_start + 1 hour)`;
the target is assessed at its end. These files had no invalid time/channel rows
and no hours triggering the memory guard.

The original record totals are 94,154,693 and 2,643,769. Counts reconcile between
the one-hour features and historical six-hour aggregates. The hourly grids contain
510 and 138 clock hours; 509 and 115 have exported events and are eligible for
the exploratory models. A grid hour without exported events has a missing event
count and unknown observation status. It is not converted to zero firing.

Feature groups are:

- **Counts:** log(1 + event count) and active observed channels.
- **Channel distribution:** natural-log entropy of channel event shares and the
  largest channel's share. This entropy is not normalized by the channel count.
- **Amplitude:** median and interquartile range of absolute `max_voltage_uv`.
  The source field is labeled microvolts; no waveform-based artifact rejection is done.
- **Timing:** pooled within-channel interval median, population-standard-deviation
  coefficient of variation, mean CV2 of adjacent intervals, and the fraction of
  channel intervals shorter than 10 ms. CV2 excludes adjacent pairs whose summed
  interval is zero. These are event-weighted channel summaries, not neuron statistics.
- **Bursts:** log(1 + channel-burst count) and the fraction of events in those bursts.
  A channel burst is a run of at least five detections connected by intervals at
  most 100 ms. Hour boundaries censor bursts. Bursts are not network synchrony.

Identical time/channel detections are counted as candidates and retained. No
electrical artifact removal, waveform sorting, neuron assignment, synchrony
analysis, or electrode-to-individual organoid assignment is performed. Missing
coverage and interruptions can affect all timing and burst measurements.

Environmental means are aggregated by clock hour only if one nonmissing incubator
identifier is observed for that source table. Temperature, CO2, and O2 remain
source-reported values; environmental associations are not causal effects.
Stimulation counts summarize exported log records. Zero logged records does not
establish absence of stimulation or a truly spontaneous hour. The no-logged-trigger
sensitivity analysis has no pre/post-stimulation exclusion guard.

Continuous recording exposure has not been verified. Raw HDF5 snippets are
short event-triggered waveform segments, and their presence does not establish
continuous monitoring. Therefore exposure-normalized firing rate and verified
recording seconds remain missing in the FinalSpark output.

## Labels and evaluation

For assessment time t, the two labels in days are:

```
batch_calendar_age_days = (t - metadata["Batch Start"]) / 1 day
days_until_recorded_experiment_endpoint = (metadata["Dead Date"] - t) / 1 day
```

Window starts must follow the recorded start date, and the full window must end
by the recorded endpoint. Labels are recomputed from metadata before modeling.
The two targets sum to a constant within one experiment; their results are not
independent confirmations of biology.

Both metadata records report NS Batch 3 and the same batch creation timestamp,
with 12 neurospheres per aggregate record. There is no individual electrode map
or independently verified individual death outcome. Endpoint causes are
"Almost no activity" and "No activity", so a neural-activity endpoint task may
be circular. The source metadata's recorded Lifespan field begins at the recording
start; it is not the biological lifespan since batch creation.

Primary modeling trains on FS369 and evaluates FS437 (509 and 115 hourly windows).
Age ranges do not overlap, and source environmental conditions differ. Repeated
windows are dependent observations, and the two aggregate experiments do not
establish broad biological replication. FS437 has been inspected in earlier
analysis; this is exploratory evaluation, not untouched future confirmation.

The prespecified one-hour models are a training-target median, count-only Ridge,
rich neural Ridge, rich neural random forest, environment-only Ridge, neural plus
environment Ridge and forest, and neural/environment/stimulation Ridge. Five
additional neural Ridge ablations remove one feature group at a time from the
full neural set. These models exclude timestamps, experiment identifiers,
elapsed recording time, and either target from the predictors.

Ridge alpha is 10. Random forests use 200 trees, maximum depth 4, minimum leaf
size 5, seed 42, and one worker. Missing-value medians, missingness indicators,
and feature scaling are fitted inside the training-only pipeline. Scores are
computed without clipping negative predictions. Hyperparameters are fixed;
all configurations, ablations, and negative outcomes are retained.

Secondary hourly protocols are FS369-to-FS437 after excluding hours with exported
stimulation logs (487/66 windows), and an 80/20 chronological FS369 holdout with
a full one-hour gap (406/102 windows). These are sensitivity checks, not
independent biological replications. A change in error across different subsets
does not by itself show model improvement. No random mixing of experiment
windows, dependent-window bootstrap intervals, or population claims are used.

The historical six-hour comparison has 85/20 transfer windows and 67/17
chronological holdout windows with a six-hour gap. Its recording-clock diagnostic
uses elapsed recording time and is explicitly non-neural. Small internal clock
errors can follow from label arithmetic. They are not evidence of neural age
prediction. Six-hour and one-hour errors use different observations and sample
counts and are not a controlled comparison of recording duration.

## Main finding

All primary hourly FS369-to-FS437 R² values are negative. For example,
count-only Ridge has batch-calendar-age MAE about 83.55 days, versus about 98.09
days for the training median. The rich neural forest has recorded-endpoint MAE
about 2.82 days, versus about 7.40 days for the training median, but its R² is
about -7.99. All model scores are saved, not just these examples.

MAE is average absolute error in days. Negative R² means the squared error
exceeds that of a constant equal to the evaluation target's own mean. This
constant uses evaluation labels and is a diagnostic benchmark, not a deployable
training-only baseline. A model can improve MAE over the training median and
still have negative R², particularly with narrow target variation and dataset
shift. The results do not establish reliable age, health, survival, or causal
environmental prediction.

## DANDI: separate rat-culture descriptive analysis

Dandiset 002020 version 0.261004.1410 contains 129 available NWB recording sessions
from seven rat dissociated cortical cultures. The reader uses the actual custom
`processing/ecephys/detected_spikes` DynamicTable, not a hypothetical Units table.
Fields include `source_channel`, `source_frame_number`,
`time_from_first_spike_seconds`, and `amplitude_native`.

All 129 files were processed on the user's computer with zero skipped files,
yielding 130,035,509 detections. Counts, active source channels, nominal-duration
rates, amplitude summaries, channel interval statistics, and fixed-ISI channel
bursts are descriptive. Chunked calculations preserve channel interval and burst
state across chunk boundaries. DANDI's timing CV uses pooled channel interval
moments; it is not interpreted as single-neuron variability.

Times are relative to the first detected spike, not recording start. Initial
silence cannot be reconstructed. `nominal_duration_s` comes from converter
provenance, so count/duration is explicitly nominal, not independently verified
continuous exposure. Source channel mappings can remain unresolved. Amplitudes
retain native units, and detections remain unsorted channel events.

The source donor-age field `P18D` with gestational reference describes the donor
at dissection, not culture age. Culture days in vitro are unavailable in the
verified metadata. Days since each culture's first processed recording are a
recording timeline, not biological age. No age, remaining-life, or FinalSpark
transfer model is evaluated on these rat cultures.

## Zenodo: separate human forebrain organoid descriptive analysis

Zenodo record 22755004 supplies MATLAB v7.3 individual-organoid files. The reader
supports loose `fs*-org*.mat` files and files inside `Indiv_organoids.zip`.
Identical copies are deduplicated by organoid ID and SHA256; conflicting content
for one ID is flagged. It does not execute the published MATLAB scripts.

The user's run processed all 45 individual recordings from 16 experiment prefixes,
with 856,988 detections and no skipped files. Their hashes and features match the
independently tested public-source files. File-specific recording durations are
200, 400, 600, 1000, 1200, 1800, or 3600 seconds. These vary beyond the general
listing description; each file's own metadata determines its descriptive rates.
Continuous coverage has not been independently established.

For `fs417-org0`, all eight channels have amplitude arrays whose lengths do not
match their spike arrays. All 33,096 valid spike times remain included; those
amplitudes are withheld. The mismatch is explicitly reported, rather than
discarding the recording or pairing incompatible arrays.

MAT amplitude units are not independently verified, so features retain source
units. The shared burst definition above is a simple fixed-ISI implementation;
it does not reproduce the publication's complete criticality, synchrony, or
population-burst analyses. Experiment prefixes may identify related recordings.

Verified per-organoid age labels and independent death labels are absent from
these individual files. Published cohort ranges or medians are not assigned to
individual recordings. Biological-age and remaining-life columns remain missing.
This is descriptive activity analysis, not external validation of lifespan.

## Source and reproducibility notes

- FinalSpark evidence: metadata and README files in `data/BDAPP_model_inputs.zip`,
  plus the hourly extraction audits. Both original README copies describe FS369;
  FS437 acquisition conventions still need their own confirmation.
- DANDI source: https://dandiarchive.org/dandiset/002020/0.261004.1410
- DANDI metadata API: https://api.dandiarchive.org/api/dandisets/002020/versions/0.261004.1410/
- Zenodo source: https://zenodo.org/records/22755004
- Zenodo DOI: https://doi.org/10.5281/zenodo.22755004

The public-check provenance files identify verified DANDI samples and Zenodo
archive checksums. The supplied derived tables were generated from the user's
actual files; no synthetic observations are used in the scientific results.
Input and result hashes document snapshots, not acquisition validity. Raw files
remain on the user's PC. The optional source readers are included for reruns;
the default project command reproduces models from the frozen derived tables.

The unresolved next research requirements are independent culture age/death
labels, recording exposure and channel-to-organoid mapping, acquisition/artifact
review, and more independent cultures/batches. These are evidence gaps, not
additional performance claims that the present code can resolve.
