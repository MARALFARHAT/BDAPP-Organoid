# BDAPP presentation materials

- `slides/`: ten 2560 x 1440 JPEG slides, in numerical order.
- `scripts/video_script.md`: the latest simple-English, slide-by-slide script with the final-project goal; approximately 6–7 minutes depending on pace.
- `figures/`: 22 PNG and SVG figures, a 22-page PDF overview, underlying tables, captions, numeric checks, and a Matplotlib generator.
- `results/finalspark_endpoint_tolerance_windows.csv`: measured endpoint tolerance scores from the Windows results already committed in this repository.
- `results/calculate_endpoint_tolerance.py`: reproduces those tolerance counts from saved predictions without fitting a model.

## Slide versions and speaking notes

The ten JPEGs preserve the presentation, with slide 8 updated to the measured Windows result used by the latest script: **84/115 endpoint predictions within two days (73.0%)** and **95/115 within three days (82.6%)**. These tolerance percentages are specific to the recorded experiment endpoint and the stated day limits.

`alternatives/BDAPP_slide_08_assumed_87_percent.jpeg` preserves the earlier clearly labeled hypothetical scenario. It is an assumed example, not a measured model score, and is not the primary slide 8.

`alternatives/BDAPP_slide_08_culture_age_benchmark.jpeg` preserves the earlier slide 8 with the separate KCL culture-time benchmark. Its primary forest MAE is 6.60 DPD and pooled R² is 0.302. Slides 9 and 10 refer to that separate benchmark as well as the FinalSpark work.

The FinalSpark endpoint is an experiment-level recorded endpoint. It is not an independently measured individual death. The original one-hour FinalSpark task and the five-minute dissociated-culture DPD benchmark have different targets and must stay separate.

The figure data use the user-reported Windows primary forest MAEs where stated. The complete-code `reference_results` are separate reference-runtime computations, with small platform differences. The tolerance CSV explicitly uses the existing Windows prediction file rather than rounding or guessing from MAE.

No video recording has been supplied. The slides and script are materials for recording it.

## Figure regeneration

With NumPy, pandas, Matplotlib and Pillow installed:

```bash
python presentation/figures/source/generate_figures.py
```

This recreates the scientific figure pack from its included source tables. Its conceptual schematic is labeled as an illustration.
