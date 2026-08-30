# CF-ESCI local human annotation interface

This is a usability layer over the frozen MVP 30.0 CSV templates. It does not regenerate samples, alter scientific fields, or expose machine answers.

## Start on Windows PowerShell

```powershell
cd C:\Projects\policyrank-cortex
python tools\cf_esci_annotation\server.py
```

Open `http://127.0.0.1:8765`.

Complete `CALIBRATION` first, selecting annotator A or B and the query or candidate-direction task. Calibration outputs remain marked and stored separately. Final annotation begins only after the guidelines are frozen. Annotator A selects only `ANNOTATOR_A_QUERY` and `ANNOTATOR_A_DIRECTION`; Annotator B selects only the corresponding B modes. A and B must work independently and must not inspect each other's completed files.

Use Save regularly. Existing compatible output files are loaded automatically so work can resume. Saves use a temporary file followed by an atomic replace and never modify the frozen source CSVs.

Answers are written under `data\cf_esci\human_validation_v1\completed\`:

- `calibration_annotations_A.csv`, `calibration_annotations_B.csv`
- `calibration_direction_annotations_A.csv`, `calibration_direction_annotations_B.csv`
- `annotator_A_query_annotations.csv`, `annotator_B_query_annotations.csv`
- `annotator_A_direction_annotations.csv`, `annotator_B_direction_annotations.csv`

Keyboard shortcuts: Alt+Left/Alt+Right move between examples; Ctrl+S saves.
