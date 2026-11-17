# Generating reports and tables

The editable report source is `reports/Kocak_YZM5110_Research.md`.
Update source details, the repository URL, and the student number there, then
regenerate the PDF. Server records are preserved in `results/server_records/`.
When adding records, update both the analyses and report interpretations;
the student should supplement the paper reading cards with their own critical reading.

```bash
python scripts/analyze_hammer_training.py
python scripts/analyze_server_results.py
python scripts/summarize_results.py
python scripts/measure_model_resources.py
```

HAMMER record analysis uses PyYAML; CRAF server analysis uses NumPy,
scikit-learn, and SciPy; result summarization uses the standard library;
model resource measurement uses PyTorch.
Outputs include `results/SUMMARY.md` and `results/model_resources.json`.
FP32 weight bytes are not a measurement of peak GPU memory; they do not
establish gradient/optimizer/cache memory usage or inference time.

PDF generation does not require GPU dependencies. You can use a separate report environment:

```bash
python -m pip install -r reports/requirements_report.txt
python scripts/build_report.py
```

DejaVu Sans/Mono fonts are required for Unicode Turkish and mathematical symbols.
On Linux, use the `fonts-dejavu-core` package; for fonts in another location, use
`python scripts/build_report.py --font-dir /absolute/path/to/fonts`.
The `--source` and `--output` options support alternative input/output locations.
PDFs use A4 pages, embedded fonts, repeated table headers, page numbers, and
clickable source links. After changing the source, visually check table row
breaks and Turkish characters again.

By default, the seven-column table in `docs/LITERATURE_MATRIX.md` is appended
as Appendix A on landscape A4 pages with continuous page numbering. Regenerate
the PDF when the matrix changes as well. `--without-matrix` produces only the
main text; keep the comparison matrix in the report submitted to the instructor.
The matrix source is now in English; existing PDFs retain their previous
content until regenerated.

When preparing the final package, render the report PDF to page images for
visual inspection. Record the verification scope and open issues in `qa/`.

To generate the CRAF technical explanation separately:

```bash
python scripts/build_report.py \
  --source docs/CRAF_TECHNICAL_EXPLANATION.md \
  --output reports/CRAF_Gelistirmesi_Teknik_Aciklama.pdf --without-matrix
```

Code blocks and source explanations are kept together across page breaks.
