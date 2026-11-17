# AI Use Disclosure

This statement discloses the AI assistance used in this YZM5110 research project, as the
course policy requires. It lists the tools that were used, what each one contributed, and
what was done without AI.

## Tools and their contributions

### OpenAI Codex (chat and sub-agents)

The exact model version was not recorded.

- **Project development:** help with choosing the topic, explaining the HAMMER/DGM4 code,
  writing the four-GPU launchers, designing the CRAF residual head, debugging, drafting
  experiment commands and interpreting the results.
- **Sub-agents:** packaging the GPU code; preparing the literature matrix and reading notes
  from primary sources; drafting the first Turkish version of the report; independent
  consistency checks.
- **Coordinating agent:** integrating files and evidence, numerical checks, CPU tests,
  building the first report PDF and verifying the package.
- **Server records (13 November 2026):** after the author brought two working folders back
  from the server, the agents analysed the source-code hashes, the six run/history records,
  the three test prediction files and the cache manifests. They then updated the report
  accordingly. The raw folders were not modified.
- **HAMMER training record:** the four original files supplied by the author (configuration,
  run arguments, the 11-epoch log and the test metrics) were archived and incorporated into
  the report after 32 consistency checks.
- **CRAF technical note:** a document explaining the v1/v2 mechanism, with excerpts from the
  actual source code, a worked numerical example and synthetic usage code. The example was
  checked on CPU; this check is not a model training run or a GPU benchmark.

### Claude (Anthropic, Claude Opus 5.5 in the Claude Code desktop app)

Used on 17 November 2026.

- **Source verification:** re-read every published number cited in the report from the
  primary sources (arXiv full texts of R1–R6 and the authors' repositories). The results are
  recorded in `docs/EXTERNAL_SOURCE_VERIFICATION.md`.
- **Additional analysis scripts:**
  - `scripts/bootstrap_uncertainty.py` → `results/bootstrap_uncertainty.json`
  - `scripts/calibration_analysis.py` → `results/calibration_analysis.json`
  - `scripts/check_report_numbers.py` → `results/report_number_check.json`

  These scripts compute bootstrap intervals, calibration and confident-error statistics
  from the stored prediction files, and check every measured number quoted in the report
  against the original run files. No model was re-run.
- **Report:** restructured and rewrote the English and Turkish versions of the report
  (`reports/build/`), produced the figures, the PDF files and the editable Word files.
- **Presentation:** wrote the English and Turkish five-minute presentation slides and the
  speaker notes.
- **Repository check:** compared the public repository with the report and listed the files
  that were missing or inconsistent.

### Other tools

- **Web search:** used to locate paper titles, versions, full texts and official code
  repositories. The literature documents state whether a paper was read in full or only
  at the abstract level.
- **Python and PDF tools:** used to verify the results supplied by the author, regenerate
  the tables and build the report.

## Work done without AI

- All GPU training and evaluation (HAMMER re-training and every CRAF run) was carried out
  by the author on the author's own 4 × RTX A6000 server.
- No AI tool connected to that server, loaded a checkpoint or ran a new training job.
- Small CPU validation tests were run separately and are reported as such. Server results
  are never presented as new runs made by an AI tool.

## Responsibility

The author remains responsible for:

- reading the cited sources;
- understanding the implementation, the loss functions, the feature flow and the data
  splits;
- checking every citation and number;
- explaining the deviations from the original study and the negative results;
- defending the conclusions in their own words.

No claim in the report is assumed to be correct because an AI tool produced it. The project
is individual work: the sub-agents are software tools, and no code, outputs or report text
were shared with or divided among other students.
