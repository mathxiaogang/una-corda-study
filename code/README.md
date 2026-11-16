# Una corda study — reproducible analysis package

Working title: *The Quiet Pedal Comes Late: Una Corda Use Separates Adult from Adolescent Elite Pianists across 1,276 Competition Performances*

Data: MAESTRO v3.0.0 (MIDI only) plus the performer table from the PID-CNN repository. MAESTRO is licensed CC BY-NC-SA 4.0, so derived files here carry the same licence.

## Run order

```bash
# 1. clean metadata (names, nationality, composer fixes, group, round, advancement, laureates, repeat entrants)
python -I code/01_clean_metadata.py <maestro-v3.0.0_performer.csv> data
# 2. per-performance pedal and dynamics features from the MIDI files
python -I code/02_extract_features.py <maestro-v3.0.0 folder with year subfolders> data/metadata_clean.csv data
# 3. all statistics (permutation tests, bootstrap, REML mixed model, robustness, H1b, RQ4, supplements)
python -I code/03_analysis.py data results
# 4. publication figures (PNG 300 dpi + PDF)
python -I code/04_figures.py data results figures/exploratory
# 5. optional: independent re-computation of the headline numbers
python -I code/05_verify_headline_numbers.py data
# 6. discovery follow-ups: prevalence, individuality (variance shares, split-half), event-locked velocity, stability
python -I code/06_discovery_followups.py <maestro-v3.0.0 folder> data results
# 7. "other pianists as the score": DTW-align performances of the same piece, consensus loudness,
#    placement agreement (kappa), una corda by loudness quartile, who decides whether (~5 min)
python -I -u code/07_cross_performance_alignment.py <maestro-v3.0.0 folder> data results save_frames=1
#    sensitivity runs (same script, results into results/sensitivity):
#    fps=5 tag=fps5 | fps=20 tag=fps20 | band=0.08 tag=band08 | band=0.20 tag=band20 | ref=longest tag=reflongest
# 8. FINAL analyses for the paper (H1a-H1c, H2a-H2c, robustness, exploratory, Holm) -> results/final/
python -I -u code/08b_h2a_bootstrap_worker.py data results seed=1 B=30   # run seed=1 and seed=2 in parallel
python -I -u code/08_final_analyses.py <maestro-v3.0.0 folder> data results
python -I code/08c_threshold_checks.py data results
# 9. FINAL publication figures Fig1-Fig6 -> figures/
python -I code/09_final_figures.py data results figures
# 10. independent re-computation of headline numbers (7 checks)
python -I code/10_verify_final_numbers.py data results
# 11. novelty extensions: shared rule vs personal threshold, within-edition signature test, style independence
python -I -u code/11_novelty_extensions.py data results
# 12. revision analyses requested in peer review (R1-R7); run parts separately, then merge
for P in R1 R2 R3 R4 R5 R6 R7; do python -I -u code/12_revision_analyses.py <maestro-v3.0.0 folder> data results part=$P; done
python -I code/12_revision_analyses.py x data results part=merge
# 13. revised figures (Figure1-Figure7, numbered as in the revised manuscript)
python -I code/13_revision_figures.py data results figures/revision
# 14. second-round revision analyses (S1-S5: pedal-purged consensus, model-implied agreement, trait vs edition,
#     threshold vs extent, event-locked consensus) -> results/revision2/
python -I -u code/14_revision2_analyses.py data results
# 15. figures of revision 2 (Figure 3 with panel B) -> figures/revision2/
python -I code/15_revision2_figures.py data results figures/revision2
# 16. completes Table 4 of the submission (presses per minute, velocity range in subsamples) -> results/submission/
python -I code/16_submission_extras.py data results
# 17. figures of the Music Perception submission (Figure S1 labels use 'piece') -> figures/submission/
python -I code/17_submission_figures.py data results figures/submission
```

## Paper

*Commonality in Placement, Individuality in Extent: A Corpus Study of Una Corda Use in 1,276 Expert Piano Performances*

Steps 1-6 produced the exploratory work that led to this design; steps 7-9 are the analyses reported in the paper.
`results/final/final_summary.json` holds every number quoted in the results.

Requirements: Python 3.10+, numpy, pandas, scipy, matplotlib. No MIDI library is needed (a small parser is included).
Full run time is under one minute on a laptop.

## Contents

| Path | What it is |
| --- | --- |
| `data/metadata_clean.csv` | 1,276 rows: cleaned pianist names (201 pianists), group (junior/senior), round, advanced to R2, top-3 laureate, repeat-entrant type, normalised nationality, corrected composer, style period |
| `data/features.csv` | Per-performance features: una corda (time share at thresholds 32/64/96, share of notes, presses per minute, episode length, velocity under/without una corda), sustain pedal (time share, partial depth, presses per minute, press lag, syncopated share), sostenuto, velocity and density |
| `data/cc67_raw_by_year.csv` | Raw soft-pedal controller value counts per edition (equipment audit) |
| `results/table1_sample.csv` | Sample description by group |
| `results/h1a_by_edition.csv` | Edition medians and composer-adjusted means |
| `results/h1a_lmm.csv` | Crossed random-effects REML mixed models (three outcomes) |
| `results/specificity_standardized.csv` | Composer-adjusted standardised senior–junior differences for 14 features |
| `results/robustness_h1a.csv`, `robust_leave_one_edition_out.csv` | Robustness checks |
| `results/h1b_*.csv` | Repeat entrants by edition and their changes |
| `results/rq4_advancement.csv` | Una corda use vs advancement to round 2 |
| `results/supp_era_by_group.csv`, `supp_bach.csv` | Style period and Bach summaries |
| `results/equipment_audit.csv` | Sensor value audit per edition |
| `results/summary.json`, `analysis_log.txt` | Key numbers and the full printed log |
| `figures/Fig1–Fig4, FigS1` | Publication figures (English labels) |

## Key design decisions

- Group (junior/senior) is constant within an edition, so the edition is the unit of inference for the group effect.
  The pre-specified primary test is an exact permutation of group labels across the 10 editions (252 splits);
  the smallest attainable two-sided p is 0.008.
- Each performance is first adjusted by its composer's mean, because the two groups play different repertoire.
- The mixed model uses crossed random intercepts for pianist, edition and composer (REML, dense Cholesky,
  implemented in `03_analysis.py` because statsmodels was unavailable); the group effect is tested with
  edition-level df = 8.
- Name fixes: `ean-Selim Abdelmoula` → Jean-Selim Abdelmoula, `u Yeon Kim` → Su Yeon Kim, `Frank Duepree` → Frank Dupree,
  `s` → Tristan Teo (restores R2 ⊆ R1 nesting in 2011; an inference, flagged as such).
- Composer fixes: *Barcarolle, Op. 60* → Chopin; *Variations and Fugue Op. 35 "Eroica"* → Beethoven.

## Not done here (needs data this environment could not download)

- Score-level control with ASAP (dynamic markings and una corda indications).
- Out-of-sample replication with competition years not in MAESTRO.

## Revision 1

`paper_src_R1/` and `review_src/` hold the sources of the revised manuscript (`una_corda_paper_R1.html`) and the review report (`una_corda_review_report.html`); `results/revision/revision_summary.json` holds every new number.

## Revision 2

`paper_src_R2/` and `review2_src/` hold the sources of `una_corda_paper_R2.html` and `una_corda_review_report_R2.html`; `results/revision2/revision2_summary.json` holds every new number of the second round.

## Revision 3

`paper_src_R3/` holds the sources of `una_corda_paper_R3.html` (repositioned around the una corda vs sustain contrast, Bernays & Traube 2014 added, four must-fix items of the full-text check resolved). No new analyses; figures from `figures/revision2/`.

## Music Perception submission

`submission_src/` holds the APA-formatted sources and build script of `una_corda_MusicPerception_submission.html` (title page with author note and open-practices statement, abstract and keywords, main text, references, supplementary material S1-S6). Upload `figures/submission/Figure1-6` (and `Figure7` as Figure S1) as separate high-resolution files.
