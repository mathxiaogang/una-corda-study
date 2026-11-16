# Una corda analysis code

Analysis code for the study on soft-pedal (una corda) use in elite piano
competition performances, based on the MAESTRO v3.0.0 MIDI corpus.

- Title in the manuscript at the time of packaging (submission folder):
  *Commonality in Placement, Individuality in Extent: Una Corda Versus Sustain
  Pedalling* (Music Perception submission).
- Title recorded in `code/README.md`:
  *Commonality in Placement, Individuality in Extent: A Corpus Study of Una Corda
  Use in 1,276 Expert Piano Performances*.

The two titles differ; the manuscript title is the one to use.

## What this repository contains

`code/` — the complete analysis pipeline, 18 numbered scripts plus the run order
and design notes in `code/README.md`:

| stage | scripts | what they do |
| --- | --- | --- |
| data preparation | 01, 02 | clean MAESTRO metadata; extract pedal and dynamics features per performance |
| statistics | 03, 05, 06 | permutation tests, bootstrap, REML mixed models, robustness, follow-ups |
| cross-performance | 07 | DTW alignment of performances of the same piece; placement agreement |
| paper analyses | 08, 08b, 08c | final hypothesis tests (H1a–H1c, H2a–H2c), threshold checks |
| figures | 04, 09, 13, 15, 17, 18 | exploratory, final, revision and submission figures |
| verification | 05, 10 | independent re-computation of headline numbers |
| extensions | 11, 12, 14, 16 | novelty extensions, first- and second-round review analyses, submission extras |

Run order, expected inputs and outputs are documented in `code/README.md`. The
commands there are written to be run **from the repository root**, e.g.

```bash
python -I code/01_clean_metadata.py <maestro-v3.0.0_performer.csv> data
```

## What this repository does not contain

- **The data.** MAESTRO v3.0.0 (MIDI) and the performer table from the PID-CNN
  repository are not redistributed here. MAESTRO is licensed CC BY-NC-SA 4.0;
  the analysis package notes that derived files carry the same licence.
- **Derived outputs.** The `data/`, `results/` and `figures/` directories named in
  `code/README.md` are produced by the scripts; they are not part of this
  repository.
- **Manuscript sources.** `paper_src_R*`, `submission_src/` and `review*_src/`
  mentioned in `code/README.md` live with the submission files and are not
  included here.

## Requirements

Python 3.10 or newer, and `numpy`, `pandas`, `scipy`, `matplotlib`
(`requirements.txt`). No MIDI library is needed: the package contains a small
parser. The full pipeline runs in under a minute on a laptop
(`code/README.md`).

## Licence

The code here carries no explicit licence from the author yet. The MAESTRO data
and files derived from it are CC BY-NC-SA 4.0. Add a licence file before sharing
the repository further if one is wanted.

## Citation

> (authors to be added once the submission is final). *Commonality in Placement,
> Individuality in Extent: Una Corda Versus Sustain Pedalling*. Code:
> https://github.com/mathxiaogang/una-corda-study
