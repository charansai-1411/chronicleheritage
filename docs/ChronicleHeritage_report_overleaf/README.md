# ChronicleHeritage report — Overleaf package

Self-contained LaTeX source for the Major Project Part 1 report. Same template and
front matter as the ARGUS report, so it matches the department format.

```
ChronicleHeritage_report.tex   the whole report in one file
cbit.png                       institute logo used on four pages
images/                        15 figures
```

## Running it in Overleaf

1. Zip this folder (the `.tex`, `cbit.png` and the `images/` folder together).
2. In Overleaf: **New Project → Upload Project** and drop the zip in.
3. Set the compiler to **pdfLaTeX** (Menu → Compiler). It is the default.
4. Compile. There is no `.bib` file — the bibliography is a `thebibliography`
   environment inside the `.tex` — so no BibTeX pass is needed. Compile twice so the
   table of contents, the list of figures and the cross-references resolve.

## Before you submit — edit these four lines

They are at the top of the `.tex`, in the block marked `Editables of the document`
(around line 28). Nothing else needs touching.

```latex
\newcommand{\thesisauthora}{Ch. Jyothi (1601-23-771-004)}
\newcommand{\thesisauthorb}{N. Sharanya (1601-23-771-014)}
\newcommand{\thesisauthorc}{Y. Charan Sai (1601-23-771-303)}
\newcommand{\thesisguide}{Ms. P. Kiranmaie}
```

The names and roll numbers appear on the title page, the declaration and the bonafide
certificate, so check them before printing. If the team has two members rather than
three, delete the `\thesisauthorc` lines in those three places as well.

The month on the title page (`NOVEMBER 2026`) and the academic year in the bonafide
certificate (`2026-27`) are also worth a look.

## Where the numbers come from

Every figure and statistic in the report is read from the running system, not typed in
by hand. The three charts are regenerated from the live database with:

```bash
python scripts/report_figures.py
```

which rewrites `images/fig_satellite_timeline.png`, `images/fig_imagery_by_monument.png`
and `images/fig_damage_distribution.png`. The screenshots were taken from the application
and the two annotated photographs are the stored reference images themselves.

## One thing to be ready for in the viva

The report states plainly, in the abstract and again in the results chapter, that the
damage detector has not been trained and that **no detection accuracy figures are
reported**. The nineteen findings in the system are hand annotations and are labelled as
such everywhere. This is deliberate: the chapter explains why, and shows the training
pipeline that is ready for the weights. Do not let anyone read the hand annotations as
model output — the report is written so that they cannot be.
