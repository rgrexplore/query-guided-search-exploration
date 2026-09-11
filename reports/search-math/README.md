# Three ways to search binary embeddings

The main deliverable is the rendered report at
`../../output/pdf/search-methods-mathematics.pdf`.

Start with the concise report at `../../output/pdf/search-methods-concise.pdf` for the main
flow and examples. This longer report keeps the complete calculations, earlier experiments,
and details needed to check the results.

Both use the same three methods: select clusters and scan, select clusters and search
bitplanes, or select clusters and look up bit patterns. The longer report also explains a
prefix tree, where removing a bit constraint includes more possible keys. It keeps exact
calculations separate from estimates and measured results. The shared vector representation
and scoring choices are attributed to Exa's public description.

## Read or rebuild

Open `report.tex` in a LaTeX editor. Chapter text is in `sections/`; all figures are already
rendered as PDFs in `figures/`. A normal PDF build does not need Node, Mermaid or a browser.

```bash
./build.sh
```

The script runs `pdflatex` three times for cross-references and copies the finished PDF to
`../../output/pdf/`. It uses the TeX packages listed at the top of `report.tex`, available in
a standard full TeX Live installation. Compilation files stay in the ignored `build/` folder.
The supplied source ZIP preserves the `reports/search-math/` directory structure.

## Diagrams and plot

Six diagrams are authored in Mermaid (`figures/*.mmd`). The rendered PDFs are included.
To change a diagram, use Mermaid CLI with `figures/mermaid.json`, for example:

```bash
npx -p @mermaid-js/mermaid-cli mmdc \
  -i figures/keys.mmd -o figures/keys.pdf \
  -c figures/mermaid.json --pdfFit -b white
```

Mermaid CLI needs its supported browser runtime. This build used Mermaid CLI with a temporary
headless Chrome process; no personal browser profile is needed. Five additional figures use Matplotlib: `recall-validation.pdf` shows the earlier quality
study; `nearest-distance.pdf` and `recall-model-checks.pdf` show the derivation checks;
`measured-cases.pdf` compares independently evaluated settings on three data distributions;
`billion-break-even.pdf` shows the conditional scan-fraction crossing.

## Check the calculations

Run with Python 3; the report checker uses the standard library:

```bash
python3 verify_math.py
```

It checks:

- the score equation, lookup example, bitplane layout and top-two query example;
- complete weighted subset ordering, including equal and zero weights;
- the six-bit conditional-recall example and twenty finite top-K settings by exhaustive counting;
- the 256-document, 16-bit example, including its nearest-distance mixture;
- code bytes, bitmap widths, split/leaf word counts and quoted archived timings;
- FiQA summary counts and the quoted branch example;
- the three quality-study candidate-recall examples and 96 exact reference checks.

Results and SHA-256 hashes are saved in `evidence/math-checks.json`. The evidence directory
contains frozen copies of the original scaling summary, FiQA summary and quality-study
observations. Their hashes identify the exact inputs to this report. They are small summary
and per-query files, not the document text or embedding arrays.

The original larger studies remain in the project `results/` directory. The small study was
run by `calculator/validate_recall.py` using cached embeddings and the C++ index. Re-running
that study requires its original cache and Python environment; compiling and checking this
report's arithmetic do not.

## Scope

The report now includes C++ implementations, measured timing costs, tests on new real queries,
and experiments with deliberately chosen query weights. It does not certify a billion-document
latency/recall prediction or complete the previously edited frontend. The source
calculator's mixed float/binary recall references and omitted work are discussed as modeling
limits. Full-key exactness is kept distinct from short-key candidate recall.

## Check recall estimates

Section 6 starts with a random-bit example and derives nearest-distance probabilities,
which documents pass the prefix check, top-K recall, and unequal query weights. Section 7 checks the
assumptions against independently generated document collections and the previously saved real-data observations.
Appendix A gives the stable binomial calculation.

From the project root, reproduce the controlled checks and their two plots with:

```bash
MPLCONFIGDIR=/private/tmp/search-math-mpl .venv/bin/python reports/search-math/check_recall_models.py \
  --trials 10000 --seed 20260910 --bootstrap-samples 10000
```

This optional study needs NumPy, SciPy, Matplotlib, threadpoolctl, and the project's compiled
`bitplane_index` module. Compiling the PDF and running `verify_math.py` do not need that module.
The source ZIP includes the report scripts and saved evidence; it does not include the C++
extension, document collection, or embedding cache. Run the optional study from the full project checkout.

The four eight-bit cases are explicit teaching setups in the script. Each uses 64 documents,
top five, three lookup bits, and one allowed prefix error. The simulation reports recall,
not index latency. Results are in `evidence/recall-model-checks.json`. Bootstrap intervals use
previously inspected check queries with the fitted correlation held fixed. They are not a fresh
real-data test or a bound on total model error.

Version checkpoints: `67bdfc7` preserves the report before this extension; `0fd1ae3` saves the
expanded derivations and checked distributions. That edition has 55 pages, nine figures and fourteen pseudocode listings. The current
measured-study edition has 72 pages, eleven figures and fifteen listings.


## Implemented study

Sections 8 and 9 connect the formulas to native work, measured timing and independently
evaluated choices. `verify_math.py` checks the saved comparison rows, 447 applicable work
predictions and the ideal-key recall example. These are additional checks, not a replacement
for the earlier distribution tests. The native implementations and full raw-case archives
are in the project checkout, outside the report source ZIP.

From the project root:

```bash
python examples/small_search.py
python -m experiments.run --config experiments/configs/controlled-small.toml --output results/my-small-check
python -m experiments.check_controlled results/controlled-fixed-2026-09-10/tuning
python reports/search-math/plot-measured-cases.py
```

The last two commands use completed results. Extract a run's `cases.zip` inside that run
folder before checking its native counters on a fresh checkout. The plotting script reads
saved independent-evaluation tables. The full dataset runs are documented in
`experiments/README.md`; ordinary PDF compilation still needs no dataset or native extension.

The current edition preserves the original report and adds evidence. The bounded parameter follow-ups, larger buffers and frozen 8M native checks are included.
The billion-document scenarios remain extrapolations, with assumptions and limitations stated explicitly. The calculator remains separate from this evidence update.


## Final evidence and calculations

The final cost section includes the retained cache/model failures, the corrected median fit,
small-posting selection work, a frozen 8M check, parameter-boundary follow-ups, a shortest-key
control and actual single-text-query timing. `verify_math.py` checks the quoted values from
frozen copies under `evidence/`.

The source ZIP also includes the small pure-Python cost modules. With NumPy and SciPy installed,
recalculate the scenarios from the saved inputs without the native extension or corpus:

```bash
python -m experiments.project_costs --inputs reports/search-math/evidence/projected-cases.json \
  --output results/my-scenarios --documents 1000000000 --ram-gb 32 64 1000
```

Ordinary PDF compilation still needs only LaTeX and the included vector figures. The full
project checkout contains all experiment drivers and case archives.

## Plain-English rewrite

The latest edition uses shorter headings and explains unfamiliar terms with examples. It
attributes the shared representation and scoring techniques to Exa's published description.
The methods and measured results are unchanged. Earlier PDFs and sources remain in Git.
The concise edition is 21 pages and the detailed reference is 74 pages.
