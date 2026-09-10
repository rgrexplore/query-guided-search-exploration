# Three ways to search binary embeddings

The main deliverable is the rendered report at
`../../output/pdf/search-methods-mathematics.pdf`.

The report develops the shared pipeline, then cluster + scan, cluster + dense bitplanes,
and cluster + weighted key lookup. It also explains literal prefix relaxation with a trie.
It separates exact identities, implementation-dependent counts, hardware cost models,
statistical assumptions and measured evidence. General formulas are parameterized; numbers
appear in labeled examples or measured tables.

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
headless Chrome process; no personal browser profile is needed. Four additional figures use Matplotlib: `recall-validation.pdf` shows the earlier quality
study; `nearest-distance.pdf` and `recall-model-checks.pdf` show the derivation checks;
`measured-cases.pdf` compares independently evaluated settings on three data distributions.

## Arithmetic and evidence checks

Run with Python 3; the report checker uses the standard library:

```bash
python3 verify_math.py
```

It checks:

- the score identity, lookup example, bitplane transpose and routed top-two example;
- complete weighted subset ordering, including equal and zero weights;
- the six-bit conditional-recall example and twenty finite top-K settings by exhaustive counting;
- the 256-document, 16-bit example, including its nearest-distance mixture;
- code bytes, bitmap widths, split/leaf word counts and quoted archived timings;
- FiQA summary counts and the quoted branch example;
- the three quality-study candidate-recall examples and 96 exact reference checks.

Results and SHA-256 hashes are saved in `evidence/math-checks.json`. The evidence directory
contains frozen copies of the original scaling summary, FiQA summary and quality-study
observations. Their hashes identify the exact inputs to this report. They are small summary
and per-query files, not the corpus text or embedding arrays.

The original larger studies remain in the project `results/` directory. The small study was
run by `calculator/validate_recall.py` using cached embeddings and the native index. Re-running
that study requires its original cache and Python environment; compiling and checking this
report's arithmetic do not.

## Scope

The report now includes native implementations, timing calibration, independent real-query
evaluation and constructed query-weight experiments. It does not certify a billion-document
latency/recall prediction or complete the previously edited frontend. The source
calculator's mixed float/binary recall references and omitted work are discussed as modeling
limits. Full-key exactness is kept distinct from short-key candidate recall.

## Recall derivation checks

Section 6 starts with a random-bit example and derives nearest-distance probabilities,
prefix survival, finite top-K recall, and the weighted version. Section 7 checks the
assumptions against independent corpora and the previously saved real-data observations.
Appendix A gives the stable binomial calculation.

From the project root, reproduce the controlled checks and their two plots with:

```bash
MPLCONFIGDIR=/private/tmp/search-math-mpl .venv/bin/python reports/search-math/check_recall_models.py \
  --trials 10000 --seed 20260910 --bootstrap-samples 10000
```

This optional study needs NumPy, SciPy, Matplotlib, threadpoolctl, and the project's compiled
`bitplane_index` module. Compiling the PDF and running `verify_math.py` do not need that module.
The source ZIP includes the report scripts and saved evidence; it does not include the native
extension, corpus, or embedding cache. Run the optional study from the full project checkout.

The four eight-bit cases are explicit teaching setups in the script. Each uses 64 documents,
top five, three lookup bits, and one allowed prefix error. The simulation reports recall,
not index latency. Results are in `evidence/recall-model-checks.json`. Bootstrap intervals use
previously inspected check queries with the fitted correlation held fixed. They are not a fresh
real-data test or a bound on total model error.

Version checkpoints: `67bdfc7` preserves the report before this extension; `0fd1ae3` saves the
expanded derivations and checked distributions. That edition has 55 pages, nine figures and fourteen pseudocode listings. The current
measured-study edition has 65 pages, ten figures and fifteen listings.


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

The current edition preserves the original report and adds evidence. Further parameter-boundary
checks and larger-working-set cost measurements are needed before stronger optimum or
billion-document timing claims. The calculator remains separate from this evidence update.
