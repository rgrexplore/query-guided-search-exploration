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

Five diagrams are authored in Mermaid (`figures/*.mmd`). The rendered PDFs are included.
To change a diagram, use Mermaid CLI with `figures/mermaid.json`, for example:

```bash
npx -p @mermaid-js/mermaid-cli mmdc \
  -i figures/keys.mmd -o figures/keys.pdf \
  -c figures/mermaid.json --pdfFit -b white
```

Mermaid CLI needs its supported browser runtime. This build used Mermaid CLI with a temporary
headless Chrome process; no personal browser profile is needed. The sixth figure,
`recall-validation.pdf`, is a standard Matplotlib plot of the saved quality-study observations.

## Arithmetic and evidence checks

Run with Python 3; the report checker uses the standard library:

```bash
python3 verify_math.py
```

It checks:

- the score identity, lookup example, bitplane transpose and routed top-two example;
- complete weighted subset ordering, including equal and zero weights;
- the six-bit binomial conditional-recall example by exhaustive counting;
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

This is the math-first deliverable. It does not certify a billion-document recall prediction,
implement a new hash enumerator, or complete the previously edited frontend. The source
calculator's mixed float/binary recall references and omitted work are discussed as modeling
limits. Full-key exactness is kept distinct from short-key candidate recall.
