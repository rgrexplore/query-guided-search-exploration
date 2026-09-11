# Plain-English review

11 September 2026. This pass changes the explanation and headings, not the methods or experiments.

- Every report section received a plain-language review.
- Shared binary representation, Matryoshka shortening, query/document scoring, and lookup-table scoring are attributed to Exa's public description.
- Unfamiliar terms are replaced or explained using examples: score bounds, reference results, temporary query memory, data bytes versus reserved memory, and probabilities of recovering results.
- Displayed equation/align blocks and algorithm listing bodies were compared with the baseline and remain unchanged. Numeric table values are unchanged; some headings and table labels are simpler.
- Added numerical examples are illustrations, not new measurements: the concise version explains a branch's maximum score, six IDs in reserved array space, recall across two stages, and rounding a proposed split depth.
- Scientific assumptions, negative findings and the distinction between forecasts and measurements remain explicit.
- All pages were visually checked. The long key-search listing and short branch listing were kept together.
- Both builds have no LaTeX warnings or over/underfull boxes; text bounds checks found no characters outside pages.
- The source bundles were extracted separately and rebuilt to the same page counts and extracted text.

Concise report: 21 pages, six figures and four listings.
Detailed report: 74 pages, eleven figures and fifteen listings.
Baseline commit: f27e1e5 (the long report was unchanged from f16ca66 at that point).

search-methods-concise.pdf SHA256: `4963c398f0790f975bcf2a935b0c9a7e9d7c52d27d8b6430265a9faccbbe0f9c`.

search-methods-mathematics.pdf SHA256: `dca8e9612e9384ce42378968536cbd159691eae31598f80535b983ac2315acd9`.

