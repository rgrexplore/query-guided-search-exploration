# Verification record

11 September 2026. This edition summarizes existing experiments; it does not add a new benchmark.

- 20 A4 pages, including references; 12-point body text.
- Six figures and four intact listings, including three inline method descriptions.
- All pages rendered and inspected. The mask arrows and final command listing were corrected after inspection.
- Final build has no LaTeX warnings, underfull/overfull boxes, or unresolved references.
- PDF text bounds were checked; no characters fell outside a page.
- The existing small native example returns IDs [3, 4] for all three methods.
- Its six full scores, ideal key, and next-key score bound match the report.
- Existing full-report arithmetic/evidence verification passed. Final-study source hashes and final selected timings/recall were checked.
- Figures use included real-target and projected-case snapshots. The normal-approximation failure and later 98.95% recall miss remain explicit.
- The source archive was extracted separately and rebuilt to the same 20-page text.

The document source starts from project checkpoint f16ca66. The draft checkpoint is f8d989f.
The original 72-page PDF and its source are unchanged.

Original edition PDF SHA256: `35ae2bede1fba275b78b26f236357df89e3ed3474dac57d48787d23d7583730a`.

## Conversational prose revision

The previous concise edition is preserved at 3d74cf4. All six sections received a prose pass,
with a personal, attributed introduction and more connected explanations.

- Displayed equations, tables, figure calls/captions, headings and pseudocode match the previous edition exactly.
- Inline mathematical expressions and numerical tokens have the same counts and values in each section. Layout-only Needspace values are excluded from the numerical comparison.
- The assumptions, scope qualifiers, negative outcomes and conclusions were reviewed against the prior text.
- Figure and evidence assets, implementation files, and benchmark results are unchanged. No new experiment was run.
- The revised PDF has 20 pages, six figures and four intact listings, with 12-point body text at standard line spacing.
- All pages were rendered and visually inspected. The key listing has an explicit space reservation to keep it intact.
- The final build has no warnings or over/underfull boxes; no text extends beyond the page.
- The refreshed source bundle was independently extracted and rebuilt with identical 20-page text.

Previous conversational-edition PDF SHA256: `78a6c0d1ade7a7b5756f3d152e624e539394670284613eea03022f91350c3fff`.

## Plain-English edition

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

