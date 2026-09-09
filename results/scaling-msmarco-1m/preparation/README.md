# MS MARCO preparation

The fixed sample contains **1,000,000 passages**, with **100 development queries** and
**200 separate evaluation queries**. The source bundle and selected row order were checked
before encoding. This is a custom scaling subset, not the full MS MARCO leaderboard corpus.

Nomic v1.5 produced full 768-dimensional vectors on the Mac GPU. Search uses the first 256
coordinates: document signs are packed into four 64-bit words, while queries stay float32.
The first 4,096 passages and all queries took 13.8 seconds. Resuming the remaining chunks,
combining the files and deriving the search arrays took 3,328.9 seconds. These are preparation
times; they are excluded from the search measurements.

Checks completed:

- All one million packed codes match the saved document signs.
- Full vectors are finite and have unit length within the checked tolerance.
- Three freshly encoded documents and three queries match their saved rows; a wrong-row control fails.
- A normal cache reuse does not load the text encoder.
- Independent sign decoding and score accumulation agree with native search on 1,000 documents
  and five queries, with no near-boundary ambiguities in that sample.

[preparation.json](preparation.json) records the source hashes, model revision, settings, array
shapes and checks. Passage text, weights and vector arrays remain in the ignored data cache.
Full per-pool references and timed search comparisons follow under the
[current protocol](../../../docs/scaling-protocol-v2.md).
