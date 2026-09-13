## 2.4 Summary

### Time complexity

| Method          | Approximate query time                                                                                             |
| --------------- | ------------------------------------------------------------------------------------------------------------------ |
| Original method | \(\displaystyle T_A \approx T_0+\frac{PN}{C}a_A\)                                                                  |
| Bitplanes       | \(\displaystyle T_B \approx T_0+PS\left\lceil\frac{N}{64C}\right\rceil b+f_B\frac{PN}{C}a_B+T_{\mathrm{other},B}\) |
| Backward Walk   | \(\displaystyle T_C \approx T_0+2PL\log_2(n+1)t_b+F_Ca_C+T_{\mathrm{ranges}}\)                                     |

### Memory complexity

| Method          | Stored index                | Temporary query memory |
| --------------- | --------------------------- | ---------------------- |
| Original method | \(M_0\)                     | \(Q_0\)                |
| Bitplanes       | \(M_0+M_{\mathrm{planes}}\) | \(Q_0+Q_B\)            |
| Backward Walk   | \(M_0+M_{\mathrm{prefix}}\) | \(Q_0+Q_C\)            |

For the current Backward Walk layout, \(M_{\mathrm{prefix}}\) contains the stored prefix keys/ranges. If one four-byte key is stored per document, the key data alone is approximately \(4N\) bytes.

### Recall

| Method          | Per-query recall                       | Exactness condition within opened clusters |       |                                                          |
| --------------- | -------------------------------------- | ------------------------------------------ | ----- | -------------------------------------------------------- |
| Original method | \(\displaystyle \frac{g_q}{K}\)        | Score all opened documents                 |       |                                                          |
| Bitplanes       | \(\displaystyle \frac{g_q}{K}r_{B,q}\) | Complete traversal using safe score bounds |       |                                                          |
| Backward Walk   | (\displaystyle \frac{                  | G_q\cap D_{q,\ell}                         | }{K}) | Relax to depth \(0\), so all opened documents are scored |

### Variable definitions

| Symbol                   | Definition                                                                     |
| ------------------------ | ------------------------------------------------------------------------------ |
| \(N\)                    | Total number of documents                                                      |
| \(C\)                    | Number of clusters                                                             |
| \(P\)                    | Number of clusters opened for one query                                        |
| \(n=N/C\)                | Average documents per cluster, assuming balanced clusters                      |
| \(K\)                    | Number of requested results                                                    |
| \(d\)                    | Number of binary dimensions                                                    |
| \(T_0\)                  | Common query work, including routing and score-table preparation               |
| \(a_A\)                  | Average time to fully process one document in Method A                         |
| \(S\)                    | Average number of Bitplane splits per opened cluster                           |
| \(b\)                    | Average time for one split-word visit                                          |
| \(f_B\)                  | Fraction of Method A's selected documents that are fully scored by Bitplanes   |
| \(a_B\)                  | Average time to fully process one surviving Bitplane document                  |
| \(T_{\mathrm{other},B}\) | Other Bitplane work, such as sorting, queues and leaf-mask processing          |
| \(x\)                    | Starting prefix depth for Backward Walk                                        |
| \(\ell\)                 | Final prefix depth reached by Backward Walk                                    |
| \(L=x-\ell+1\)           | Number of prefix depths visited                                                |
| \(t_b\)                  | Average time for one key comparison during prefix-range lookup                 |
| \(F_C\)                  | Number of distinct documents scored by Backward Walk                           |
| \(a_C\)                  | Average time to fully process one Backward Walk candidate                      |
| \(T_{\mathrm{ranges}}\)  | Prefix-range lookup/update and new-slice bookkeeping cost                      |
| \(M_0\)                  | Stored memory shared by the methods, such as packed rows, IDs and routing data |
| \(M_{\mathrm{planes}}\)  | Additional stored Bitplane index                                               |
| \(M_{\mathrm{prefix}}\)  | Additional stored Backward Walk prefix index                                   |
| \(Q_0\)                  | Common temporary query memory                                                  |
| \(Q_B\)                  | Additional temporary Bitplane branch/mask memory                               |
| \(Q_C\)                  | Additional temporary Backward Walk range/prefix state                          |
| \(G_q\)                  | Exact reference top-\(K\) document IDs for query \(q\)                         |
| \(g_q\)                  | Number of reference top-\(K\) documents contained in the opened clusters       |
| \(r_{B,q}\)              | Fraction of those routed reference documents recovered by Bitplanes            |
| \(D_{q,\ell}\)           | Documents exposed by Backward Walk by final prefix depth \(\ell\)              |

That’s enough for the summary — no extra prose needed.
