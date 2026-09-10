"""Direct prefix-key routing control for the constructed-weight experiments."""
import numpy as np
import bitplane_index


class DirectPrefixRouter:
    """Rank occupied sign prefixes without scanning a float centroid matrix.

    Cluster labels are the bit patterns themselves, in ascending order. The
    native full-key search gives the same weighted prefix score and ID tie rule
    as an independent exhaustive ranking of those patterns.
    """
    def __init__(self, labels, dimensions):
        self.labels=np.ascontiguousarray(labels,dtype=np.int64)
        self.dimensions=dimensions
        codes=np.ascontiguousarray(self.labels.astype(np.uint64)[:,None])
        assignments=np.zeros(len(labels),dtype=np.int64)
        self.index=bitplane_index.KeyIndex(codes,assignments,dimensions,dimensions)
        self.payload_bytes=self.index.info()['logical_bytes']+self.labels.nbytes

    def select(self, query, probes):
        probes=min(probes,len(self.labels))
        result=self.index.search(np.ascontiguousarray(query[:,:self.dimensions]),
                                 np.zeros((len(query),1),dtype=np.int64),
                                 candidate_limit=probes,candidate_target=0,key_limit=0)
        return np.ascontiguousarray(self.labels[result['rows']],dtype=np.int64)
