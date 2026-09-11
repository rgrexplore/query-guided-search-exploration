"""Numerical what-if cases. This calculates a model; it does not run a search benchmark."""
import json
import math
from pathlib import Path

RAM = 32_000_000_000
RESERVE = 0.10 * RAM
K = 100
WORD_BITS = 64
GROUP_BITS = 4
SETUP_NS = 20_000
CENTROID_NS = 130       # Assumed cost for one 256-dimensional centroid plus selection.
SCORE_TERM_NS = 0.4     # Rounded from the previous reference cost fit.
BITMAP_WORD_NS = 0.5
NODE_NS = 100
SORT_COMPARE_NS = 2
HASH_NS = 100
POSTING_NS = 2
GATHER_EXTRA_NS = 100   # Extra cost per scattered document; applied to BOTH B and C.
QUERY_BYTES = 65_536
ID_BYTES = 8
POSTING_BYTES = 8
ROUTE_DIMS = 256
SLOT_BYTES = 32
HASH_LOAD = 0.7


def scan(n, d, probes):
    """Optimize C for fixed probes in a flat-centroid, balanced-cluster model."""
    row_ns = math.ceil(d / GROUP_BITS) * SCORE_TERM_NS
    raw_code_bytes = math.ceil(d / WORD_BITS) * (WORD_BITS // 8)
    fixed_bytes = n * (raw_code_bytes + ID_BYTES) + QUERY_BYTES + 8
    per_cluster_bytes = ROUTE_DIMS * 4 + 8 + 16
    max_c = min(n * probes // K, n, math.floor((RAM-RESERVE-fixed_bytes)/per_cluster_bytes))
    if max_c < probes:
        return None
    continuous = math.sqrt(row_ns * n * probes / CENTROID_NS)
    candidates = {probes, max_c, max(probes, min(max_c, math.floor(continuous))),
                  max(probes, min(max_c, math.ceil(continuous)))}
    options = []
    for clusters in candidates:
        selected = n * probes / clusters
        routing_ns = CENTROID_NS * clusters if clusters > 1 else 0
        index = n * (raw_code_bytes + ID_BYTES) + (clusters+1)*8
        if clusters > 1:
            index += clusters * ROUTE_DIMS * 4
        scratch = QUERY_BYTES + (clusters*16 if clusters > 1 else 0)
        options.append(dict(clusters=clusters, probes=probes, documents_scored=selected,
                            ms=(SETUP_NS+routing_ns+selected*row_ns)/1e6,
                            index_bytes=index, scratch_bytes=scratch,
                            budgeted_ram_bytes=index+scratch+RESERVE,
                            continuous_optimum=continuous))
    return min(options, key=lambda x:x['ms'])


def strong_branch(n, d, important_bits):
    """One exact matching path: strong bits dominate the total omitted contribution."""
    scored = math.ceil(n / (2**important_bits))
    assert math.floor(n/(2**important_bits)) >= K
    leaf = 2**math.ceil(math.log2(scored))
    nodes = important_bits + 1
    words = math.ceil(n/WORD_BITS)
    score_ns = scored * (math.ceil(d/GROUP_BITS) * SCORE_TERM_NS + GATHER_EXTRA_NS)
    order_ns = d*math.log2(d)*SORT_COMPARE_NS
    ns = SETUP_NS + order_ns + nodes*words*BITMAP_WORD_NS + nodes*NODE_NS + score_ns
    index = n*(math.ceil(d/WORD_BITS)*8+ID_BYTES) + 16 + d*words*8
    # Pending alternative masks, capacity allowance, parent and child buffers.
    scratch = QUERY_BYTES + (2*important_bits+3)*words*8 + 2*important_bits*64
    a_threshold = ((ns-SETUP_NS)**2) / (4*math.ceil(d/GROUP_BITS)*SCORE_TERM_NS*n)
    return dict(clusters=1,probes=1,leaf_size=leaf,split_nodes=important_bits,
                leaf_nodes=1,nodes=nodes,documents_scored=scored,bitmap_words=nodes*words,
                ms=ns/1e6,index_bytes=index,scratch_bytes=scratch,
                budgeted_ram_bytes=index+scratch+RESERVE,
                fits=index+scratch+RESERVE<=RAM,
                centroid_cost_threshold_ns=a_threshold, order_ns=order_ns,
                bitmap_word_threshold_ns=(scan(n,d,1)['ms']*1e6-SETUP_NS-order_ns-nodes*NODE_NS-score_ns)/(nodes*words))


def strong_key(n, d, prefix_bits):
    """The ideal prefix has enough rows and a bound certifies the unseen keys cannot win."""
    found = math.ceil(n/2**prefix_bits)
    assert math.floor(n/2**prefix_bits)>=K
    tail_groups = math.ceil((d-prefix_bits)/GROUP_BITS)
    ns = SETUP_NS + prefix_bits*math.log2(prefix_bits)*SORT_COMPARE_NS + HASH_NS + found*POSTING_NS + found*(tail_groups*SCORE_TERM_NS+GATHER_EXTRA_NS)
    codes = n*math.ceil(d/WORD_BITS)*8
    ids = n*ID_BYTES
    postings = n*POSTING_BYTES
    slots = math.ceil(min(n,2**prefix_bits)/HASH_LOAD)*SLOT_BYTES
    index = codes+ids+postings+slots+16
    scratch = QUERY_BYTES + 96
    return dict(clusters=1,probes=1,prefix_bits=prefix_bits,keys_tried=1,
                documents_scored=found,tail_score_groups=tail_groups,ms=ns/1e6,
                index_bytes=index,scratch_bytes=scratch,budgeted_ram_bytes=index+scratch+RESERVE,
                fits=index+scratch+RESERVE<=RAM)


def weak_branch(n,d,baseline):
    """The best choice when pruning is useless: choose leaves large enough to scan."""
    c,p=baseline['clusters'],baseline['probes']
    small,extra=divmod(n,c)
    plane_words=(c-extra)*math.ceil(small/64)+extra*math.ceil((small+1)/64)
    words_per_probe=math.ceil(math.ceil(n/c)/64)
    overhead_ns=p*(NODE_NS+words_per_probe*BITMAP_WORD_NS)+d*math.log2(d)*SORT_COMPARE_NS
    index=baseline['index_bytes']+d*plane_words*8
    scratch=baseline['scratch_bytes']+2*p*(64+words_per_probe*8)
    return dict(clusters=c,probes=p,leaf_size=2**math.ceil(math.log2(math.ceil(n/c))),
                nodes=p,documents_scored=baseline['documents_scored'],ms=baseline['ms']+overhead_ns/1e6,
                index_bytes=index,scratch_bytes=scratch,budgeted_ram_bytes=index+scratch+RESERVE)


def main():
    ordinary=scan(1_000_000,256,32)
    cases=[dict(name='No useful pruning',N=1_000_000,D=256,baseline=ordinary,
                branch=weak_branch(1_000_000,256,ordinary),
                key='Uninformative prefix: best fallback is the same scan; no justified key-search gain.'),
           dict(name='Query-adaptive selective bits',N=1_000_000,D=768,
                baseline_optimistic_one_probe=scan(1_000_000,768,1),
                branch=strong_branch(1_000_000,768,13),
                key='Important coordinates are outside the fixed short prefix; its best useful fallback is scan.'),
           dict(name='Useful fixed 12-bit prefix',N=1_000_000,D=256,
                baseline_optimistic_one_probe=scan(1_000_000,256,1),
                branch=strong_branch(1_000_000,256,12),key=strong_key(1_000_000,256,12)),
           dict(name='Useful fixed 16-bit prefix',N=100_000_000,D=256,
                baseline_optimistic_one_probe=scan(100_000_000,256,1),
                branch=strong_branch(100_000_000,256,16),key=strong_key(100_000_000,256,16)),
           dict(name='Useful fixed 20-bit prefix',N=500_000_000,D=256,
                baseline_optimistic_one_probe=scan(500_000_000,256,1),
                branch=strong_branch(500_000_000,256,20),key=strong_key(500_000_000,256,20))]
    output=dict(assumptions={k: v for k,v in globals().items() if k.isupper() and isinstance(v,(float,int))},
                cases=cases,one_billion_256bit_minimum=dict(codes=32e9,ids=8e9,index_minimum=40e9,ram=RAM,feasible=False),
                approximate_document_limits=dict(A=math.floor((RAM-RESERVE)/40),B=math.floor((RAM-RESERVE)/72),C=math.floor((RAM-RESERVE)/48)))
    # Check that the derivative's integer choice beats nearby cluster counts.
    for case in cases:
        a=case.get('baseline',case.get('baseline_optimistic_one_probe'))
        if a:
            b=math.ceil(case['D']/4)*SCORE_TERM_NS
            for c in range(max(a['probes'],a['clusters']-25),a['clusters']+26):
                value=(SETUP_NS+CENTROID_NS*c+b*case['N']*a['probes']/c)/1e6
                assert a['ms']<=value+1e-10
    assert not output['cases'][-1]['branch']['fits']
    assert output['cases'][-1]['key']['fits']
    path=Path(__file__).with_name('results.json')
    path.write_text(json.dumps(output,indent=2)+'\n')
    for c in cases:
        a=c.get('baseline',c.get('baseline_optimistic_one_probe'))
        print(c['name'], 'A=',round(a['ms'],6),'C_clusters=',a['clusters'],
              'B=',round(c['branch']['ms'],6),'B_RAM_GB=',round(c['branch']['budgeted_ram_bytes']/1e9,4),
              'C=',round(c['key']['ms'],6) if isinstance(c['key'],dict) else 'scan fallback',
              'C_RAM_GB=',round(c['key']['budgeted_ram_bytes']/1e9,4) if isinstance(c['key'],dict) else '-')
    print('B router-cost threshold:',cases[1]['branch']['centroid_cost_threshold_ns'],'ns/centroid')
    print('Document ceilings, before directory/temporary overhead:',output['approximate_document_limits'])


if __name__=='__main__':
    main()
