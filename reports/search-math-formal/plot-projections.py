"""Draw the conditional scan-fraction crossing from saved scenario calculations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPORT=Path(__file__).resolve().parent
ROOT=REPORT.parents[1]
source=ROOT/'results/projected-cases-2026-09-11/results.json'
rows=json.loads(source.read_text())['rows']
chosen=[r for r in rows if r['documents']==1000000000 and r['ram_gb']==1000]
scan=next(r for r in chosen if r['method']=='scan_all' and r['branch_scale']==1)
base=next(r for r in chosen if r['method']=='branch' and r['branch_scale']==1)
slow=next(r for r in chosen if r['method']=='branch' and r['branch_scale']==2)
fraction=np.geomspace(.0001,1,200)
fig,ax=plt.subplots(figsize=(7,4.2))
ax.plot(fraction*100,scan['modeled_ms']*fraction,label='Scan scoring; routing excluded',color='#3978bb')
ax.axhline(base['modeled_ms'],label='Global bitplanes: modeled cost',color='#d77832')
ax.axhline(slow['modeled_ms'],label='Global bitplanes: 2x variable cost',color='#d77832',linestyle='--')
ax.axvline(base['break_even_scan_fraction']*100,color='#888888',linestyle=':',linewidth=1)
ax.set_xscale('log');ax.set_yscale('log')
ax.set_xlabel('Fraction of corpus scored by scan (%)')
ax.set_ylabel('Query cost (ms; log scale)')
ax.grid(alpha=.2)
ax.spines[['top','right']].set_visible(False)
ax.legend(fontsize=9,loc='upper left')
fig.tight_layout()
fig.savefig(REPORT/'figures/billion-break-even.pdf',bbox_inches='tight')
fig.savefig(REPORT/'figures/billion-break-even.png',dpi=170,bbox_inches='tight')
