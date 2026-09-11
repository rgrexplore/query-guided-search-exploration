"""Draw the small examples and a tradeoff plot from the saved real results."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / 'figures'
INK = '#17354b'
COLORS = {'scan': '#3573a3', 'branch': '#bb5b25', 'keys': '#7453a4'}
plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})


def save(fig, name):
    fig.savefig(FIGURES / f'{name}.pdf', bbox_inches='tight', pad_inches=.12)
    fig.savefig(FIGURES / f'{name}.png', dpi=150, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)


fig, ax = plt.subplots(figsize=(8.0, 2.65))
ax.set(xlim=(0, 10), ylim=(0, 3.2))
ax.axis('off')
for x, y, label, live in [(2.8, 2.35, 'Parent: 6,400 live positions', 1),
                         (.2, .65, 'Matching child: 80 live', 80/6400),
                         (5.2, .65, 'Other child: 6,320 live', 6320/6400)]:
    ax.text(x+2.15, y+.58, label, ha='center', color=INK, fontsize=11)
    ax.add_patch(Rectangle((x, y), 4.3, .36, color='#e7edf1'))
    ax.add_patch(Rectangle((x, y), 4.3*live, .36, color='#397798'))
    ax.add_patch(Rectangle((x, y), 4.3, .36, fill=False, edgecolor=INK))
    ax.text(x+2.15, y-.24, '100 words allocated', ha='center', fontsize=10)
for x in [2.35, 7.35]:
    ax.add_patch(FancyArrowPatch((4.95,1.94), (x,1.58), arrowstyle='->', mutation_scale=12, color=INK))
ax.text(5, .04, 'Shading shows the fraction of live positions, not their actual bit order.',
        ha='center', fontsize=9, color='#555555')
save(fig, 'masks')

fig, ax = plt.subplots(figsize=(8.0, 2.45))
ax.axis('off')
rows = [['10', '0.0', '0', '3, 4'], ['11', '0.4', 'empty', 'empty'],
        ['00', '0.5', '5', 'empty'], ['01', '0.9', '1', '2']]
table = ax.table(cellText=rows, colLabels=['Key order', 'Known penalty', 'Cluster 0 IDs', 'Cluster 1 IDs'],
                 loc='center', cellLoc='center', colWidths=[.2,.24,.28,.28])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1,1.85)
for (r,c), cell in table.get_celld().items():
    cell.set_edgecolor('#ccd8df')
    cell.set_facecolor('#edf4f7' if r in (0,1) else 'white')
    if r==0: cell.set_text_props(weight='bold', color=INK)
ax.text(.5, 1.02, 'Query key: coordinates 2 and 3 = 10', ha='center', transform=ax.transAxes, color=INK)
save(fig, 'keys')

# The figure uses the frozen 200-query evaluation, not the later 100-query control.
with (ROOT / 'evidence' / 'real-targets.csv').open() as f:
    data=[r for r in csv.DictReader(f) if r['selection']=='conservative' and int(r['documents'])==1000000]
fig, ax=plt.subplots(figsize=(7.5,3.25))
for method,label,marker in [('scan','A: full scan','o'),('branch','B: bitplanes','s'),('keys','C: key lookup','^')]:
    rows=sorted((r for r in data if r['method']==method),key=lambda r:float(r['target']))
    ax.plot([100*float(r['recall']) for r in rows], [float(r['p50_ms']) for r in rows],
            color=COLORS[method], marker=marker, label=label, linewidth=1)
ax.set(xlabel='Measured binary Recall@100 (%)',ylabel='Median retrieval time (ms)',
       xlim=(80,100.3))
ax.grid(alpha=.2)
ax.legend(frameon=False, loc='upper left')
fig.tight_layout()
save(fig, 'real-tradeoff')

# All three outcomes of the tiny binomial example, not a simulation.
fig, ax = plt.subplots(figsize=(7.5, 1.65))
ax.axis('off')
table=ax.table(cellText=[['0 matches', '1/4', '0'], ['1 match', '1/2', '1'], ['2 matches', '1/4', '1']],
               colLabels=['Ideal-group size Z', 'Probability', 'Recall@1'], loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1,1.7)
for (r,c),cell in table.get_celld().items():
    cell.set_edgecolor('#ccd8df')
    cell.set_facecolor('#edf4f7' if r==0 else 'white')
    if r==0: cell.set_text_props(weight='bold', color=INK)
save(fig,'recall-example')

# Draw only the declared global B point and an optimistic scan line.
# The scan line ignores routing time; its recall remains a separate requirement.
import json
with (ROOT / 'evidence' / 'projected-cases.json').open() as f:
    projected=json.load(f)
rows=[r for r in projected['rows'] if r['documents']==1000000000 and r['ram_gb']==1000 and r['branch_scale']==1]
scan=next(r for r in rows if r['method']=='scan_all')
branch=next(r for r in rows if r['method']=='branch')
fraction=[i/10000 for i in range(101)]
fig,ax=plt.subplots(figsize=(7.5,2.85))
ax.plot([100*f for f in fraction],[f*scan['modeled_ms'] for f in fraction],color=COLORS['scan'],label='A: scan time, excluding routing')
ax.axhline(branch['modeled_ms'],color=COLORS['branch'],label='B: global-path forecast')
cross=100*branch['modeled_ms']/scan['modeled_ms']
ax.axvline(cross,color='#777777',linestyle=':',linewidth=1)
ax.text(cross+.025,10,f'{cross:.2f}% of documents',fontsize=10)
ax.set(xlabel='Fraction of the billion documents scored by A (%)',ylabel='Predicted time (ms)',xlim=(0,1),ylim=(0,270))
ax.legend(frameon=False,loc='upper left',fontsize=10)
ax.grid(alpha=.2)
fig.tight_layout()
save(fig,'billion-comparison')
