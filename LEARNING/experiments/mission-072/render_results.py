#!/usr/bin/env python3
"""Render descriptive figures from stored exact MISSION-072 evidence.

Usage: MPLCONFIGDIR=/tmp/mission072-mpl python3 render_results.py
Key Input Dependencies: completed results.json, raw/L32.csv.gz, matplotlib;
    floating point is used only for visualization, never an empirical decision.
Key Functions: main().
"""
import csv
from fractions import Fraction as F
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


# Plot stored measurements without rerunning or extending the experiment.
def main() -> None:
    """Show reset costs near the target and best sampled performance across starts."""
    cases = json.loads((ROOT/'results.json').read_text())['cases']
    rows = list(csv.DictReader(gzip.open(ROOT/'raw/L32.csv.gz','rt')))
    fig,axes = plt.subplots(1,2,figsize=(11,4.6))
    colors = ('#176B9C','#D77D24','#258270')
    for cost,color in zip(('0/1','1/1','8/1'),colors):
        selected = [row for row in rows if row['start']=='1' and row['cost']==cost]
        selected.sort(key=lambda row:F(row['r']))
        axes[0].plot([float(F(row['r'])) for row in selected],
                     [float(F(row['exact_mean'])) for row in selected],'.-',color=color,
                     label=f'Reset time c={int(F(cost))}')
        selected_cases = [case for case in cases if case['L']==32 and case['cost']==cost]
        axes[1].plot([case['start'] for case in selected_cases],
                     [min(1,case['best_grid_ratio']) for case in selected_cases],'.-',color=color)
    axes[0].axhline(63,color='#666666',linestyle='--',linewidth=1,label='No reset: 63')
    axes[0].set_xscale('log')
    axes[0].set(xlabel='Reset probability per event',ylabel='Expected time to target',
                title='Start one step from target (L=32)',ylim=(0,135))
    axes[0].legend(frameon=False,fontsize=8)
    axes[1].axhline(1,color='#666666',linestyle='--',linewidth=1)
    axes[1].set(xlabel='Starting distance from target',ylabel='Best sampled time / no-reset time',
                title='Benefit disappears farther from the target',ylim=(0,1.05),xlim=(1,32))
    for axis in axes:
        axis.grid(alpha=.2)
        axis.spines[['top','right']].set_visible(False)
    fig.suptitle('Restarting helps only when the clock and location permit it',fontsize=13)
    fig.text(.5,.01,'Exact finite reflected-walk experiment; best of 26 sampled rates or no reset. Not a continuous-rate optimum.',
             ha='center',fontsize=8,color='#555555')
    fig.tight_layout(rect=(0,.04,1,.94))
    fig.savefig(ROOT/'restart-costs.png',dpi=180)
    fig.savefig(ROOT/'restart-costs.svg')
    print('Rendered restart-costs.png and restart-costs.svg from stored data.')


if __name__ == '__main__':
    main()
