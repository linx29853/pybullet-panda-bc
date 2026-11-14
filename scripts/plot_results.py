"""Plot recorded training and held-out evaluation metrics."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f8fafc','axes.facecolor':'#f8fafc'})
with (ROOT/'runs/bc_100/loss.csv').open() as f:rows=list(csv.DictReader(f))
epochs=[int(r['epoch']) for r in rows]
fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
for col,label,color in [('train_scaled_mse','Train','#2563eb'),('val_scaled_mse','Validation','#ea580c')]:axs[0].plot(epochs,[float(r[col]) for r in rows],label=label,color=color,lw=2)
axs[0].set(title='Action prediction loss',xlabel='Epoch',ylabel='Scaled action MSE',yscale='log');axs[0].legend();axs[0].grid(alpha=.2)
axs[1].plot(epochs,[float(r['val_rmse_rad']) for r in rows],color='#7c3aed',lw=2)
axs[1].set(title='Validation action error',xlabel='Epoch',ylabel='RMSE (rad)');axs[1].grid(alpha=.2)
fig.suptitle('100 demonstrations | 80 train / 20 validation trajectories',fontweight='bold');fig.savefig(ROOT/'assets/training_curves.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
for ax,seed in zip(axs,[10000,20000]):
    data=json.loads((ROOT/f'results/eval_seed{seed}.json').read_text());rr=data['reports'];n=len(rr)
    ax.plot(range(n),[v['final_distance_m'] for v in rr],color='#2563eb',marker='o',ms=3,lw=1,label='Final distance')
    failed=[v for v in rr if not v['success']]
    if failed:ax.scatter([v['episode'] for v in failed],[v['final_distance_m'] for v in failed],color='#dc2626',marker='x',s=60,zorder=4,label='Timeout')
    ax.axhline(.015,color='#059669',ls='--',label='0.015 m threshold');ax.set(title=f"Seed {seed}: {sum(v['success'] for v in rr)}/{n} success",xlabel='Episode',ylabel='Final distance (m)',yscale='log');ax.legend(fontsize=9);ax.grid(alpha=.2)
fig.suptitle('Closed-loop evaluation on new targets',fontweight='bold');fig.savefig(ROOT/'assets/evaluation_curves.png',dpi=160);plt.close(fig)
print('Saved training_curves.png and evaluation_curves.png')
