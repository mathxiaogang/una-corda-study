# Independent re-computation of headline numbers (different code path from 03_analysis.py)
import sys, itertools, numpy as np, pandas as pd
d=sys.argv[1]
m=pd.read_csv(f"{d}/metadata_clean.csv"); f=pd.read_csv(f"{d}/features.csv"); D=m.merge(f,on="midi_filename")
assert len(D)==1276 and D.pianist.nunique()==201
# composer-adjusted edition means via groupby-apply
D["r"]=D.soft_frac-D.groupby("composer").soft_frac.transform("mean")
E=D.groupby(["year","group"]).r.mean().reset_index()
s=E[E.group=="senior"].r.values; j=E[E.group=="junior"].r.values
obs=s.mean()-j.mean(); allv=E.r.values
cnt=sum(abs(allv[list(c)].mean()-np.delete(allv,list(c)).mean())>=abs(obs)-1e-12 for c in itertools.combinations(range(10),5))
print(f"H1a diff={obs*100:.2f}pp perm p={cnt/252:.4f} sep={s.min()>j.max()}")
# two-stage check of LMM: regress edition means of residualised (on covariates) outcome
from numpy.linalg import lstsq
X=np.column_stack([np.ones(len(D)),(D.nps-D.nps.mean())/D.nps.std(),(D.vel_mean-D.vel_mean.mean())/D.vel_mean.std(),np.log(D.duration)])
b=lstsq(X,D.r.values,rcond=None)[0]; D["r2"]=D.r-X[:,1:]@b[1:]
E2=D.groupby(["year","group"]).r2.mean().reset_index()
print(f"two-stage (covariate-adjusted) diff={ (E2[E2.group=='senior'].r2.mean()-E2[E2.group=='junior'].r2.mean())*100:.2f}pp")
# counts quoted in the doc
print("soft>95%:", (D.soft_frac>0.95).sum(), "| Bach n:", (D.composer=='Johann Sebastian Bach').sum(),
      "| Bach sus<5%:", ((D.composer=='Johann Sebastian Bach')&(D.sus_frac<0.05)).sum())
b=D[D.composer=='Johann Sebastian Bach']; print("Bach no-sustain share by group:", b.groupby('group').sus_frac.apply(lambda x:(x<0.05).mean()).round(3).to_dict())
print("median episode length (s):", round(D.soft_episode_med_s.median(),2))
both=D.groupby('canonical_title_label').group.nunique(); M=D[D.canonical_title_label.isin(both[both==2].index)]
pm=M.groupby(['canonical_title_label','group']).soft_frac.mean().unstack(); print("matched pieces:", len(pm), "senior higher:", int((pm.senior>pm.junior).sum()), "mean diff", round((pm.senior-pm.junior).mean()*100,2))
print("edition medians:", D.groupby('year').soft_frac.median().round(3).to_dict())
r1=D[D['round']=='R1'].drop_duplicates(['year','pianist']); print("R1 pianist-years:", len(r1), "advanced:", int(r1.advanced_R2.sum()))
