"""Replay f32 on nb3_ensemble:r4 along the reference steps: first non-finite Taylor coefficient per order, next to E2 blow-up shares."""
import math, sys, numpy as np, pandas as pd
sys.path.insert(0,'analysis')
from common import traj_rows
from msps.systems import load_case
from msps.arith import get_arith
from msps.runner.references import Reference
from msps.integrators.taylor import TaylorStepper
tr=traj_rows("E2"); t=tr[(tr.case=='nb3_ensemble:r4')&(tr.schedule=='shared')&(tr.arith=='f32')&(tr.tol==1e-3)]
print(t.groupby('n').agg(tp=('tpred','median'),blown=('status',lambda x:(x=='blown_up').mean()),stop=('status',lambda x:(x=='stopped_err').mean()),tend=('t_end','median')).round(3).to_string())
print(t.assign(same=np.isclose(t.tpred,t.t_end)).groupby('n').same.mean().round(2).to_dict())
case=load_case('nb3_ensemble:r4'); f=get_arith('f32'); f.activate()
ov={}
for n in (8,16,24,32):
    s=TaylorStepper(case,n,f); first=[]; maxc=[]
    for ic in range(0,30):
        ref=Reference(case,ic); z=[f.convert(v) for v in case.ic(ic)]
        hs=[f.convert(h) for h in ref.steps]; times=ref.step_times(); hit=np.nan; m=0
        for k,h in enumerate(hs):
            c=s.coeffs(z)
            vals=[abs(float(v)) for ci in c for v in ci]
            if not all(math.isfinite(v) for v in vals): hit=float(times[k]); break
            m=max(m,max(vals))
            z=f.post_step(s.horner(c,h))
            if float(times[k])>6: break
        first.append(hit); maxc.append(m)
    first=np.array(first); ov[n]=dict(ics_nonfinite_before_t6=int(np.isfinite(first).sum()),median_first=float(np.nanmedian(first)) if np.isfinite(first).any() else None,median_maxc=float(np.median(maxc)))
print(pd.DataFrame(ov).T.to_string())
