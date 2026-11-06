"""Replay dd on nbN_circle:N_8_hp #0 along the reference steps until a Taylor coefficient goes non-finite. Argument: order."""
import math, sys
from fractions import Fraction
from msps.systems import load_case
from msps.arith import get_arith
from msps.runner.references import Reference
from msps.integrators.taylor import TaylorStepper
case=load_case("nbN_circle:N_8_hp"); ic=0; n=int(sys.argv[1]) if len(sys.argv)>1 else 32
ref=Reference(case,ic)
dd=get_arith("dd"); dd.activate()
s=TaylorStepper(case,n,dd); z=[dd.convert(v) for v in case.ic(ic)] if hasattr(case,'ic') else None
hs=[dd.convert(h) for h in ref.steps]; times=ref.step_times()
for k,h in enumerate(hs):
    c=s.coeffs(z)
    bad=[(i,j) for i,ci in enumerate(c) for j,v in enumerate(ci) if not (math.isfinite(v.hi) and math.isfinite(v.lo))]
    if bad or k%200==0:
        mx=max(abs(v.hi) for ci in c for v in ci if math.isfinite(v.hi))
        mnz=min(abs(v.lo) for ci in c for v in ci if v.lo!=0 and math.isfinite(v.lo)) if True else 0
        print(k,float(times[k]),'hlog2',math.log2(float(h)),'max|c|%.2e min|lo|%.2e'%(mx,mnz),'nbad',len(bad),bad[:6], [ (repr(c[i][j])) for i,j in bad[:2]])
    if bad: break
    z=dd.post_step(s.horner(c,h))
