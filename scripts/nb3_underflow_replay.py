"""Replay nb3_ensemble:r4 with the E5 schedule (f64, t <= 300): how many step underflows have the lifted u = 1/r off from the positions."""
import math, warnings; warnings.filterwarnings('ignore')
from fractions import Fraction
from msps.systems import load_case
from msps.arith import get_arith
from msps.integrators.taylor import TaylorStepper
from msps.schedules import AdaptiveDyadic
case=load_case('nb3_ensemble:r4'); a=get_arith('f64'); a.activate(); S=case.system; f=a.to_float; names=[str(x) for x in S.states]
for n in (4,6,8,12,16):
    und=lifted=coll=0; N=30
    for ic in range(N):
        st=TaylorStepper(case,n,a); sc=AdaptiveDyadic(8,Fraction(1,2)); z=[a.convert(v) for v in case.ic(ic)]; t=Fraction(0); k=0; hmin=Fraction(1,2**100)
        while t<300 and k<300000:
            c=st.coeffs(z); h=min(sc.next_h(t,z,c,st),300-t)
            if h<hmin:
                und+=1
                v={m:f(z[names.index(m)]) for m in names}
                c2=[v[m]*math.hypot(v['qx'+m[1]]-v['qx'+m[3]],v['qy'+m[1]]-v['qy'+m[3]]) for m in names if m.startswith('u')]
                r=min(math.hypot(v['qx'+m[1]]-v['qx'+m[3]],v['qy'+m[1]]-v['qy'+m[3]]) for m in names if m.startswith('u'))
                if max(abs(x-1) for x in c2)>1: lifted+=1
                if r<1e-6: coll+=1
                break
            z=a.post_step(st.horner(c,a.convert(h))); t+=h; k+=1
    print(f'n={n} ICs={N} underflow={und} of_which_u*r_off_by_more_than_1={lifted} with_rmin<1e-6={coll}',flush=True)
