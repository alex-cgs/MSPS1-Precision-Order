"""Pilot experiments: truncation order n x arithmetic precision on the planar 3-body problem.

Run:  python3 experiments.py            (full run, ~10 min on 2 cores)
      python3 experiments.py --resume   (reuse phases already stored in results.json / ref.pkl)
"""
import json, math, time, pickle
from fractions import Fraction
from multiprocessing import Pool
from taylor3b import make_schedule, integrate, taylor_coeffs, horner, make_converter, energy, constraint_drift, initial_state

T = 120
N_LIST = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28]
ARITHS = ['float', 'mpfr:80', 'mpfr:113', 'mpfr:160']
BITS = {'float': 53, 'mpfr:80': 80, 'mpfr:113': 113, 'mpfr:160': 160}


def pos_err(z, zr):
    return max(abs(float(z[i] - zr[i])) for i in range(6))


def summarize(states, ref, sched, wall):
    ts = [0.0]
    for h in sched:
        ts.append(ts[-1] + float(h))
    E0 = energy(ref[0])
    err, dE, dC = [], [], []
    for k, z in enumerate(states):
        zr = ref[k]
        try:
            err.append(pos_err(z, zr)); dE.append(abs(float(energy(z) - E0)))
        except (OverflowError, ValueError):
            err.append(float('inf')); dE.append(float('inf'))
        dC.append(constraint_drift(z))
    return dict(t=ts[:len(states)], err=err, dE=dE, dC=dC, wall=wall, steps=len(states) - 1)


def run_one(args):
    n, arith = args
    sched, ref = pickle.load(open('ref.pkl', 'rb'))
    states, wall = integrate(n, arith, sched)
    out = summarize(states, ref, sched, wall)
    out.update(n=n, arith=arith, bits=BITS[arith])
    return out


if __name__ == '__main__':
    import os, sys
    resume = '--resume' in sys.argv and os.path.exists('results.json')
    results = json.load(open('results.json')) if resume else {}
    def save(): json.dump(results, open('results.json', 'w'))

    # 1. Reference run + shared step schedule
    if 'ref' in results and os.path.exists('ref.pkl'):
        sched, ref = pickle.load(open('ref.pkl', 'rb'))
    else:
        t0 = time.perf_counter(); sched, ref = make_schedule(T, n=56, bits=400)
        results['ref'] = dict(n=56, bits=400, steps=len(sched), wall=time.perf_counter() - t0,
                              hmin=float(min(sched)), hmax=float(max(sched)))
        pickle.dump((sched, ref), open('ref.pkl', 'wb')); save()
        print('reference done', results['ref'], flush=True)
    ts = [0.0]
    for h in sched:
        ts.append(ts[-1] + float(h))

    # 2. Check the reference with a more accurate run on the same schedule
    if 'ref_check' not in results:
        chk, _ = integrate(64, 'mpfr:500', sched)
        results['ref_check'] = dict(max_pos_diff=max(pos_err(chk[k], ref[k]) for k in range(len(ref))),
                                    diff_at_T=pos_err(chk[-1], ref[-1]))
        save(); print('reference check', results['ref_check'], flush=True)

    # 3. Amplification: perturb vx1 by 1e-20 and follow the separation
    if 'lyap' not in results:
        z0 = initial_state(); z0[6] += Fraction(1, 10 ** 20)
        pert, _ = integrate(56, 'mpfr:400', sched, z0=z0)
        results['lyap'] = dict(t=ts, sep=[pos_err(pert[k], ref[k]) for k in range(len(ref))])
        save(); print('lyapunov run done', flush=True)

    # 4. Local (one-step) truncation error vs n, from ~60 states along the reference orbit, 400-bit arithmetic
    if 'local' not in results:
        conv = make_converter('mpfr:400')
        idx = list(range(0, len(sched), max(1, len(sched) // 60)))
        local = {}
        for n in N_LIST + [32, 36]:
            errs = []
            for k in idx:
                c = taylor_coeffs(ref[k], n)
                z1 = [horner(ci, conv(sched[k])) for ci in c]
                errs.append(max(pos_err(z1, ref[k + 1]), 1e-300))
            local[n] = sorted(math.log10(e) for e in errs)
        results['local'] = local
        save(); print('local errors done', flush=True)

    # 5. Sweep n x arithmetic
    if 'sweep' not in results:
        jobs = [(n, a) for a in ARITHS for n in N_LIST]
        with Pool(2) as pool:
            results['sweep'] = pool.map(run_one, jobs, chunksize=1)
        save(); print('sweep done', flush=True)

    # 6. Exact rationals: bit growth (n = 8). Step 5 would need ~30x the memory of step 4 (GBs).
    if 'exact' not in results:
        conv = make_converter('exact')
        z = [conv(v) for v in initial_state()]
        bits, walls = [], []
        t0 = time.perf_counter()
        for k, h in enumerate(sched):
            c = taylor_coeffs(z, 8)
            hq = conv(h)
            z = [horner(ci, hq) for ci in c]
            bits.append(max(int(v.p).bit_length() + int(v.q).bit_length() for v in z))
            walls.append(time.perf_counter() - t0)
            if walls[-1] > 120 or len(bits) >= 4:
                break
        exact_err = pos_err([float(v) for v in z[:6]], [float(v) for v in ref[len(bits)][:6]])
        results['exact'] = dict(n=8, bits=bits, wall=walls, steps=len(bits), t_reached=float(sum(sched[:len(bits)])), err_vs_ref=exact_err)
        save(); print('exact done', len(bits), 'steps', flush=True)
    print('saved results.json')
