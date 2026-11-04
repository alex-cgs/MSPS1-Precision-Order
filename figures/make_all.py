"""Rebuild every figure from results/ (AGENT_PROMPT section 9). Each draws only when its table exists."""
import fig_e01_e04 as a
import fig_e02_e03 as b
import fig_others as c

for fn in (a.local_law, a.exact_growth, b.tpred, b.crossover, b.pred_vs_meas, c.stats_vs_traj, c.escapes, c.representation, c.methods, c.hardware, c.spurious, c.dimension, c.indicator):
    try:
        fn()
    except Exception as e:  # one missing study must not stop the others; the failure is reported
        print("FIGURE FAILED", fn.__name__, repr(e))
print("figures done")
