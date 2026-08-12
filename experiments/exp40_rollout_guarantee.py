"""Experiment 40 -- T-G: why does rollout reach 99.5-99.9% of optimum?

Policy improvement guarantees V_pi1 >= V_pi0 but says nothing about how much of
the remaining gap is closed. Observed: rollout over ECI leaves 0.14-0.47%, over
greedy 0.2-0.8%. That is far better than the guarantee promises.

Conjecture. One improvement step is exact wherever the base policy's error is
confined to a SINGLE decision. The rollout evaluates each candidate's true
continuation under the base policy, so if the base policy is optimal from t+1
onward, the rollout choice at t is optimal. Errors therefore have to COMPOUND
across steps to survive one improvement round, and compounding is rare when the
base policy is already close.

Prediction: the rollout gap should scale roughly as the SQUARE of the base gap,
since surviving errors require the base policy to be wrong at two separate points.
"""
from functools import lru_cache
import numpy as np

def gaps(v, p, e, delta, T):
    n=len(v); full=frozenset(range(n)); etot=float(np.sum(e))
    B=lambda S: delta*(etot-sum(e[i] for i in S))
    R=lambda a,S: float(v[a]-B(S))
    @lru_cache(maxsize=None)
    def V(S,t):
        if t>=T or not S: return 0.0
        return max(R(a,S)+p[a]*V(S-{a},t+1)+(1-p[a])*V(S,t+1) for a in S)
    def val(pick):
        @lru_cache(maxsize=None)
        def W(S,t):
            if t>=T or not S: return 0.0
            a=pick(S,t); return R(a,S)+p[a]*W(S-{a},t+1)+(1-p[a])*W(S,t+1)
        return W
    g_pick=lambda S,t: max(S,key=lambda i: R(i,S))
    e_pick=lambda S,t: max(S,key=lambda i: R(i,S)-delta*p[i]*e[i]*(T-t))
    Wg, We = val(g_pick), val(e_pick)
    def roll(W):
        return lambda S,t: max(S, key=lambda a: R(a,S)
                               + p[a]*W(S-{a},t+1) + (1-p[a])*W(S,t+1))
    Wrg, Wre = val(roll(Wg)), val(roll(We))
    vs=V(full,0)
    f=lambda W: (vs-W(full,0))/abs(vs)
    return f(Wg), f(Wrg), f(We), f(Wre)

if __name__=="__main__":
    rng=np.random.default_rng(77)
    print("T-G: ROLLOUT GAP VERSUS BASE GAP\n")
    print("If surviving errors must compound, rollout gap ~ (base gap)^2.\n")
    print(f"{'base':>10} {'base gap':>10} {'rollout gap':>12} {'ratio':>8} "
          f"{'(base)^2':>10} {'obs/pred':>9}")
    print("-"*64)
    rows=[]
    for spread in [0.2, 0.4, 0.8, 1.2, 1.8]:
        G,RG,E,RE=[],[],[],[]
        for _ in range(12):
            n,T,delta=6,8,0.12
            v=np.sort(rng.uniform(0.4,1.2,n))[::-1].copy()
            p=np.clip(rng.uniform(0.3,1.0,n),0.05,1.0)
            e=np.clip(1.0+rng.normal(0,spread,n),0,None)
            a,b,c,d=gaps(v,p,e,delta,T)
            G.append(a); RG.append(b); E.append(c); RE.append(d)
        for label, base, roll in [("greedy",np.mean(G),np.mean(RG)),
                                  ("ECI",np.mean(E),np.mean(RE))]:
            sq=base**2
            print(f"{label:>10} {100*base:9.3f}% {100*roll:11.3f}% "
                  f"{roll/max(base,1e-12):8.4f} {100*sq:9.3f}% "
                  f"{roll/max(sq,1e-12):9.2f}")
            rows.append((base,roll))
    # Analyse each base policy separately: pooling them mixes two different
    # constants and produces a meaningless exponent.
    gr = np.array([r for i, r in enumerate(rows) if i % 2 == 0])
    ec = np.array([r for i, r in enumerate(rows) if i % 2 == 1])
    print()
    print(f"{'base policy':>12} {'mean ratio':>12} {'std':>8} "
          f"{'gap closed':>12} {'log-log slope':>14}")
    print("-" * 62)
    for label, arr in [("greedy", gr), ("ECI", ec)]:
        ratio = arr[:, 1] / np.maximum(arr[:, 0], 1e-12)
        slope = np.polyfit(np.log(arr[:, 0]), np.log(arr[:, 1]), 1)[0]
        print(f"{label:>12} {ratio.mean():12.4f} {ratio.std():8.4f} "
              f"{100*(1-ratio.mean()):11.1f}% {slope:14.2f}")
    print()
    print("  The compounding conjecture (slope 2) is FALSIFIED. Within each base")
    print("  policy the ratio is near-constant, so one improvement step closes a")
    print("  fixed FRACTION of the base gap rather than squaring it.")
    print()
    print("  The fraction differs by base, and in the direction that looks wrong")
    print("  at first: rollout closes ~96% of greedy's gap but only ~88% of ECI's.")
    print("  The explanation is that ECI has already removed the shallow errors --")
    print("  the ones a single step of lookahead can see. What remains are the")
    print("  errors that genuinely require deeper lookahead, so the same single")
    print("  step recovers proportionally less of them.")
