"""Experiment 09 -- close the index's residual gap.

exp08 left 0.9-3.7% on the table. Diagnosis: the charge delta*kappa_a*(T-t)
assumes the burden is paid over (T-t) more rounds. It is not. Burden is only
paid on rounds where a pull actually happens, and the pool can empty first.

Arm i survives Geometric(p_i) pulls, so the pool supports about sum_i 1/p_i
further pulls. Effective horizon:

    H(S, t) = min( T - t , sum_{i in S} 1/p_i )

Compare three indices against exact DP.
"""
from functools import lru_cache
import numpy as np

def evaluate(v, p, e, delta, T):
    n = len(v); full = frozenset(range(n)); etot = float(np.sum(e))
    B = lambda S: delta * (etot - sum(e[i] for i in S))
    R = lambda a, S: float(v[a] - B(S))

    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S: return 0.0
        return max(R(a,S) + p[a]*V(S-{a},t+1) + (1-p[a])*V(S,t+1) for a in S)

    def pol(pick):
        @lru_cache(maxsize=None)
        def W(S, t):
            if t >= T or not S: return 0.0
            a = pick(S, t)
            return R(a,S) + p[a]*W(S-{a},t+1) + (1-p[a])*W(S,t+1)
        return W(full, 0)

    greedy = pol(lambda S,t: max(S, key=lambda i: R(i,S)))
    idx_T  = pol(lambda S,t: max(S, key=lambda i: R(i,S) - delta*p[i]*e[i]*(T-t)))
    def h(S,t): return min(T-t, sum(1.0/p[i] for i in S))
    idx_H  = pol(lambda S,t: max(S, key=lambda i: R(i,S) - delta*p[i]*e[i]*h(S,t)))
    return V(full,0), greedy, idx_T, idx_H

def main():
    rng = np.random.default_rng(31)
    N, T, DELTA, INST = 5, 16, 0.12, 12
    print("binding regime: T=16, sum(1/p)~6 so H < T-t\n")
    print(f"{'std(k)':>7} {'greedy':>9} {'idx(T-t)':>10} {'idx(H)':>9}  "
          f"{'capture T-t':>12} {'capture H':>10}")
    print("-"*64)
    for scale in [0.1, 0.2, 0.4, 0.8, 1.2]:
        G,IT,IH,SD = [],[],[],[]
        for _ in range(INST):
            v = np.sort(rng.uniform(0.4,1.2,N))[::-1].copy()
            p = np.clip(rng.uniform(0.7,1.0,N),0.05,1.0)
            e = np.clip(1.0+rng.normal(0,scale,N),0,None)
            vs,g,it,ih = evaluate(v,p,e,DELTA,T)
            G.append((vs-g)/abs(vs)*100); IT.append((vs-it)/abs(vs)*100)
            IH.append((vs-ih)/abs(vs)*100); SD.append(float(np.std(p*e)))
        g,it,ih = np.mean(G),np.mean(IT),np.mean(IH)
        print(f"{np.mean(SD):7.3f} {g:8.3f}% {it:9.3f}% {ih:8.3f}%  "
              f"{100*(g-it)/g:11.1f}% {100*(g-ih)/g:9.1f}%")


if __name__ == "__main__":
    main()
