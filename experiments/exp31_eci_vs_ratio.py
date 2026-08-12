"""Experiment 31 -- is ECI actually better than a simple ratio heuristic?

exp30 found `ratio` (argmax v_a / (1 + p_a e_a)) competitive with ECI, and ahead
of it at moderate dispersion. That is an honest problem for the paper's
algorithmic claim and must be characterised, not buried.

Hypothesis. ECI's charge delta*kappa_a*(T-t) is horizon-dependent; ratio's is not.
So ratio should do well when the horizon is long relative to the pool (the charge
is roughly constant over the useful part of the episode) and badly when the
horizon is short or the episode is near its end, where the correct charge shrinks
toward zero and ratio keeps penalising.

Tested against exact DP so the comparison is against truth, not against greedy.
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
    def pol(pick):
        @lru_cache(maxsize=None)
        def W(S,t):
            if t>=T or not S: return 0.0
            a=pick(S,t)
            return R(a,S)+p[a]*W(S-{a},t+1)+(1-p[a])*W(S,t+1)
        return W(full,0)
    vs=V(full,0)
    g   = pol(lambda S,t: max(S,key=lambda i: R(i,S)))
    eci = pol(lambda S,t: max(S,key=lambda i: R(i,S)-delta*p[i]*e[i]*(T-t)))
    rat = pol(lambda S,t: max(S,key=lambda i: R(i,S)/(1+p[i]*e[i])))
    f=lambda x: (vs-x)/abs(vs)*100
    return f(g), f(eci), f(rat)

if __name__=="__main__":
    rng=np.random.default_rng(61)
    print("Gap to exact optimum (%), lower is better. n=6.\n")
    print(f"{'T':>4} {'std(k)':>7} {'greedy':>9} {'ECI':>8} {'ratio':>8} {'winner':>10}")
    print("-"*52)
    for T in [4, 8, 16, 30]:
        for spread in [0.5, 1.5]:
            G,E_,Rt,SD=[],[],[],[]
            for _ in range(12):
                v=np.sort(rng.uniform(0.4,1.2,6))[::-1].copy()
                p=np.clip(rng.uniform(0.3,1.0,6),0.05,1.0)
                e=np.clip(1.0+rng.normal(0,spread,6),0,None)
                a,b,c=gaps(v,p,e,0.12,T)
                G.append(a); E_.append(b); Rt.append(c); SD.append(np.std(p*e))
            e_,r_=np.mean(E_),np.mean(Rt)
            w = "ECI" if e_ < r_-0.05 else ("ratio" if r_ < e_-0.05 else "tie")
            print(f"{T:4d} {np.mean(SD):7.3f} {np.mean(G):8.2f}% {e_:7.2f}% "
                  f"{r_:7.2f}% {w:>10}")
    print("\nECI is derived from Theorem 1 and carries the horizon term; ratio is")
    print("a scale-free heuristic with no derivation. Where they tie, ECI's value")
    print("is that it is the closed form the theory predicts, not that it wins.")
