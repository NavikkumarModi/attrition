"""Experiment 46 -- the sequential multi-agent model is degenerate.

THEOREM (sequential equivalence). On a shared consumable pool, if agents act one
after another within a round and each observes the state left by its predecessors,
then m agents over T rounds is exactly one learner over m*T pulls.

Proof. The state is the surviving set together with the accumulated burden. Agent
i's decision at position j of round t faces exactly the state that a single learner
would face at pull index (t*m + j), and its immediate reward v_a - B(S) is the same
function of that state. The transition -- destruction with probability p_a -- is
also identical. The two processes therefore have the same MDP, and any policy in
one induces a policy in the other with equal value. QED

Consequence, and it is not a small one: decentralised greedy agents behave exactly
like a single greedy learner, so a "price of anarchy" computed in this model does
not measure decentralisation. The 1.297 reported in exp32 came from comparing
against a planner restricted to DISTINCT arms per round while the agents could
re-target within the round -- a difference of action spaces, not of incentives.

The tell was already visible: at zero kappa dispersion, where Theorem 1 forces
decentralisation to be costless, exp45 measured PoA = 1.0230 rather than 1.
"""
from functools import lru_cache
import numpy as np

def single_learner(v, p, e, delta, H):
    n=len(v); full=frozenset(range(n)); etot=float(np.sum(e))
    B=lambda S: delta*(etot-sum(e[i] for i in S)); R=lambda a,S: float(v[a]-B(S))
    @lru_cache(maxsize=None)
    def V(S,t):
        if t>=H or not S: return 0.0
        return max(R(a,S)+p[a]*V(S-{a},t+1)+(1-p[a])*V(S,t+1) for a in S)
    def pol(pick):
        @lru_cache(maxsize=None)
        def W(S,t):
            if t>=H or not S: return 0.0
            a=pick(S,t); return R(a,S)+p[a]*W(S-{a},t+1)+(1-p[a])*W(S,t+1)
        return W(full,0)
    return V(full,0), pol(lambda S,t: max(S,key=lambda i: R(i,S)))

def sequential_agents(v, p, e, delta, T, m, seeds=4000):
    n=len(v); tot=[]
    for s in range(seeds):
        rng=np.random.default_rng(9000+s); alive=np.ones(n,bool); acc=0.0
        for _ in range(T):
            if not alive.any(): break
            for _ in range(m):
                idx=np.flatnonzero(alive)
                if idx.size==0: break
                a=int(idx[np.argmax(v[idx])])
                acc += v[a]-delta*float(e[~alive].sum())
                if rng.random()<p[a]: alive[a]=False
        tot.append(acc)
    return float(np.mean(tot))

if __name__=="__main__":
    print("SEQUENTIAL EQUIVALENCE: m agents x T rounds == 1 learner x m*T pulls\n")
    print(f"{'m':>3} {'T':>3} {'m*T':>5} {'sequential agents':>19} "
          f"{'single greedy':>15} {'difference':>12}")
    print("-"*62)
    rng=np.random.default_rng(5); ok=True
    for m,T in [(1,6),(2,3),(3,2),(2,4),(4,2)]:
        D,G=[],[]
        for _ in range(4):
            n=6; v=np.sort(rng.uniform(0.4,1.2,n))[::-1].copy()
            p=np.clip(rng.uniform(0.3,0.9,n),0.05,1.0)
            e=np.clip(1.0+rng.normal(0,1.0,n),0,None)
            D.append(sequential_agents(v,p,e,0.12,T,m))
            G.append(single_learner(v,p,e,0.12,m*T)[1])
        d,g=np.mean(D),np.mean(G); ok &= abs(d-g)<0.03
        print(f"{m:3d} {T:3d} {m*T:5d} {d:19.4f} {g:15.4f} {abs(d-g):12.5f}")
    print(f"\n  equivalence holds: {'CONFIRMED' if ok else 'FALSIFIED'}")
    print("  Sequential shared-pool consumption therefore has NO price of anarchy.")
    print("  The cost of decentralisation can only appear under simultaneity or")
    print("  private observation -- which is what the simultaneous model supplies.")
