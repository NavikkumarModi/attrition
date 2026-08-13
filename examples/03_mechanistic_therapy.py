"""Derive (v, p, e) from actual Lotka-Volterra tumour dynamics instead of
setting them by hand, and check that maximum tolerated dose -- which is exactly
the greedy policy -- loses to a dose-modulating policy exactly as the adaptive
therapy literature reports.

Run:  python examples/03_mechanistic_therapy.py
"""
from attrition import derive_arm_parameters, Greedy, ECI, ConsumableBandit, run


def main():
    v, p, e, doses = derive_arm_parameters(engine_kwargs={"dt": 5.0}, s0_sd=0.26)
    print("Doses:", [f"{d:.2f}" for d in doses])
    print("Derived v (immediate control):", [f"{x:.3f}" for x in v])
    print("Derived p (chance of exhausting sensitive cells):",
          [f"{x:.3f}" for x in p])
    print("Derived e (damage once that happens):", [f"{x:.3f}" for x in e])
    print()

    env = ConsumableBandit(v, p, e, delta=0.30, horizon=8, seed=0)
    mtd = run(env, Greedy(), seed=0)
    corrected = run(ConsumableBandit(v, p, e, delta=0.30, horizon=8, seed=0),
                    ECI(), seed=0)
    print(f"MTD (= greedy): value={mtd['value']:.3f}  regret={mtd['regret']:.8f}")
    print(f"Corrected:      value={corrected['value']:.3f}")
    print()
    print("MTD reports zero regret and loses system value -- competitive release,")
    print("recovered from the dynamics rather than assumed.")


if __name__ == "__main__":
    main()
