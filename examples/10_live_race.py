"""Live terminal demo: watch Greedy and ECI's system value diverge, round by
round, on any registered scenario -- built-in, mechanistic, or real-data.
No new dependency: redraws with plain ANSI escape codes, the same "no CDN,
no server" philosophy as dashboard.py.

Both policies run on separately-seeded-but-matched ConsumableBandit copies
of the same scenario, stepped in lockstep, so what's visible each round is
exactly what makes them diverge: Greedy chasing the highest immediate value,
ECI paying down the externality charge as it goes.

Built to be recorded with asciinema:
    asciinema rec docs/casts/<name>.cast \\
      -c "python examples/10_live_race.py <scenario> --speed 0.1"

Run:  python examples/10_live_race.py [scenario] [--seed N] [--speed SECONDS]
      python examples/10_live_race.py --list      # see available scenarios
"""

import argparse
import sys
import time

from attrition import ConsumableBandit, Greedy, ECI, SCENARIOS, get_arrays

BAR_WIDTH = 40
N_LINES = 9


def _bar(value, max_value, width=BAR_WIDTH, fill="#"):
    n = 0 if max_value <= 0 else max(0, min(width, int(width * value / max_value)))
    return fill * n + "-" * (width - n)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", nargs="?", default="adaptive-therapy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=0.12,
                        help="seconds between rounds (default 0.12)")
    parser.add_argument("--list", action="store_true",
                        help="list available scenarios and exit")
    args = parser.parse_args()

    if args.list:
        for name in sorted(SCENARIOS):
            print(f"  {name}")
        return

    if args.scenario not in SCENARIOS:
        print(f"unknown scenario {args.scenario!r}; --list to see options",
              file=sys.stderr)
        sys.exit(1)

    v, p, e, kw = get_arrays(args.scenario)
    spec = SCENARIOS[args.scenario]
    delta, horizon = kw["delta"], kw["horizon"]

    env_g = ConsumableBandit(v, p, e, delta=delta, horizon=horizon, seed=args.seed)
    env_e = ConsumableBandit(v, p, e, delta=delta, horizon=horizon, seed=args.seed)
    greedy, eci = Greedy(), ECI()

    print(f"\033[1m{args.scenario}\033[0m -- {spec['phenomenon']}")
    print(spec["description"])
    print(f"{len(v)} arms, delta={delta}, horizon={horizon}")
    time.sleep(1.5)
    print("\n" * N_LINES, end="")

    gv = gr = ev = er = 0.0
    for t in range(horizon):
        if not env_g.done():
            arm = greedy.select(env_g.state())
            _, rv, rg, _ = env_g.step(arm)
            gv += rv
            gr += rg
        if not env_e.done():
            arm = eci.select(env_e.state())
            _, rv, rg, _ = env_e.step(arm)
            ev += rv
            er += rg

        max_v = max(gv, ev, 1.0)
        gap = (gv - ev) / max(abs(ev), 1e-9)
        sys.stdout.write(f"\033[{N_LINES}F\033[J")
        print(f"round {t + 1:>3}/{horizon}   arms alive:  "
              f"greedy={int(env_g.alive.sum()):>3}   eci={int(env_e.alive.sum()):>3}")
        print()
        print(f"  greedy  value {gv:8.3f}  |{_bar(gv, max_v)}|")
        print(f"          regret{gr:8.3f}")
        print()
        print(f"  eci     value {ev:8.3f}  |{_bar(ev, max_v)}|")
        print(f"          regret{er:8.3f}")
        print()
        print(f"  system value gap (greedy vs eci): {gap:+.1%}")
        sys.stdout.flush()
        time.sleep(args.speed)

    verdict = ("greedy KEPT UP with eci here"
              if gap > -0.02 else
              f"greedy left {abs(gap):.1%} of system value on the table -- "
              f"while posting {'ZERO' if gr < 1e-9 else f'only {gr:.3f}'} "
              f"private regret the whole time.")
    print(f"\nDone. {verdict}")


if __name__ == "__main__":
    main()
