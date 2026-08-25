"""Level 1/2 agents and the voter-to-voter social network, together --
the two pieces of the population-engine plan flagged as genuinely open at
the end of the previous session (FINDINGS.md#31).

Three tiers, same contact-intensity sequence, compared directly:

  Level 0  (population.py's SyntheticVoter) -- a fixed draw, never
           responds to anything. Not run here; it's the baseline the other
           two tiers are meant to add response *to*.
  Level 1  (agents.py's BehaviouralVoter) -- cheap, no LLM, the tier meant
           to scale to thousands of voters, using the same rho-decay
           pressure rule already proven for DegradingPool.
  Level 2  (agents.py's LLM-driven VOTER_PERSONAS) -- a handful of
           personas reasoning via the same LLMPolicy/State/MockLLMClient
           machinery already proven for campaigns, the tier meant to be
           deep, not wide.

Then the voter-to-voter social network (network.py): does word-of-mouth
propagation through Level-1 voters add a measurable effect beyond direct
campaign contact alone? Scoped precisely against
examples/15_campaign_peer_influence.py's campaign-to-campaign network --
this is a different causal channel, not a bigger version of that one.

This script deliberately runs the sequence all the way to maximum
intensity, to show the full adoption curve -- see examples/17's docstring
for why that's the wrong default once the point is "does this flip a real
seat by a believable margin" rather than "does the mechanism work end to
end": sustained maximum intensity saturates the population near 100%
regardless of resistance, which is a real, checked property of the model,
not a realistic campaign-swing magnitude.

Run:  python examples/16_voter_social_network.py [constituency name]
"""
import sys

from attrition.election import (simulate_level1_contact, simulate_level2_contact,
                                simulate_level1_contact_with_network,
                                real_margin_pct, build_constituency)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Ahmedabad East"
    constituency = build_constituency(name)
    margin = real_margin_pct(constituency)
    seq = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(__doc__)
    print(f"Constituency: {name} ({constituency.state}), real margin={margin:.2f}pp\n")

    print("Level 1 (behavioural, n=3000, no LLM):\n")
    fractions, final1 = simulate_level1_contact(name, seq, n_voters=3000, seed=0)
    print("  switched fraction by round:", [f"{x:.3f}" for x in fractions])
    print(f"  final: {final1:.3f}\n")

    print("Level 2 (LLM-driven personas, MockLLMClient, no API key needed):\n")
    results = simulate_level2_contact(name, seq)
    for persona_name, decisions in results.items():
        first_switch = next((i for i, d in enumerate(decisions) if d), None)
        marker = f"switches at round {first_switch}" if first_switch is not None \
            else "never switches"
        print(f"  {persona_name:18s} {marker}")
    print()

    print("Voter-to-voter social network (n=500, sparse random graph, "
          "peer_conversion_prob=0.05):\n")
    _, final_networked, final_direct = simulate_level1_contact_with_network(
        name, seq, n_voters=500, seed=0)
    print(f"  direct contact only:      {final_direct:.3f}")
    print(f"  with peer propagation:    {final_networked:.3f}")
    print(f"  network's marginal lift:  {final_networked - final_direct:+.3f}\n")

    print("Reading this: Level 1 gives a smooth, aggregate adoption curve; "
          "Level 2 shows WHERE personas diverge\n(the point of paying for "
          "LLM reasoning instead of a cheaper rule -- see how differently "
          "voter-skeptical\nresponds to the same sequence). The network "
          "layer adds a real, separately-reported lift on top of\ndirect "
          "contact, not folded into a single conflated number. Re-run with "
          "a different constituency\nname to see how the real margin "
          "changes the Level-1/network baseline (see examples/12).")


if __name__ == "__main__":
    main()
