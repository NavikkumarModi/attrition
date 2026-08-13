# Contributing to ATTRITION

## The one rule that matters

**Every theorem has exactly one regression test, and no result ships without one.**
If you add a claim — theoretical, empirical, or "we checked and it's false" — add
the test that would fail if the claim stopped holding.

## Before you start

Read `THEORY.md` first. It is the lab notebook: every result, every falsified
hypothesis, every correction, in the order they happened. A change that
contradicts something recorded there needs to either fix the record or be wrong.

## Workflow

```bash
git clone https://github.com/<user>/attrition
cd attrition
pip install -e ".[dev]"
pytest tests/ -v          # must pass before and after your change
```

## Adding a new experiment

1. Number it sequentially (`experiments/expNN_description.py`).
2. Guard the driver: everything that prints or runs long computation goes under
   `if __name__ == "__main__":`. Importing an experiment module must be silent —
   CI checks this.
3. State the hypothesis being tested in the module docstring *before* running it,
   the way every experiment in this repository does. If the hypothesis is wrong,
   say so in the docstring once you know — falsified hypotheses stay in the record,
   they are not deleted.
4. If the experiment supports a claim in `THEORY.md` or the paper, add a test in
   `tests/test_theorems.py` that would fail if the claim broke.

## Adding a domain or engine

New domains go in `attrition/domains.py` (reduced-form) or `attrition/engines.py`
(mechanistic). State the fit quality honestly in the docstring — "strong",
"moderate", or "partial" — the way the existing four domains do. A domain that
turns out *not* to exhibit the phenomenon (like `platform-trial`) is exactly as
valuable to include as one that does, provided the reason is understood and
recorded.

## Changing the paper

The three builds (`no-regret-is-not-no-harm.tex`, `arxiv-...tex`,
`jmlr-...tex`) share `body.tex`, and the JMLR build additionally includes
`sections_long.tex`. Edit the shared files, not the wrapper `.tex` files, unless
the change is genuinely format-specific. Rebuild all three and check page counts
and error/warning logs before committing:

```bash
cd paper && make
```

## Style

- Report raw values alongside percentages wherever the denominator can approach
  zero — this project has twice manufactured a spurious finding by dividing by a
  near-zero optimum. See `THEORY.md`'s ECI section for what that looked like and
  why it was corrected.
- Prefer an honest "verified numerically, not proven" over an unjustified "proven"
  every time. Six of this project's claims are exact proofs; the rest say so.
- If you disprove something already in the record, keep the disproof — don't
  silently delete the earlier claim.
