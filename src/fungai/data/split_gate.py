"""Gate an observation-level run on the split invariants it depends on."""

from __future__ import annotations

import pandas as pd

from fungai.data.leakage import observation_split_violations


class SplitGateError(RuntimeError):
    pass


def run_split_gate(protocol, frames: dict[str, pd.DataFrame]) -> None:
    """Raise SplitGateError listing every failed invariant; return silently if clean."""
    problems = observation_split_violations(protocol, frames)
    if problems:
        raise SplitGateError(
            "observation-level split invariants failed:\n  - " + "\n  - ".join(problems)
        )
