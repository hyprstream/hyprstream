#!/usr/bin/env python3
"""P0.7 portability proof, Python arm: run the STOCK `typesafe-sdk` client against the
stub facade via TYPESAFE_BASE_URL (S6a §4 — the SDK, not the adapter, is the artifact
that speaks the /v1/systemone wire).

Requires: TYPESAFE_BASE_URL (facade address), TYPESAFE_API_KEY (any value).
Exit 0 on success; assertions fail otherwise.
"""

import os
import sys

from typesafe_sdk import (
    Choice,
    Noul,
    Score,
    TypeSafeClient,
    TypeSafeUnprocessableEntityError,
)

STUB_VERSION = "jev-stub-1.0.0"


def main() -> int:
    base_url = os.environ["TYPESAFE_BASE_URL"]
    with TypeSafeClient() as client:
        # GET /v1/models: alias + versioned id listed.
        models = client.models.list()
        names = [m.name for m in models.models]
        assert STUB_VERSION in names, f"versioned id listed: {names}"

        # Golden mixed request: all three primitives, alias model (resolution echoes
        # the versioned id per the contract).
        result = client.system_one(
            state="The refund arrived two weeks late and the box was crushed.",
            model="jev-latest",
            questions={
                "is_refund": Noul(
                    instructions="The customer wants money back.",
                    criteria={"true": "Explicit refund request", "false": None},
                ),
                "tone": Choice(
                    instructions="Classify the tone.",
                    criteria={"angry": "Hostile message", "calm": None, "pleading": "Begging"},
                ),
                "severity": Score(
                    instructions="How severe?",
                    criteria=["cosmetic", "usable", "unusable"],
                ),
            },
        )
        assert result.model == STUB_VERSION, f"alias resolves to versioned id: {result.model}"

        noul = result.answers["is_refund"]
        assert noul.type == "noul" and 0.0 <= noul.noul <= 1.0, noul

        tone = result.answers["tone"]
        assert tone.type == "choice"
        assert set(tone.probabilities) == {"angry", "calm", "pleading"}
        assert abs(sum(tone.probabilities.values()) - 1.0) <= 1e-2, tone.probabilities
        assert tone.choice == max(tone.probabilities, key=tone.probabilities.get)
        assert 0.0 <= tone.confidence <= 1.0

        severity = result.answers["severity"]
        assert severity.type == "score"
        assert abs(sum(severity.probabilities.values()) - 1.0) <= 1e-2
        expected = sum(i * p for i, p in severity.probabilities.items())
        assert abs(severity.score - expected) < 1e-4, (severity.score, expected)
        assert severity.legend[0] == "cosmetic", "legend echoes criteria"
        assert 0.0 <= severity.confidence <= 1.0

        assert result.usage.input_tokens > 0 and result.usage.output_tokens > 0

        # Server-side 422 with the FastAPI detail shape surfaces as the SDK's typed error.
        try:
            client.system_one(
                state="x",
                questions={"bad": {"type": "noul", "criteria": {"yes": "not a jev-1 key"}}},
                retry=None,
            )
        except TypeSafeUnprocessableEntityError as error:
            assert error.status == 422, error.status
        else:
            raise AssertionError("invalid noul criteria must raise UnprocessableEntity")

        # Determinism on the wire: the identical request, identical answers.
        again = client.system_one(
            state="The refund arrived two weeks late and the box was crushed.",
            model="jev-latest",
            questions={
                "is_refund": Noul(
                    instructions="The customer wants money back.",
                    criteria={"true": "Explicit refund request", "false": None},
                ),
            },
        )
        assert again.answers["is_refund"].noul == result.answers["is_refund"].noul

    print(f"PYTHON SDK SMOKE OK against {base_url} (model={result.model})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
