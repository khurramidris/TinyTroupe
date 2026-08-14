"""Offline smoke test for the H&M benchmark's real TinyTroupe execution path.

No network/model call is made. The TinyTroupe ActionGenerator client is replaced by a
small deterministic fake so this exercises TinyPerson construction, persona injection,
listen_and_act(), TALK extraction, JSON parsing, and resumable JSONL writing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "examples" / "hm_benchmark" / "hm_benchmark.py"
spec = importlib.util.spec_from_file_location("hm_benchmark", MODULE)
hm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hm)


class FakeClient:
    def send_message(self, messages, **kwargs):
        # The fake deliberately follows TinyTroupe v0.7's multi-action contract:
        # THINK -> TALK -> DONE, with the benchmark payload inside TALK content.
        payload = {
            "actions": [
                {"type": "THINK", "content": "This looks moderately suitable for me.", "target": ""},
                {
                    "type": "TALK",
                    "content": json.dumps({"p_buy": 0.42, "reason": "Moderate fit with my shopping profile."}),
                    "target": "",
                },
                {"type": "DONE", "content": "Decision recorded.", "target": ""},
            ],
            "cognitive_state": {
                "goals": "Evaluate the product",
                "context": ["Independent shopping decision"],
                "attention": "Product fit",
                "emotions": "Neutral",
            },
        }
        return {"role": "assistant", "content": json.dumps(payload)}


def main() -> None:
    # ActionGenerator imports `client` into its own module namespace, so patch there.
    import tinytroupe.agent.action_generator as action_generator_module

    action_generator_module.client = lambda: FakeClient()

    with tempfile.TemporaryDirectory(prefix="hm_tt_offline_") as td:
        root = Path(td)
        bundle = root / "bundle"
        bundle.mkdir()

        persona = {
            "persona_key": "TT-C-p00",
            "arm": "TT-C",
            "persona_idx": 0,
            "persona_id": "p00",
            "n_customers": 1,
            "population_weight": 1.0,
            "persona": {
                "name": "H&M segment 00",
                "age": "25-34",
                "nationality": None,
                "country_of_residence": None,
                "occupation": None,
                "shopping_profile": {
                    "age_group": "25-34",
                    "engagement_level": "mid",
                    "typical_paid_price_tier": "mid",
                    "most_common_historical_product_type": "Trousers",
                },
            },
        }
        personas_path = bundle / "personas.jsonl"
        personas_path.write_text(json.dumps(persona, sort_keys=True) + "\n", encoding="utf-8")

        query = pd.DataFrame([
            {
                "query_id": "TT-C-a123-p00",
                "arm": "TT-C",
                "article_id": 123,
                "persona_idx": 0,
                "persona_key": "TT-C-p00",
                "product_prompt": hm._product_prompt(pd.Series({"prod_name": "Test trousers", "colour_group_name": "Black"})),
            }
        ])
        queries_path = bundle / "queries.csv"
        query.to_csv(queries_path, index=False)

        q_sha = hm._sha256_file(queries_path)
        p_sha = hm._sha256_file(personas_path)
        plan_sha = hashlib.sha256(f"{q_sha}:{p_sha}".encode()).hexdigest()
        (bundle / "manifest.json").write_text(
            json.dumps({"query_plan_sha256": plan_sha, "n_products": 1}), encoding="utf-8"
        )

        output = root / "responses.jsonl"
        args = argparse.Namespace(
            bundle=bundle,
            output=output,
            cache_file=root / "cache.json",
            model="gpt-5-mini",
            temperature=None,
            reasoning_effort=None,
            workers=1,
            replicate=1,
            arms=None,
            max_products=None,
            max_queries=1,
            resume=True,
        )
        hm.run(args)

        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 1, rows
        row = rows[0]
        assert row["status"] == "ok", row
        assert row["query_id"] == "TT-C-a123-p00"
        assert row["p_buy"] == 0.42
        assert row["raw_talk"].startswith("{")
        assert row["query_plan_sha256"] == plan_sha
        print("Offline TinyTroupe H&M run-path smoke passed:", json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
