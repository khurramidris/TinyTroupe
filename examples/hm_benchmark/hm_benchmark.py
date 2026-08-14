from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEED = 20260815
N_PERSONAS = 50
OUTCOME_DAYS = 28
CUTOFF = pd.Timestamp("2019-09-20")
TEST_START = pd.Timestamp("2019-10-18")
TEST_END = pd.Timestamp("2020-01-31")
ARTICLE_DATASET = "microsoft/hnm-search-data"
ARTICLE_REVISION = "ac35fedf926b4a7e7ca4a4303ee275db866abd8f"
DEFAULT_PRODUCTS = 50
EPS = 1e-12
BOOTSTRAPS = 2000


def _read_csv_from_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX/")]
        if not members:
            raise RuntimeError(f"No CSV found in {path}")
        with zf.open(sorted(members)[0]) as fh:
            return pd.read_csv(fh)


def _qcut(values: pd.Series, labels: list[str]) -> pd.Series:
    out = pd.qcut(values, q=len(labels), labels=labels, duplicates="drop")
    if out.nunique(dropna=True) >= 2:
        return out
    ranked = values.rank(method="average")
    n_bins = min(len(labels), int(ranked.nunique()))
    return pd.qcut(ranked, q=n_bins, labels=labels[:n_bins], duplicates="drop")


def _persona_assignment(customer_features: pd.DataFrame, cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cf = customer_features.copy()
    cf["customer_id"] = cf["customer_id"].astype(str)
    cf["txn_count"] = pd.to_numeric(cf["txn_count"], errors="coerce").fillna(0.0)
    cf["mean_price"] = pd.to_numeric(cf["mean_price"], errors="coerce").fillna(0.0)
    cf["age"] = pd.to_numeric(cf["age"], errors="coerce")
    active = cf[cf["txn_count"] > 0].copy()
    active["age_bin"] = pd.cut(
        active["age"], bins=[16, 25, 35, 45, 55, 200], right=False,
        labels=["16-24", "25-34", "35-44", "45-54", "55+"],
    )
    span_months = max(((pd.Timestamp("2019-09-19") - pd.Timestamp("2018-09-01")).days + 1) / 30.4375, 1e-6)
    active["txn_per_month"] = active["txn_count"] / span_months
    active["engagement_bin"] = _qcut(active["txn_per_month"], ["low", "mid", "high"])
    active["price_tier"] = _qcut(active["mean_price"], ["low", "mid", "high"])
    top_types = active["top_product_type"].value_counts().head(100).index
    active["taste_bucket"] = np.where(active["top_product_type"].isin(top_types), active["top_product_type"], "OTHER")

    frozen = cells.head(N_PERSONAS).copy().reset_index(drop=True)
    frozen["persona_idx"] = np.arange(len(frozen), dtype=int)
    keys = ["age_bin", "engagement_bin", "price_tier", "taste_bucket"]
    for col in keys:
        active[col] = active[col].astype("string")
        frozen[col] = frozen[col].astype("string")
    active = active.merge(frozen[keys + ["persona_id", "persona_idx"]], on=keys, how="left")
    mapped = active.dropna(subset=["persona_idx"]).copy()
    mapped["persona_idx"] = mapped["persona_idx"].astype(int)
    return mapped, frozen


def _buyer_counts(tx: pd.DataFrame, customer_to_persona: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    first_sale = tx.groupby("article_id")["date"].min().sort_index()
    tmp = tx[["article_id", "customer_id", "date"]].copy()
    tmp["launch_date"] = tmp["article_id"].map(first_sale)
    tmp["days_from_launch"] = (tmp["date"] - tmp["launch_date"]).dt.days
    tmp = tmp[(tmp["days_from_launch"] >= 0) & (tmp["days_from_launch"] < OUTCOME_DAYS)]
    tmp = tmp.drop_duplicates(["article_id", "customer_id"])
    tmp["persona_idx"] = tmp["customer_id"].map(customer_to_persona)
    tmp = tmp.dropna(subset=["persona_idx"]).copy()
    tmp["persona_idx"] = tmp["persona_idx"].astype(int)
    counts = tmp.groupby(["article_id", "persona_idx"], as_index=False).size().rename(columns={"size": "buyer_count"})
    return first_sale, counts


def _load_articles(article_ids: set[int]) -> tuple[pd.DataFrame, str]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(repo_id=ARTICLE_DATASET, repo_type="dataset", revision=ARTICLE_REVISION)
    candidates = [
        f for f in files
        if "article" in f.lower() and (f.lower().endswith(".parquet") or f.lower().endswith(".csv")) and "image" not in f.lower()
    ]
    if not candidates:
        raise RuntimeError(f"No article metadata file found in {ARTICLE_DATASET}@{ARTICLE_REVISION}")

    def priority(name: str) -> tuple:
        lower = name.lower()
        return (
            0 if lower.startswith("articles/") else 1,
            0 if "raw/articles" in lower else 1,
            0 if lower.endswith(".parquet") else 1,
            len(name),
            name,
        )

    filename = sorted(candidates, key=priority)[0]
    local = hf_hub_download(repo_id=ARTICLE_DATASET, repo_type="dataset", revision=ARTICLE_REVISION, filename=filename)
    path = Path(local)
    art = pd.read_parquet(path) if filename.lower().endswith(".parquet") else pd.read_csv(path)
    art["article_id"] = pd.to_numeric(art["article_id"], errors="coerce").astype("Int64")
    art = art[art["article_id"].isin(article_ids)].copy()
    art["article_id"] = art["article_id"].astype(np.int64)
    art = art.drop_duplicates("article_id")
    fingerprint = hashlib.sha256(f"{ARTICLE_DATASET}@{ARTICLE_REVISION}:{filename}".encode()).hexdigest()
    return art, fingerprint


def _safe_list(series: pd.Series, n: int) -> list[str]:
    vals = series.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if vals.empty:
        return []
    return vals.value_counts().head(n).index.tolist()


def _clean_text_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _product_prompt(row: pd.Series) -> str:
    fields = [
        ("Product name", "prod_name"),
        ("Product type", "product_type_name"),
        ("Colour", "colour_group_name"),
        ("Graphical appearance", "graphical_appearance_name"),
        ("Garment group", "garment_group_name"),
        ("Section", "section_name"),
        ("Description", "detail_desc"),
    ]
    parts = []
    for label, key in fields:
        value = _clean_text_value(row.get(key, ""))
        if value:
            parts.append(f"{label}: {value}")
    product = "\n".join(parts)
    return (
        "You are making an independent H&M shopping decision. Consider this newly appearing trouser product:\n\n"
        f"{product}\n\n"
        "Estimate your own relative affinity for buying this product using only your persona and the product information above. "
        "Do not infer popularity, inventory, recommendations, other customers' behavior, or any future outcome. "
        "Use a probability-like affinity between 0 and 1; it is a comparative score for this benchmark, not a calibrated conversion probability.\n\n"
        "Respond with a single SPEAK action whose content is exactly one JSON object in this form: "
        '{"p_buy": <number from 0 to 1>, "reason": "<=25 words"}. Then finish with DONE.'
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _run_hm_baseline(hm_repo: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    script = hm_repo / "scripts" / "research_hm_exp007_confirmation.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing H&M Experiment 007 script: {script}")
    cmd = [
        sys.executable,
        str(script),
        "--transactions", str(hm_repo / "outputs/products/txns_trousers_online.csv.zip"),
        "--customer-features", str(hm_repo / "outputs/personas/customer_features.csv.zip"),
        "--persona-cells", str(hm_repo / "outputs/personas/persona_cells.csv"),
        "--output-dir", str(out_dir),
    ]
    subprocess.run(cmd, cwd=hm_repo / "scripts", check=True)


def prepare(args: argparse.Namespace) -> None:
    hm_repo = args.hm_repo.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    baseline_dir = out / "hm_exp007_baseline"
    if not args.skip_baseline:
        _run_hm_baseline(hm_repo, baseline_dir)
    required = [baseline_dir / "qualified_test_articles.csv", baseline_dir / "product_metrics.csv", baseline_dir / "summary.json"]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing baseline outputs: {missing}")

    tx = _read_csv_from_zip(hm_repo / "outputs/products/txns_trousers_online.csv.zip")
    tx = tx.rename(columns={"t_dat": "date"})
    tx["date"] = pd.to_datetime(tx["date"])
    tx["customer_id"] = tx["customer_id"].astype(str)
    tx["article_id"] = pd.to_numeric(tx["article_id"], errors="raise").astype(np.int64)
    tx["price"] = pd.to_numeric(tx["price"], errors="coerce")
    if "sales_channel_id" in tx and not (tx["sales_channel_id"] == 1).all():
        raise AssertionError("Expected committed trouser transaction source to be online-only")

    customer_features = _read_csv_from_zip(hm_repo / "outputs/personas/customer_features.csv.zip")
    cells = pd.read_csv(hm_repo / "outputs/personas/persona_cells.csv")
    mapped, frozen = _persona_assignment(customer_features, cells)
    customer_to_persona = mapped.set_index("customer_id")["persona_idx"]
    _, counts = _buyer_counts(tx, customer_to_persona)

    qualified = pd.read_csv(baseline_dir / "qualified_test_articles.csv", parse_dates=["launch_date"])
    qualified["article_id"] = qualified["article_id"].astype(np.int64)
    qualified = qualified[(qualified["launch_date"] >= TEST_START) & (qualified["launch_date"] <= TEST_END)].copy()
    # Match the frozen H&M Stage-B selection exactly: support first, article_id tie-break only.
    qualified = qualified.sort_values(["mapped_buyers", "article_id"], ascending=[False, True])
    selected = qualified.head(args.n_products).copy()
    if len(selected) < args.n_products:
        raise RuntimeError(f"Only {len(selected)} qualified products available; requested {args.n_products}")
    selected_ids = selected["article_id"].astype(int).tolist()

    articles, article_fingerprint = _load_articles(set(int(x) for x in tx["article_id"].unique()))
    article_lookup = articles.set_index("article_id")
    missing_articles = [a for a in selected_ids if a not in article_lookup.index]
    if missing_articles:
        raise RuntimeError(f"Missing article metadata for selected IDs: {missing_articles[:10]}")

    pre = tx[tx["date"] < CUTOFF][["article_id", "customer_id", "price"]].drop_duplicates(["article_id", "customer_id"])
    pre["persona_idx"] = pre["customer_id"].map(customer_to_persona)
    pre = pre.dropna(subset=["persona_idx"]).copy()
    pre["persona_idx"] = pre["persona_idx"].astype(int)
    enrich_cols = ["article_id", "prod_name", "colour_group_name", "graphical_appearance_name", "garment_group_name", "section_name", "product_type_name", "detail_desc"]
    pre = pre.merge(articles[enrich_cols], on="article_id", how="left")

    persona_rows: list[dict[str, Any]] = []
    frozen = frozen.reset_index(drop=True)
    total_population = float(pd.to_numeric(frozen["n_customers"], errors="coerce").fillna(0).sum())
    for pidx, cell in frozen.iterrows():
        g = pre[pre["persona_idx"] == int(pidx)].copy()
        coarse_profile = {
            "age_group": str(cell["age_bin"]),
            "engagement_level": str(cell["engagement_bin"]),
            "typical_paid_price_tier": str(cell["price_tier"]),
            "most_common_historical_product_type": str(cell["taste_bucket"]),
        }
        # Keep TT-R aligned to the actual rich evidence rendered by the frozen H&M Stage-B prompt.
        # Do not expose raw H&M normalized prices or support counts as additional treatment variables.
        rich_evidence = {
            "top_colours": _safe_list(g["colour_group_name"], 4),
            "top_graphical_appearances": _safe_list(g["graphical_appearance_name"], 3),
            "top_garment_groups": _safe_list(g["garment_group_name"], 3),
            "top_sections": _safe_list(g["section_name"], 3),
            "representative_products": _safe_list(g["prod_name"], 5),
            "evidence_cutoff": CUTOFF.date().isoformat(),
        }
        n_customers = int(cell["n_customers"])
        pop_weight = n_customers / total_population if total_population else 0.0
        for arm in ("TT-C", "TT-R"):
            persona = {
                "name": f"H&M segment {int(pidx):02d}",
                "age": str(cell["age_bin"]),
                "nationality": "Not observed in the H&M benchmark data",
                "country_of_residence": "Not observed in the H&M benchmark data",
                "occupation": "Synthetic H&M customer segment; occupation unobserved",
                "shopping_profile": coarse_profile,
                "evidence_provenance": {
                    "source": "H&M public transactions and pre-cutoff persona-cell construction",
                    "observed_or_derived_before": CUTOFF.date().isoformat(),
                    "future_behavior_included": False,
                    "unobserved_demographics_are_not_inferred": True,
                },
            }
            if arm == "TT-R":
                persona["historical_trouser_style_evidence"] = rich_evidence
            persona_rows.append({
                "persona_key": f"{arm}-p{int(pidx):02d}",
                "arm": arm,
                "persona_idx": int(pidx),
                "persona_id": str(cell["persona_id"]),
                "n_customers": n_customers,
                "population_weight": pop_weight,
                "persona": persona,
            })

    personas_path = out / "personas.jsonl"
    with personas_path.open("w", encoding="utf-8") as fh:
        for row in persona_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    products = articles[articles["article_id"].isin(selected_ids)].copy()
    products = products.merge(selected[["article_id", "launch_date", "mapped_buyers"]], on="article_id", how="left")
    products["product_prompt"] = products.apply(_product_prompt, axis=1)
    product_cols = ["article_id", "launch_date", "mapped_buyers", "prod_name", "product_type_name", "colour_group_name", "graphical_appearance_name", "garment_group_name", "section_name", "detail_desc", "product_prompt"]
    products = products[product_cols].sort_values(["launch_date", "article_id"]).reset_index(drop=True)
    products.to_csv(out / "products.csv", index=False)

    queries = []
    persona_df = pd.DataFrame([{k: v for k, v in r.items() if k != "persona"} for r in persona_rows])
    for _, prod in products.iterrows():
        for _, pr in persona_df.iterrows():
            queries.append({
                "query_id": f"{pr['arm']}-a{int(prod['article_id'])}-p{int(pr['persona_idx']):02d}",
                "arm": pr["arm"],
                "article_id": int(prod["article_id"]),
                "persona_idx": int(pr["persona_idx"]),
                "persona_key": pr["persona_key"],
                "product_prompt": prod["product_prompt"],
            })
    queries_df = pd.DataFrame(queries).sort_values(["arm", "article_id", "persona_idx"]).reset_index(drop=True)
    queries_path = out / "queries.csv"
    queries_df.to_csv(queries_path, index=False, quoting=csv.QUOTE_MINIMAL)

    query_sha = _sha256_file(queries_path)
    persona_sha = _sha256_file(personas_path)
    plan_sha = hashlib.sha256(f"{query_sha}:{persona_sha}".encode()).hexdigest()

    selected_counts = counts[counts["article_id"].isin(selected_ids)].copy()
    selected_counts.to_csv(out / "buyer_counts.csv", index=False)

    baseline_products = pd.read_csv(baseline_dir / "product_metrics.csv")
    baseline_products = baseline_products[baseline_products["article_id"].isin(selected_ids)].copy()
    baseline_products.to_csv(out / "baseline_product_metrics.csv", index=False)
    subset_metrics = []
    for model, g in baseline_products.groupby("model"):
        subset_metrics.append({
            "model": model,
            "buyer_nll": float(g["nll_sum"].sum() / g["n_buyers"].sum()),
            "macro_js": float(g["js_divergence"].mean()),
            "macro_tv": float(g["tv_distance"].mean()),
            "buyers": int(g["n_buyers"].sum()),
            "articles": int(g["article_id"].nunique()),
        })
    pd.DataFrame(subset_metrics).sort_values("buyer_nll").to_csv(out / "baseline_metrics_subset.csv", index=False)

    baseline_summary = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = {
        "scientific_status": "EXPLORATORY_RETROSPECTIVE",
        "reason": "Experiment 007 missed its preregistered 1% effect gate (0.7213%), and its outcomes were already inspected before this TinyTroupe test.",
        "hm_repo_head": _git_head(hm_repo),
        "hm_source_branch_expected": "research/07-expanded-cold-start-confirmation",
        "hm_experiment": "007_expanded_cold_start_confirmation",
        "model_information_cutoff": CUTOFF.date().isoformat(),
        "test_launch_window": [TEST_START.date().isoformat(), TEST_END.date().isoformat()],
        "outcome_days": OUTCOME_DAYS,
        "n_personas": N_PERSONAS,
        "n_products": int(len(products)),
        "n_queries": int(len(queries_df)),
        "arms": ["TT-C", "TT-R"],
        "aggregation": "population_weight * p_buy, normalized across 50 persona cells for each product",
        "query_plan_sha256": plan_sha,
        "queries_file_sha256": query_sha,
        "personas_file_sha256": persona_sha,
        "article_dataset": ARTICLE_DATASET,
        "article_revision": ARTICLE_REVISION,
        "article_fingerprint": article_fingerprint,
        "baseline_exp007_decision": baseline_summary.get("exp007_decision"),
        "selection_rule": f"Top {args.n_products} qualified Experiment 007 products by mapped unique buyers; ties by article_id.",
        "labels_not_in_query_plan": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["scientific_status", "n_products", "n_queries", "query_plan_sha256"]}, indent=2))


def _parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in SPEAK content")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object")
    return obj


def _load_personas(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            out[row["persona_key"]] = row
    return out


def run(args: argparse.Namespace) -> None:
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    queries_path = bundle / "queries.csv"
    personas_path = bundle / "personas.jsonl"
    current_plan_sha = hashlib.sha256(f"{_sha256_file(queries_path)}:{_sha256_file(personas_path)}".encode()).hexdigest()
    if current_plan_sha != manifest["query_plan_sha256"]:
        raise RuntimeError("Query/persona bundle hash does not match frozen manifest")

    queries = pd.read_csv(queries_path)
    if args.arms:
        queries = queries[queries["arm"].isin(args.arms)]
    if args.max_products is not None:
        keep_ids = sorted(queries["article_id"].unique())[: args.max_products]
        queries = queries[queries["article_id"].isin(keep_ids)]
    if args.max_queries is not None:
        queries = queries.head(args.max_queries)
    personas = _load_personas(personas_path)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists() and args.resume:
        with output.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("replicate") == args.replicate and row.get("status") == "ok":
                    done.add(row["query_id"])
    queries = queries[~queries["query_id"].isin(done)].copy()
    if queries.empty:
        print("No pending queries.")
        return

    from tinytroupe import config_manager
    from tinytroupe.agent.action_generator import ActionGenerator
    from tinytroupe.agent.tiny_person import TinyPerson

    TinyPerson.communication_display = False
    config_manager.update("model", args.model)
    config_manager.update("cache_api_calls", True)
    config_manager.update("cache_file_name", str(args.cache_file.resolve()))
    config_manager.update("max_concurrent_model_calls", args.workers)
    if args.temperature is not None:
        config_manager.update("temperature", args.temperature)
    if args.reasoning_effort is not None:
        config_manager.update("reasoning_effort", args.reasoning_effort)

    write_lock = threading.Lock()

    def one(row: pd.Series) -> dict[str, Any]:
        qid = str(row["query_id"])
        persona_row = personas[str(row["persona_key"])]
        unique_name = f"{qid}-r{args.replicate}"
        action_generator = ActionGenerator(
            max_attempts=1,
            enable_quality_checks=False,
            enable_regeneration=False,
            enable_direct_correction=False,
            continue_on_failure=True,
            enable_multi_action_output=True,
        )
        agent = TinyPerson(name=unique_name, action_generator=action_generator, enable_basic_action_repetition_prevention=False)
        persona = json.loads(json.dumps(persona_row["persona"]))
        persona["name"] = unique_name
        agent.include_persona_definitions(persona)
        t0 = time.time()
        record: dict[str, Any] = {
            "query_id": qid,
            "arm": str(row["arm"]),
            "article_id": int(row["article_id"]),
            "persona_idx": int(row["persona_idx"]),
            "persona_key": str(row["persona_key"]),
            "replicate": int(args.replicate),
            "model": args.model,
            "query_plan_sha256": manifest["query_plan_sha256"],
        }
        try:
            actions = agent.listen_and_act(str(row["product_prompt"]), return_actions=True, communication_display=False)
            speak_contents = []
            for item in actions or []:
                action = item.get("action", {}) if isinstance(item, dict) else {}
                if action.get("type") == "SPEAK":
                    speak_contents.append(str(action.get("content", "")))
            if not speak_contents:
                raise ValueError(f"No SPEAK action returned; actions={actions}")
            raw = speak_contents[-1]
            payload = _parse_json_object(raw)
            p_buy = float(payload["p_buy"])
            if not math.isfinite(p_buy) or p_buy < 0.0 or p_buy > 1.0:
                raise ValueError(f"p_buy outside [0,1]: {p_buy}")
            record.update({"status": "ok", "p_buy": p_buy, "reason": str(payload.get("reason", ""))[:500], "raw_speak": raw})
        except Exception as exc:
            record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            record["elapsed_seconds"] = time.time() - t0
            try:
                TinyPerson.all_agents.pop(unique_name, None)
            except Exception:
                pass
        return record

    with output.open("a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one, row): str(row["query_id"]) for _, row in queries.iterrows()}
            completed = 0
            for fut in as_completed(futures):
                rec = fut.result()
                with write_lock:
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")
                    fh.flush()
                completed += 1
                if completed % 25 == 0 or completed == len(futures):
                    ok = "ok" if rec.get("status") == "ok" else "error"
                    print(f"completed {completed}/{len(futures)}; latest={ok}; {rec['query_id']}")


def _normalize_affinities(p_buy: np.ndarray, weights: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p_buy, dtype=float), 0.0, 1.0)
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    mass = w * np.maximum(p, 1e-6)
    if mass.sum() <= 0:
        mass = np.maximum(w, EPS)
    mass = np.maximum(mass, EPS)
    return mass / mass.sum()


def _js(actual: np.ndarray, pred: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    a = a / a.sum()
    p = np.clip(p, EPS, None)
    p = p / p.sum()
    m = 0.5 * (a + p)
    left = np.where(a > 0, a * np.log(a / m), 0.0)
    right = p * np.log(p / m)
    return float(0.5 * left.sum() + 0.5 * right.sum())


def _evaluate_counts(counts: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    c = np.asarray(counts, dtype=float)
    p = np.clip(np.asarray(pred, dtype=float), EPS, None)
    p = p / p.sum()
    n = float(c.sum())
    actual = c / n
    order = np.argsort(-p)
    return {
        "n_buyers": int(n),
        "nll_sum": float(np.sum(c * -np.log(p))),
        "nll": float(np.sum(c * -np.log(p)) / n),
        "top1_hits": float(c[order[0]]),
        "top5_hits": float(c[order[:5]].sum()),
        "js_divergence": _js(actual, p),
        "tv_distance": float(0.5 * np.abs(actual - p).sum()),
    }


def _bootstrap_diff(product_df: pd.DataFrame, model: str, baseline: str, n_boot: int = BOOTSTRAPS) -> tuple[float, float, float]:
    m = product_df[product_df["model"] == model].set_index("article_id")
    b = product_df[product_df["model"] == baseline].set_index("article_id")
    ids = sorted(set(m.index).intersection(b.index))
    if not ids:
        return float("nan"), float("nan"), float("nan")
    diff_sum = (m.loc[ids, "nll_sum"] - b.loc[ids, "nll_sum"]).to_numpy(dtype=float)
    n = m.loc[ids, "n_buyers"].to_numpy(dtype=float)
    point = float(diff_sum.sum() / n.sum())
    rng = np.random.default_rng(SEED)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(ids), size=len(ids))
        boots[i] = diff_sum[idx].sum() / n[idx].sum()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def _delta_for_ids(product_df: pd.DataFrame, model: str, baseline: str, ids: list[int]) -> float:
    m = product_df[(product_df["model"] == model) & (product_df["article_id"].isin(ids))].set_index("article_id")
    b = product_df[(product_df["model"] == baseline) & (product_df["article_id"].isin(ids))].set_index("article_id")
    common = sorted(set(m.index).intersection(b.index))
    if not common:
        return float("nan")
    diff = (m.loc[common, "nll_sum"] - b.loc[common, "nll_sum"]).to_numpy(float)
    n = m.loc[common, "n_buyers"].to_numpy(float)
    return float(diff.sum() / n.sum())


def evaluate(args: argparse.Namespace) -> None:
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    personas = pd.DataFrame(_load_personas(bundle / "personas.jsonl").values())
    persona_weights = personas[["arm", "persona_idx", "population_weight"]].drop_duplicates()
    products = pd.read_csv(bundle / "products.csv", parse_dates=["launch_date"])
    counts_df = pd.read_csv(bundle / "buyer_counts.csv")
    baseline = pd.read_csv(bundle / "baseline_product_metrics.csv")

    records = []
    with args.responses.resolve().open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    resp = pd.DataFrame(records)
    if resp.empty:
        raise RuntimeError("No responses found")
    resp = resp[resp["status"] == "ok"].copy()
    if args.replicate is not None:
        resp = resp[resp["replicate"] == args.replicate]

    count_map: dict[int, np.ndarray] = {}
    for aid, g in counts_df.groupby("article_id"):
        v = np.zeros(N_PERSONAS, dtype=float)
        v[g["persona_idx"].to_numpy(int)] = g["buyer_count"].to_numpy(float)
        count_map[int(aid)] = v

    product_rows: list[dict[str, Any]] = []
    completeness = []
    for (rep, arm, aid), g in resp.groupby(["replicate", "arm", "article_id"]):
        g = g.drop_duplicates("persona_idx", keep="last")
        expected = N_PERSONAS
        completeness.append({"replicate": int(rep), "arm": arm, "article_id": int(aid), "valid_personas": int(g["persona_idx"].nunique()), "expected_personas": expected})
        if g["persona_idx"].nunique() != expected:
            continue
        w = persona_weights[persona_weights["arm"] == arm].set_index("persona_idx")["population_weight"]
        g = g.set_index("persona_idx").reindex(range(N_PERSONAS))
        pred = _normalize_affinities(g["p_buy"].to_numpy(float), w.reindex(range(N_PERSONAS)).to_numpy(float))
        metrics = _evaluate_counts(count_map[int(aid)], pred)
        product_rows.append({"replicate": int(rep), "article_id": int(aid), "model": arm, **metrics})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(completeness).to_csv(args.output_dir / "response_completeness.csv", index=False)
    if not product_rows:
        raise RuntimeError("No complete product-arm blocks with all 50 persona responses")
    llm_products = pd.DataFrame(product_rows)
    reps = sorted(llm_products["replicate"].unique())
    if len(reps) != 1:
        raise RuntimeError(f"Evaluate one replicate at a time; found {reps}. Use --replicate.")
    rep = int(reps[0])
    llm_products = llm_products[llm_products["replicate"] == rep].drop(columns=["replicate"])

    common_ids = sorted(set(products["article_id"].astype(int)).intersection(*[
        set(llm_products[llm_products["model"] == arm]["article_id"].astype(int)) for arm in ["TT-C", "TT-R"]
    ]))
    if len(common_ids) < manifest["n_products"] and not args.allow_incomplete:
        raise RuntimeError(f"Only {len(common_ids)}/{manifest['n_products']} products complete in both arms. Re-run missing queries or pass --allow-incomplete.")

    llm_products = llm_products[llm_products["article_id"].isin(common_ids)].copy()
    baseline = baseline[baseline["article_id"].isin(common_ids)].copy()
    all_products = pd.concat([baseline, llm_products], ignore_index=True, sort=False)
    all_products.to_csv(args.output_dir / "all_product_metrics.csv", index=False)

    summary_rows = []
    for model, g in all_products.groupby("model"):
        row = {
            "model": model,
            "buyer_nll": float(g["nll_sum"].sum() / g["n_buyers"].sum()),
            "macro_js": float(g["js_divergence"].mean()),
            "macro_tv": float(g["tv_distance"].mean()),
            "buyers": int(g["n_buyers"].sum()),
            "articles": int(g["article_id"].nunique()),
        }
        if "top1_hits" in g.columns and g["top1_hits"].notna().any():
            row["buyer_top1"] = float(g["top1_hits"].fillna(0).sum() / g["n_buyers"].sum())
            row["buyer_top5"] = float(g["top5_hits"].fillna(0).sum() / g["n_buyers"].sum())
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values("buyer_nll")
    summary_df.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    metric_map = summary_df.set_index("model").to_dict(orient="index")

    p3 = "P3_supervised_content"
    if p3 not in metric_map:
        raise RuntimeError(f"Expected baseline {p3}; got {list(metric_map)}")

    products_order = products[products["article_id"].isin(common_ids)].sort_values(["launch_date", "article_id"])["article_id"].astype(int).tolist()
    split = len(products_order) // 2
    early, late = products_order[:split], products_order[split:]

    comparisons = {}
    for arm in ["TT-C", "TT-R"]:
        point, lo, hi = _bootstrap_diff(all_products, arm, p3)
        rel = float((metric_map[p3]["buyer_nll"] - metric_map[arm]["buyer_nll"]) / metric_map[p3]["buyer_nll"])
        js_ratio = float(metric_map[arm]["macro_js"] / metric_map[p3]["macro_js"])
        early_delta = _delta_for_ids(all_products, arm, p3, early)
        late_delta = _delta_for_ids(all_products, arm, p3, late)
        comparisons[f"{arm}_vs_P3"] = {
            "delta_nll": point,
            "bootstrap_95_ci": [lo, hi],
            "relative_nll_improvement": rel,
            "macro_js_ratio": js_ratio,
            "early_delta_nll": early_delta,
            "late_delta_nll": late_delta,
            "meets_exploratory_effect_gate": bool(rel >= 0.01 and hi < 0.0 and js_ratio <= 1.05 and early_delta < 0.0 and late_delta < 0.0),
        }

    point, lo, hi = _bootstrap_diff(all_products, "TT-R", "TT-C")
    comparisons["TT-R_vs_TT-C"] = {
        "delta_nll": point,
        "bootstrap_95_ci": [lo, hi],
        "relative_nll_improvement": float((metric_map["TT-C"]["buyer_nll"] - metric_map["TT-R"]["buyer_nll"]) / metric_map["TT-C"]["buyer_nll"]),
        "early_delta_nll": _delta_for_ids(all_products, "TT-R", "TT-C", early),
        "late_delta_nll": _delta_for_ids(all_products, "TT-R", "TT-C", late),
        "favorable_with_ci": bool(hi < 0.0),
    }

    result = {
        "scientific_status": "EXPLORATORY_RETROSPECTIVE_ONLY",
        "confirmatory_claim_allowed": False,
        "replicate": rep,
        "query_plan_sha256": manifest["query_plan_sha256"],
        "common_products": len(common_ids),
        "metrics": summary_df.to_dict(orient="records"),
        "comparisons": comparisons,
        "interpretation_rule": "A positive result is hypothesis-generating only because Experiment 007 missed its preregistered gate and the cohort outcomes were already inspected before this test.",
    }
    (args.output_dir / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# H&M × TinyTroupe exploratory benchmark",
        "",
        "**Scientific status: retrospective exploratory only. This is not fresh confirmatory evidence.**",
        "",
        f"Replicate: `{rep}`  ",
        f"Products complete in both arms: **{len(common_ids)}**  ",
        f"Frozen query-plan SHA-256: `{manifest['query_plan_sha256']}`",
        "",
        "## Metrics",
        "",
        "| Model | Buyer NLL ↓ | Macro JS ↓ | Macro TV ↓ |",
        "|---|---:|---:|---:|",
    ]
    for r in summary_df.to_dict(orient="records"):
        lines.append(f"| {r['model']} | {r['buyer_nll']:.6f} | {r['macro_js']:.6f} | {r['macro_tv']:.6f} |")
    lines += ["", "## Paired comparisons", ""]
    for name, c in comparisons.items():
        lines.append(f"### {name}")
        lines.append(f"- ΔNLL: **{c['delta_nll']:.6f}**")
        lines.append(f"- Product-cluster bootstrap 95% CI: **[{c['bootstrap_95_ci'][0]:.6f}, {c['bootstrap_95_ci'][1]:.6f}]**")
        if "relative_nll_improvement" in c:
            lines.append(f"- Relative NLL improvement: **{100*c['relative_nll_improvement']:.3f}%**")
        lines.append(f"- Early-half ΔNLL: **{c['early_delta_nll']:.6f}**")
        lines.append(f"- Late-half ΔNLL: **{c['late_delta_nll']:.6f}**")
        lines.append("")
    lines += [
        "## Claim limit",
        "",
        "Even if TinyTroupe beats P3 here, the result is exploratory because this cohort was previously inspected and the frozen H&M gate did not authorize a confirmatory LLM challenge. A positive result should be frozen and carried to a new untouched dataset/cohort.",
    ]
    (args.output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["comparisons"], indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Retrospective H&M cold-start buyer-composition benchmark inside TinyTroupe.")
    sub = p.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--hm-repo", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--n-products", type=int, default=DEFAULT_PRODUCTS)
    prep.add_argument("--skip-baseline", action="store_true")
    prep.set_defaults(func=prepare)

    runp = sub.add_parser("run")
    runp.add_argument("--bundle", type=Path, required=True)
    runp.add_argument("--output", type=Path, required=True)
    runp.add_argument("--cache-file", type=Path, default=Path("hm_tinytroupe_api_cache.json"))
    runp.add_argument("--model", default="gpt-5-mini")
    runp.add_argument("--temperature", type=float, default=None)
    runp.add_argument("--reasoning-effort", default=None)
    runp.add_argument("--workers", type=int, default=4)
    runp.add_argument("--replicate", type=int, default=1)
    runp.add_argument("--arms", nargs="*", choices=["TT-C", "TT-R"])
    runp.add_argument("--max-products", type=int, default=None)
    runp.add_argument("--max-queries", type=int, default=None)
    runp.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    runp.set_defaults(func=run)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--bundle", type=Path, required=True)
    ev.add_argument("--responses", type=Path, required=True)
    ev.add_argument("--output-dir", type=Path, required=True)
    ev.add_argument("--replicate", type=int, default=None)
    ev.add_argument("--allow-incomplete", action="store_true")
    ev.set_defaults(func=evaluate)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "output_dir"):
        args.output_dir.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
