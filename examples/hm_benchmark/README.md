# H&M × TinyTroupe exploratory benchmark

This experiment plugs the frozen H&M cold-start buyer-composition task from `khurramidris/LLM-demand-simulator` into TinyTroupe.

## Scientific status

**Retrospective exploratory only.** H&M Experiment 007 found a reproducible customer-style signal, but it missed its preregistered minimum practical-effect gate: P2 improved buyer NLL over P1 by 0.7213%, below the frozen 1.0% threshold. The Experiment 007 cohort has also already been inspected. Therefore this TinyTroupe experiment can generate engineering evidence and hypotheses, but it cannot be presented as fresh confirmatory validation.

The benchmark remains narrow: it predicts **which frozen customer/persona groups appear among subsequent buyers of newly appearing H&M trouser products**. It does not estimate exposure-conditioned conversion, inventory-conditioned demand, absolute demand, or causal price elasticity.

## Arms

- **TT-C** — a TinyPerson built from the coarse H&M persona-cell fields: age group, engagement level, typical paid-price tier, and most common historical product type.
- **TT-R** — the same TinyPerson plus the pre-2019-09-20 trouser style evidence actually rendered by the frozen H&M Stage-B prompt: top colours, graphical appearances, garment groups, sections, and representative product names.

No future/test behavior enters either persona. Unknown demographics are explicitly left unknown rather than invented. Raw H&M normalized prices and persona support counts are deliberately not added to TT-R because the frozen Stage-B prompt did not render them; this keeps the treatment aligned to the original H&M experimental design.

For every product/persona pair, a fresh TinyPerson makes an independent decision. Agents do not talk to each other and do not carry memory from one product to the next. The response is a single `p_buy` affinity in `[0,1]`. For each product, the 50 affinities are converted to buyer-composition probabilities using the frozen persona-cell population weights:

`predicted_mass_i = population_weight_i * p_buy_i`

followed by normalization across the 50 cells. This matches the H&M Stage-B design and avoids treating the 50 persona cells as equally sized.

## Frozen comparison set

By default the preparation step uses the 50 highest-support qualified products from the independent Experiment 007 window (2019-10-18 through 2020-01-31), with `article_id` as the frozen support-tie breaker. That gives:

- 50 products
- 50 persona cells
- 2 TinyTroupe arms
- **5,000 logical TinyTroupe evaluations**

The H&M baseline code is rerun unchanged first, then the exact selected-product P0/P1/P2/P3 product-level metrics are carried into this bundle. P3 remains the primary conventional hurdle.

## 1. Prepare the benchmark

From a checkout of this TinyTroupe branch, with the H&M repository available locally:

```bash
python examples/hm_benchmark/hm_benchmark.py prepare \
  --hm-repo ../LLM-demand-simulator \
  --output-dir outputs/hm_tinytroupe
```

The command reruns H&M Experiment 007, reconstructs the frozen top-50 population and labels, loads the locked H&M article metadata revision, writes the TinyTroupe personas/products, and freezes a label-blind `queries.csv` + `personas.jsonl` plan. A SHA-256 digest is stored in `manifest.json` before labels are written.

## 2. Run TinyTroupe

Set the normal TinyTroupe/OpenAI credentials, then:

```bash
python examples/hm_benchmark/hm_benchmark.py run \
  --bundle outputs/hm_tinytroupe \
  --output outputs/hm_tinytroupe/responses.jsonl \
  --model gpt-5-mini \
  --workers 4
```

The runner is resumable. Successful query IDs already present in the JSONL are skipped. TinyTroupe API caching is enabled as an additional guard against duplicated calls.

For a smoke test:

```bash
python examples/hm_benchmark/hm_benchmark.py run \
  --bundle outputs/hm_tinytroupe \
  --output outputs/hm_tinytroupe/smoke.jsonl \
  --model gpt-5-mini \
  --workers 2 \
  --max-queries 10
```

## 3. Evaluate

After both arms are complete:

```bash
python examples/hm_benchmark/hm_benchmark.py evaluate \
  --bundle outputs/hm_tinytroupe \
  --responses outputs/hm_tinytroupe/responses.jsonl \
  --output-dir outputs/hm_tinytroupe/evaluation \
  --replicate 1
```

The evaluator reports buyer NLL, macro Jensen-Shannon divergence, macro total-variation distance, TinyTroupe top-1/top-5 accuracy, product-cluster bootstrap uncertainty versus P3, chronological-half consistency, and TT-R versus TT-C.

The exploratory H&M Stage-B hurdle is retained for interpretability:

- at least 1% relative buyer-NLL improvement versus P3;
- product-cluster bootstrap 95% CI for `NLL_TT - NLL_P3` strictly below zero;
- macro JS no more than 5% worse than P3;
- favorable direction in both chronological halves;
- TT-R must also beat TT-C with paired product-level uncertainty before richer personas are described as helpful.

Even if these are met, `evaluation.json` deliberately keeps `confirmatory_claim_allowed: false`. A positive result must be frozen and carried to a genuinely untouched cohort/dataset before being used as scientific validation.

## Why this is inside TinyTroupe rather than a generic prompt loop

Each query creates an actual `TinyPerson`, injects the arm-specific persona into TinyTroupe's persona model, presents the cold product as a stimulus, and uses TinyTroupe's action-generation path. Quality-check/regeneration loops are disabled so the representation—not extra judge/correction calls—is the intended arm difference. Every product starts from a fresh agent to prevent product-order memory leakage.
