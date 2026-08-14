import importlib.util
from pathlib import Path

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "examples" / "hm_benchmark" / "hm_benchmark.py"
spec = importlib.util.spec_from_file_location("hm_benchmark", MODULE)
hm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hm)


def test_parse_json_object_plain_and_fenced():
    assert hm._parse_json_object('{"p_buy": 0.7, "reason": "x"}')["p_buy"] == 0.7
    assert hm._parse_json_object('```json\n{"p_buy": 0.2, "reason": "x"}\n```')["p_buy"] == 0.2


def test_population_weighted_normalization():
    pred = hm._normalize_affinities(np.array([0.5, 0.5]), np.array([0.8, 0.2]))
    assert np.allclose(pred, [0.8, 0.2])
    pred2 = hm._normalize_affinities(np.array([1.0, 0.25]), np.array([0.5, 0.5]))
    assert np.allclose(pred2, [0.8, 0.2])


def test_evaluate_counts_prefers_correct_prediction():
    counts = np.array([9.0, 1.0])
    good = hm._evaluate_counts(counts, np.array([0.9, 0.1]))
    bad = hm._evaluate_counts(counts, np.array([0.1, 0.9]))
    assert good["nll"] < bad["nll"]
    assert good["js_divergence"] < bad["js_divergence"]


def test_product_prompt_omits_missing_values():
    row = hm.pd.Series({
        "prod_name": "Tailored trousers",
        "product_type_name": np.nan,
        "colour_group_name": "Black",
        "graphical_appearance_name": None,
        "detail_desc": "nan",
    })
    prompt = hm._product_prompt(row)
    assert "Product name: Tailored trousers" in prompt
    assert "Colour: Black" in prompt
    assert "nan" not in prompt.lower()
    assert "Product type:" not in prompt
