import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.data import load_data, split_data
from src.models import SEARCH_SPACES, get_models
from src.preprocessing import NumericCoercer, build_preprocessor
from src.tune import make_pipeline


@pytest.fixture(scope="module")
def data():
    X, y = load_data()
    return split_data(X, y)


def test_split_is_stratified_and_disjoint(data):
    X_tr, X_te, y_tr, y_te = data
    assert abs(y_tr.mean() - y_te.mean()) < 0.01
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert len(X_tr) + len(X_te) == 7043


def test_numeric_coercer_handles_blank_strings():
    df = pd.DataFrame({"TotalCharges": ["10.5", " ", "3"]})
    out = NumericCoercer(["TotalCharges"]).fit_transform(df)
    assert out["TotalCharges"].isna().sum() == 1
    assert out["TotalCharges"].iloc[0] == 10.5


def test_preprocessor_output_numeric_no_nan(data):
    X_tr, X_te, _, _ = data
    pre = build_preprocessor().fit(X_tr)
    out = pre.transform(X_te)
    assert not np.isnan(out).any()
    assert out.dtype.kind == "f"


def test_preprocessor_handles_unseen_category(data):
    X_tr, X_te, _, _ = data
    pre = build_preprocessor().fit(X_tr)
    new = X_te.head(3).copy()
    new["Contract"] = "Five year"  # never seen
    assert pre.transform(new).shape[1] == pre.transform(X_te.head(3)).shape[1]


def test_every_model_has_a_search_space_with_valid_params():
    for name, model in get_models().items():
        _, space = SEARCH_SPACES[name]
        pipe = make_pipeline(model)
        valid = pipe.get_params().keys()
        for param in space:
            assert param in valid, f"{param} is not a parameter of {name}"


@pytest.mark.parametrize("name", ["LogisticRegression", "XGBoost"])
def test_pipeline_cv_beats_chance(data, name):
    X_tr, _, y_tr, _ = data
    pipe = make_pipeline(get_models()[name])
    score = cross_val_score(pipe, X_tr.head(2000), y_tr.head(2000), cv=3, scoring="roc_auc").mean()
    assert score > 0.75


def test_ai_suggested_space_has_valid_params_and_sane_ranges():
    from src.models import AI_SUGGESTED_XGB_SPACE

    pipe = make_pipeline(get_models()["XGBoost"])
    valid = pipe.get_params().keys()
    assert all(p in valid for p in AI_SUGGESTED_XGB_SPACE)
    # same parameter set as the original XGBoost space -> a fair comparison
    assert set(AI_SUGGESTED_XGB_SPACE) == set(SEARCH_SPACES["XGBoost"][1])
    depth = AI_SUGGESTED_XGB_SPACE["model__max_depth"].rvs(200, random_state=0)
    assert depth.min() >= 3 and depth.max() <= 6
