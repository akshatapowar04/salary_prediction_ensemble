import numpy as np

from data import generate_dataset, split_data
from models import get_models, make_pipeline


def test_dataset_shape_and_no_missing():
    df = generate_dataset(n=200)
    assert df.shape == (200, 8)
    assert df.isnull().sum().sum() == 0


def test_split_is_80_20():
    X_train, X_test, _, _ = split_data(generate_dataset(n=200))
    assert len(X_train) == 160 and len(X_test) == 40


def test_all_models_fit_and_predict():
    X_train, X_test, y_train, _ = split_data(generate_dataset(n=200))
    models = get_models()
    assert len(models) == 8
    for name, (_, est) in models.items():
        pred = make_pipeline(est).fit(X_train, y_train).predict(X_test)
        assert pred.shape == (len(X_test),) and np.isfinite(pred).all(), name
