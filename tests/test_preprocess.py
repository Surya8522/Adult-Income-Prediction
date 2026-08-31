import pandas as pd

from src.preprocess import prepare_data


def test_prepare_data():

    data = pd.DataFrame({
        "age": [25, 35],
        "education": ["Bachelors", "Masters"],
        "income": ["<=50K", ">50K"]
    })

    X, y = prepare_data(data)

    assert "income" not in X.columns
    assert len(X) == 2
    assert list(y) == [0, 1]