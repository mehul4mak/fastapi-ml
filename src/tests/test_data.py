import pytest

from ml.data import process_data


@pytest.mark.parametrize("training", [True, False])
def test_process_data(df, cat_features, label, encoder, lb, training):
    # ACT
    X_test, y_test, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        training=training,
        label=label,
        encoder=encoder,
        lb=lb,
    )

    # ASSERT

    assert X_test.shape[0] == df.shape[0]
    assert y_test.shape[0] == df.shape[0]

    assert encoder is not None
    assert lb is not None


@pytest.mark.parametrize("training", [True, False])
def test_process_data_encoder(df, cat_features, label, encoder, training):
    # ACT

    e1 = encoder.transform(df[cat_features].values)
    X_test, y_test, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        training=training,
        label=label,
        encoder=encoder,
    )
    e2 = encoder.transform(df[cat_features].values)
    # ASSERT
    assert encoder is not None
    assert e1.all() == e2.all()


@pytest.mark.parametrize("training", [True, False])
def test_process_data_lb(df, cat_features, label, encoder, lb, training):
    # ACT

    e1 = lb.transform(df[label].values)
    X_test, y_test, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        training=training,
        encoder=encoder,
        label=label,
        lb=lb,
    )
    e2 = lb.transform(df[label].values)
    # ASSERT
    assert lb is not None
    assert e1.all() == e2.all()
