import pandas as pd
import numpy as np
import joblib

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, FunctionTransformer
from sklearn.linear_model import LogisticRegression

from pytorch_tabnet.tab_model import TabNetClassifier

from src.config import RANDOM_SEED
from src.models.transforms import cyclical_transform
from src.models.stacked_model import StackedModel


PARAMS = {
    'learning_rate': 0.050623168535997125,
    'max_iter': 1390,
    'max_leaf_nodes': 2260,
    'max_depth': 9,
    'min_samples_leaf': 41,
    'l2_regularization': 1.1298645707113427e-07,
    'max_bins': 213,
}

CATEGORICAL_FEATURES = [
    'job', 'marital', 'education',
    'default', 'housing', 'loan',
    'contact', 'poutcome',
]

NUMERICAL_FEATURES = [
    'age', 'balance', 'campaign',
    'pdays', 'previous',
]

CYCLICAL_FEATURES = [
    'month_sin', 'month_cos',
    'day_sin', 'day_cos',
]


def train_and_save_model(train_file: str, model_out: str):

    train = pd.read_parquet(train_file)

    X_train = train.drop(columns=['y'])
    y_train = train['y']

    # -----------------------------
    # Step 1: Feature engineering
    # -----------------------------

    cyclical_transformer = FunctionTransformer(cyclical_transform)

    preprocessor_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
            ('num', PowerTransformer(method='yeo-johnson'), NUMERICAL_FEATURES),
            ('cyclical', 'passthrough', CYCLICAL_FEATURES),
        ],
        remainder='passthrough'
    )

    preprocessing_pipeline = Pipeline([
        ("cyclical", cyclical_transformer),
        ("preprocessor", preprocessor_transformer),
    ])

    # Fit preprocessing
    X_processed = preprocessing_pipeline.fit_transform(X_train)

    # -----------------------------
    # Step 2: Train LightGBM
    # -----------------------------

    lgbm_model = LGBMClassifier(
        **PARAMS,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )

    lgbm_model.fit(X_processed, y_train)

    p_lgbm = lgbm_model.predict_proba(X_processed)[:, 1]

    # -----------------------------
    # Step 3: Train TabNet
    # -----------------------------

    tabnet_model = TabNetClassifier(seed=RANDOM_SEED)

    tabnet_model.fit(
        X_processed,
        y_train.values,
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
    )

    p_tabnet = tabnet_model.predict_proba(X_processed)[:, 1]

    # -----------------------------
    # Step 4: Train Meta Model
    # -----------------------------

    Z = np.column_stack((p_lgbm, p_tabnet))

    meta_model = LogisticRegression(random_state=RANDOM_SEED)

    meta_model.fit(Z, y_train)

    # -----------------------------
    # Step 5: Build Stacked Model
    # -----------------------------

    stacked_model = StackedModel(
        preprocessor=preprocessing_pipeline,
        lgbm_model=lgbm_model,
        tabnet_model=tabnet_model,
        meta_model=meta_model
    )

    # Save model
    joblib.dump(stacked_model, model_out)

    return model_out