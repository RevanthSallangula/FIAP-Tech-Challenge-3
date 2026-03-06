import pandas as pd
import numpy as np
import joblib

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from pytorch_tabnet.tab_model import TabNetClassifier

from src.config import RANDOM_SEED
from src.models.transforms import cyclical_transform
from src.models.stacked_model import StackedModel


PARAMS = {
    "learning_rate": 0.03,
    "n_estimators": 800,
    "max_depth": 8,
    "num_leaves": 128,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "class_weight": "balanced"
}


CATEGORICAL_FEATURES = [
    'job','marital','education',
    'default','housing','loan',
    'contact','poutcome'
]

NUMERICAL_FEATURES = [
    'age','balance','campaign',
    'pdays','previous'
]

CYCLICAL_FEATURES = [
    'month_sin','month_cos',
    'day_sin','day_cos'
]


def train_and_save_model(train_file: str, model_out: str):

    df = pd.read_parquet(train_file)

    X = df.drop(columns=["y"])
    y = df["y"]

    # --------------------------
    # Train / validation split
    # --------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # --------------------------
    # Preprocessing pipeline
    # --------------------------

    cyclical_transformer = FunctionTransformer(cyclical_transform)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", PowerTransformer(method="yeo-johnson"), NUMERICAL_FEATURES),
            ("cyclical", "passthrough", CYCLICAL_FEATURES),
        ],
        remainder="passthrough"
    )

    preprocessing_pipeline = Pipeline([
        ("cyclical", cyclical_transformer),
        ("preprocessor", preprocessor)
    ])

    X_train_proc = preprocessing_pipeline.fit_transform(X_train)
    X_val_proc = preprocessing_pipeline.transform(X_val)

    # --------------------------
    # LightGBM
    # --------------------------

    lgbm_model = LGBMClassifier(
        **PARAMS,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    lgbm_model.fit(X_train_proc, y_train)

    p_lgbm = lgbm_model.predict_proba(X_val_proc)[:,1]

    # --------------------------
    # TabNet
    # --------------------------

    tabnet_model = TabNetClassifier(seed=RANDOM_SEED)

    tabnet_model.fit(
        X_train_proc,
        y_train.values,
        eval_set=[(X_val_proc, y_val.values)],
        max_epochs=100,
        patience=15,
        batch_size=1024,
        virtual_batch_size=128
    )

    p_tabnet = tabnet_model.predict_proba(X_val_proc)[:,1]

    # --------------------------
    # Meta features
    # --------------------------

    Z_val = np.column_stack((p_lgbm, p_tabnet))

    # --------------------------
    # Meta learner
    # --------------------------

    meta_model = LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_SEED
    )

    meta_model.fit(Z_val, y_val)

    # --------------------------
    # Final stacked model
    # --------------------------

    stacked_model = StackedModel(
        preprocessor=preprocessing_pipeline,
        lgbm_model=lgbm_model,
        tabnet_model=tabnet_model,
        meta_model=meta_model
    )

    joblib.dump(stacked_model, model_out)

    return model_out