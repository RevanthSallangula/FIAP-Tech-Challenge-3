import numpy as np


class StackedModel:
    """
    Hybrid stacked model that combines:
    - preprocessing pipeline
    - LightGBM base model
    - TabNet base model
    - Logistic Regression meta learner
    """

    def __init__(self, preprocessor, lgbm_model, tabnet_model, meta_model):
        self.preprocessor = preprocessor
        self.lgbm_model = lgbm_model
        self.tabnet_model = tabnet_model
        self.meta_model = meta_model

    def _prepare_features(self, X):
        """
        Apply preprocessing pipeline.
        """
        X_proc = self.preprocessor.transform(X)

        # ensure dense matrix for TabNet
        return np.array(X_proc)

    def _meta_features(self, X):
        """
        Generate meta features from base model probabilities.
        """
        X_proc = self._prepare_features(X)

        p_lgbm = self.lgbm_model.predict_proba(X_proc)[:, 1]
        p_tabnet = self.tabnet_model.predict_proba(X_proc)[:, 1]

        # expanded meta features
        Z = np.column_stack([
            p_lgbm,
            p_tabnet,
            p_lgbm * p_tabnet,
            p_lgbm - p_tabnet
        ])

        return Z

    def predict_proba(self, X):
        """
        Return final probability predictions.
        """
        Z = self._meta_features(X)
        return self.meta_model.predict_proba(Z)

    def predict(self, X):
        """
        Return class prediction.
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)