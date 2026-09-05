"""
Layer 1, explanation half: SHAP values for individual scored transactions.

A judge does not trust "0.83, blocked." They trust "0.83, blocked -- because
the amount is 6x this transaction's usual range, the device was never seen
before, and it happened at 3am." SHAP gives us that per-transaction, per-
feature breakdown, and because it decomposes the model's actual decision
(not a post-hoc guess), it is honest about what XGBoost really keyed on.

TreeExplainer is exact and fast for tree ensembles (no sampling, unlike
KernelSHAP), so this is cheap enough to run per-request in the live API.
"""
import numpy as np
import shap

# A handful of raw feature names get a human phrase; everything else falls
# back to a generic "<name> was <value>" so new / engineered columns never
# crash the explainer.
FEATURE_GLOSSARY = {
    "TransactionAmt": "transaction amount",
    "log_amt": "transaction amount (log scale)",
    "amt_cents": "the cents portion of the amount",
    "hour": "hour of day",
    "dayofweek": "day of week",
    "id_missing_frac": "fraction of device/identity fields missing",
    "card1": "card identifier",
    "card2": "card identifier (secondary)",
    "card4": "card network",
    "card6": "card type (credit/debit)",
    "addr1": "billing region code",
    "P_emaildomain": "purchaser email domain",
    "R_emaildomain": "recipient email domain",
    "DeviceType": "device type",
    "DeviceInfo": "device info",
    "ProductCD": "product category",
}


def _phrase(feature, value, shap_value):
    label = FEATURE_GLOSSARY.get(feature, feature)
    direction = "pushed risk UP" if shap_value > 0 else "pushed risk DOWN"
    if value is None or (isinstance(value, float) and np.isnan(value)):
        val_str = "missing"
    else:
        val_str = f"{value:.2f}" if isinstance(value, (int, float, np.floating)) else str(value)
    return f"{label} = {val_str} ({direction}, contribution {shap_value:+.3f})"


class Explainer:
    def __init__(self, xgb_model, feature_names):
        # model_output='raw' matches predict_proba's underlying margin;
        # TreeExplainer is exact for XGBoost, no background dataset needed.
        self.explainer = shap.TreeExplainer(xgb_model)
        self.feature_names = feature_names

    def explain_row(self, X_row, top_k=3):
        """X_row: a single-row DataFrame in the model's feature order (i.e.
        the output of features.transform on one transaction).

        Returns a list of top_k dicts: {feature, value, shap_value, phrase},
        ranked by absolute SHAP contribution -- the features that moved this
        specific score the most, in either direction.
        """
        shap_values = self.explainer.shap_values(X_row)
        sv = np.asarray(shap_values)[0]
        vals = X_row.iloc[0]

        order = np.argsort(-np.abs(sv))[:top_k]
        out = []
        for i in order:
            fname = self.feature_names[i]
            fval = vals.iloc[i]
            out.append({
                "feature": fname,
                "value": None if (isinstance(fval, float) and np.isnan(fval)) else float(fval),
                "shap_value": float(sv[i]),
                "phrase": _phrase(fname, fval, sv[i]),
            })
        return out
