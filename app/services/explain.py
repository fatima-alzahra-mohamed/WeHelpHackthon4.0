import numpy as np
import pandas as pd

def feature_importance_explanation(model, X_row: pd.DataFrame, top_k: int = 8):
    # RandomForest has feature_importances_
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return {"method": "none", "top_features": []}

    vals = X_row.iloc[0].values.astype(float)
    contrib = vals * importances

    idx = np.argsort(np.abs(contrib))[-top_k:][::-1]
    top = []
    for i in idx:
        top.append({
            "feature": X_row.columns[i],
            "z_value": float(vals[i]),
            "importance": float(importances[i]),
            "contribution_proxy": float(contrib[i]),
        })

    return {"method": "feature_importance_proxy", "top_features": top}
