import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest

def survive_select(survive_data, data, top_n=50):
    print("Formatting survival data for RSF...")
    y = np.array([(bool(e), float(t)) for e, t in zip(survive_data['OS'], survive_data['OS.time'])],
                 dtype=[('status', 'bool'), ('time', 'float')])
    
    X = data.values
    features = data.columns

    print("Fitting Random Survival Forest...")
    rsf = RandomSurvivalForest(n_estimators=100, n_jobs=-1, random_state=42)
    rsf.fit(X, y)

    # Manual Permutation Importance
    print("Calculating feature importance manually...")
    baseline_score = rsf.score(X, y)
    scores = []

    # We loop through each feature, shuffle it, and see how much the C-index drops
    for i in range(X.shape[1]):
        save = X[:, i].copy()
        np.random.seed(42)
        np.random.shuffle(X[:, i])
        
        shuffled_score = rsf.score(X, y)
        scores.append(baseline_score - shuffled_score)
        
        # Put the feature back to normal
        X[:, i] = save

    importances = np.array(scores)
    indices = np.argsort(importances)[-top_n:]
    
    selected_data = data.iloc[:, indices]
    print(f"Success! Manual importance complete. Selected top {top_n} features.")
    return selected_data