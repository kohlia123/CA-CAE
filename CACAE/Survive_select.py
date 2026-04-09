def survive_select(survive_data, data, p_thresh, lasso_alpha=0.05):

    from lifelines import CoxPHFitter
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    time_col = survive_data['OS.time'].values
    event_col = survive_data['OS'].values

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    lasso = Lasso(alpha=lasso_alpha)
    lasso.fit(scaled_data, time_col)

    selected_features_indices = [i for i, coef in enumerate(lasso.coef_) if coef != 0]
    print(f"Features after Lasso: {len(selected_features_indices)}")

    selected_features = []

    for i in selected_features_indices:
        cpf = CoxPHFitter()
        temp_df = survive_data.copy()
        temp_df['feature'] = data.iloc[:, i].values
        cpf.fit(temp_df, 'OS.time', 'OS')

        if cpf.summary['p'].values[0] <= p_thresh:
            selected_features.append(i)

    selected_data = data.iloc[:, selected_features]

    selected_data.index = data.index
    print(f"Features after P-value test: {len(selected_features)}")

    return selected_data
