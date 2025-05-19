def sarimax_prediction(arret_libelle, year, vacances_tuples):
    import numpy as np
    import pandas as pd
    import holidays
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Charger les données
    df = pd.read_csv('final_data/df_all_data.csv', parse_dates=['JOUR'])

    # Nettoyage et sélection
    df = df[['JOUR', 'NB_VALD', 'IS_FERIE', 'DAY_OF_WEEK', 'DAY_OF_YEAR', 'WEEK_OF_YEAR', 'IS_WEEKEND', 'IS_VACANCE', 'LIBELLE_ARRET']].dropna()
    df = df[df['NB_VALD'] >= 0]
    df = df[df['NB_VALD'] < df['NB_VALD'].quantile(0.99)]
    df = df[df['LIBELLE_ARRET'] == arret_libelle]
    
    if df.empty:
        print(f"Aucune donnée trouvée pour l'arrêt : {arret_libelle}")
        return None

    # Index temporel
    df = df.set_index('JOUR').asfreq('D')
    df['NB_VALD'] = df['NB_VALD'].fillna(0)
    df = df.fillna(0)

    # Dépendante & exogènes
    y = df['NB_VALD'].astype(float)
    exog = df[['IS_FERIE', 'DAY_OF_WEEK', 'DAY_OF_YEAR', 'WEEK_OF_YEAR', 'IS_WEEKEND', 'IS_VACANCE']].astype(float)

    # Normalisation
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    # Split data
    split_idx = int(len(y_scaled) * 0.8)
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    exog_train, exog_test = exog.iloc[:split_idx], exog.iloc[split_idx:]

    # Entraînement SARIMAX
    #anciennes valeurs : (2, 1, 2), (1, 1, 1, 7)
    #gridserch order=(0, 1, 0), seasonal_order=(1, 0, 1, 7)
    try:
        model = SARIMAX(
            y_train,
            order=(1, 1, 1),
            seasonal_order=(2, 0, 1, 7),
            exog=exog_train,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
    except Exception as e:
        print(f"Erreur entraînement SARIMAX pour {arret_libelle} : {e}")
        return None

    # Prédiction test
    pred = results.get_forecast(steps=len(y_test), exog=exog_test)
    y_pred_scaled = pred.predicted_mean
    y_pred = scaler.inverse_transform(y_pred_scaled.values.reshape(-1, 1)).flatten()
    y_test_denorm = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Évaluation
    mask = (~np.isnan(y_test_denorm)) & (y_test_denorm != 0)
    mae = mean_absolute_error(y_test_denorm[mask], y_pred[mask])
    rmse = np.sqrt(mean_squared_error(y_test_denorm[mask], y_pred[mask]))
    mape = (np.mean(np.abs((y_test_denorm[mask] - y_pred[mask]) / y_test_denorm[mask])) * 100) if np.all(y_test_denorm[mask] != 0) else np.nan

    print(f"Évaluation pour l'arrêt {arret_libelle} :")
    print(f" - MAE : {mae:.2f}")
    print(f" - RMSE : {rmse:.2f}")
    print(f" - MAPE : {mape if not np.isinf(mape) else 'NaN'}%")

    # Générer données futures pour l’année demandée
    future_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
    fr_holidays = holidays.France(years=[year])
    exog_future = pd.DataFrame(index=future_dates)
    exog_future['IS_FERIE'] = future_dates.to_series().apply(lambda d: int(d in fr_holidays))
    exog_future['DAY_OF_WEEK'] = future_dates.dayofweek
    exog_future['DAY_OF_YEAR'] = future_dates.dayofyear
    exog_future['WEEK_OF_YEAR'] = future_dates.isocalendar().week.astype(int)
    exog_future['IS_WEEKEND'] = exog_future['DAY_OF_WEEK'].apply(lambda x: 1 if x in [5, 6] else 0)

    # Calcul de IS_VACANCE pour l'année future
    vacance_dates = set()
    for start, end in vacances_tuples:
        vacance_dates.update(pd.date_range(start, end, freq='D'))
    exog_future['IS_VACANCE'] = exog_future.index.isin(vacance_dates).astype(int)

    try:
        pred_future = results.get_forecast(steps=len(future_dates), exog=exog_future.astype(float))
        y_pred_scaled = pred_future.predicted_mean
        y_pred_future = scaler.inverse_transform(y_pred_scaled.values.reshape(-1, 1)).flatten()
    except Exception as e:
        print(f"Erreur prédiction future pour {arret_libelle} : {e}")
        return None

    # Résultat final
    df_result = pd.DataFrame({
        'ds': future_dates,
        'NB_VALD_pred': y_pred_future,
        'LIBELLE_ARRET': arret_libelle
    })

    return df_result