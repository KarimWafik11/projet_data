import numpy as np
import pandas as pd
import holidays
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def light_gbm_model(arret_libelle, year, vacances_tuples):
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
    X = df[['IS_FERIE', 'DAY_OF_WEEK', 'DAY_OF_YEAR', 'WEEK_OF_YEAR', 'IS_WEEKEND', 'IS_VACANCE']].astype(float)

    # Normalisation
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    # Split data
    split_idx = int(len(y_scaled) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

    # Entraînement LightGBM
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'seed': 42
    }
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        num_boost_round=100,
        #early_stopping_rounds=10,
        #verbose_eval=False
    )

    # Prédiction test
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_denorm = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Évaluation
    mask = (~np.isnan(y_test_denorm)) & (y_test_denorm != 0)
    mae = mean_absolute_error(y_test_denorm[mask], y_pred[mask])
    rmse = np.sqrt(mean_squared_error(y_test_denorm[mask], y_pred[mask]))
    mape = (np.mean(np.abs((y_test_denorm[mask] - y_pred[mask]) / y_test_denorm[mask])) * 100) if np.all(y_test_denorm[mask] != 0) else np.nan

    print(f"Évaluation LightGBM pour l'arrêt {arret_libelle} :")
    print(f" - MAE : {mae:.2f}")
    print(f" - RMSE : {rmse:.2f}")
    print(f" - MAPE : {mape if not np.isinf(mape) else 'NaN'}%")

    # Générer données futures pour l’année demandée
    future_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
    fr_holidays = holidays.France(years=[year])
    X_future = pd.DataFrame(index=future_dates)
    X_future['IS_FERIE'] = future_dates.to_series().apply(lambda d: int(d in fr_holidays))
    X_future['DAY_OF_WEEK'] = future_dates.dayofweek
    X_future['DAY_OF_YEAR'] = future_dates.dayofyear
    X_future['WEEK_OF_YEAR'] = future_dates.isocalendar().week.astype(int)
    X_future['IS_WEEKEND'] = X_future['DAY_OF_WEEK'].apply(lambda x: 1 if x in [5, 6] else 0)

    # Calcul de IS_VACANCE pour l'année future
    vacance_dates = set()
    for start, end in vacances_tuples:
        vacance_dates.update(pd.date_range(start, end, freq='D'))
    X_future['IS_VACANCE'] = X_future.index.isin(vacance_dates).astype(int)

    # Prédiction future
    y_pred_future_scaled = model.predict(X_future)
    y_pred_future = scaler.inverse_transform(y_pred_future_scaled.reshape(-1, 1)).flatten()

    # Résultat final
    df_result = pd.DataFrame({
        'ds': future_dates,
        'NB_VALD_pred': y_pred_future,
        'LIBELLE_ARRET': arret_libelle
    })

    return df_result
