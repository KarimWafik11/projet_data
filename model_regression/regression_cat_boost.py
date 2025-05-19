import pandas as pd
import numpy as np
import holidays
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

def regression_cat_boost(year, vacances_tuples):
    # Charger les données
    df = pd.read_csv('final_data/df_all_data.csv', parse_dates=['JOUR'])

    # S'assurer que LIBELLE_ARRET est bien une chaîne de caractères
    df['LIBELLE_ARRET'] = df['LIBELLE_ARRET'].astype(str)

    # Harmoniser les noms de colonnes
    if 'is_weekend' in df.columns:
        df = df.rename(columns={
            'is_weekend': 'IS_WEEKEND',
            'is_ferie': 'IS_FERIE',
            'is_vacance': 'IS_VACANCE'
        })

    # S'assurer que toutes les features existent
    for col in ['IS_FERIE', 'DAY_OF_WEEK', 'DAY_OF_YEAR', 'WEEK_OF_YEAR', 'IS_WEEKEND', 'IS_VACANCE']:
        if col not in df.columns:
            if col == 'DAY_OF_WEEK':
                df['DAY_OF_WEEK'] = df['JOUR'].dt.dayofweek
            elif col == 'DAY_OF_YEAR':
                df['DAY_OF_YEAR'] = df['JOUR'].dt.dayofyear
            elif col == 'WEEK_OF_YEAR':
                df['WEEK_OF_YEAR'] = df['JOUR'].dt.isocalendar().week.astype(int)
            elif col == 'IS_WEEKEND':
                df['IS_WEEKEND'] = df['JOUR'].dt.dayofweek.isin([5, 6]).astype(int)
            elif col == 'IS_FERIE':
                jours_feries = pd.to_datetime(list(holidays.France(years=df['JOUR'].dt.year.unique()).keys()))
                df['IS_FERIE'] = df['JOUR'].isin(jours_feries).astype(int)
            elif col == 'IS_VACANCE':
                df['IS_VACANCE'] = 0  # à adapter si besoin

    # Définir X et y
    features = ['IS_FERIE', 'DAY_OF_WEEK', 'DAY_OF_YEAR', 'WEEK_OF_YEAR', 'IS_WEEKEND', 'IS_VACANCE', 'LIBELLE_ARRET']
    X = df[features]
    y = df['NB_VALD']

    # Split (réduire la taille du test pour accélérer)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Grille réduite pour accélérer
    param_grid = {
        'iterations': [100],
        'learning_rate': [0.1],
        'depth': [4]
    }

    model = CatBoostRegressor(cat_features=['LIBELLE_ARRET'], verbose=0, early_stopping_rounds=20)

    grid = GridSearchCV(model, param_grid, cv=2, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train, eval_set=(X_test, y_test))

    print("Meilleurs paramètres :", grid.best_params_)

    # Évaluation
    y_pred = grid.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'MSE : {mse:.2f}')
    print(f'RMSE : {rmse:.2f}')
    print(f'MAE : {mae:.2f}')

    # Prédictions sur une année
    dates_prediction = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31')
    arrets_echantillon = df['LIBELLE_ARRET'].unique()

    df_prediction = pd.MultiIndex.from_product(
        [dates_prediction, arrets_echantillon],
        names=['JOUR', 'LIBELLE_ARRET']
    ).to_frame(index=False)

    df_prediction['LIBELLE_ARRET'] = df_prediction['LIBELLE_ARRET'].astype(str)
    df_prediction['DAY_OF_WEEK'] = df_prediction['JOUR'].dt.dayofweek
    df_prediction['DAY_OF_YEAR'] = df_prediction['JOUR'].dt.dayofyear
    df_prediction['WEEK_OF_YEAR'] = df_prediction['JOUR'].dt.isocalendar().week.astype(int)
    df_prediction['IS_WEEKEND'] = df_prediction['DAY_OF_WEEK'].isin([5, 6]).astype(int)
    jours_feries_pred = pd.to_datetime(list(holidays.France(years=year).keys()))
    df_prediction['IS_FERIE'] = df_prediction['JOUR'].isin(jours_feries_pred).astype(int)

    # Calcul de IS_VACANCE à partir de vacances_tuples
    vacance_dates = set()
    for start, end in vacances_tuples:
        vacance_dates.update(pd.date_range(start, end, freq='D'))
    df_prediction['IS_VACANCE'] = df_prediction['JOUR'].isin(vacance_dates).astype(int)

    df_prediction = df_prediction[['JOUR', 'IS_FERIE', 'DAY_OF_WEEK', 'DAY_OF_YEAR', 'WEEK_OF_YEAR', 'IS_WEEKEND', 'IS_VACANCE', 'LIBELLE_ARRET']]

    X_pred = df_prediction[features]
    df_prediction['NB_VALD_pred'] = grid.predict(X_pred)

    return df_prediction