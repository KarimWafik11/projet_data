import warnings
import logging

# Supprimer les warnings (affichage console)
warnings.filterwarnings("ignore")

# Supprimer les logs de niveau WARNING et inférieur
logging.getLogger('statsmodels').setLevel(logging.ERROR)

import pandas as pd
import numpy as np
import holidays
import re
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def sarimax_model_rer_e(arret, dates_exclues):

    try:
        df = pd.read_csv('final_data/df_all_data.csv', parse_dates=['JOUR'])
    except Exception as e:
        print(f"Erreur lors du chargement du fichier CSV : {e}")
        return None

    # Nettoyer l'arrêt demandé
    arret_clean = re.sub(r'\s+', '', re.sub(r'\s*\([^)]*\)', '', str(arret)).replace('-', ''))
    df['LIBELLE_ARRET_clean'] = df['LIBELLE_ARRET'].apply(
        lambda x: re.sub(r'\s+', '', re.sub(r'\s*\([^)]*\)', '', str(x)).replace('-', ''))
    )
    df = df[df['LIBELLE_ARRET_clean'] == arret_clean].copy()

    if df.empty:
        print(f"Aucune donnée trouvée pour l'arrêt : {arret}")
        return None

    # Préparation des données
    df = df.rename(columns={'JOUR': 'ds', 'NB_VALD': 'y'})
    df = df[['ds', 'y', 'LIBELLE_ARRET', 'LIBELLE_ARRET_clean', 'IS_VACANCE']].sort_values('ds')
    df['DAY_OF_WEEK'] = df['ds'].dt.weekday / 6
    df['IS_WEEKEND'] = (df['ds'].dt.weekday >= 5).astype(int)

    # Événements
    fr_holidays = holidays.France(years=range(2015, 2025))


    features = ['DAY_OF_WEEK', 'IS_WEEKEND', 'IS_VACANCE']

    # Log + normalisation
    df['y_log'] = np.log1p(df['y'])
    y_scaler = MinMaxScaler()
    df['y_scaled'] = y_scaler.fit_transform(df[['y_log']])

    # Train / test split
    train_df = df[(df['ds'].dt.year >= 2015) & (df['ds'].dt.year <= 2022)].copy()
    test_df = df[(df['ds'] >= pd.Timestamp('2023-01-01')) & (df['ds'] <= pd.Timestamp('2023-03-31'))].copy()

    # Exclusion spécifique des 23/09 et 24/09 pour certaines gares travaux
    arrets_exclus = []
    if len(dates_exclues) > 0:
        for date in dates_exclues:
            test_df = test_df[test_df['ds'] != pd.Timestamp(date)]



    if train_df.dropna().shape[0] < 2 or test_df.dropna().shape[0] < 2:
        print("Pas assez de données pour entraîner ou tester SARIMAX.")
        return None


    # Exogènes
    exog_train = train_df[features]
    exog_test = test_df[features]

    # Entraînement du modèle SARIMAX
    model = SARIMAX(train_df['y_scaled'], order=(2, 1, 2), seasonal_order=(1, 0, 1, 7),
                    exog=exog_train, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    # Indices pour prédiction (utiliser des indices entiers)
    start_idx = len(train_df)
    end_idx = start_idx + len(test_df) - 1

    pred_scaled = results.predict(start=start_idx, end=end_idx, exog=exog_test)
    pred_log = y_scaler.inverse_transform(pred_scaled.values.reshape(-1,1)).flatten()
    pred = np.expm1(pred_log)

    # MAPE
    mape = mean_absolute_percentage_error(test_df['y'], pred) * 100

    # Visualisation

    """
    plt.figure(figsize=(12, 6))

    # Tracé des valeurs réelles
    plt.plot(test_df['ds'], test_df['y'], label='Valeurs réelles', color='blue', linewidth=1.5)

    # Tracé des prédictions SARIMAX
    plt.plot(test_df['ds'], pred, label='Prédictions SARIMAX', color='orange', linewidth=1.5)

    # Formatage pour l'axe des dates avec précision journalière
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))  # tous les 2 jours
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

    plt.title(f"Prédictions SARIMAX pour l'été 2023 – arrêt {arret}")
    plt.xlabel("Date")
    plt.ylabel("Nombre de validations")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    """

    #Résultat
    df_result = test_df[['ds', 'y', 'LIBELLE_ARRET']].copy()
    df_result['yhat'] = np.round(pred).astype(int)

    #f"{mape:.2f}%"
    return df_result
