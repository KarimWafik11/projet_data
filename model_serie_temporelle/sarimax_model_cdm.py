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

def sarimax_prediction_cdm(year, arret):
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
    euro_2016_start = pd.Timestamp('2016-06-10')
    euro_2016_end = pd.Timestamp('2016-07-10')
    cdm_rugby_2023_start = pd.Timestamp('2023-09-08')
    cdm_rugby_2023_end = pd.Timestamp('2023-10-28')
    evenement_dates = [
        '2015-05-23', '2015-05-26', '2017-07-15', '2017-07-16', '2017-07-25',
        '2018-07-14', '2019-06-07', '2019-06-08', '2022-05-21', '2022-07-29',
        '2015-11-13', '2017-03-28', '2017-10-07', '2017-10-10', '2018-03-23',
        '2018-06-09', '2019-06-02', '2019-09-07', '2019-09-10', '2020-09-08',
        '2021-06-08', '2021-09-01', '2021-09-04', '2021-09-07', '2022-05-28',
        '2016-02-06', '2017-02-12', '2018-02-03', '2019-02-01', '2020-02-02',
        '2021-02-14', '2022-02-12', '2021-11-20', '2015-06-13', '2016-06-24',
        '2017-06-04', '2018-06-02', '2019-06-15', '2021-06-25', '2022-06-24'
    ]
    evenement_dates = set(pd.to_datetime(evenement_dates))
    
    def is_evenement(date):
        if euro_2016_start <= date <= euro_2016_end:
            return 1
        if cdm_rugby_2023_start <= date <= cdm_rugby_2023_end:
            return 1
        if date in evenement_dates:
            return 1
        return 0

    df['IS_EVENEMENT'] = df['ds'].apply(is_evenement)
    features = ['DAY_OF_WEEK', 'IS_WEEKEND', 'IS_VACANCE', 'IS_EVENEMENT']

    # Log + normalisation
    df['y_log'] = np.log1p(df['y'])
    y_scaler = MinMaxScaler()
    df['y_scaled'] = y_scaler.fit_transform(df[['y_log']])

    # Train / test split
    train_df = df[(df['ds'].dt.year >= 2015) & (df['ds'].dt.year <= 2022)].copy()
    test_df = df[(df['ds'] >= cdm_rugby_2023_start) & (df['ds'] <= cdm_rugby_2023_end)].copy()

    # Exclusion spécifique des 23/09 et 24/09 pour certaines gares travaux
    arrets_exclus = ['AULNAYSOUSBOIS', 'SEVRANLIVRY', 'VERTGALANT', 'VILLEPARISISMITRYLENEUF', 'MITRYCLAYE']
    if df['LIBELLE_ARRET_clean'].iloc[0] in arrets_exclus:
        test_df = test_df[~test_df['ds'].isin([pd.Timestamp('2023-09-23'), pd.Timestamp('2023-09-24')])]

    # Exclusion spécifique du 30/09/2023 pour certains arrêts - travaux
    arrets_exclus_30sept = ['LESBACONNETS', 'MASSYVERRIERES']
    if df['LIBELLE_ARRET_clean'].iloc[0] in arrets_exclus_30sept:
        test_df = test_df[test_df['ds'] != pd.Timestamp('2023-09-30')]

   

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
    plt.plot(test_df['ds'], test_df['y'], label='Valeurs réelles', color='blue')
    plt.plot(test_df['ds'], pred, label='Prédictions SARIMAX', color='orange')
    plt.title(f"Prédictions SARIMAX pour 2023 (CDM Rugby) – arrêt {arret}")
    plt.xlabel("Date")
    plt.ylabel("Nombre de validations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    """

    # Résultat
    df_result = test_df[['ds', 'y']].copy()
    df_result['yhat'] = np.round(pred).astype(int)
    
    return df_result, f"{mape:.2f}%"
