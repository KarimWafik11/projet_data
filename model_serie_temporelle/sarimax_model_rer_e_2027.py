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
from sklearn.preprocessing import MinMaxScaler
import datetime

def sarimax_model_rer_e_2027(arret, dates_exclues=[]):
    """
    Fonction qui prédit les validations pour le RER E jusqu'en 2027 pour un arrêt donné
    
    Args:
        arret (str): Nom de l'arrêt du RER E
        dates_exclues (list): Liste de dates à exclure du dataset (format 'YYYY-MM-DD')
        
    Returns:
        pd.DataFrame: DataFrame avec les prédictions jusqu'en 2027
    """
    try:
        df = pd.read_csv('final_data/df_rer_e.csv', parse_dates=['JOUR'])
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
    df = df[['ds', 'y', 'LIBELLE_ARRET', 'LIBELLE_ARRET_clean', 'IS_VACANCE', 'DAY_OF_WEEK', 'IS_WEEKEND']].sort_values('ds')

    # Événements
    all_years = list(range(2015, 2030))  # Inclure les années jusqu'à 2027
    fr_holidays = holidays.France(years=all_years)

    # Features pour le modèle
    features = ['DAY_OF_WEEK', 'IS_WEEKEND', 'IS_VACANCE']

    # Log + normalisation
    df['y_log'] = np.log1p(df['y'])
    y_scaler = MinMaxScaler()
    df['y_scaled'] = y_scaler.fit_transform(df[['y_log']])

    # Données d'entraînement (toutes les données disponibles)
    train_df = df.copy()

    # Génération des données pour la prédiction du 1er trimestre 2027 uniquement
    start_prediction = pd.Timestamp('2027-01-01')
    end_date = pd.Timestamp('2027-03-31')
    
    # Création de la série temporelle future
    future_dates = pd.date_range(start=start_prediction, end=end_date)
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Ajout du nom de l'arrêt
    future_df['LIBELLE_ARRET'] = arret
    future_df['LIBELLE_ARRET_clean'] = arret_clean
    
    # Calcul des features pour les dates futures
    future_df['DAY_OF_WEEK'] = future_df['ds'].dt.weekday / 6
    future_df['IS_WEEKEND'] = (future_df['ds'].dt.weekday >= 5).astype(int)
    
    # Vacances scolaires jusqu'en 2027 (estimation basée sur les patterns habituels de la zone C)
    # Pour simplifier, nous utilisons un pattern estimé pour les vacances scolaires
    # Dans un cas réel, il faudrait mettre à jour ces données avec les calendriers officiels
    vacances_futures = [
        # 2024
        ('2024-02-17', '2024-03-04'),
        ('2024-04-13', '2024-04-29'),
        ('2024-07-06', '2024-09-02'),
        ('2024-10-19', '2024-11-04'),
        ('2024-12-21', '2025-01-06'),
        # 2025
        ('2025-02-15', '2025-03-03'),
        ('2025-04-12', '2025-04-28'),
        ('2025-07-05', '2025-09-01'),
        ('2025-10-18', '2025-11-03'),
        ('2025-12-20', '2026-01-05'),
        # 2026
        ('2026-02-14', '2026-03-02'),
        ('2026-04-11', '2026-04-27'),
        ('2026-07-04', '2026-08-31'),
        ('2026-10-17', '2026-11-02'),
        ('2026-12-19', '2027-01-04'),
        # 2027
        ('2027-02-13', '2027-03-01'),
        ('2027-04-10', '2027-04-26'),
        ('2027-07-03', '2027-08-30'),
        ('2027-10-16', '2027-11-01'),
        ('2027-12-18', '2028-01-03'),
        # 2028
        ('2028-02-12', '2028-02-28'),
        ('2028-04-08', '2028-04-24'),
        ('2028-07-08', '2028-09-04'),
        ('2028-10-21', '2028-11-06'),
        ('2028-12-23', '2029-01-08'),
        # 2029
        ('2029-02-17', '2029-03-05'),
        ('2029-04-14', '2029-04-30'),
        ('2029-07-07', '2029-09-03'),
        ('2029-10-20', '2029-11-05'),
        ('2029-12-22', '2030-01-07'),
    ]
    
    # Création d'un ensemble de dates de vacances
    vacance_dates = set()
    for start, end in vacances_futures:
        vacance_dates.update(pd.date_range(start, end, freq='D'))
    
    # Ajout de la colonne IS_VACANCE
    future_df['IS_VACANCE'] = future_df['ds'].isin(vacance_dates).astype(int)
    
    # Préparation des données pour l'entraînement
    exog_train = train_df[features]
    
    # Entraînement du modèle SARIMAX
    model = SARIMAX(train_df['y_scaled'], order=(2, 1, 2), seasonal_order=(1, 0, 1, 7),
                   exog=exog_train, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    
    # Préparation des données exogènes pour la prédiction
    exog_future = future_df[features]
    
    # Indices pour prédiction
    start_idx = len(train_df)
    end_idx = start_idx + len(future_df) - 1
    
    # Prédiction
    pred_scaled = results.predict(start=start_idx, end=end_idx, exog=exog_future)
    
    # Transformation inverse pour obtenir les prédictions finales
    pred_log = y_scaler.inverse_transform(pred_scaled.values.reshape(-1, 1)).flatten()
    pred = np.expm1(pred_log)
    
    # Création du DataFrame de résultat
    future_df['y'] = 0  # Valeur placeholder pour les valeurs réelles (inconnues pour le futur)
    future_df['yhat'] = np.round(pred).astype(int)
    
    # Résultat
    result_df = future_df[['ds', 'LIBELLE_ARRET', 'yhat']].copy()
    
    return result_df
