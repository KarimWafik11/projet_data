import pandas as pd
import numpy as np
import holidays
from prophet import Prophet
import matplotlib.pyplot as plt


def prophet_prediction(year, arret):
    # Charger les données
    df = pd.read_csv('final_data/df_all_data.csv', parse_dates=['JOUR'])

    # Filtrer sur l'arrêt demandé
    df_arret = df[df['LIBELLE_ARRET'].astype(str) == str(arret)].copy()

    # Vérifier que des données existent pour cet arrêt
    if df_arret.empty:
        print(f"Aucune donnée trouvée pour l'arrêt '{arret}'.")
        return None

    # Nettoyer les NaN
    df_arret = df_arret[['JOUR', 'NB_VALD']].dropna()
    # Supprimer les valeurs négatives ou aberrantes
    df_arret = df_arret[df_arret['NB_VALD'] >= 0]
    df_arret = df_arret[df_arret['NB_VALD'] < df_arret['NB_VALD'].quantile(0.99)]
    if df_arret.empty:
        print(f"Aucune donnée exploitable pour l'arrêt '{arret}'.")
        return None

    df_arret = df_arret.rename(columns={'JOUR': 'ds', 'NB_VALD': 'y'})
    df_arret = df_arret[['ds', 'y']].sort_values('ds')

    # Ajouter les jours fériés français sur toute la période utile
    min_year = df_arret['ds'].dt.year.min()
    jours_feries = holidays.France(years=range(min_year, year+1))
    holidays_df = pd.DataFrame({
        'ds': pd.to_datetime(list(jours_feries.keys())),
        'holiday': 'fr_holiday'
    })

    # Instancier et ajuster le modèle
    model = Prophet(
        holidays=holidays_df,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    try:
        model.fit(df_arret, algorithm='LBFGS')
    except Exception as e:
        print(f"Erreur lors de l'entraînement Prophet : {e}")
        return None

    # Générer les dates futures pour l'année demandée
    future = pd.DataFrame({'ds': pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')})

    # Prédire
    forecast = model.predict(future)
    result = future.copy()
    result['NB_VALD_pred'] = forecast['yhat']

    # Visualisation des prédictions
    plt.figure(figsize=(10, 6))
    plt.plot(result['ds'], result['NB_VALD_pred'], label='Prédictions')
    plt.xlabel('Date')
    plt.ylabel('NB_VALD_pred')
    plt.title(f'Prédictions pour l\'année {year}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return result