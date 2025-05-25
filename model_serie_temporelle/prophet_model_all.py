import pandas as pd
import numpy as np
import holidays
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from itertools import product
import re


def prophet_prediction_all(year, arret_liste):
    try:
        # 1. Charger les données
        df = pd.read_csv('final_data/df_all_data.csv', parse_dates=['JOUR'])
    except Exception as e:
        print(f"Erreur lors du chargement du fichier CSV : {e}")
        return None

    # Nettoyer l'arrêt demandé
    #arret_clean = re.sub(r'\s+', '', re.sub(r'\s*\([^)]*\)', '', str(arret)).replace('-', ''))

    # Nettoyer la colonne LIBELLE_ARRET pour la comparaison
    df['LIBELLE_ARRET_clean'] = df['LIBELLE_ARRET'].apply(
        lambda x: re.sub(r'\s+', '', re.sub(r'\s*\([^)]*\)', '', str(x)).replace('-', ''))
    )
    df = df[df['LIBELLE_ARRET_clean'].isin(arret_liste)].copy()
    if df.empty:
        print(f"Aucune donnée trouvée pour l'arrêt : {arret_liste}")
        return None

    # 2. Renommer pour Prophet
    df = df.rename(columns={'JOUR': 'ds', 'NB_VALD': 'y'})
    df = df[['ds', 'y', 'LIBELLE_ARRET', 'IS_VACANCE']].sort_values('ds')
    df['DAY_OF_WEEK'] = df['ds'].dt.weekday / 6
    df['IS_WEEKEND'] = (df['ds'].dt.weekday >= 5).astype(int)

    # Détection des vacances
    fr_holidays = holidays.France(years=range(2015, 2025))
    vacances_connues = pd.to_datetime(list(fr_holidays.keys()))

    # Ajouter la colonne IS_EVENEMENT
    euro_2016_start = pd.Timestamp('2016-06-10')
    euro_2016_end = pd.Timestamp('2016-07-10')
    cdm_rugby_2023_start = pd.Timestamp('2023-09-08')
    cdm_rugby_2023_end = pd.Timestamp('2023-10-28')

    # Dates événements Stade de France (concerts, foot, rugby)
    evenement_dates = [
        # Concerts majeurs
        '2015-05-23', '2015-05-26',  # AC/DC
        '2017-07-15', '2017-07-16',  # Coldplay
        '2017-07-25',                # U2
        '2018-07-14',                # Beyoncé & Jay-Z
        '2019-06-07', '2019-06-08',  # BTS
        '2022-05-21',                # Indochine
        '2022-07-29',                # Ed Sheeran
        # Football hors Euro
        '2015-11-13',                # France – Allemagne
        '2017-03-28',                # France – Espagne (amical)
        '2017-10-07',                # France – Bulgarie (qualif CM)
        '2017-10-10',                # France – Biélorussie (qualif CM)
        '2018-03-23',                # France – Colombie (amical)
        '2018-06-09',                # France – USA (amical)
        '2019-06-02',                # France – Bolivie (amical)
        '2019-09-07',                # France – Albanie (qualif Euro)
        '2019-09-10',                # France – Andorre (qualif Euro)
        '2020-09-08',                # France – Croatie (Ligue des nations)
        '2021-06-08',                # France – Bulgarie (amical)
        '2021-09-01',                # France – Bosnie (qualif CM)
        '2021-09-04',                # France – Ukraine (qualif CM)
        '2021-09-07',                # France – Finlande (qualif CM)
        '2022-05-28',                # Finale Ligue des champions
        # Rugby - Six Nations (exemples, à compléter si besoin)
        '2016-02-06',                # France – Italie
        '2017-02-12',                # France – Ecosse
        '2018-02-03',                # France – Irlande
        '2019-02-01',                # France – Pays de Galles
        '2020-02-02',                # France – Angleterre
        '2021-02-14',                # France – Irlande
        '2022-02-12',                # France – Irlande
        # Rugby - autres
        '2021-11-20',                # France – Nouvelle-Zélande
        # Finales Top 14 (exemples, à compléter si besoin)
        '2015-06-13', '2016-06-24', '2017-06-04', '2018-06-02', '2019-06-15', '2021-06-25', '2022-06-24'
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

    # 4. Logarithmiser la variable cible si dispersion élevée
    df['y_log'] = np.log1p(df['y'])  # log(1 + y)

    # 5. Normalisation
    y_scaler = MinMaxScaler()
    df['y_scaled'] = y_scaler.fit_transform(df[['y_log']])

    # 6. Filtrer les années pour entraînement
    train_df = df[(df['ds'].dt.year >= 2015) & (df['ds'].dt.year <= 2022)]

    # Vérification du nombre de lignes pour Prophet
    if train_df.dropna().shape[0] < 2:
        print(f"⛔ Pas assez de données pour entraîner Prophet pour l'arrêt '{arret_liste}'.")
        return None

    # Jours fériés pour Prophet
    holidays_df = pd.DataFrame({
        'ds': pd.to_datetime(list(fr_holidays.keys())),
        'holiday': 'fr_holiday'
    })

    # 7. Modèle Prophet avec régressseurs
    m = Prophet(
        holidays=holidays_df,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    features = ['DAY_OF_WEEK', 'IS_WEEKEND', 'IS_VACANCE', 'IS_EVENEMENT']
    for feat in features:
        m.add_regressor(feat)

    m.fit(train_df[['ds', 'y_scaled'] + features].rename(columns={'y_scaled': 'y'}))

    # Prédiction mois par mois
    mape_by_month = []
    median_by_month = []
    for month in range(1, 13):
        # Générer le futur pour le mois courant
        start = pd.Timestamp(f'{year}-{month:02d}-01')
        if month == 12:
            end = pd.Timestamp(f'{year+1}-01-01') - pd.Timedelta(days=1)
        else:
            end = pd.Timestamp(f'{year}-{month+1:02d}-01') - pd.Timedelta(days=1)
        n_days = (end - start).days + 1

        future = pd.DataFrame({'ds': pd.date_range(start, end)})
        future['DAY_OF_WEEK'] = future['ds'].dt.weekday / 6
        future['IS_WEEKEND'] = (future['ds'].dt.weekday >= 5).astype(int)
        # Définir les intervalles de vacances 2023
        vacances_2023 = [
            ('2023-02-18', '2023-03-06'),
            ('2023-04-22', '2023-05-09'),
            ('2023-07-08', '2023-09-03'),
            ('2023-10-21', '2023-11-06'),
            ('2023-12-23', '2024-01-08')
        ]
        vacances_2023_ranges = [
            (pd.to_datetime(start_v), pd.to_datetime(end_v)) for start_v, end_v in vacances_2023
        ]
        def is_vacance(date):
            return any(start_v <= date <= end_v for start_v, end_v in vacances_2023_ranges)
        future['IS_VACANCE'] = future['ds'].apply(is_vacance).astype(int)
        # Reprendre la fonction is_evenement définie plus haut
        euro_2016_start = pd.Timestamp('2016-06-10')
        euro_2016_end = pd.Timestamp('2016-07-10')
        cdm_rugby_2023_start = pd.Timestamp('2023-09-08')
        cdm_rugby_2023_end = pd.Timestamp('2023-10-28')
        evenement_dates = [
            '2015-05-23', '2015-05-26', '2017-07-15', '2017-07-16', '2017-07-25', '2018-07-14',
            '2019-06-07', '2019-06-08', '2022-05-21', '2022-07-29', '2015-11-13', '2017-03-28',
            '2017-10-07', '2017-10-10', '2018-03-23', '2018-06-09', '2019-06-02', '2019-09-07',
            '2019-09-10', '2020-09-08', '2021-06-08', '2021-09-01', '2021-09-04', '2021-09-07',
            '2022-05-28', '2016-02-06', '2017-02-12', '2018-02-03', '2019-02-01', '2020-02-02',
            '2021-02-14', '2022-02-12', '2021-11-20', '2015-06-13', '2016-06-24', '2017-06-04',
            '2018-06-02', '2019-06-15', '2021-06-25', '2022-06-24'
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
        future['IS_EVENEMENT'] = future['ds'].apply(is_evenement)

        forecast = m.predict(future)
        forecast['yhat_log'] = forecast['yhat']
        forecast['yhat'] = np.expm1(y_scaler.inverse_transform(forecast[['yhat_log']]))
        forecast['ds'] = pd.to_datetime(forecast['ds']).dt.normalize()
        df['ds'] = pd.to_datetime(df['ds']).dt.normalize()
        df_merged = forecast[['ds', 'yhat']].merge(df[['ds', 'y']], on='ds', how='left')
        test_df = df_merged.dropna(subset=['y', 'yhat'])
        if not test_df.empty:
            # Calcul du MAPE mensuel (moyenne sur le mois)
            mape = mean_absolute_percentage_error(test_df['y'], test_df['yhat'])
            mape_by_month.append((month, mape))
            # Calcul de la médiane journalière du MAPE sur le mois
            daily_mape = np.abs((test_df['y'] - test_df['yhat']) / test_df['y'])
            median_mape = np.median(daily_mape)
            median_by_month.append((month, median_mape))
        else:
            mape_by_month.append((month, None))
            median_by_month.append((month, None))
            print(f"MAPE pour {year}-{month:02d} : Pas de données")

    # Affichage trié par MAPE croissant (hors None)
    mape_sorted = sorted([(m, v) for m, v in mape_by_month if v is not None], key=lambda x: x[1])
    
    map_result = []
    # Afficher le mois avec la meilleure (plus basse) MAPE et la médiane journalière de ce mois
    if mape_sorted:
        best_month, best_mape = mape_sorted[0]
        # Récupérer la médiane journalière pour ce mois
        best_median = next((med for m, med in median_by_month if m == best_month), None)
        #print(f"\nMeilleur mois : {best_month:02d} avec MAPE {best_mape*100:.2f}%")
        map_result.append((best_month, str(best_mape*100)+"%"))
        # Calculer la meilleure (plus basse) médiane journalière parmi tous les mois
        all_medians = [med for _, med in median_by_month if med is not None]
        if all_medians:
            min_median = min(all_medians)
            min_median_month = next(m for m, med in median_by_month if med == min_median)
            #print(f"Meilleure médiane journalière tous mois confondus : Mois {min_median_month:02d} avec {min_median*100:.2f}%")
            map_result.append((min_median_month, str(min_median*100)+"%"))

    else:
        print("\nAucun mois avec données valides pour le MAPE.")

    

    return map_result


def find_best_prophet_params(train_df, features, holidays_df):
    """
    Recherche des meilleurs hyperparamètres Prophet via une grille simple.
    Retourne le meilleur ensemble de paramètres et le score associé.
    """
    param_grid = {  # à ajuster selon vos besoins
        'changepoint_prior_scale': [0.01, 0.1, 0.5],
        'seasonality_prior_scale': [1.0, 5.0, 10.0],
        'holidays_prior_scale': [1.0, 5.0, 10.0]
    }
    all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    best_mape = float('inf')
    best_params = None

    for params in all_params:
        m = Prophet(
            holidays=holidays_df,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            **params
        )
        for feat in features:
            m.add_regressor(feat)
        m.fit(train_df[['ds', 'y_scaled'] + features].rename(columns={'y_scaled': 'y'}))
        # Validation croisée on 2018
        try:
            df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days', parallel="processes")
            df_p = performance_metrics(df_cv)
            mape = df_p['mape'].mean()
            if mape < best_mape:
                best_mape = mape
                best_params = params
        except Exception as e:
            print(f"Erreur avec paramètres {params}: {e}")
            continue

    print(f"Meilleurs hyperparamètres Prophet: {best_params} (MAPE={best_mape:.4f})")
    return best_params


def get_top_errors(df_merged, top_n=10):
    """
    Retourne les indices des dates où l'erreur absolue entre la prédiction et la réalité est la plus forte.
    """
    df_merged = df_merged.dropna(subset=['y', 'yhat']).copy()
    df_merged['abs_error'] = np.abs(df_merged['y'] - df_merged['yhat'])
    top_errors = df_merged.sort_values('abs_error', ascending=False).head(top_n)
    #print(f"\nTop {top_n} erreurs de prédiction :")
    #print(top_errors[['ds', 'y', 'yhat', 'abs_error']])
    list_date_error = top_errors['ds'].tolist()
    #dates_2020 = liste1 = pd.date_range(start="2020-01-01", end="2020-12-31").to_list()
    
    return []
