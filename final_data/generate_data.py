import pandas as pd
import unicodedata, re, holidays

def generate_data():
    years = ['2016', '2017', '2018', '2019', '2021', '2022', '2023']

    col_names = ['JOUR', 'CODE_STIF_TRNS', 'CODE_STIF_RES', 'CODE_STIF_ARRET',
                 'LIBELLE_ARRET', 'ID_REFA_LDA', 'CATEGORIE_TITRE', 'NB_VALD']

    df1_2015 = pd.read_csv('data/data-rf-2015/2015_S1_NB_FER.csv', sep=';', names=col_names, skiprows=1, low_memory=False, encoding='ISO-8859-1')
    df2_2015 = pd.read_csv('data/data-rf-2015/2015_S2_NB_FER.csv', sep=';', names=col_names, skiprows=1, low_memory=False, encoding='ISO-8859-1')

    df = pd.concat([df1_2015, df2_2015])
    for year in years:
        df_per_year1 = pd.read_csv(f'data/data-rf-{year}/{year}_S1_NB_FER.txt', sep='\t', names=col_names, skiprows=1, low_memory=False, encoding='ISO-8859-1')
        
        separator = '\t'

        df_per_year2 = pd.read_csv(f'data/data-rf-{year}/{year}_S2_NB_FER.txt', sep=separator, names=col_names, skiprows=1, low_memory=False, encoding='ISO-8859-1')
        df_per_year = pd.concat([df_per_year1, df_per_year2], ignore_index=True)
        df = pd.concat([df, df_per_year], ignore_index=True)
    
    df = df[['JOUR', 'LIBELLE_ARRET', 'NB_VALD']]

    df.loc[:, 'JOUR'] = pd.to_datetime(df['JOUR'], dayfirst=True, errors='coerce')

    df = df.copy()
    df.loc[df['NB_VALD'] == 'Moins de 5', 'NB_VALD'] = 4
    df.loc[:, 'NB_VALD'] = df['NB_VALD'].fillna(0)
    df.loc[:, 'NB_VALD'] = df['NB_VALD'].astype(int)
    df = df.groupby(['JOUR', 'LIBELLE_ARRET'])['NB_VALD'].sum().reset_index()

    df.loc[:, 'LIBELLE_ARRET_clean'] = df['LIBELLE_ARRET'].apply(
        lambda x: re.sub(r'\s+', '', re.sub(r'\s*\([^)]*\)', '', str(x)).replace('-', ''))
    )

    # Lecture des données de stations
    station_df = pd.read_csv('data/station_data/emplacement_gare.csv', sep=';')
    station_df = station_df[['nom_long', 'nom_ZdA', 'nom_ZdC','res_com']]

    df_final = pd.DataFrame()

    # Merge des données de station avec les données des arrêts
    for col in ['nom_long', 'nom_ZdA', 'nom_ZdC']:
        station_df.loc[:, f'{col}_clean'] = station_df[col].apply(
            lambda x: re.sub(
                r'\s+', '',
                re.sub(
                    r'\([^)]*\)', '',
                    unicodedata.normalize('NFKD', str(x))
                    .encode('ASCII', 'ignore')
                    .decode('utf-8')
                    .upper()
                    .replace('-', '')
                )
            )
        )

        df_tamp = pd.merge(df, station_df, left_on='LIBELLE_ARRET_clean', right_on=f'{col}_clean', how='inner')
        df_final = pd.concat([df_final, df_tamp], ignore_index=True)

        df_merged = pd.merge(df, station_df, left_on='LIBELLE_ARRET_clean', right_on=f'{col}_clean', how='left')
        df_non_jointes = df_merged[df_merged['nom_long'].isna()]
        df = df_non_jointes[['JOUR', 'LIBELLE_ARRET', 'LIBELLE_ARRET_clean', 'NB_VALD']]

    df_final = df_final[['JOUR', 'LIBELLE_ARRET', 'LIBELLE_ARRET_clean', 'NB_VALD', 'res_com']]
    
    # Création d'une correspondance pour les noms de stations
    correspondance = {
        'AEROPORT CHARLES DE GAULLE 1': 'Terminal 1',
        'AEROPORT CHARLES DE GAULLE 2-TGV': 'Terminal 2 - Gare TGV',
        'ASNIERES': 'Asnières-sur-Seine',
        'AUBERVILLIERS-PANTIN (QUATRE CHEMINS)': 'Aubervilliers Pantin - Quatre Chemins',
        'AUSTERLITZ': 'Gare d\'Austerlitz',
        'BAGNEUX-SUR-LOING': 'Bagneaux-sur-Loing',
        'BLANC-MESNIL': 'Le Blanc-Mesnil',
        'BOBIGNY-PANTIN (RAYMOND QUENEAU)': 'Bobigny-Pantin - Raymond Queneau',
        'BOUTIGNY-SUR-ESSONNE': 'Boutigny',
        "CHAUSSEE D'ANTIN (LA FAYETTE)": 'Chaussée d\'Antin - La Fayette',
        'CHENAY-GAGNY': 'Le Chénay-Gagny',
        'CHESSY - MARNE-LA-VALLEE': 'Marne-la-Vallée-Chessy',
        'COUDRAY-MONTCEAUX': 'Le Coudray-Montceaux',
        'CRECY-EN-BRIE-LA-CHAPELLE': 'Crécy-la-Chapelle',
        'CRETEIL POMP': 'Créteil Pompadour',
        'CRETEIL-P. LAC': 'Créteil - Pointe du Lac',
        'FR. POPULAIRE ': 'Front Populaire',
        'FRATERNELLE': 'Rungis-La Fraternelle',
        'GABRIEL PERI-ASNIERES-GENNEVILLIERS': 'Gabriel Péri',
        'GARE DE GENNEVILLIERS': 'Gennevilliers',
        'LA DEFENSE-GRANDE ARCHE': 'La Défense',
        'LES AGNETTES-ASNIERES-GENNEVILLIERS': 'Les Agnettes',
        'M. MONTROUGE  ': 'Mairie de Montrouge',
        "MERY-VAL D'OISE": 'Méry-sur-Oise',
        'PIERRE CURIE': 'Pierre et Marie Curie',
        'SAINT-MANDE-TOURELLE': 'Saint-Mandé',
        'SOUPPES': 'Souppes-Château-Landon',
        'SAINT-GERMAIN-BEL-AIR-FOURQUEUX': 'Fourqueux - Bel Air',
    }

    for key, value in correspondance.items():
        result = station_df[
            (station_df['nom_long'] == value) |
            (station_df['nom_ZdA'] == value) |
            (station_df['nom_ZdC'] == value)
        ]

        if len(result) > 0:
            correspondance[key] = result.iloc[0]

    keys = correspondance.keys()
    df = df.copy()
    for idx, row in df.iterrows():
        libelle = row['LIBELLE_ARRET']
        if libelle in keys:
            df.loc[idx, 'res_com'] = correspondance[libelle]['res_com']

    df_final = pd.concat([df_final, df], ignore_index=True)

    # Transformation de la colonne res_com en colonnes binaires
    df_dummies = df_final['res_com'].str.get_dummies(sep=' / ')

    # Fusion avec les données de station
    df_lignes_stations = pd.concat([df_final[['LIBELLE_ARRET']], df_dummies], axis=1)

    df_lignes_stations = df_lignes_stations.melt(
        id_vars='LIBELLE_ARRET',
        var_name='ligne_res',
        value_name='present'
    )

    df_lignes_stations = df_lignes_stations[df_lignes_stations['present'] == 1]
    df_lignes_stations = df_lignes_stations.drop(columns='present')
    df_lignes_stations = df_lignes_stations.drop_duplicates().reset_index(drop=True)

    df_final = df_final.drop(columns=['res_com'])

    # Ajout de la saisonnalité : jour de l'année et semaine de l'année
    df_final = df_final.copy()
    df_final['DAY_OF_YEAR'] = df_final['JOUR'].dt.dayofyear
    df_final['WEEK_OF_YEAR'] = df_final['JOUR'].dt.isocalendar().week
    df_final['DAY_OF_WEEK'] = df_final['JOUR'].dt.dayofweek  # Transformation de is_weekend en day_of_week

    # Création des jours fériés et week-end
    years = df_final['JOUR'].dt.year.unique()
    fr_holidays = holidays.France(years=years)
    holidays_dates = pd.to_datetime(list(fr_holidays.keys()))
    df_final['IS_FERIE'] = df_final['JOUR'].isin(holidays_dates).astype(int)
    df_final['IS_WEEKEND'] = df_final['DAY_OF_WEEK'].isin([5, 6]).astype(int)  # Transformation en weekend

    # Ajout de la colonne IS_VACANCE (vacances scolaires zone C, hors 2020)
    vacances_zone_c = [
        # 2015
        ('2014-12-20', '2015-01-05'),
        ('2015-02-14', '2015-03-02'),
        ('2015-04-18', '2015-05-04'),
        ('2015-07-04', '2015-08-31'),
        ('2015-10-17', '2015-11-02'),
        ('2015-12-19', '2016-01-04'),
        # 2016
        ('2016-02-20', '2016-03-07'),
        ('2016-04-16', '2016-05-02'),
        ('2016-07-05', '2016-08-31'),
        ('2016-10-19', '2016-11-02'),
        ('2016-12-17', '2017-01-03'),
        # 2017
        ('2017-02-04', '2017-02-20'),
        ('2017-04-01', '2017-04-18'),
        ('2017-07-08', '2017-09-03'),
        ('2017-10-21', '2017-11-06'),
        ('2017-12-23', '2018-01-08'),
        # 2018
        ('2018-02-17', '2018-03-05'),
        ('2018-04-14', '2018-04-30'),
        ('2018-07-07', '2018-09-02'),
        ('2018-10-20', '2018-11-05'),
        ('2018-12-22', '2019-01-07'),
        # 2019
        ('2019-02-23', '2019-03-11'),
        ('2019-04-20', '2019-05-06'),
        ('2019-07-06', '2019-09-01'),
        ('2019-10-19', '2019-11-04'),
        ('2019-12-21', '2020-01-06'),
        # 2021
        ('2021-02-13', '2021-03-01'),
        ('2021-04-17', '2021-05-03'),
        ('2021-07-06', '2021-09-02'),
        ('2021-10-23', '2021-11-08'),
        ('2021-12-18', '2022-01-03'),
        # 2022
        ('2022-02-19', '2022-03-07'),
        ('2022-04-23', '2022-05-09'),
        ('2022-07-07', '2022-09-01'),
        ('2022-10-22', '2022-11-07'),
        ('2022-12-17', '2023-01-03'),
        # 2023
        ('2023-02-18', '2023-03-06'),
        ('2023-04-22', '2023-05-09'),
        ('2023-07-08', '2023-09-03'),
        ('2023-10-21', '2023-11-06'),
        ('2023-12-23', '2024-01-08'),
    ]
    vacance_dates = set()
    for start, end in vacances_zone_c:
        vacance_dates.update(pd.date_range(start, end, freq='D'))
    df_final['IS_VACANCE'] = df_final['JOUR'].isin(vacance_dates).astype(int)

    # Enregistrement du dataframe final
    df_lignes_stations.loc[:, 'LIBELLE_ARRET_clean'] = df_lignes_stations['LIBELLE_ARRET'].apply(
        lambda x: re.sub(r'\s+', '', re.sub(r'\s*\([^)]*\)', '', str(x)).replace('-', ''))
    )

    # Pour éviter le doublon de colonne lors du reset_index, change l'agg
    df_lignes_stations = df_lignes_stations.groupby(['LIBELLE_ARRET_clean', 'ligne_res'], as_index=False).agg({
        'LIBELLE_ARRET': 'first'
    })
    df_lignes_stations['LIBELLE_ARRET'] = df_lignes_stations['LIBELLE_ARRET'].str.strip()

    df_lignes_stations.to_csv('final_data/df_lignes_ferrees.csv', index=False, encoding='ISO-8859-1')
    df_final.to_csv('final_data/df_all_data.csv', index=False, encoding='ISO-8859-1')