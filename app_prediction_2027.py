import time
import pandas as pd
from model_serie_temporelle.sarimax_model_rer_e_2027 import sarimax_model_rer_e_2027

start = time.time()


# Ajout des futures stations du RER E Ouest
stations_rer_e_ouest = [
    "NANTERRE-LA-FOLIE",
    "HOUILLES-CARRIERES-SUR-SEINE",
    "POISSY",
    "VILLENNES-SUR-SEINE",
    "VERNOUILLET-VERNEUIL",
    "LES CLAIRIERES-DE-VERNEUIL",
    "LES MUREAUX",
    "AUBERGENVILLE-ELISABETHVILLE",
    "EPONE-MEZIERES",
    "MANTES-STATION",
    "MANTES-LA-JOLIE"
]

# Chargement des stations du RER E
df = pd.read_csv('final_data/df_lignes_ferrees.csv')
rer_e = list(df[df['ligne_res'] == 'RER E']['LIBELLE_ARRET'].unique())

# Calcul du nombre de correspondances par station
def calculer_correspondances(df, rer_e):
    """
    Calcule le nombre de correspondances par station en excluant les tramways
    """
    # Grouper par station et compter les lignes qui ne sont pas des tramways
    correspondances = {}
    
    for station in rer_e:
        # Filtrer les lignes pour cette station
        lignes_station = df[df['LIBELLE_ARRET'] == station]['ligne_res'].tolist()
        
        # Compter uniquement les lignes non-tramway (les 4 premiers caractères ne sont pas 'TRAM')
        unique_lines = set()
        for ligne in lignes_station:
            if ligne[:4] != 'TRAM':
                unique_lines.add(ligne)
        
        correspondances[station] = len(unique_lines)
    
    return correspondances

# Fusion des listes de stations
all_stations = list(set(rer_e + stations_rer_e_ouest))

# Calcul des correspondances pour toutes les stations
correspondances = calculer_correspondances(df, all_stations)

# Pour les stations de l'extension ouest qui n'existent pas encore dans df_lignes_ferrees.csv,
# on leur attribue 1 correspondance par défaut (sera incrémenté plus tard pour 2027)
for station in stations_rer_e_ouest:
    if station not in correspondances:
        correspondances[station] = 1

# Affichage du nombre de correspondances par station
print("\nNombre de correspondances par station:")
for station, count in sorted(correspondances.items()):
    if station in stations_rer_e_ouest:
        print(f"  {station}: {count} (future station de l'extension ouest)")
    else:
        print(f"  {station}: {count}")



# Dates à exclure pour certaines stations (événements spéciaux, travaux, etc.)
station_dates_exclues = {
    "PANTIN": ['2023-01-19', '2023-01-31', '2023-03-07', '2023-03-08', '2023-03-25', '2023-03-26'],
    "MAGENTA":[ '2023-01-19', '2023-01-31', '2023-03-07', '2023-03-08'],
    "ROSA PARKS":['2023-03-25'],
    "VILLIERS-SUR-MARNE-PLESSIS-TREVISE":['2023-01-14', '2023-02-18', '2023-02-19'],
    "LE RAINCY-VILLEMOMBLE-MONTFERMEIL":['2023-02-04', '2023-02-05'],
    "LES BOULLEREAUX-CHAMPIGNY":['2023-01-14', '2023-01-15', '2023-01-21', '2023-01-22', '2023-02-18', '2023-02-25', '2023-02-26', '2023-03-18', '2023-03-19', '2023-03-25'],
    "ROISSY EN BRIE":['2023-01-14', '2023-01-15', '2023-02-18', '2023-02-19'],
    "EMERAINVILLE-PONTAULT-COMBAULT":['2023-01-14', '2023-01-15', '2023-02-18', '2023-02-19'],
    "OZOIR-LA-FERRIERE":['2023-01-14', '2023-01-15', '2023-02-18', '2023-02-19'],
    'VILLENNES-SUR-SEINE':['2023-02-18', '2023-03-11', '2023-03-12'],
    'LES MUREAUX':['2023-02-19', '2023-03-11'],
    'AUBERGENVILLE-ELISABETHVILLE':['2023-01-28', '2023-01-29', '2023-02-11', '2023-02-12', '2023-02-18', '2023-02-25', '2023-02-26', '2023-03-04','2023-03-05','2023-03-11','2023-03-12'],
    'EPONE-MEZIERES': ['2023-03-11','2023-03-12', '2023-03-04','2023-03-05']
}

# Initialisation du DataFrame pour stocker les résultats
final_df = pd.DataFrame()

# Traitement pour chaque station
for station in all_stations:
    print(f"Traitement de la station: {station}")
    
    # Gestion de cas spécial pour Nanterre-La-Folie
    if station == "NANTERRE-LA-FOLIE":
        station_for_model = 'NANTERRE-PREFECTURE'
    else:
        station_for_model = station
    
    # Récupération des dates à exclure pour la station si applicable
    if station in station_dates_exclues:
        dates_exclues = station_dates_exclues[station]
    else:
        dates_exclues = []
    
    # Exécution du modèle de prédiction
    result = sarimax_model_rer_e_2027(station_for_model, dates_exclues)
    
    # Restauration du nom original de la station si modifié
    if station == "NANTERRE-LA-FOLIE" and result is not None:
        result['LIBELLE_ARRET'] = station
    
    # Ajout des résultats au DataFrame final
    if result is not None:
        final_df = pd.concat([final_df, result], ignore_index=True)

# Sauvegarde des résultats
final_df.to_csv('final_data/prediction_rer_e_2027.csv', index=False)

# Calcul de la durée d'exécution
duration = time.time() - start
print(f"Temps d'exécution: {duration:.2f} secondes")

# Analyse rapide des résultats
print("\nAnalyse des prédictions:")
print(f"Nombre total de prédictions: {len(final_df)}")
print(f"Période de prédiction: de {final_df['ds'].min()} à {final_df['ds'].max()}")
print(f"Nombre total de validations prédites: {final_df['yhat'].sum():,}".replace(',', ' '))

# Chargement et analyse des données historiques pour comparaison
print("\nAnalyse des données historiques pour comparaison:")
historique_df = pd.read_csv('final_data/df_all_data.csv', parse_dates=['JOUR'])

# Renommer les colonnes pour correspondre à notre format
historique_df = historique_df.rename(columns={
    'JOUR': 'ds',
    'NB_VALD': 'y'
})

# Filtrer pour ne garder que les stations du RER E existantes (hors extension ouest)
stations_existantes = set(rer_e) - set(stations_rer_e_ouest)
historique_df_rer_e = historique_df[historique_df['LIBELLE_ARRET'].isin(stations_existantes)]

# Calculer l'affluence totale par année pour les stations existantes
historique_df_rer_e['year'] = historique_df_rer_e['ds'].dt.year
historique_df_rer_e['quarter'] = historique_df_rer_e['ds'].dt.quarter
yearly_totals = historique_df_rer_e.groupby('year')['y'].sum()

print("\nValidations historiques par année pour le RER E (stations existantes):")
for year, total in yearly_totals.items():
    print(f"  {year}: {total:,} validations".replace(',', ' '))

# Calculer l'affluence pour le 1er trimestre de chaque année (stations existantes)
q1_totals = historique_df_rer_e[historique_df_rer_e['quarter'] == 1].groupby('year')['y'].sum()
print("\nValidations historiques pour le 1er trimestre de chaque année (stations existantes):")
for year, total in q1_totals.items():
    print(f"  Q1 {year}: {total:,} validations".replace(',', ' '))

# Ajuster les prédictions en fonction du nombre de correspondances
# Pour les stations de l'extension ouest, on ajoute 1 correspondance car elles seront desservies par le RER E en 2027
correspondances_2027 = correspondances.copy()
for station in stations_rer_e_ouest:
    correspondances_2027[station] += 1

# Création d'un DataFrame avec les prédictions ajustées selon les correspondances
adjusted_df = final_df.copy()
adjusted_df['nb_correspondances'] = adjusted_df['LIBELLE_ARRET'].map(correspondances_2027)
adjusted_df['yhat_adjusted'] = adjusted_df.apply(lambda row: row['yhat'] / row['nb_correspondances'], axis=1)

# Prédiction pour Q1 2027 pour les stations existantes uniquement (avec ajustement des correspondances)
pred_brut_existantes = final_df[~final_df['LIBELLE_ARRET'].isin(stations_rer_e_ouest)]['yhat'].sum()
pred_ajuste_existantes = adjusted_df[~adjusted_df['LIBELLE_ARRET'].isin(stations_rer_e_ouest)]['yhat_adjusted'].sum()

print(f"\nPrédiction pour Q1 2027 (stations existantes uniquement):")
print(f"  Brut: {pred_brut_existantes:,} validations".replace(',', ' '))
print(f"  Ajusté par correspondances: {pred_ajuste_existantes:,} validations".replace(',', ' '))

# Calculer le taux de croissance par rapport au dernier Q1 disponible (stations existantes)
derniere_annee = q1_totals.index.max()
dernier_q1 = q1_totals[derniere_annee]

# Ajuster les données historiques avec les correspondances
historique_df_rer_e['nb_correspondances'] = historique_df_rer_e['LIBELLE_ARRET'].map(correspondances)
historique_df_rer_e['y_adjusted'] = historique_df_rer_e.apply(lambda row: row['y'] / row['nb_correspondances'], axis=1)
dernier_q1_ajuste = historique_df_rer_e[(historique_df_rer_e['year'] == derniere_annee) & 
                                      (historique_df_rer_e['quarter'] == 1)]['y_adjusted'].sum()

croissance_brute = ((pred_brut_existantes / dernier_q1) - 1) * 100
croissance_ajustee = ((pred_ajuste_existantes / dernier_q1_ajuste) - 1) * 100
print(f"Taux de croissance par rapport à Q1 {derniere_annee} (stations existantes):")
print(f"  Brut: {croissance_brute:.2f}%")
print(f"  Ajusté par correspondances: {croissance_ajustee:.2f}%")

# Prédiction pour Q1 2027 avec l'extension ouest (toutes les stations)
pred_brut_total = final_df['yhat'].sum()
pred_ajuste_total = adjusted_df['yhat_adjusted'].sum()
print(f"\nPrédiction pour Q1 2027 (avec extension ouest):")
print(f"  Brut: {pred_brut_total:,} validations".replace(',', ' '))
print(f"  Ajusté par correspondances: {pred_ajuste_total:,} validations".replace(',', ' '))

# Contribution des stations de l'extension ouest
pred_brut_ouest = final_df[final_df['LIBELLE_ARRET'].isin(stations_rer_e_ouest)]['yhat'].sum()
pred_ajuste_ouest = adjusted_df[adjusted_df['LIBELLE_ARRET'].isin(stations_rer_e_ouest)]['yhat_adjusted'].sum()
print(f"Contribution des stations de l'extension ouest:")
print(f"  Brut: {pred_brut_ouest:,} validations".replace(',', ' '))
print(f"  Ajusté par correspondances: {pred_ajuste_ouest:,} validations".replace(',', ' '))

pourcentage_ouest_brut = (pred_brut_ouest / pred_brut_total) * 100
pourcentage_ouest_ajuste = (pred_ajuste_ouest / pred_ajuste_total) * 100
print(f"Pourcentage de l'extension ouest dans le total:")
print(f"  Brut: {pourcentage_ouest_brut:.2f}%")
print(f"  Ajusté par correspondances: {pourcentage_ouest_ajuste:.2f}%")

# Analyse détaillée des stations de l'extension ouest
print("\nPrédictions détaillées pour les stations de l'extension ouest (Q1 2027):")
stations_ouest_df = adjusted_df[adjusted_df['LIBELLE_ARRET'].isin(stations_rer_e_ouest)]
stations_details = []

for station in stations_rer_e_ouest:
    station_df = stations_ouest_df[stations_ouest_df['LIBELLE_ARRET'] == station]
    if not station_df.empty:
        brut = station_df['yhat'].sum()
        ajuste = station_df['yhat_adjusted'].sum()
        nb_corresp = station_df['nb_correspondances'].iloc[0]
        stations_details.append((station, brut, ajuste, nb_corresp))

# Trier par validations brutes décroissantes
stations_details.sort(key=lambda x: x[1], reverse=True)

# Afficher les détails
for station, brut, ajuste, nb_corresp in stations_details:
    print(f"  {station} ({nb_corresp} correspondances):")
    print(f"    Validations brutes: {brut:,}".replace(',', ' '))
    print(f"    Validations ajustées: {ajuste:,}".replace(',', ' '))

# Calcul de l'augmentation totale par rapport au dernier trimestre historique
augmentation_brute = ((pred_brut_total / dernier_q1) - 1) * 100
augmentation_ajustee = ((pred_ajuste_total / dernier_q1_ajuste) - 1) * 100

print(f"\nAugmentation totale attendue Q1 2027 vs Q1 {derniere_annee} (avec extension ouest):")
print(f"  Brut: {augmentation_brute:.2f}%")
print(f"  Ajusté par correspondances: {augmentation_ajustee:.2f}%")

print(f"\nDécomposition de l'augmentation brute:")
print(f"  {croissance_brute:.2f}% de croissance naturelle")
print(f"  {augmentation_brute - croissance_brute:.2f}% grâce à l'extension ouest")

print(f"\nDécomposition de l'augmentation ajustée:")
print(f"  {croissance_ajustee:.2f}% de croissance naturelle")
print(f"  {augmentation_ajustee - croissance_ajustee:.2f}% grâce à l'extension ouest")

# Top 10 des stations avec le plus de validations (ajustées)
print("\nTop 10 des stations avec le plus de validations ajustées (Q1 2027):")
top_stations = adjusted_df.groupby('LIBELLE_ARRET').agg({
    'yhat': 'sum',
    'yhat_adjusted': 'sum',
    'nb_correspondances': 'first'
}).sort_values('yhat_adjusted', ascending=False).head(10)

for idx, (station, row) in enumerate(top_stations.iterrows(), 1):
    print(f"  {idx}. {station} ({int(row['nb_correspondances'])} correspondances):")
    print(f"     Validations brutes: {row['yhat']:,.0f}".replace(',', ' '))
    print(f"     Validations ajustées: {row['yhat_adjusted']:,.0f}".replace(',', ' '))

# Affichage des statistiques par année
yearly_stats = final_df.copy()
yearly_stats['year'] = yearly_stats['ds'].dt.year
yearly_totals = yearly_stats.groupby('year')['yhat'].sum()

print("\nValidations prédites par année:")
for year, total in yearly_totals.items():
    print(f"  {year}: {total:,} validations".replace(',', ' '))

# Affichage des stations avec le plus de validations prédites
station_totals = final_df.groupby('LIBELLE_ARRET')['yhat'].sum().sort_values(ascending=False)

print("\nTop 5 des stations avec le plus de validations prédites:")
for station, total in station_totals.head(5).items():
    print(f"  {station}: {total:,} validations".replace(',', ' '))

# Analyse de l'affluence sans les stations ouest
print("\nAnalyse de l'impact de l'extension ouest:")
# Créer un masque pour filtrer les stations existantes vs stations ouest
is_ouest = final_df['LIBELLE_ARRET'].isin(stations_rer_e_ouest)

# Total des validations pour le RER E actuel (sans les stations ouest)
df_sans_ouest = final_df[~is_ouest].copy()
total_sans_ouest = df_sans_ouest['yhat'].sum()
print(f"Total des validations du RER E actuel (2023-2027): {total_sans_ouest:,}".replace(',', ' '))

# Total des validations pour les nouvelles stations ouest
df_ouest = final_df[is_ouest].copy()
total_ouest = df_ouest['yhat'].sum()
print(f"Total des validations des stations de l'extension ouest (2023-2027): {total_ouest:,}".replace(',', ' '))

# Pourcentage d'augmentation grâce à l'extension ouest
pourcentage = (total_ouest / total_sans_ouest) * 100
print(f"L'extension ouest représente {pourcentage:.2f}% de trafic supplémentaire")

# Analyse par année pour les deux groupes de stations
print("\nValidations par année pour le RER E actuel vs extension ouest:")
df_sans_ouest['year'] = df_sans_ouest['ds'].dt.year
df_ouest['year'] = df_ouest['ds'].dt.year

totals_sans_ouest = df_sans_ouest.groupby('year')['yhat'].sum()
totals_ouest = df_ouest.groupby('year')['yhat'].sum()

for year in range(2023, 2030):
    sans_ouest = totals_sans_ouest.get(year, 0)
    ouest = totals_ouest.get(year, 0)
    total = sans_ouest + ouest
    if total > 0:
        pct_ouest = (ouest / total) * 100
        print(f"  {year}: {sans_ouest:,} (RER E actuel) + {ouest:,} (extension ouest) = {total:,} validations ({pct_ouest:.1f}% ouest)".replace(',', ' '))

duration = time.time() - start