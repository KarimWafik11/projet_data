import time, re
import pandas as pd
from model_serie_temporelle.sarimax_model_rer_e import sarimax_model_rer_e

start = time.time()


df = pd.read_csv('final_data/df_lignes_ferrees.csv')
final_result = {}
cpt = 1
rer_e = list(df[df['ligne_res'] == 'RER E']['LIBELLE_ARRET_clean'].unique())

df = pd.read_csv('final_data/df_all_data.csv')
rer_e = list(df['LIBELLE_ARRET'].unique())

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

station_exclues = station_dates_exclues.keys()
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
rer_e = rer_e+stations_rer_e_ouest
final_df = pd.DataFrame()

for station in rer_e:
    print(station)
    if station == "NANTERRE-LA-FOLIE":
        station = 'NANTERRE-PREFECTURE'
    if station in station_exclues:
        dates_exclues = station_dates_exclues[station]
    else:
        dates_exclues = []

    result = sarimax_model_rer_e(station, dates_exclues)

    if station == 'NANTERRE-PREFECTURE':
        station = "NANTERRE-LA-FOLIE"

    final_df.pd.concat([final_df, result], ignore_index=True)
#final_df.to_csv('final_data/df_rer_e.csv', index=False)




print(f"Temps d'exécution : {time.time() - start}")

