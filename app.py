import time
import pandas as pd
from final_data.generate_data import generate_data
from model_regression.regression_cat_boost import regression_cat_boost
from model_serie_temporelle.model_sarimax import sarimax_prediction
from model_serie_temporelle.light_gbm_model import light_gbm_model

start = time.time()

#generate_data()    #permet de (re)générer le fichier df_concatenat.csv contenant toutes les données
#df = pd.read_csv('final_data/df_all_data.csv')
#dff = df[(df['LIBELLE_ARRET'] == 'JAVEL') & (df['JOUR'].str.contains('-12-31'))]
#print(dff)    #affiche le nombre de validation pour l'arrêt JAVEL le 1er janvier

year = 2024
vacances_2024 = [
    # Vacances d'hiver
    ('2024-02-10', '2024-02-26'),
    # Vacances de printemps
    ('2024-04-06', '2024-04-22'),
    # Vacances d'été
    ('2024-07-06', '2024-09-02'),
    # Vacances de la Toussaint
    ('2024-10-19', '2024-11-04'),
    # Vacances de Noël
    ('2024-12-21', '2025-01-06'),
]


print("light_gbm")
result = light_gbm_model("JAVEL", year, vacances_2024)

"""
print("sarimax")
result = sarimax_prediction("JAVEL", year, vacances_2024)
print("cat_boost")
result = regression_cat_boost(year, vacances_2024)    #prédire 2017 avec le modele de regression linear
"""
print(result)


print(time.time() - start)