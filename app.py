import time, re
import pandas as pd
from final_data.generate_data import generate_data
from model_serie_temporelle.prophet_model_by_arret import prophet_prediction_by_arret
from model_serie_temporelle.prophet_model_all import prophet_prediction_all
from model_serie_temporelle.prophet_model import prophet_prediction

start = time.time()

#generate_data()    #permet de (re)généré le fichier df_concatenat.csv contenant toutes les données
#df = pd.read_csv('final_data/df_all_data.csv')
#dff = df[(df['LIBELLE_ARRET'] == 'JAVEL') & (df['JOUR'].str.contains('-12-31'))]
#print(dff)    #affiche le nombre de validation pour l'arrêt JAVEL le 1er janvier

year = 2023
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


#result = prophet_prediction(year, 'LA PLAINE-STADE DE FRANCE')
#print(result)


lignes = ['METRO 12', 'RER C', 'RER A', 'TRAIN L', 'RER B', 'METRO 4', 'METRO 2', 'METRO 9', 'METRO 3', 'TRAIN J', 'TER', 'ORLYVAL', 'METRO 1', 'METRO 11', 'METRO 7', 'TRAIN K', 'METRO 10', 'METRO 5', 'TRAIN H', 'TRAIN R', 'METRO 8', 'RER D', 'METRO 13', 'METRO 6', 'TRAIN N', 'METRO 14', 'TRAIN V', 'METRO 7bis', 'RER E', 'TRAIN P', 'TRAIN U', 'METRO 3bis']

"""
final_result = {}
cpt = 1
for ligne in lignes:
    df_ligne = df_rer[df_rer['ligne_res'] == ligne].copy()
    liste_station = list(df_ligne['LIBELLE_ARRET_clean'].unique())
    
    print(cpt, ligne)
    result = prophet_prediction_all(year, liste_station)
    final_result[ligne] = result

    cpt += 1

print(final_result)

"""

#result = prophet_prediction_cdm(year, 'LA PLAINE-STADE DE FRANCE')
#result = prophet_prediction_cdm(year, 'VILLEPARISIS-MITRY-LE-NEUF')





print(f"Temps d'exécution : {time.time() - start}")



"""

[('AEROPORT CDG1', '57.94%'), ('AEROPORT CHARLES DE GAULLE 2-TGV', '84.82%'), ('ANTONY', '11.67%'), ('ARCUEIL-CACHAN', '19.02%'), ('AULNAY-SOUS-BOIS', '11.22%'), ('BAGNEUX', '19.92%'), ('BLANC-MESNIL', '13.81%'), ('BOURG-LA-REINE', '13.01%'), ('BURES-SUR-YVETTE', '58.22%'), ('CHATELET-LES HALLES', '30.24%'), ('CITE UNIVERSITAIRE', '27.35%'), ('COURCELLE-SUR-YVETTE', '70.75%'), ('DENFERT-ROCHEREAU', '14.23%'), ('DRANCY', '23.24%'), ('FONTAINE-MICHALON', '36.53%'), ('FONTENAY-AUX-ROSES', '13.47%'), ('GARE DU NORD', '9.87%'), ('GENTILLY', '15.65%'), ('GIF-SUR-YVETTE', '33.65%'), ('LA COURNEUVE-AUBERVILLIERS', '42.27%'), ('LA CROIX-DE-BERNY-FRESNES', '28.16%'), ('LA HACQUINIERE', '50.30%'), ('LAPLACE', '17.24%'), ('LA PLAINE-STADE DE FRANCE', '21.26%'), ('LE BOURGET', '9.83%'), ('LE GUICHET', '38.04%'), ('LES BACONNETS', '10.73%'), ('LOZERE', '22.47%'), ('LUXEMBOURG', '65.08%'), ('MASSY-PALAISEAU', '13.56%'), ('MASSY-VERRIERES', '38.09%'), ('MITRY-CLAYE', '24.63%'), ('ORSAY-VILLE', '18.24%'), ('PALAISEAU', '18.92%'), ('PALAISEAU-VILLEBON', '34.53%'), ('PARC-DE-SCEAUX', '14.94%'), ('PARC DES EXPOSITIONS', '29.06%'), ('PORT ROYAL', '64.11%'), ('ROBINSON', '19.08%'), ('SAINT-MICHEL', '31.15%'), ('SAINT-MICHEL NOTRE DAME', '50.84%'), ('SAINT-REMY-LES-CHEVREUSE', '35.02%'), ('SCEAUX', '23.65%'), ('SEVRAN-BEAUDOTTES', '15.56%'), ('SEVRAN-LIVRY', '13.91%'), ('VERT-GALANT', '22.50%'), ('VILLEPARISIS-MITRY-LE-NEUF', '36.68%'), ('VILLEPINTE', '22.30%')]

"""