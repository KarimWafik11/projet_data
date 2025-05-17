import time

import pandas as pd
import unicodedata, re

start = time.time()


df = pd.read_csv('data-rf-2016/2016S1_NB_FER.txt', sep='\t')
df = df[['JOUR', 'LIBELLE_ARRET', 'NB_VALD']]
df = df.groupby(['JOUR', 'LIBELLE_ARRET'])['NB_VALD'].sum().reset_index()
df.loc[:, 'LIBELLE_ARRET_clean'] = df['LIBELLE_ARRET'].apply(
    lambda x: re.sub(r'\s+', '',
                     re.sub(r'\s*\([^)]*\)', '', str(x)).replace('-', '')
                     )
)

station_df = pd.read_csv('station_data/emplacement_gare.csv', sep=';')
station_df = station_df[['nom_long', 'nom_ZdA', 'nom_ZdC','res_com']]

df_final = pd.DataFrame()

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

    # Faire une jointure gauche pour garder toutes les lignes de df
    df_merged = pd.merge(df, station_df, left_on='LIBELLE_ARRET_clean', right_on=f'{col}_clean', how='left')

    # Garder uniquement les lignes non matchées (celles où 'res_com' est NaN)
    df_non_jointes = df_merged[df_merged['nom_long'].isna()]
    df = df_non_jointes[['JOUR', 'LIBELLE_ARRET', 'LIBELLE_ARRET_clean', 'NB_VALD']]

df_final = df_final[['JOUR', 'LIBELLE_ARRET', 'NB_VALD', 'res_com']]

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

df = df.copy()  # Créer une copie complète du DataFrame au début
for idx, row in df.iterrows():
    libelle = row['LIBELLE_ARRET']
    if libelle in keys:
        df.loc[idx, 'res_com'] = correspondance[libelle]['res_com']

df_final = pd.concat([df_final, df], ignore_index=True)
df_final = df_final.drop(columns='LIBELLE_ARRET_clean')



#print(time.time() - start)

