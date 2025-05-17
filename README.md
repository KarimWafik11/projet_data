Projet Data – Karim Wafik – Lyes Sid Ali – Mohamed Sehrane

Nous avons récupéré les données depuis :
https://data.iledefrance-mobilites.fr/explore/dataset/histo-validations-reseau-ferre/information/
Il s'agit de l'historique (de 2015 à 2023) des validations de titres de transport par station sur l'ensemble du réseau ferré (métro, RER, tram) d'Île-de-France.

Nous disposons de :

Un fichier CSV/TXT contenant le nombre de validations par station et par jour ;

Un autre fichier détaillant la répartition horaire des validations, exprimée en pourcentage par rapport au total journalier (issu du premier fichier).

Exemple :

Fichier 1 → Date : 01/01/2015 ; Station : Jussieu ; nb_validations = 500

Fichier 2 → Date : 01/01/2015 ; Station : Jussieu ; Heure : 9h–10h ; pourcentage : 0,2
→ On a donc : 0,2 × 500 = 100 validations de titres entre 9h et 10h à Jussieu.

Ces deux fichiers ne concernent qu’une seule année.
Pour les neuf années de données, il y a donc 9 × 2 fichiers.

Nous avons également couplé ces données avec un fichier CSV unique :
https://data.iledefrance-mobilites.fr/explore/dataset/emplacement-des-gares-idf-data-generalisee/table/?sort=nom_zdc&location=11,48.8751,2.35966&basemap=jawg.streets

Ce fichier permet de faire le lien avec les stations du premier jeu de données afin de connaître, pour chaque station, les lignes de RER, métro ou tram qui la desservent.


But du projet : Prédire le nombre de validation aux heures de pointes pour tous les rer.

Etape 1 : Nettoyage des données et couplage des données entre les 2 sets (test d'abord sur une annnée)
Etape 2 : Choix du modèle de regression à implementer
Etape 3 : Alimenter le modèle avec toutes les données


