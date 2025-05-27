
# Prédiction de trafic du RER E jusqu'en 2029

Ce projet vise à prédire le trafic du RER E jusqu'en 2029, y compris l'impact de son extension à l'ouest.

## Installation

1. Créez un environnement virtuel Python :
   ```
   python -m venv .venv
   ```

2. Activez l'environnement virtuel :
   - Windows : `.venv\Scripts\activate`
   - Linux/Mac : `source .venv/bin/activate`

3. Installez les dépendances :
   ```
   pip install -r requirements.txt
   ```

## Modèles disponibles

- `sarimax_model_rer_e.py` : Modèle pour prédire le trafic sur une courte période (test sur Q1 2023)
- `sarimax_model_rer_e_2029.py` : Modèle pour prédire le trafic jusqu'en 2029

## Utilisation

Pour générer des prédictions jusqu'en 2029 :
