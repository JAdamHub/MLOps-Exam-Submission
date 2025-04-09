# Bitcoin Dashboard

Dette projekt indeholder en applikation til at vise historiske Bitcoin-priser og forudsige fremtidige priser ved hjælp af machine learning-modeller.

## Komponenter

Projektet består af to hovedkomponenter:

1. **FastAPI Backend** - Håndterer data og modeller, og leverer endpoints til både historiske data og prognoseforudsigelser.
2. **Streamlit Frontend** - Giver en brugervenlig grænseflade til at visualisere data og interagere med API'en.

## Installation

Sørg for, at du har Python 3.9+ installeret. Installer derefter de nødvendige pakker:

```bash
pip install -r requirements.txt
```

## Brug

Der er to måder at starte applikationen på:

### Metode 1: Brug det automatiske startscript

Kør følgende kommando i terminalen:

```bash
./start_bitcoin_app.sh
```

Dette script vil:
- Starte FastAPI-serveren på port 8000
- Starte Streamlit-appen på port 8501
- Konfigurere dem til at kommunikere med hinanden

### Metode 2: Start komponenterne manuelt

#### Start FastAPI Backend

```bash
uvicorn src.api.bitcoin_api:app --host 0.0.0.0 --port 8000 --reload
```

#### Start Streamlit Frontend

```bash
export API_URL="http://localhost:8000"  # Sæt API URL miljøvariabel
streamlit run src/streamlit/bitcoin_app.py
```

## Funktioner

Applikationen indeholder følgende funktioner:

- **Dashboard:** Viser historiske Bitcoin-priser med interaktive grafer
- **Forudsigelser:** Laver forudsigelser for Bitcoin-prisen for de kommende 1, 3 og 7 dage
- **Modelmetrikker:** Viser performance-metrikker og feature importance for ML-modellerne

## API Endpoints

- `GET /price/history` - Henter historiske Bitcoin-priser
- `POST /predict` - Forudsiger Bitcoin-prisen baseret på input features
- `GET /metrics` - Henter model performance metrikker
- `GET /features` - Henter listen over features, der bruges af modellerne

## Datakilder

Applikationen bruger Bitcoin-prisdata fra CSV-filen placeret i `data/intermediate/combined/bitcoin_macro_combined_trading_days.csv`.

## Modeller

Modellerne er placeret i `models/`-mappen:
- `xgboost_model_1d.joblib` - Model til 1-dags forudsigelser
- `xgboost_model_3d.joblib` - Model til 3-dags forudsigelser
- `xgboost_model_7d.joblib` - Model til 7-dags forudsigelser

## Fejlfinding

Hvis du oplever problemer med at starte applikationen:

1. Sørg for, at portene 8000 og 8501 ikke allerede er i brug
2. Kontroller, at du har de korrekte stier til data og modeller
3. Tjek log-output for fejlmeddelelser 