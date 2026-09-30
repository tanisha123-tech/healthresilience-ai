# HealthResilience AI

**BRICS 2026 — Resilience**
**Track 3 — Smart Health & Supply Chain Resilience**

## Problem
In large-scale public health networks, critical resources like medicines, staff, and beds are often misallocated. Unpredictable demand spikes (e.g., during pandemics or natural disasters) lead to stock-outs at some Primary Health Centres (PHCs) while others have surplus, leading to avoidable healthcare crises.

## Solution
HealthResilience AI is a predictive healthcare resource and supply-chain resilience platform. It uses machine learning to forecast demand, predict stock-out probabilities, and intelligently recommend resource redistribution across a pan-India network of PHCs. 

## Architecture
The application is built using Streamlit for the frontend and Python for the backend, powered by a suite of XGBoost models. It uses Firebase Firestore as a NoSQL database for syncing and persisting AI predictions and alerts.

## AI Models
1. **Demand Forecasting (XGBRegressor):** Predicts the 7-day demand for specific medicines at each PHC based on current stock, consumption, patient footfall, seasonality, and emergency indices.
2. **Stock-out Risk (XGBClassifier):** Predicts the probability of a stock-out occurring within the next 7 days.
3. **SHAP Explainability:** Provides real-time feature importance, showing exactly why a specific PHC is predicted to be at high risk (e.g., low staff, high bed occupancy, critical stock levels).

## Resource Redistribution Logic
Instead of a black-box model, redistribution uses an optimization scoring approach. When a PHC is predicted to have a critical shortage, the system identifies candidates with a projected surplus of the same medicine. It calculates a source score based on:
- Projected surplus quantity
- Haversine distance (using latitude/longitude)
- Lead time

The best suitable source is then recommended for transfer.

## Emergency Mode
A simulation toggle that instantly stress-tests the network. When activated, patient footfall and emergency demand indices increase drastically. The AI models recalculate predictions in real-time, identifying new high-risk PHCs and generating updated redistribution recommendations.

## Federated Learning Concept
Demonstrates a 5-node BRICS simulated network (India, Brazil, Russia, China, South Africa). It illustrates how local healthcare nodes can train models on local data and share model updates without centralizing or exposing sensitive raw patient records.

## Tech Stack
- **Frontend/Backend:** Python, Streamlit
- **Data & AI:** Pandas, NumPy, XGBoost, SHAP, Scikit-Learn
- **Visualization:** Plotly
- **Database:** Google Firebase Firestore, Firebase Admin SDK

## Setup & Installation

```powershell
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate synthetic dataset
python generate_data.py
```

### Firebase / Firestore Configuration
To enable database persistence:
1. Create a Firebase project and a Firestore database.
2. Generate a Service Account JSON key from the Firebase Console.
3. You can configure credentials in two ways:
   - **Streamlit Secrets:** Create a `.streamlit/secrets.toml` file and add your Firebase credentials under a `[firebase]` block.
   - **Environment Variable:** Set the `FIREBASE_CREDENTIALS` environment variable to the stringified JSON content.
*Note: The app will run in a safe demo/fallback mode if Firebase credentials are not found.*

## Run Instructions

```powershell
streamlit run app.py
```

## Demo Flow
1. **India-wide Dashboard:** View the All India map and overall metrics.
2. **Select Critical PHC:** Expand a high-risk PHC alert.
3. **View Predictions:** Observe predicted demand and stock-out probability.
4. **SHAP Explainability:** See the AI model's reasoning for the risk score.
5. **Redistribution:** Review the AI-recommended resource transfer from a surplus PHC.
6. **Emergency Simulation:** Activate Emergency Mode to see real-time recalculations and increased network stress.
7. **Federated Learning:** Scroll down to view the BRICS federated learning node simulation.

## Disclaimer
**The PHC data included in this repository (`data/phc_data.csv`) is 100% synthetic and generated for demonstration purposes only. It does not represent real government or patient data.**
