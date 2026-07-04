# ⚽ FIFA World Cup 2026 Predictor & Tournament Simulator

An end-to-end Machine Learning application that predicts international football match outcomes and simulates the entire FIFA World Cup 2026 tournament using historical match data, FIFA rankings, team form, Elo ratings, and advanced feature engineering.

---

# 🚀 Features

## 🎯 Match Predictor
- Predicts Home Win / Draw / Away Win probabilities
- Ensemble of multiple Machine Learning models
- Uses FIFA Rankings
- Elo Ratings
- Recent Form (2-Year Weighted)
- Head-to-Head Statistics
- Confederation Strength
- Neutral Venue Support
- Interactive Probability Visualization
- Team Comparison Dashboard
- Last 5 Match History
- Head-to-Head Match History

---

## 🏆 Tournament Simulator
- Complete FIFA World Cup 2026 Simulation
- 48-Team Tournament Format
- Automatic Group Stage Simulation
- Knockout Stage Simulation
- Round of 32
- Round of 16
- Quarter Finals
- Semi Finals
- Final
- Tournament Winner Prediction

---

## 🎲 Monte Carlo Simulation
Run hundreds or thousands of tournament simulations to estimate:

- World Cup Winner Probability
- Final Appearance Probability
- Semi Final Probability
- Quarter Final Probability
- Round of 16 Probability
- Round of 32 Probability

---

# 🧠 Machine Learning

The prediction engine uses an ensemble of multiple ML models.

Current pipeline includes:

- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Extra Trees

The best-performing model is automatically selected for predictions.

---

# 📊 Features Used

The model incorporates multiple football-specific features:

- FIFA World Ranking
- Elo Rating
- Recent Form
- Goals Scored
- Goals Conceded
- Goal Difference
- Head-to-Head Record
- Confederation Strength
- Neutral Venue
- Tournament Calibration

---

# 📂 Project Structure

```
FIFA_WC_2026_Project
│
├── app.py
├── requirements.txt
│
├── data/
│
├── models/
│   ├── best_model.pkl
│   ├── all_models.pkl
│   ├── feature_columns.pkl
│
└── src/
    ├── config.py
    ├── feature_engineering.py
    ├── simulation.py
    └── train_models.py
```

---

# 🛠 Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost
- Plotly
- Matplotlib

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/AbhinavReddy84/FIFA_WC_2026_Project.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📈 Current Capabilities

✅ Match Prediction

✅ Probability Estimation

✅ Team Comparison

✅ Tournament Simulation

✅ Monte Carlo Tournament Simulation

---

# 🔮 Future Enhancements

### 🤖 Generative AI Match Analysis
- AI-generated pre-match reports
- Tactical analysis
- Strength & weakness explanations
- Match summaries
- Tournament insights using Large Language Models

---

### 🌍 Transfermarkt Integration

Planned integration of Transfermarkt data including:

- Squad Market Value
- Player Market Value
- Injuries
- Suspensions
- Expected Starting XI
- Average Squad Age
- Player Availability
- Club Form
- Transfer Activity

These features will be incorporated into the prediction model to improve real-world accuracy.

---

### 📊 Explainable AI

- SHAP Feature Importance
- Local Prediction Explanations
- Global Feature Importance
- Interactive Explainability Dashboard

---

### ☁ Deployment

- Streamlit Cloud
- Docker Support
- REST API
- Hugging Face Spaces

---

# 📌 Disclaimer

This project is intended for educational and research purposes.

Predictions are generated using historical data and statistical machine learning models and should not be considered guaranteed outcomes.

---

# 👨‍💻 Author

**Abhinav Reddy**

B.Tech CSE (AI & ML)

Manipal Institute of Technology

GitHub:
https://github.com/AbhinavReddy84
