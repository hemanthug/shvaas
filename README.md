# Shvaas

Hyperlocal air quality prediction using machine learning and environmental data.

Shvaas is an applied machine learning project focused on predicting air pollution at a hyperlocal level by combining historical air quality measurements with weather and environmental signals.

This project began with a simple goal: learning machine learning by building something real. Instead of tutorials or toy datasets, the idea was to learn by working through real world data challenges and understanding how models behave under messy, imperfect conditions.

The initial focus is PM2.5 prediction across Los Angeles, but the long term vision is to move toward environmental intelligence and risk aware predictions.

---

## 📑 Table of Contents

- [🚀 Why This Project Exists](#-why-this-project-exists)
- [🎯 Goals](#-goals)
- [📊 Data Sources](#-data-sources)
- [🧠 Modeling Approach](#-modeling-approach)
- [📈 What I’ve Learned So Far](#-what-ive-learned-so-far)
- [🔭 Where This Is Going](#-where-this-is-going)
- [🛠️ Tech Stack](#️-tech-stack)
- [📂 Project Structure](#-project-structure)
- [📌 Project Status](#-project-status)

---

## 🚀 Why This Project Exists

Air quality looks simple until you try to model it. A baseline model that uses previous pollutant values performs surprisingly well because pollution has strong persistence. But that only captures continuity, not the underlying environmental behavior.

This project explores questions like:

- Does weather actually improve predictions?
- Why does the same model behave differently across locations?
- Can we move beyond simple forecasting toward meaningful environmental insight?

---

## 🎯 Goals

- Build a strong PM2.5 baseline model using real sensor data  
- Integrate weather and wind signals to capture transport and dispersion effects  
- Evaluate performance at the site level, not just global averages  
- Understand where environmental features help and where they add noise  
- Move toward regime aware modeling and risk based insights  

---

## 📊 Data Sources

- EPA AQS hourly air quality measurements  
- NOAA Global Hourly weather datasets  
- Station metadata including geographic coordinates  

### Features Used

- Historical pollutant concentration  
- Temperature, humidity, pressure  
- Wind speed and direction  
- Time based cyclic features (hour and daily patterns)  
- Spatial station mapping  

---

## 🧠 Modeling Approach

Current experiments include:

- Random Forest regression baseline  
- Temporal and environmental feature engineering  
- Site level performance analysis  
- Comparison between persistence driven models and weather enhanced models  

### Key Insight So Far

Adding weather does not uniformly improve performance. Some sites improve significantly while others degrade, suggesting that pollution behavior is regime dependent rather than universal.

---

## 📈 What I’ve Learned So Far

- Persistence is an extremely strong baseline  
- Aggregate metrics can hide important local behavior  
- Weather driven transport matters in some regions but introduces noise in others  
- Evaluating models geographically changes how you interpret results  

---

## 🔭 Where This Is Going

Planned next steps:

- Regime classification for site specific modeling  
- Residual-spike source attribution to infer practical upwind source zones  
- Risk index generation instead of pure concentration prediction  
- Hyperlocal interpolation between monitoring stations  
- Uncertainty estimation and confidence bands  
- Expansion beyond PM2.5  

---

## 🛠️ Tech Stack

- Python  
- pandas  
- NumPy  
- scikit learn  
- matplotlib  

---

## 📂 Project Structure

```text
shvaas/
├── data/               # Raw, interim, processed datasets (git-ignored contents, .gitkeep kept)
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/               # Data & pipeline guides
├── src/                # Core code
│   ├── data_ingestion/ # Raw → interim cleaning/aggregation
│   ├── features/       # Feature engineering / prep
│   ├── models/         # Baseline + weather-aware models
│   └── visualization/  # Plots and exploratory utilities
├── reports/            # Generated outputs
│   └── figures/
├── requirements.txt
├── README.md
└── CONTRIBUTING.md
```

## 📌 Project Status  

Active learning and experimentation project. This repository documents the journey of learning machine learning through building a real world system and iterating based on observed behavior rather than assumptions.
