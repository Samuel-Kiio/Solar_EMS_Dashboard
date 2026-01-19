# Energy Management System for Solar-Optimised Load Scheduling
<p align="center"> <img src="https://github.com/user-attachments/assets/f8e3a1ff-d47b-40b0-8473-10a486394058" alt="EMS" width="400"> </p>

## Overview

This repository presents an Energy Management System (EMS) developed for a university campus in Kenya to support data-driven monitoring, forecasting, and operational decision-making for on-site solar photovoltaic (PV) systems.

The system integrates a machine-learning-based solar energy forecasting model with a rule-based load scheduling algorithm to support forward-looking analysis, performance oversight, and optimisation of controllable electrical loads.
An interactive Streamlit dashboard provides transparent visualisation of forecasts, projected energy production, and optimised load schedules.

The project demonstrates how predictive analytics and supervisory-style dashboards can be applied to improve resource utilisation, enhance operational visibility, and support evidence-based management decisions within regulated infrastructure environments.

## Project Objectives

i.) To develop a next-day solar energy forecasting model using machine learning and meteorological data to support proactive monitoring.

ii.) To design an analytical load scheduling approach that aligns controllable demand with forecasted solar availability.

iii.) To improve solar PV self-consumption performance and reduce unplanned export to the grid.

iv.) To provide a transparent, user-friendly dashboard to support operational oversight and decision-making.

## System Architecture

The EMS is structured around four core analytical components:

### 1. Solar Forecasting Module

Utilises an XGBoost-based predictive model to estimate next-day solar PV energy production.

Trained on historical irradiance, weather variables, and PV output data.

Meteorological forecast inputs are obtained via the Open-Meteo API.

Produces 30-minute resolution forecasts over a 24-hour horizon to support short-term planning and monitoring.

### 2. Load Modelling

Campus electrical demand is categorised into:

Base load (non-deferrable): lighting, ICT infrastructure, and essential services.

Controllable loads (deferrable): including laundry machines, dryers, dishwashers, ovens, water heaters, and ventilation systems.

This classification supports structured analysis of demand flexibility and controllability.

### 3. Load Scheduling Algorithm

A heuristic, rule-based scheduling algorithm aligns controllable loads with periods of high forecasted solar generation.

The scheduler enforces realistic operational constraints, including:

i.) Daylight operating windows (06:00–18:00)
ii.) Contiguous runtime requirements
iii.) Device-specific scheduling rules (e.g., ovens completing operation before midday)

The algorithm is intentionally designed for transparency, interpretability, and low computational complexity, supporting auditability and practical deployment.

### 4. Streamlit Analytics Dashboard

The dashboard provides visualisation of:

Next-day irradiance forecasts

Predicted solar energy production

Optimised load schedules presented in a timeline (Gantt-style) format

The interface supports forward-looking oversight, enabling energy managers to anticipate system behaviour and assess operational decisions prior to execution.

## Technologies Used

Python

XGBoost for predictive modelling

Pandas & NumPy for data processing and validation

Streamlit for analytics dashboard development

Plotly for interactive visualisation

Open-Meteo API for meteorological forecast data

## Results
<img width="940" height="449" alt="image" src="https://github.com/user-attachments/assets/9ea18b97-76f6-4ddf-b524-bda6f6fa0eb8" />

## Key outcomes from the project include:

High-accuracy day-ahead solar energy forecasts (R² ≈ 0.92).

Effective alignment of controllable demand with forecasted solar generation peaks.

Reduced mismatch between PV production and campus electricity demand.

A complete, modular EMS pipeline suitable for extension and operational scaling.

## Example Load Scheduling Output
<img width="986" height="391" alt="image" src="https://github.com/user-attachments/assets/a55008f0-c33b-4036-9703-45eebe407d28" />
## Potential Extensions

Integration with smart meters, PLCs, or IoT-based switching systems for automated control

Transition to real-time or near-real-time monitoring with rolling forecast updates

Incorporation of economic and tariff-based optimisation criteria

Extension to multi-building or campus-scale microgrid supervision

## Repository Structure
├── app.py
├── models/
│   └── xgb_model.pkl
├── utils/
│   ├── fetch_openmeteo_forecast.py
│   ├── prediction_pipeline.py
│   └── scheduler.py
├── data/
│   └── load_data.csv
├── requirements.txt
└── README.md

## Author

Samuel Kiio Kyalo
Graduate Electrical & Electronics Engineer

## Post Script

A detailed technical and analytical project report is available upon request.
