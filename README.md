# Energy Management System for Solar - Optimized Load Scheduling

<p align="center">
  <img src="https://github.com/user-attachments/assets/f8e3a1ff-d47b-40b0-8473-10a486394058" alt="EMS" width="400">
</p>

## Overview

This repository hosts a **smart Energy Management System (EMS)** developed for a **university campus in Kenya** to improve on-site solar energy utilization through **day-ahead solar forecasting and intelligent load scheduling**.

The system integrates a **machine learning–based solar energy prediction model** with a **heuristic load scheduling algorithm** to align controllable electrical loads with periods of high solar photovoltaic (PV) generation. 
An interactive **Streamlit web dashboard** is used to visualize forecasts, energy production, and optimized load schedules.

The project demonstrates how data-driven energy management can increase **solar self-consumption**, reduce dependency on grid electricity, and support institutional sustainability goals.


## Project Objectives

i.) To Forecast next-day solar energy production using **machine learning (XGBoost)** and meteorological data.

ii.) To Optimize scheduling of **controllable campus loads** to coincide with peak solar availability.

iii.) To improve on-site **solar PV self-consumption** and reduce energy export to the grid.

iv.) To provide a **user-friendly dashboard** for operational decision-making.


## System Architecture

The EMS consists of four main components:

### 1. A Solar Forecasting Module

* This module uses **XGBoost model** to predict next-day PV energy production.
* It is trained on historical irradiance, weather, and PV output data.
* The meteorological inputs are sourced via the **Open-Meteo API**.
* The module outputs **30-minute resolution forecasts** for 24 hours ahead.

### 2. Load Modelling

* Campus electrical demand were split into:

  1.  **Base load** (non-deferrable: lighting, ICT, essential services)
  2.  **Controllable loads** These include deferrable appliances such as laundry machines, dryers, dishwashers, ovens, water heaters, and ventilation.

### 3. Load Scheduling Algorithm

* A **heuristic-based scheduler** prioritizes running controllable loads during periods of high forecasted solar production.
* It Enforces realistic operational constraints, including:

  i.) Daylight-only operation (06:00–18:00)
  
  ii.) Contiguous runtime for appliances

  iii.) Device-specific time windows for example ovens finishing before midday.
* The algorithim is designed for **transparency, low computational cost, and practical deployment**

### 4. A Simple Streamlit Dashboard

* This dashboard Visualizes:

  1.  Next-day irradiance forecast.
  2.  Predicted solar energy production.
  3.  Optimized load schedule in a Gantt-style timeline.
* This dashboard could enable energy managers to understand and anticipate energy flows before the operating day


##  Technologies Used

1.  **Python**
2.  **XGBoost** machine learning model.
3.  **Pandas & NumPy** for data processing.
4.  **Streamlit** for a simple web dashboard.
5.  **Plotly** for interactive visualizations.
6.  **Open-Meteo API** to access weather & irradiance forecasts.


##  Results from the Project

<img width="940" height="449" alt="image" src="https://github.com/user-attachments/assets/9ea18b97-76f6-4ddf-b524-bda6f6fa0eb8" />


* Accurate day-ahead solar energy forecasts (R² ≈ 0.92)
* Successful alignment of controllable loads with solar generation peaks
* Reduced mismatch between PV production and campus demand
* A fully functional, code-complete EMS pipeline ready for real-world extension


##  Potential Extensions

* Integration with **smart meters, PLCs, or IoT switches** for automatic load control
* Real-time EMS operation with continuous forecast updates
* Economic optimization incorporating energy tariffs and demand charges
* Scaling to multi-building or multi-campus microgrids

---

## The Repository Structure 

```bash
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
```

---

##  Academic Context

This project was developed as a **Master’s capstone project** in the field of **Sustainable Energy Transitions**, focusing on practical energy management solutions for institutional solar PV systems in Kenya.

---

##  Author

**Samuel Kiio Kyalo**
Electrical & Electronics Engineer
