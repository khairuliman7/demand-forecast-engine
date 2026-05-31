# Geo-Category Demand Forecaster & Smart Allocation Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Forecasting-red.svg)

## 📌 Project Overview
An end-to-end Machine Learning microservice built to optimize supply chain logistics. This system leverages the Brazilian Olist e-commerce dataset to predict future demand for specific product categories across different states, and automatically recommends optimal inventory allocations to minimize delivery times and avoid stockouts.

Unlike traditional static ML notebooks, this project is designed as a production-grade, containerized AI microservice using **FastAPI, Docker, and XGBoost**.

## 💼 Business Value & Profit Drivers
This engine is designed to generate measurable ROI for warehouse and logistics teams:
1. **Smart Forward-Deployment (Cheaper Freight):** By accurately forecasting regional demand (e.g., 500 units of "Health & Beauty" in Minas Gerais), the business can ship bulk pallets to local distribution centers ahead of time, heavily reducing individual parcel freight costs.
2. **Minimizing Stockouts (Capturing Lost Revenue):** The Smart Allocation endpoint acts as an early warning system, analyzing current stock levels and lead times to generate explicit `SHIP_INVENTORY` actions before a warehouse runs dry.
3. **Reducing Dead Stock (Capital Efficiency):** The time-series engine detects dying trends, preventing the business from blindly restocking low-velocity items and freeing up capital.

## 🏗️ Architecture & Stack
* **Modeling:** XGBoost, Scikit-Learn (Chronological Train-Test Split)
* **Data Engineering:** Pandas, Numpy (Weekly Time-Series Aggregation)
* **Serving Layer:** FastAPI (Asynchronous API endpoints with Pydantic Data Contracts)
* **UI/Frontend:** Gradio (For Stakeholder Demonstration)
* **Environment & MLOps:** `uv` (Package Management), MLflow (Experiment Tracking), Docker (Containerization)

## 🚀 API Endpoints
The backend engine provides two distinct decoupled endpoints:

* `POST /predict` - **Demand Forecaster**: Accepts state and category data, returning the raw projected volume for the upcoming week.
* `POST /allocate` - **Smart Allocation Engine**: Accepts demand requests alongside `current_stock_on_hand` and `lead_time`, returning a concrete business action (e.g., "Target Inventory is 100, you have 50 in stock, so SHIP_INVENTORY of 50 units").

## 💻 Local Setup & Execution

**1. Clone the repository:**
```bash
git clone [https://github.com/YourUsername/demand-allocation-engine.git](https://github.com/YourUsername/demand-allocation-engine.git)
cd demand-allocation-engine