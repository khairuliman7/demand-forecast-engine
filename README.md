# Category Demand Forecaster and Allocation Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Forecasting-red.svg)
![Hugging Face](https://img.shields.io/badge/Deployed-HuggingFace-yellow.svg)

## 📌 Project Overview
An end-to-end Machine Learning microservice built to optimize supply chain logistics. Leveraging the Brazilian Olist e-commerce dataset, this system predicts future demand for specific product categories across different states and provides automated inventory allocation recommendations.

Designed as a production-grade microservice, this engine replaces static analysis with a containerized, asynchronous API, ensuring scalability and reproducibility.

## 💼 Business Value & Profit Drivers
This engine is engineered to solve core logistics challenges and deliver measurable ROI:
* **Smart Forward-Deployment:** Reduces freight costs by enabling bulk shipment to distribution centers based on regional demand forecasts.
* **Minimizing Stockouts:** Acts as an early warning system to trigger `SHIP_INVENTORY` actions before a warehouse runs dry.
* **Optimizing Storage Capacity:** By forecasting velocity per category, the system informs optimal warehouse slotting, ensuring high-demand goods are prioritized for easily accessible storage locations.
* **Capital Efficiency:** Identifies low-velocity trends to prevent overstocking and capital tied up in dead stock.

## 🏗️ Architecture & Stack
* **Modeling:** XGBoost, Scikit-Learn (Chronological Train-Test Split)
* **Data Engineering:** Pandas, Numpy (Weekly Time-Series Aggregation & Feature Engineering)
* **Serving Layer:** FastAPI (Asynchronous endpoints, Pydantic validation)
* **Deployment:** Docker (Containerized Microservice) hosted on Hugging Face Spaces
* **Environment & MLOps:** `uv` (Package Management), MLflow (Experiment Tracking & Model Governance)
* **Monitoring/Logging:** SQLite-based transaction logging with UUID traceability

## 🚀 API Endpoints
The backend engine exposes three primary routes to handle prediction, allocation, and audit trails:
* `POST /predict`: Accepts state/category inputs and returns projected demand volume.
* `POST /allocate`: Accepts demand data, current stock, and lead times to return an actionable inventory replenishment plan.
* `GET /logs`: Provides a fully auditable history of all inferences, including UUIDs, timestamps, input parameters, and generated allocation actions.

## 🛡️ Model Governance & Evaluation
To ensure production readiness, the model includes:
* **Traceability:** Every prediction is logged with a unique UUID, allowing for a full audit trail of model performance and decision history.
* **Automated Pipeline:** The training pipeline handles feature engineering and hyperparameter tuning, with MLflow acting as the immutable logbook to track experiment metrics (RMSE/MAE) and versioning.
* **Validation:** Chronological split testing ensures the model is evaluated on unseen future data, preventing data leakage.

## 💻 Local Setup & Execution

**1. Clone the repository:**
```bash
git clone [https://github.com/YourUsername/demand-allocation-engine.git](https://github.com/YourUsername/demand-allocation-engine.git)
cd demand-allocation-engine