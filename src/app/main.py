import os
import uuid
import sqlite3
import json
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import xgboost as xgb
import gradio as gr
import requests

# Create an absolute path so the DB never gets lost!
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "logs", "inference_logs.db")

# ---------------------------------------------------------
# 1. THE START-UP LIFESPAN
# ---------------------------------------------------------
ml_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        encoder_path = "artifacts/encoder.joblib"
        model_path = "artifacts/xgb_model.json"
        
        if os.path.exists(encoder_path):
            ml_artifacts["encoder"] = joblib.load(encoder_path)
        if os.path.exists(model_path):
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            ml_artifacts["model"] = model
            
        print("✅ Machine Learning artifacts loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
    yield 
    ml_artifacts.clear()

app = FastAPI(
    title="Smart Allocation & Demand Forecast API",
    description="Predicts demand and recommends optimal warehouse restock volumes.",
    version="1.0.0",
    lifespan=lifespan
)

# ---------------------------------------------------------
# NEW: DATABASE SETUP (ENTERPRISE LOGGING)
# ---------------------------------------------------------
def init_db():
    """Creates a local SQLite database inside the logs folder to track all predictions."""
    # 1. Safely create the logs directory if it doesn't exist
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True) # Update this line too!
    
    # 2. Connect to the database INSIDE the logs folder
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            trace_id TEXT PRIMARY KEY,
            timestamp TEXT,
            endpoint TEXT,
            input_data TEXT,
            prediction REAL
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Enterprise Logging Database Initialized.")

# Run setup immediately when the script loads
init_db()

# ---------------------------------------------------------
# 2. THE DATA CONTRACTS
# ---------------------------------------------------------
class DemandRequest(BaseModel):
    state: str = Field(..., example="SP")
    product_category_name: str = Field(..., example="health_beauty")
    month: int = Field(..., ge=1, le=12, example=11)
    week_of_year: int = Field(..., ge=1, le=53, example=45)
    demand_lag_1wk: float = Field(..., example=150.0)
    demand_lag_2wk: float = Field(..., example=145.0)
    demand_lag_3wk: float = Field(..., example=140.0)
    demand_lag_4wk: float = Field(..., example=135.0)
    rolling_mean_4wk: float = Field(..., example=142.5)
    rolling_std_4wk: float = Field(..., example=5.5)

class AllocationRequest(DemandRequest):
    current_stock_on_hand: int = Field(..., example=50, description="Current units in the state warehouse")
    lead_time_weeks: int = Field(default=1, example=1, description="Weeks it takes to ship new inventory")
    safety_stock_factor: float = Field(default=0.2, example=0.2, description="20% buffer for unexpected spikes")

class AllocationResponse(BaseModel):
    trace_id: str          # Added for Traceability
    timestamp: str         # Added for Traceability
    state: str
    category: str
    predicted_demand: float
    target_inventory: int
    recommended_allocation: int
    action: str

# ---------------------------------------------------------
# 3. HELPER FUNCTION: THE ML BRAIN
# ---------------------------------------------------------
def get_prediction(request: DemandRequest) -> float:
    """Helper function to run the XGBoost model."""
    input_data = pd.DataFrame([{
        "state": request.state,
        "product_category_name": request.product_category_name,
        "month": request.month,
        "week_of_year": request.week_of_year,
        "demand_lag_1wk": request.demand_lag_1wk,
        "demand_lag_2wk": request.demand_lag_2wk,
        "demand_lag_3wk": request.demand_lag_3wk,
        "demand_lag_4wk": request.demand_lag_4wk,
        "rolling_mean_4wk": request.rolling_mean_4wk,
        "rolling_std_4wk": request.rolling_std_4wk
    }])

    encoder = ml_artifacts.get("encoder")
    model = ml_artifacts.get("model")
    
    if encoder is None or model is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")
        
    categorical_cols = ["state", "product_category_name"]
    encoded_cats = encoder.transform(input_data[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))
    input_data = input_data.drop(columns=categorical_cols)
    
    final_features = pd.concat([input_data, encoded_df], axis=1)
    final_features = final_features[model.feature_names_in_]
    
    prediction = model.predict(final_features)
    return max(0.0, float(prediction[0]))

# ---------------------------------------------------------
# 4. THE ENDPOINTS (UPDATED WITH LOGGING)
# ---------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_demand(request: DemandRequest):
    """Returns the raw demand prediction and logs the transaction."""
    try:
        # 1. Generate Audit Trail Identifiers
        trace_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # 2. Get Prediction
        pred = get_prediction(request)
        
        # 3. Log to SQLite
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (trace_id, timestamp, endpoint, input_data, prediction) VALUES (?, ?, ?, ?, ?)",
            (trace_id, timestamp, "/predict", request.json() if hasattr(request, 'json') else request.model_dump_json(), float(pred))
        )
        conn.commit()
        conn.close()

        # 4. Return Data + Trace ID
        return {
            "trace_id": trace_id,
            "timestamp": timestamp,
            "predicted_demand": round(pred, 2), 
            "state": request.state, 
            "category": request.product_category_name
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/allocate", response_model=AllocationResponse)
async def allocate_inventory(request: AllocationRequest):
    """The Final Engine: Recommends exact shipping actions based on AI forecasts."""
    try:
        # 1. Generate Audit Trail Identifiers
        trace_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # 2. Ask the AI for the forecast
        predicted_demand = get_prediction(request)
        
        # 3. Supply Chain Logic
        safety_stock = predicted_demand * request.safety_stock_factor
        target_inventory = (predicted_demand * request.lead_time_weeks) + safety_stock
        allocation_needed = target_inventory - request.current_stock_on_hand
        final_allocation = max(0, int(round(allocation_needed)))
        action = "SHIP_INVENTORY" if final_allocation > 0 else "DO_NOTHING_SUFFICIENT_STOCK"

        # 4. Log to SQLite
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (trace_id, timestamp, endpoint, input_data, prediction) VALUES (?, ?, ?, ?, ?)",
            (trace_id, timestamp, "/allocate", request.json() if hasattr(request, 'json') else request.model_dump_json(), float(predicted_demand))
        )
        conn.commit()
        conn.close()

        # 5. Return Full Business Action + Trace ID
        return AllocationResponse(
            trace_id=trace_id,
            timestamp=timestamp,
            state=request.state,
            category=request.product_category_name,
            predicted_demand=round(predicted_demand, 2),
            target_inventory=int(round(target_inventory)),
            recommended_allocation=final_allocation,
            action=action
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
"""
logs
"""

import sqlite3
from fastapi import APIRouter

import json # (Just double-checking this is imported at the top of your file!)

@app.get("/logs")
def view_logs():
    """Fetches the logs and formats them cleanly for the user."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10")
        rows = cursor.fetchall()
        conn.close()
        
        formatted_logs = []
        
        for row in rows:
            trace_id, timestamp, endpoint, input_data_str, prediction = row
            
            # 1. Parse the stringified JSON back into a Python dictionary
            try:
                features = json.loads(input_data_str)
            except:
                features = input_data_str # Fallback if parsing fails
            
            # 2. Build the base tidy dictionary
            log_entry = {
                "logging-id": trace_id,
                "time created": timestamp,
                "operation": endpoint.replace("/", ""), # Removes the slash to just say 'predict' or 'allocate'
                "features": features
            }
            
            # 3. Add endpoint-specific formatting
            if endpoint == "/predict":
                log_entry["prediction"] = round(prediction, 2)
                
            elif endpoint == "/allocate" and isinstance(features, dict):
                # We dynamically recalculate the business logic for the logs!
                stock = features.get("current_stock_on_hand", 0)
                lead_time = features.get("lead_time_weeks", 1)
                safety_factor = features.get("safety_stock_factor", 0.2)
                
                safety_stock = prediction * safety_factor
                target_inventory = (prediction * lead_time) + safety_stock
                allocation_needed = target_inventory - stock
                final_allocation = max(0, int(round(allocation_needed)))
                action = "SHIP_INVENTORY" if final_allocation > 0 else "DO_NOTHING_SUFFICIENT_STOCK"
                
                log_entry["action"] = action
                log_entry["target inventory"] = int(round(target_inventory))
                log_entry["must ship"] = final_allocation
                
            formatted_logs.append(log_entry)
            
        return formatted_logs
        
    except Exception as e:
        return {"error": f"Could not read logs: {str(e)}"}

# ---------------------------------------------------------
# 5. GRADIO FRONTEND
# ---------------------------------------------------------
metadata_path = "artifacts/metadata.json"

if os.path.exists(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        states = metadata.get("states", ["SP", "RJ", "MG", "RS"])
        categories = metadata.get("categories", ["bed_bath_table", "health_beauty", "sports_leisure"])
    print("✅ Metadata loaded from artifacts/metadata.json")
else:
    states = ["SP", "RJ", "MG", "RS"] 
    categories = ["bed_bath_table", "health_beauty", "sports_leisure", "computers_accessories"]

def fetch_demand(state, category, month, week, lag1, lag2, lag3, lag4, roll_mean, roll_std):
    try:
        payload = {
            "state": state, "product_category_name": category, "month": int(month),
            "week_of_year": int(week), "demand_lag_1wk": float(lag1), "demand_lag_2wk": float(lag2),
            "demand_lag_3wk": float(lag3), "demand_lag_4wk": float(lag4),
            "rolling_mean_4wk": float(roll_mean), "rolling_std_4wk": float(roll_std)
        } 
        response = requests.post("http://127.0.0.1:7860/predict", json=payload) # Ensure port matches your uvicorn!
        if response.status_code != 200:
            return f"🚨 Backend Error: {response.text}"
        data = response.json()
        return f"🆔 Trace ID: {data.get('trace_id')}\n📊 Predicted Demand: {round(data.get('predicted_demand', 0), 2)} units"
    except Exception as e:
        return f"Error connecting to backend: {str(e)}"

def fetch_allocation(state, category, month, week, lag1, lag2, lag3, lag4, roll_mean, roll_std, stock, lead_time):
    try:
        payload = {
            "state": state, "product_category_name": category, "month": int(month),
            "week_of_year": int(week), "demand_lag_1wk": float(lag1), "demand_lag_2wk": float(lag2),
            "demand_lag_3wk": float(lag3), "demand_lag_4wk": float(lag4),
            "rolling_mean_4wk": float(roll_mean), "rolling_std_4wk": float(roll_std),
            "current_stock_on_hand": stock, "lead_time_weeks": lead_time,
            "safety_stock_factor": 0.2
        }
        response = requests.post("http://127.0.0.1:7860/allocate", json=payload) # Ensure port matches your uvicorn!
        if response.status_code != 200:
            return f"🚨 Backend Error: {response.text}"
        data = response.json()
        
        return f"🆔 Trace ID: {data.get('trace_id')}\n🚨 ACTION: {data.get('action')}\n🎯 Target Inventory: {data.get('target_inventory')} units\n📦 Must Ship: {data.get('recommended_allocation')} units"
    except Exception as e:
        return f"Error connecting to backend: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft()) as ui:
    gr.Markdown("# 📦 Geo-Category Demand & Allocation Engine (Auditable)")
    
    with gr.Tab("1. Demand Forecaster"):
        with gr.Row():
            state_in = gr.Dropdown(choices=states, label="State", value=states[0] if states else None)
            cat_in = gr.Dropdown(choices=categories, label="Product Category", value=categories[0] if categories else None)
        with gr.Accordion("Advanced Features", open=False):
            with gr.Row():
                month_in = gr.Slider(1, 12, value=11, step=1, label="Month")
                week_in = gr.Slider(1, 52, value=45, step=1, label="Week of Year")
            with gr.Row():
                lag1_in, lag2_in, lag3_in, lag4_in = gr.Number(value=150.0, label="Lag 1 Wk"), gr.Number(value=145.0, label="Lag 2 Wk"), gr.Number(value=140.0, label="Lag 3 Wk"), gr.Number(value=135.0, label="Lag 4 Wk")
            with gr.Row():
                rm_in, rs_in = gr.Number(value=142.5, label="Rolling Mean"), gr.Number(value=5.5, label="Rolling Std Dev")

        predict_btn = gr.Button("🔮 Run Forecast", variant="primary")
        forecast_out = gr.Textbox(label="AI Forecast Result", lines=3)
        predict_btn.click(fn=fetch_demand, inputs=[state_in, cat_in, month_in, week_in, lag1_in, lag2_in, lag3_in, lag4_in, rm_in, rs_in], outputs=forecast_out)

    with gr.Tab("2. Smart Allocation Engine"):
        with gr.Row():
            alloc_state_in = gr.Dropdown(choices=states, label="State", value=states[0] if states else None)
            alloc_cat_in = gr.Dropdown(choices=categories, label="Product Category", value=categories[0] if categories else None)
        with gr.Accordion("Advanced Features", open=False):
            with gr.Row():
                a_month_in = gr.Slider(1, 12, value=11, step=1, label="Month")
                a_week_in = gr.Slider(1, 52, value=45, step=1, label="Week of Year")
            with gr.Row():
                a_lag1_in, a_lag2_in, a_lag3_in, a_lag4_in = gr.Number(value=150.0, label="Lag 1 Wk"), gr.Number(value=145.0, label="Lag 2 Wk"), gr.Number(value=140.0, label="Lag 3 Wk"), gr.Number(value=135.0, label="Lag 4 Wk")
            with gr.Row():
                a_rm_in, a_rs_in = gr.Number(value=142.5, label="Rolling Mean"), gr.Number(value=5.5, label="Rolling Std Dev")

        with gr.Row():
            stock_in = gr.Slider(minimum=0, maximum=1000, value=50, step=1, label="Current Stock on Hand")
            lead_in = gr.Slider(minimum=1, maximum=14, value=2, step=1, label="Lead Time (Weeks)")
            
        allocate_btn = gr.Button("🚚 Run Allocation Logic", variant="primary")
        allocation_out = gr.Textbox(label="Supply Chain Recommendation", lines=5)
        allocate_btn.click(fn=fetch_allocation, inputs=[alloc_state_in, alloc_cat_in, a_month_in, a_week_in, a_lag1_in, a_lag2_in, a_lag3_in, a_lag4_in, a_rm_in, a_rs_in, stock_in, lead_in], outputs=allocation_out)

app = gr.mount_gradio_app(app, ui, path="/")