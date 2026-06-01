import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import mlflow
import mlflow.xgboost
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from itertools import product
import warnings
warnings.filterwarnings('ignore')

def engineer_features(df):
    """Handles Day 2 and Day 3 logic: Aggregation, Sparsity, and Lag Features"""
    print("⚙️ Running Feature Engineering Pipeline...")
    
    # 1. Time-Series Aggregation
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['week'] = df['order_purchase_timestamp'].dt.to_period('W-MON').dt.start_time
    
    # Assuming each row in df_master is an item sold
    actual_demand = df.groupby(['week', 'customer_state', 'product_category_name']).size().reset_index(name='demand_volume')

    # 2. Handling Sparsity (The Zero-Demand Problem)
    weeks = actual_demand['week'].unique()
    states = actual_demand['customer_state'].unique()
    categories = actual_demand['product_category_name'].unique()
    
    grid = list(product(weeks, states, categories))
    df_grid = pd.DataFrame(grid, columns=['week', 'customer_state', 'product_category_name'])
    
    df_timeseries = pd.merge(df_grid, actual_demand, 
                         on=['week', 'customer_state', 'product_category_name'], 
                         how='left')
    df_timeseries['demand_volume'] = df_timeseries['demand_volume'].fillna(0).astype(int)
    df_timeseries = df_timeseries.sort_values(by=['customer_state', 'product_category_name', 'week']).reset_index(drop=True)
    df_timeseries = pd.DataFrame(df_timeseries)

    print(f"Time-series shape: {df_timeseries.shape}")

    # Sort chronologically for lag features
    df_timeseries = df_timeseries.sort_values(by=['customer_state', 'product_category_name', 'week']).reset_index(drop=True)

    # 3. Generating Lag and Rolling Features
    print("🧠 Building Lag and Rolling Memory Features...")
    df_timeseries['month'] = df_timeseries['week'].dt.month
    df_timeseries['week_of_year'] = df_timeseries['week'].dt.isocalendar().week.astype(int)
    
    group_keys = ['customer_state', 'product_category_name']

    for i in [1, 2, 3, 4]:
        df_timeseries[f'demand_lag_{i}wk'] = df_timeseries.groupby(group_keys)['demand_volume'].shift(i)
    
    df_timeseries['rolling_mean_4wk'] = df_timeseries.groupby(group_keys)['demand_volume'].transform(
        lambda x: x.shift(1).rolling(window=4).mean()
    )

    df_timeseries['rolling_std_4wk'] = df_timeseries.groupby(group_keys)['demand_volume'].transform(
        lambda x: x.shift(1).rolling(window=4).std()
    )
    
    # Rename state column to match our API contract
    df_final = df_timeseries.rename(columns={'customer_state': 'state'})
    
    return df_final


def run_training_pipeline():
    print("🚀 Starting Demand Forecast Training Pipeline with MLflow Governance...")

    # Set up MLflow using your SQLite database!
    mlflow.set_tracking_uri("sqlite:///mlruns.db") 
    mlflow.set_experiment("Demand_Forecaster_XGBoost")

    # ---------------------------------------------------------
    # 1. LOAD MASTER DATA & ENGINEER FEATURES
    # ---------------------------------------------------------
    data_path = "data/processed/Masterfile.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Could not find raw master data at {data_path}")
        return

    print("📊 Loading raw master data...")
    df_raw = pd.read_csv(data_path)
    
    # Transform raw data into the predictive matrix
    df = engineer_features(df_raw)

    target_col = 'demand_volume'
    cols_to_drop = [target_col, 'week'] 
    df = df.sort_values(by=['week'])
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    y = df[target_col]

    # ---------------------------------------------------------
    # 2. ENCODE CATEGORICAL VARIABLES
    # ---------------------------------------------------------
    print("🔠 Encoding categorical features...")
    categorical_cols = ['state', 'product_category_name']
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(X[categorical_cols])
    
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))
    X = X.drop(columns=categorical_cols)
    X = pd.concat([X, encoded_df], axis=1)

    # ---------------------------------------------------------
    # 3. CHRONOLOGICAL TRAIN-TEST SPLIT
    # ---------------------------------------------------------
    print("✂️ Splitting data chronologically...")
    split_idx = int(len(df) * 0.6)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # ---------------------------------------------------------
    # 4. START MLFLOW RUN & TRAIN MODEL (PHASE 3 GOVERNANCE)
    # ---------------------------------------------------------
    with mlflow.start_run(run_name="Automated_Pipeline_Run"):
        
        # Define hyperparams explicitly
        params = {
            "n_estimators": 140,
            "learning_rate": 0.054722,
            "max_depth": 3,
            "subsample": 0.673707,
            "colsample_bytree": 0.790652,
            "min_child_weight": 8,
            "random_state": 42,
            "objective": 'reg:squarederror'
        }
        
        # 1. Log ALL parameters to MLflow
        mlflow.log_params(params)
        
        print("🤖 Training XGBoost Model...")
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        # ---------------------------------------------------------
        # 5. EVALUATE & LOG METRICS
        # ---------------------------------------------------------
        print("📈 Evaluating model...")
        predictions = model.predict(X_test)
        predictions = np.clip(predictions, a_min=0, a_max=None)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        
        print(f"✅ Test RMSE: {rmse:.2f}")
        print(f"✅ Test MAE:  {mae:.2f}")
        
        # 2. Log ALL metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        # 3. Log the model artifact to MLflow
        mlflow.xgboost.log_model(model, artifact_path="xgboost-model")

        # ---------------------------------------------------------
        # 6. SAVE ARTIFACTS FOR FASTAPI OVERRIDE
        # ---------------------------------------------------------
        print("💾 Saving artifacts to artifacts/ folder for FastAPI...")
        os.makedirs("artifacts", exist_ok=True)
        
        joblib.dump(encoder, "artifacts/encoder.joblib")
        model.save_model("artifacts/xgb_model.json")
        
        print(f"🎉 Pipeline Complete! Every parameter, metric, and model is now permanently logged in MLflow!")

if __name__ == "__main__":
    run_training_pipeline()