import os
import sqlite3
import pandas as pd

# Start looking from the folder where this script is saved
search_dir = os.path.dirname(os.path.abspath(__file__))
found_db_path = None

print("🕵️‍♂️ Scanning project folders for 'inference_logs.db'...")

# Walk through all folders and subfolders
for root, dirs, files in os.walk(search_dir):
    if "inference_logs.db" in files:
        found_db_path = os.path.join(root, "inference_logs.db")
        break

if found_db_path:
    print(f"\n🎉 FOUND IT! The database was hiding exactly here:\n👉  {found_db_path}\n")
    try:
        # Connect to wherever it was found
        conn = sqlite3.connect(found_db_path)
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
        conn.close()

        print(f"✅ Successfully read {len(df)} logged predictions!\n")
        print(df[['timestamp', 'endpoint', 'prediction']]) 
        
    except Exception as e:
        print(f"❌ Database found, but could not read it: {e}")
else:
    print("\n❌ CRITICAL: Could not find 'inference_logs.db' ANYWHERE in this project.")
    print("If you see this, your FastAPI server has not actually run the init_db() function yet.")