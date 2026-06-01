Test FastAPI
uvicorn src.app.main:app --reload

Test Hugging Face
huggingface-cli upload khairuliman7/demand-forecast-engine deploy_folder/ . --repo-type space
huggingface-cli upload khairuliman7/demand-forecast-engine deploy_folder/ / --repo-type space

This is the hugging face links
https://huggingface.co/spaces/khairuliman7/demand-forecast-engine
https://khairuliman7-demand-forecast-engine.hf.space

Logs link
https://khairuliman7-demand-forecast-engine.hf.space/logs

Retrain the training pipeline
python -m src.models.train_pipeline
mlflow ui --backend-store-uri sqlite:///mlruns.db