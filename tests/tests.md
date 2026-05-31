Test FastAPI
uvicorn src.app.main:app --reload

Test Hugging Face
huggingface-cli upload khairuliman7/demand-forecast-engine deploy_folder/ . --repo-type space
huggingface-cli upload khairuliman7/demand-forecast-engine deploy_folder/ / --repo-type space