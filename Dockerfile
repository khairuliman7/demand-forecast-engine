# 1. Use an official, lightweight Python base image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Explicitly tell Linux that /app is the root for Python imports
ENV PYTHONPATH=/app

# 4. Copy the requirements file first (leverages caching)
COPY requirements.txt .

# 5. Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy all your code and artifacts into the container
COPY . .

# 7. Expose the specific port Hugging Face requires
EXPOSE 7860

# 8. Run the FastAPI server on port 7860
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "7860"]