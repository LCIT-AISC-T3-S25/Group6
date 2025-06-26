FROM python:3.9-slim
 
WORKDIR /app
 
COPY . /app
 
# Install system dependencies

RUN apt-get update && apt-get install -y build-essential \
&& rm -rf /var/lib/apt/lists/*
 
# Python dependencies

RUN pip install --upgrade pip

RUN pip install --no-cache-dir flask tensorflow numpy pillow scikit-learn
 
# Expose the Flask port (can be changed per model)

EXPOSE 5020
 
CMD ["python", "app.py"]
