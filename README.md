# Clothing Size Prediction API

This repository contains an API for predicting clothing sizes based on user input using machine learning. The API is built using FastAPI and includes endpoints for training the model, making predictions, and classifying text using HuggingFace models.

## Files in the Repository

- `api.py`: The FastAPI application file containing all the API endpoints.
- `function.py`: Contains the main functions for training the model and making predictions.
- `requirements.txt`: List of dependencies required for the project.
- `notebook.ipynb`: Jupyter notebook containing the code to train the model and interact with the API.
- `sizes.csv`: Dataset used for training the model.
- `app.py`: Streamlit application to interact with the API.
- `model.h5`: The trained model file (will be generated automatically).

## How to Use

1. **Clone the Repository**:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Import Files into Jupyter Notebook**:
    Open the `notebook.ipynb` file in Jupyter Notebook. Import all the files from the repository except for `model.h5` as it will be generated automatically.

3. **Run the Notebook**:
    Execute each cell in the notebook one by one. This will train the model using `sizes.csv` and generate `model.h5`.

4. **Run the FastAPI Server**:
    After running the notebook, you can start the FastAPI server.
    ```python
    import subprocess

    # Start the FastAPI server
    process = subprocess.Popen(["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
    print(f"FastAPI server started with PID: {process.pid}")
    ```

## Example Requests

### Predict Size

```python
import requests

# URL of the prediction endpoint
url = "http://localhost:8000/predict"

# Data for the prediction
data = {
    "weight": 70.0,
    "age": 34,
    "height": 180.0
}

# Send a POST request to predict clothing size
response = requests.post(url, json=data)
print(response.json())
```
