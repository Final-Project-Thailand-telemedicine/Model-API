from fastapi import FastAPI
import torch

app = FastAPI()

#test Fast Api

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Load your trained model
# model = torch.load("model.pth")
# model.eval()  # Set model to evaluation mode

# @app.post("/predict/")
# async def predict(input_data: dict):
#     # Convert input_data to tensor (example assumes input_data is a dict with 'data' key)
#     input_tensor = torch.tensor(input_data['data'])
    
#     # Perform model prediction
#     with torch.no_grad():
#         prediction = model(input_tensor)
    
#     # Convert prediction to a Python list (to be JSON serializable)
#     return {"prediction": prediction.tolist()}

# To run the app: `uvicorn filename:app --reload`
