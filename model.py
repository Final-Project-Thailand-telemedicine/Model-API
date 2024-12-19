from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import random

app = FastAPI()

#test Fast Api

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    # Read the file contents
    file_content = await file.read()
    
    # Process the file content here (e.g., run a prediction model)
    # For now, we'll just return a message with the filename and file size+
    file_size = len(file_content)
    print("file size :",file_size)
    mockwound_state = random.randint(1, 4)

    # Return a response
    return JSONResponse(content={
        "wound_state": mockwound_state,
    })

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
