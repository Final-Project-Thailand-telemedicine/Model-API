# Use the official Python image as the base image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the application files to the container
COPY . .

# Expose FastAPI port
EXPOSE 8080

# Use uvicorn to run FastAPI in Docker
CMD ["uvicorn", "model:app", "--host", "0.0.0.0", "--port", "8080"]
