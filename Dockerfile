# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by Python packages
# e.g., RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
# Note: dev-requirements.txt is excluded as it contains development tools
# that are not needed in the production Docker image
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's source code from your host to your image filesystem.
COPY . /app/

# Set PYTHONPATH to include src directory for imports
# This allows imports like "from data import ..." to work when running from src/
ENV PYTHONPATH=/app/src

# Inform Docker that the container listens on the specified port at runtime.
# EXPOSE 8000

# Define the command to run your app
# This will be overridden by AML, but is good practice for local testing
# CMD ["python", "src/train.py"]


