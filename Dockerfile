# Dockerfile.yaml

# Use the official Python image with version 3.10
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt .

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask API application code into the container
COPY app.py .

# Expose the internal container port (5000)
EXPOSE 5000

# Specify the command to run the application using Gunicorn
# Binding Gunicorn to 0.0.0.0:5000, which will map to host's 8000 as defined in docker run
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

# Note: Use --network bridge in docker run command to enable bridge communication if needed.
