# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the crisisfacts folder (symlink will resolve to the actual folder)

# Install system dependencies (including Git)
# RUN apt-get update && apt-get install -y git && apt-get clean

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
    # && pip install --upgrade git+https://github.com/allenai/ir_datasets.git@crisisfacts

EXPOSE 5001
# Run database.py when the container launches to load the data
# CMD ["python", "upload.py"]
CMD ["bash", "-c", "python upload.py && python server.py"]