# Use the official Python 3.8 slim image from the Docker Hub
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the requirements.txt file into the container at /usr/src/app
COPY requirements.txt .

# Install the dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Expose port 80 to allow external traffic to reach the container
EXPOSE 80

# Set the command to run the application
CMD ["python", "inference.py"]
