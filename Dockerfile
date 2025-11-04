# Use a lightweight Python base image
FROM python:3.12-slim

# Set working directory in container
WORKDIR /app

# Copy your project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create logs directory
RUN mkdir -p logs

# Command to run your main script
CMD ["python", "dags/delay_pipeline_dag.py"]