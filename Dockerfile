FROM apache/spark:3.3.1

# Install pip
USER root
RUN apt-get update && apt-get install -y python3-pip

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy the application script
COPY app /app

# Expose Spark UI port
EXPOSE 4040

# Command to run the application
CMD ["spark-submit", "/app/spark_ds_app.py"]
