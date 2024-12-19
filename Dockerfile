FROM apache/spark:3.3.1

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the application script
COPY ~/spark_ds_project/app /app/spark_ds_app.py

# Expose Spark UI port
EXPOSE 4040

# Command to run the application
CMD ["spark-submit", "/app/spark_ds_app.py"]
