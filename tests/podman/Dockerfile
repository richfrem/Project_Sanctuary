FROM python:3.11-slim

WORKDIR /app

# Install Flask
RUN pip install --no-cache-dir flask

# Copy the app
COPY app.py .

# Expose port
EXPOSE 5001

# Run the app
CMD ["python", "app.py"]
