# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install basic OS dependencies (if needed)
RUN apt-get update && apt-get install -y \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements/requirements_train.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements_train.txt
RUN pip install pre-commit==2.13.0


# Copy the entire repo into the container
COPY . .

# Default command (optional: change depending on what you want to do)
# For example, if you want to run a Streamlit app:
# CMD ["streamlit", "run", "your_app.py"]

EXPOSE 8501
EXPOSE 6006

# If you just want to spawn a bash shell for manual testing:
CMD ["/bin/bash"]
