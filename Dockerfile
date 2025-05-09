# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install basic OS dependencies (if needed)
RUN apt-get update && apt-get install -y \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/requirements_train.txt .

RUN pip install --upgrade pip
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements_train.txt
RUN pip install pre-commit==2.13.0


COPY . .

EXPOSE 8501
EXPOSE 6006

# If you just want to spawn a bash shell for manual testing:
CMD ["/bin/bash"]
