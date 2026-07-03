# Use an official lightweight Python image
FROM python:3.12-slim

# Install system dependencies required for C++ compilation and CMake
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project into the container
COPY . /app

# Install the package. 
# Because we kept setup.py and pyproject.toml, this command will automatically
# trigger CMake to compile the C++ engine (mga_module) specifically for this Linux container,
# and then install the Python wrapper and FastAPI dependencies.
RUN pip install -v --no-cache-dir .

# Expose the port the FastAPI server will run on
EXPOSE 8000

# Set environment variables for production
ENV PYTHONUNBUFFERED=1

# Command to run the FastAPI web server
CMD ["uvicorn", "avarent.main:app", "--host", "0.0.0.0", "--port", "8000"]
