FROM continuumio/miniconda3

# Set working directory
WORKDIR /

# Copy environment and install
COPY environment.yaml .
RUN conda env create -f environment.yaml
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Copy source and install pip deps (if any)
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8000

# Command to run the app
CMD ["conda", "run", "-n", "myenv", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
