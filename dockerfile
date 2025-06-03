FROM continuumio/miniconda3

WORKDIR /app

# Copier les fichiers nécessaires
COPY environment.yaml .

# Créer l’environnement Conda
RUN conda env create -f environment.yaml

# Activer l’environnement et installer le reste
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

COPY . .

# Expose le port utilisé par uvicorn
EXPOSE 8000

# Commande pour démarrer l'app
CMD ["conda", "run", "-n", "myenv", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
