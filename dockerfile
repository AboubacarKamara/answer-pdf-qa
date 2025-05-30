# Utiliser une image Python officielle légère
FROM python:3.11-slim

# Définir le répertoire de travail dans le container
WORKDIR /app

# Copier les fichiers requirements.txt (ou poetry.lock / pyproject.toml) dans le container
COPY requirements.txt .

# Installer les dépendances (avec pip)
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code source dans le container
COPY . .

# Exposer le port sur lequel uvicorn va tourner
EXPOSE 8000

# Lancer l'application avec uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
