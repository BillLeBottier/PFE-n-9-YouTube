FROM python:3.9-slim

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .
COPY . .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Variable d'environnement pour le port
ENV PORT 8080

# Commande pour démarrer l'application
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app 