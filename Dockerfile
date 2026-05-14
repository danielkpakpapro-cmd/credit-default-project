FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Projet 5 - Défaut de Prêt"
LABEL version="1.0.0"

# Répertoire de travail
WORKDIR /app

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le projet
COPY . .

# Exposer le port
EXPOSE 8000

# Démarrer l'API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
