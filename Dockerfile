# 1. Utiliser une image Python officielle légère (adapte la version si besoin)
FROM python:3.9-slim

# 2. Définir le dossier de travail dans le conteneur
WORKDIR /app

# 3. Copier uniquement le fichier des dépendances en premier (bonne pratique Docker)
COPY requirements.txt .

# 4. Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copier tout le reste du code (dont app.py)
COPY . .

# 6. Exposer le port que Flask utilise (par défaut 5000)
EXPOSE 5000

# 7. La commande pour lancer l'application en production avec Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]