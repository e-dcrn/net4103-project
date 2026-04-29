#!/bin/bash
echo "Création de l'environnement virtuel..."
python3 -m venv env

echo "Activation de l'environnement..."
source env/bin/activate

echo "Mise à jour de pip..."
pip install --upgrade pip

echo "Installation des dépendances exactes..."
pip install -r requirements.txt

echo "Installation terminée ! Tapez 'source env/bin/activate' pour l'utiliser."