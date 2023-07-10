import sqlite3

# Connexion à la base de données (création si elle n'existe pas)
conn = sqlite3.connect('/home/hyont-nick/DATA_ANALYST/Soutenance/php/dbtest.db')

# Création d'un curseur pour exécuter les requêtes SQL
cursor = conn.cursor()

# Création de la table utilisateurs avec les champs spécifiés
cursor.execute('''
    CREATE TABLE IF NOT EXISTS utilisateurs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nom TEXT,
        email TEXT,
        motdepasse TEXT
    )
''')

# Fermeture de la connexion à la base de données
conn.close()
