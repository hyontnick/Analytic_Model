import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import shutil
import os
from tkinter import ttk
import webbrowser
import subprocess
import logging


def import_file():
    # Ouvrir la boîte de dialogue pour sélectionner le fichier à importer
    file_path = filedialog.askopenfilename()

    # Vérifier si le fichier est au format CSV
    if file_path.endswith('.csv'):
        # Définir le répertoire de destination
        destination_path = '/home/hyont-nick/DATA_ANALYST/Soutenance/data/'

        # Copier le fichier vers le répertoire de destination
        shutil.copy(file_path, destination_path)
        show_status_message("Fichier importé avec succès !", "green", "info")
    else:
        show_status_message("Veuillez sélectionner un fichier CSV.", "red", "warning")

logging.basicConfig(filename='output.log', level=logging.INFO)
def execute_code():
    # Supprimer le contenu du fichier de log
    with open('output.log', 'w'):
        pass
    # Exécuter le fichier python
    try:
        process = subprocess.Popen(['python3', '/home/hyont-nick/DATA_ANALYST/Soutenance/app/model.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        logging.info(output.decode('utf-8'))

        # subprocess.run(['python3', '/home/hyont-nick/DATA_ANALYST/Soutenance/app/model.py']) #, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        show_status_message("chargement des données terminer.", "blue", "info")
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur s'est produite lors de l'exécution du code :\n{str(e)}")

def execute_code_assistance():
    # Supprimer le contenu du fichier de log
    #with open('output.log', 'w'):
        #pass
    # Exécuter le fichier python
    try:
        process = subprocess.Popen(['python3', '/home/hyont-nick/DATA_ANALYST/Soutenance/app/assistance.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #output, error = process.communicate()
        #logging.info(output.decode('utf-8'))

        # subprocess.run(['python3', '/home/hyont-nick/DATA_ANALYST/Soutenance/app/model.py']) #, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        show_status_message("L'assistant est pret.", "blue", "info")
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur s'est produite lors de l'exécution du code :\n{str(e)}")

def show_status_message(message, color, icon):
    status_label.config(text=message, fg=color)
    status_label.after(500, toggle_status_message)

def toggle_status_message():
    current_state = status_label.cget("state")
    if current_state == tk.NORMAL:
        status_label.config(state=tk.DISABLED)
    else:
        status_label.config(state=tk.NORMAL)
    status_label.after(500, toggle_status_message)

def launch_webpage(url):
    webbrowser.open(url)

def open_webpage():
    process = subprocess.Popen(['python3', '/home/hyont-nick/DATA_ANALYST/Soutenance/app/server.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    webpage_url = 'http://localhost:8000'  # Remplacez par l'URL de la page que vous souhaitez ouvrir
    launch_webpage(webpage_url)

# Créer la fenêtre principale de l'interface graphique
root = tk.Tk()
root.title("Analytic Model")
root.resizable(False, False)  # Empêcher le redimensionnement de la fenêtre

# Charger l'image pour l'arrière-plan
background_image = tk.PhotoImage(file="/home/hyont-nick/DATA_ANALYST/Soutenance/app/img/fond.png")  # Remplacez 'chemin/vers/votre/image.png' par le chemin de votre image

# Créer un canvas pour afficher l'image en arrière-plan
canvas = tk.Canvas(root, width=980, height=400)
canvas.pack(fill=tk.BOTH, expand=True)

# Afficher l'image en arrière-plan
canvas.create_image(0, 0, anchor=tk.NW, image=background_image)

# Créer un cadre pour les éléments de l'interface
frame = tk.Frame(root, bg='white', padx=20, pady=20)
frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Créer une étiquette pour afficher le statut
status_label = tk.Label(frame, text="", font=("Arial", 14, "bold"), fg="red")
status_label.pack()

# Création du nouveau conteneur pour les boutons
button_container = tk.Frame(frame)
button_container.pack(pady=10)

# Charger les images pour les icônes
import_icon = tk.PhotoImage(file="/home/hyont-nick/DATA_ANALYST/Soutenance/app/img/import_csv.png")
execute_icon = tk.PhotoImage(file="/home/hyont-nick/DATA_ANALYST/Soutenance/app/img/load_server.png")
web_icon = tk.PhotoImage(file="/home/hyont-nick/DATA_ANALYST/Soutenance/app/img/view.png")
chatbot_icon = tk.PhotoImage(file="/home/hyont-nick/DATA_ANALYST/Soutenance/app/img/chatbot_icone.png")

# Bouton pour importer un fichier
import_button = ttk.Button(button_container, text="importer les données", command=import_file, image=import_icon, compound=tk.LEFT)
import_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

# Bouton pour exécuter le code
execute_button = ttk.Button(button_container, text="charger les données", command=execute_code, image=execute_icon, compound=tk.LEFT)
execute_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

# Bouton pour ouvrir la page web
open_button = ttk.Button(button_container, text="vusualisation", command=open_webpage, image=web_icon, compound=tk.LEFT)
open_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

# Bouton personnalisé
chatbot_button = ttk.Button(button_container, text="AM assistant virtuel", command=execute_code_assistance, image=chatbot_icon, compound=tk.LEFT)
chatbot_button.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

# Charge l'icône au format .gif ou .png
icon_image = tk.PhotoImage(file="/home/hyont-nick/DATA_ANALYST/Soutenance/app/img/icone.png")
# Configure l'icône de la fenêtre
root.iconphoto(True, icon_image)

# Lancer la boucle principale de l'interface graphique
root.mainloop()
