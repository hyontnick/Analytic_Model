import tkinter as tk
from tkinter import ttk
import sqlite3
import re
import subprocess

def register():
    # Fonction pour enregistrer les informations de l'utilisateur
    username = username_entry.get()
    password = password_entry.get()
    confirm_password = confirm_password_entry.get()
    access_code = access_code_entry.get()

    # Vérification des erreurs spécifiques à chaque champ
    username_error = ""
    password_error = ""
    confirm_password_error = ""
    access_code_error = ""

    if not username:
        username_error = "Veuillez saisir un nom d'utilisateur."
    if not password:
        password_error = "Veuillez saisir un mot de passe."
    elif len(password) < 8:
        password_error = "Le mot de passe doit contenir au moins 8 caractères."
    elif not re.search("[a-z]", password):
        password_error = "Le mot de passe doit contenir au moins une lettre minuscule."
    elif not re.search("[A-Z]", password):
        password_error = "Le mot de passe doit contenir au moins une lettre majuscule."
    elif not re.search("[0-9]", password):
        password_error = "Le mot de passe doit contenir au moins un chiffre."
    elif not re.search("[^a-zA-Z0-9]", password):
        password_error = "Le mot de passe doit contenir au moins un caractère spécial."

    if password != confirm_password:
        confirm_password_error = "Les mots de passe ne correspondent pas."

    if len(access_code) != 12:
        access_code_error = "Le code d'accès doit contenir 12 chiffres."

    if username_error or password_error or confirm_password_error or access_code_error:
        register_status_label.config(text="")
        username_error_label.config(text=username_error)
        password_error_label.config(text=password_error)
        confirm_password_error_label.config(text=confirm_password_error)
        access_code_error_label.config(text=access_code_error)
    else:
        conn = sqlite3.connect("login_data.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users VALUES (?, ?, ?)", (username, password, access_code))
        conn.commit()
        conn.close()

        register_status_label.config(text="Enregistrement réussi.",foreground="blue")
        register_status_label.place(relx=0.5, rely=0.80, anchor=tk.CENTER)
        username_error_label.config(text="")
        password_error_label.config(text="")
        confirm_password_error_label.config(text="")
        access_code_error_label.config(text="")

root = tk.Tk()
root.title("Analytic Model")
root.geometry("718x404")
root.resizable(False, False)

def open_login_window():
    # Fonction pour ouvrir une nouvelle fenêtre de connexion
    login_window = tk.Toplevel(root)
    login_window.title("Connexion")
    login_window.geometry("300x250")
    login_window.resizable(False, False)

    login_frame = ttk.Frame(login_window)
    login_frame.pack(pady=20)
    #login_window.configure(bg="lightblue")
    
    # background_image = tk.PhotoImage(file="/home/hyont-nick/DATA_ANALYST/Soutenance/app/img/fond.png")
    # background_label = ttk.Label(login_frame, image=background_image)
    # background_label.place(x=0, y=0, relwidth=1, relheight=1)

    login_username_label = ttk.Label(login_frame, text="Nom d'utilisateur:", image=images[0], compound=tk.LEFT)
    login_username_label.grid(row=0, column=0, sticky=tk.W)
    login_username_entry = ttk.Entry(login_frame)
    login_username_entry.grid(row=0, column=1)

    login_password_label = ttk.Label(login_frame, text="Mot de passe:", image=images[1], compound=tk.LEFT)
    login_password_label.grid(row=1, column=0, sticky=tk.W)
    login_password_entry = ttk.Entry(login_frame, show="*")
    login_password_entry.grid(row=1, column=1)

    login_status_label = ttk.Label(login_frame, text="")
    login_status_label.grid(row=2, columnspan=2, pady=10)

    def login():
        # Fonction pour vérifier les informations de connexion
        username = login_username_entry.get()
        password = login_password_entry.get()

        conn = sqlite3.connect("login_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user is None:
            login_status_label.config(text="Nom d'utilisateur incorrect.", foreground="red")
        elif password != user[1]:
            login_status_label.config(text="Mot de passe incorrect.", foreground="red")
        else:
            login_status_label.config(text="Connexion réussie.", foreground="green")
            root.destroy()  # Ferme la fenêtre "Formulaire d'inscription"
            subprocess.run(["python3", "/home/hyont-nick/DATA_ANALYST/Soutenance/app/analyse.py"])  # Exécute le code Python depuis le fichier "autre_script.py"

    login_button = ttk.Button(login_frame, text="Connexion", image=images[5], compound=tk.LEFT, command=login)
    login_button.grid(row=3, columnspan=2, pady=10)


# # Créer l'étiquette avec l'image
# username_label = ttk.Label(register_frame, text="Nom d'utilisateur:", image=username_image, compound=tk.LEFT)
# username_label.grid(row=0, column=0, sticky=tk.W)


# root = tk.Tk()
# root.title("Formulaire d'inscription bomm")
# root.geometry("718x404")
# root.resizable(False, False)

def load_images(image_paths):
    images = []
    for image_path in image_paths:
        image = tk.PhotoImage(file=image_path)
        images.append(image)
    return images

# Liste des chemins d'accès des images PNG pour chaque étiquette
image_paths = [
    "/home/hyont-nick/DATA_ANALYST/Soutenance/app/img/user.png",
    "/home/hyont-nick/DATA_ANALYST/Soutenance/app/img/password.png",
    "/home/hyont-nick/DATA_ANALYST/Soutenance/app/img/acces_code.png",
    "/home/hyont-nick/DATA_ANALYST/Soutenance/app/img/code_acces.png",
    "/home/hyont-nick/DATA_ANALYST/Soutenance/app/img/save.png",
    "/home/hyont-nick/DATA_ANALYST/Soutenance/app/img/connect.png"
]

# Charger les images
images = load_images(image_paths)


register_canvas = tk.Canvas(root, width=718, height=404)
register_canvas.pack()

background_image = tk.PhotoImage(file="/home/hyont-nick/DATA_ANALYST/Soutenance/app/img/inscri.png")
register_canvas.create_image(0, 0, anchor=tk.NW, image=background_image)

register_frame = ttk.Frame(register_canvas)
register_frame.place(relx=0.5, rely=0.3, anchor=tk.CENTER)


#username_label = ttk.Label(register_frame, text="Nom d'utilisateur:")
username_label = ttk.Label(register_frame, text="Nom d'utilisateur:", image=images[0], compound=tk.LEFT)
username_label.grid(row=0, column=0, sticky=tk.W)
username_entry = ttk.Entry(register_frame)
username_entry.grid(row=0, column=1)

password_label = ttk.Label(register_frame, text="Mot de passe:", image=images[1], compound=tk.LEFT)
password_label.grid(row=1, column=0, sticky=tk.W)
password_entry = ttk.Entry(register_frame, show="*")
password_entry.grid(row=1, column=1)

confirm_password_label = ttk.Label(register_frame, text="Confirmer le mot de passe:", image=images[2], compound=tk.LEFT)
confirm_password_label.grid(row=2, column=0, sticky=tk.W)
confirm_password_entry = ttk.Entry(register_frame, show="*")
confirm_password_entry.grid(row=2, column=1)

access_code_label = ttk.Label(register_frame, text="Code d'accès:", image=images[3], compound=tk.LEFT)
access_code_label.grid(row=3, column=0, sticky=tk.W)
access_code_entry = ttk.Entry(register_frame, show="*")
access_code_entry.grid(row=3, column=1)

register_button = ttk.Button(register_canvas, text="S'inscrire", command=register, image=images[4], compound=tk.LEFT)
register_button.place(relx=0.5, rely=0.65, anchor=tk.CENTER)

register_status_label = ttk.Label(register_canvas, text="")
register_status_label.place(relx=0.5, rely=0.85, anchor=tk.CENTER)

username_error_label = ttk.Label(register_frame, text="", foreground="red")
username_error_label.grid(row=0, column=2, padx=10)
password_error_label = ttk.Label(register_frame, text="", foreground="red")
password_error_label.grid(row=1, column=2, padx=10)
confirm_password_error_label = ttk.Label(register_frame, text="", foreground="red")
confirm_password_error_label.grid(row=2, column=2, padx=10)
access_code_error_label = ttk.Label(register_frame, text="", foreground="red")
access_code_error_label.grid(row=3, column=2, padx=10)

# Charger l'image pour le lien de connexion
link_image = tk.PhotoImage(file="/home/hyont-nick/DATA_ANALYST/Soutenance/app/img/connect.png")

# Étiquette pour le lien de connexion
login_link_label = ttk.Label(register_canvas, text="Déjà inscrit ? Connectez-vous", image=link_image, compound=tk.LEFT, foreground="blue", cursor="hand2")
login_link_label.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

login_link_label.bind("<Button-1>", lambda e: open_login_window())

# Charge l'icône au format .gif ou .png
icon_image = tk.PhotoImage(file="/home/hyont-nick/DATA_ANALYST/Soutenance/app/img/icone.png")
# Configure l'icône de la fenêtre
root.iconphoto(True, icon_image)

root.mainloop()
