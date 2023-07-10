import tkinter as tk
import subprocess

def execute_code():
    subprocess.run(["python3", "/home/hyont-nick/DATA_ANALYST/Soutenance/app/main.py"])

def on_icon_click(event):
    execute_code()

root = tk.Tk()
root.title("A M")
root.geometry("128x128")
root.resizable(False, False)

# Chargement de l'icône
icon_image = tk.PhotoImage(file="/home/hyont-nick/DATA_ANALYST/Soutenance/app/img/icone.png")
root.iconphoto(True, icon_image)

# Création d'un label qui représente l'icône
icon_label = tk.Label(root, image=icon_image)
icon_label.pack()

# Lorsque l'utilisateur clique sur l'icône, la fonction on_icon_click sera appelée
icon_label.bind("<Button-1>", on_icon_click)

root.mainloop()
