# import tkinter as tk
# from tkinter import font

# root = tk.Tk()

# # Créer une police personnalisée
# custom_font = font.Font(family="Arial", size=12, weight="bold")

# # Créer un label avec la police personnalisée
# label = tk.Label(root, text="Texte avec une police personnalisée", font=custom_font)
# label.pack()

# root.mainloop()
import tkinter as tk
from tkinter import font

root = tk.Tk()

# Récupérer la liste de toutes les polices disponibles
all_fonts = font.families()

# Afficher les polices
for font_family in all_fonts:
    label = tk.Label(root, text=f"Texte avec la police '{font_family}'", font=(font_family, 12))
    label.pack()

root.mainloop()

