import tkinter as tk
from tkinter import ttk
import openai
import time

openai.api_key = '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'

chemin_image = "/home/hyont-nick/DATA_ANALYST/Soutenance/app/img/chatbot.png"

COULEUR_UTILISATEUR = "blue"
COULEUR_ASSISTANT = "green"

def demander_chat(question):
    chat_complet = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": question}]
    )

    reponse = chat_complet.choices[0].message.content

    return reponse

def afficher_progressivement(texte, delai, tag):
    for caractere in texte:
        dialogue_text.insert(tk.END, caractere, (tag,))
        dialogue_text.update()
        time.sleep(delai)

def soumettre_question():
    question = question_entree.get()
    if question.strip() != "":
        dialogue_text.insert(tk.END, "Utilisateur : " + question + "\n", (COULEUR_UTILISATEUR,))
        dialogue_text.update()

        progress_bar.start()  # Démarrer la barre de progression

        fenetre.update()  # Mettre à jour l'interface graphique pour afficher la barre de progression

        reponse = demander_chat(question)

        dialogue_text.insert(tk.END, "AM Assistant : ", (COULEUR_ASSISTANT,))
        dialogue_text.update()
        afficher_progressivement(reponse, 0.02, COULEUR_ASSISTANT)
        dialogue_text.insert(tk.END, "\n\n", (COULEUR_ASSISTANT,))
        dialogue_text.update()

        progress_bar.stop()  # Arrêter la barre de progression

        question_entree.delete(0, tk.END)

def copier_reponse():
    reponse = dialogue_text.get("1.0", tk.END)
    reponse = reponse.strip()
    fenetre.clipboard_clear()
    fenetre.clipboard_append(reponse)

fenetre = tk.Tk()
fenetre.title("AM Assistant virtuel")
fenetre.geometry("610x490")
fenetre.resizable(False, False)
fenetre.configure(bg="white")

image = tk.PhotoImage(file=chemin_image)

background_label = tk.Label(fenetre, image=image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

style = ttk.Style(fenetre)
style.configure("TFrame", background="white", borderwidth=0, relief="solid", bordercolor="gray", padding=10)
style.configure("TButton", background="#007bff", foreground="white", padding=10, font=("Arial", 12, "bold"))
style.map("TButton", background=[("active", "#0056b3")])

dialogue_frame = ttk.Frame(fenetre, style="TFrame")
dialogue_frame.pack(padx=20, pady=20, fill="both")

dialogue_text = tk.Text(dialogue_frame, height=20, font=("Arial", 12, "bold"))
dialogue_text.tag_configure(COULEUR_UTILISATEUR, foreground=COULEUR_UTILISATEUR, font=("Arial", 12, "bold"))
dialogue_text.tag_configure(COULEUR_ASSISTANT, foreground=COULEUR_ASSISTANT, font=("Arial", 12, "bold"))
dialogue_text.pack(fill="both", expand=True)
dialogue_scrollbar = ttk.Scrollbar(dialogue_frame, orient=tk.VERTICAL, command=dialogue_text.yview)
dialogue_text.configure(yscrollcommand=dialogue_scrollbar.set)
dialogue_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

saisie_frame = ttk.Frame(fenetre, style="TFrame")
saisie_frame.pack(padx=20, pady=(0, 20), fill="both")

question_label = ttk.Label(saisie_frame, text="Question :", background="white")
question_label.pack(side=tk.LEFT)

question_entree = ttk.Entry(saisie_frame, font=("Arial", 12, "bold"))
question_entree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

soumettre_bouton = ttk.Button(saisie_frame, text="Envoyer", command=soumettre_question, style="TButton")
soumettre_bouton.pack(side=tk.RIGHT)

dialogue_frame.columnconfigure(0, weight=1)

copier_bouton = ttk.Button(dialogue_frame, text="Copier", command=copier_reponse, style="TButton")
copier_bouton.pack(side=tk.RIGHT, padx=(0, 10))

progress_bar = ttk.Progressbar(dialogue_frame, mode="indeterminate")
progress_bar.pack(fill="x")

icon_image = tk.PhotoImage(file="/home/hyont-nick/DATA_ANALYST/Soutenance/app/img/icone.png")
fenetre.iconphoto(True, icon_image)

fenetre.mainloop()
