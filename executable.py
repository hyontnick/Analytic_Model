import webbrowser

def launch_webpage(url):
    webbrowser.open(url)

if __name__ == '__main__':
    webpage_url = 'file:///home/hyont-nick/DATA_ANALYST/Soutenance/app/mono.html'  # Remplacez par l'URL de la page que vous souhaitez ouvrir
    launch_webpage(webpage_url)
