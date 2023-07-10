from http.server import BaseHTTPRequestHandler, HTTPServer
import sqlite3
import urllib.parse
import re

# Classe de gestionnaire de requêtes
class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('inscription.html', 'rb') as file:
                self.wfile.write(file.read())
        elif self.path == '/connexion.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('connexion.html', 'rb') as file:
                self.wfile.write(file.read())
        elif self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('index.html', 'rb') as file:
                self.wfile.write(file.read())
        elif self.path == '/script.js':
            # Serve the JavaScript file
            self.send_response(200)
            self.send_header('Content-type', 'application/javascript')
            self.end_headers()
            with open('script.js', 'rb') as file:
                self.wfile.write(file.read())
        elif self.path.endswith('.css'):
            # Serve the CSS file
            self.send_response(200)
            self.send_header('Content-type', 'text/css')
            self.end_headers()
            with open(self.path[1:], 'rb') as file:
                self.wfile.write(file.read())
        elif self.path.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            # Serve the image file
            extension = self.path.split('.')[-1]
            content_type = 'image/' + extension
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.end_headers()
            with open(self.path[1:], 'rb') as file:
                self.wfile.write(file.read())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write('Fichier non trouvé.'.encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        data = urllib.parse.parse_qs(post_data)

        if self.path == '/inscription':
            username = data['username'][0]
            password = data['password'][0]
            confirm_password = data['confirm_password'][0]
            access_code = data['access_code'][0]

            # Vérification des conditions de sécurité du mot de passe
            if len(password) < 8 or not re.search(r'[A-Z]', password) or not re.search(r'[a-z]', password) or not re.search(r'\d', password) or not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                self.send_response(400)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write('Le mot de passe ne respecte pas les conditions de sécurité.'.encode('utf-8'))
                return

            # Vérification de correspondance du mot de passe confirmé
            if password != confirm_password:
                self.send_response(400)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write('Le mot de passe confirmé ne correspond pas au mot de passe.'.encode('utf-8'))
                return

            # Vérification du code d'accès
            if not re.match(r'^\d{12}$', access_code):
                self.send_response(400)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write('Le code d\'accès doit contenir 12 chiffres.'.encode('utf-8'))
                return

            conn = sqlite3.connect('login_data.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, password, access_code) VALUES (?, ?, ?)',
                           (username, password, access_code))
            conn.commit()
            conn.close()

            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write('Inscription réussie.'.encode('utf-8'))

        elif self.path == '/connexion':
            username = data['username'][0]
            password = data['password'][0]

            conn = sqlite3.connect('login_data.db')
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
            result = cursor.fetchone()
            conn.close()

            if result is not None:
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write('Connexion réussie.'.encode('utf-8'))
            else:
                self.send_response(401)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write('Identifiants incorrects.'.encode('utf-8'))


# Démarrer le serveur
def run():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, RequestHandler)
    print('Serveur démarré sur le port 8000...')
    httpd.serve_forever()


run()
