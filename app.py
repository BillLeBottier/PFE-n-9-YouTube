from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Les fichiers video uploadés vont être stocké dans ce dossier
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route de la page principale
@app.route('/')
def index():
    return render_template('index.html')

# Route pour le traitement du fichier vidéo uploadé
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Vérification de d'upload du fichier
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return 'Vidéo téléchargée et sous-titrée avec succès!'  # lancer le processus d'ajout de sous-titres.

if __name__ == '__main__':
    app.run(debug=True)