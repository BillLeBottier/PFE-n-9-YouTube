from flask import Flask, render_template, request, redirect, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os
import whisper
from datetime import timedelta
import subprocess

app = Flask(__name__)

# fichier stocké dans uploads
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = whisper.load_model("medium")  # Ou tiny, base, small, medium, large ou large-v2 selon la config du PC et des GPU 

# Route homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Sécurité pour le nom du fichier
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # génère le fichier .vtt
            vtt_path = transcribe_video(filepath)

            # Ajout des sous-titres
            subtitled_video_path = add_subtitles_to_video(filepath, vtt_path)

            return redirect(url_for('download_file', filename=os.path.basename(subtitled_video_path)))

    return redirect(url_for('index'))

# Route pour télécharger la vidéo traitée 
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def transcribe_video(filepath):
    # Utilisation de whisper pour timestamp
    result = model.transcribe(filepath, task="transcribe")

    # étape de création du fichier .vtt
    segments = result["segments"]
    vtt_content = "WEBVTT\n\n"  # Ajoute l'en-tête VTT

    for segment in segments:

        start = timedelta(seconds=segment["start"])
        end = timedelta(seconds=segment["end"])
        content = segment["text"].strip()  # Retirer les espaces inutiles

        # Formater les timestamps pour VTT
        start_str = f"{int(start.total_seconds() // 3600):02}:{int((start.total_seconds() % 3600) // 60):02}:{int(start.total_seconds() % 60):02}.{int(start.microseconds / 1000):03}"
        end_str = f"{int(end.total_seconds() // 3600):02}:{int((end.total_seconds() % 3600) // 60):02}:{int(end.total_seconds() % 60):02}.{int(end.microseconds / 1000):03}"

        vtt_content += f"{start_str} --> {end_str}\n{content}\n\n"

    vtt_path = filepath.rsplit('.', 1)[0] + ".vtt"

    with open(vtt_path, "w", encoding="utf-8-sig") as vtt_file:
        vtt_file.write(vtt_content)


    return vtt_path

def add_subtitles_to_video(video_path, vtt_path):  # Modifier ici aussi

    # Sortie pour la vidéo sous-titrée
    output_path = video_path.rsplit('.', 1)[0] + "_subtitled.mp4"

    # Imprimer le chemin du fichier VTT pour débogage
    print(f"Chemin du fichier VTT : {vtt_path}")
    
    # ffmpeg pour ajouter les sous-titres
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"subtitles='{vtt_path}'",
        "-c:a", "copy",
        output_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Erreur FFmpeg: {result.stderr}")

    return output_path

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)