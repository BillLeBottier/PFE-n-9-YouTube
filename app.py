from flask import Flask, render_template, request, redirect, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os
import whisper
import srt
from datetime import timedelta
import subprocess

app = Flask(__name__)

# fichier stocké ds uploads
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = whisper.load_model("medium")  # Ou tiny , base , small, medium , large ou large-v2 selon la config du pc et des gpu 

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

            # génère le fichier .srt
            srt_path = transcribe_video(filepath)

            # Ajout des sous-titre
            subtitled_video_path = add_subtitles_to_video(filepath, srt_path)

            
            return redirect(url_for('download_file', filename=os.path.basename(subtitled_video_path)))
    
    return redirect(url_for('index'))

# Route pour dl la vidéo traitée 
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def transcribe_video(filepath):
    # Utilisation de whisper pour timestamp
    result = model.transcribe(filepath, task="transcribe")

    # étape de création du fichier .srt
    segments = result["segments"]
    subtitles = []

    for i, segment in enumerate(segments):
        start = timedelta(seconds=segment["start"])
        end = timedelta(seconds=segment["end"])
        content = segment["text"]
        subtitle = srt.Subtitle(index=i, start=start, end=end, content=content)
        subtitles.append(subtitle)

    srt_content = srt.compose(subtitles)
    srt_path = filepath.rsplit('.', 1)[0] + ".srt"

    with open(srt_path, "w") as srt_file:
        srt_file.write(srt_content)

    return srt_path

def add_subtitles_to_video(video_path, srt_path):
    # Sortie pour la vidéo sous titrée
    output_path = video_path.rsplit('.', 1)[0] + "_subtitled.mp4"
    
    # ffmpeg pour ajouter les sous titres
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"subtitles={srt_path}",
        "-c:a", "copy",
        output_path
    ]

    subprocess.run(command, check=True)
    return output_path

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)