from flask import Flask, render_template, request, redirect, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os
import whisper
from datetime import timedelta
import subprocess

app = Flask(__name__)

# Fichier stocké dans uploads
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
        if 'file' not in request.files or 'language' not in request.form or 'style' not in request.form:
            return redirect(request.url)
        
        file = request.files['file']
        language = request.form['language']  # Récupère la langue choisie
        style = request.form['style']        # Récupère le style de sous-titres choisi

        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Transcrire la vidéo en utilisant la langue choisie
            vtt_path = transcribe_video(filepath, language)

            # Ajouter les sous-titres à la vidéo avec le style sélectionné
            subtitled_video_path = add_subtitles_to_video(filepath, vtt_path, style)

            return redirect(url_for('download_file', filename=os.path.basename(subtitled_video_path)))

    return redirect(url_for('index'))

# Route pour télécharger la vidéo traitée 
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def transcribe_video(filepath, language):
    # Utilise Whisper pour transcrire la vidéo avec la langue sélectionnée
    result = model.transcribe(filepath, task="transcribe", language=language)

    # Création du fichier .vtt
    segments = result["segments"]
    vtt_content = "WEBVTT\n\n"

    for segment in segments:
        start = timedelta(seconds=segment["start"])
        end = timedelta(seconds=segment["end"])
        content = segment["text"].strip()

        # Formatage des timestamps pour VTT
        start_str = f"{int(start.total_seconds() // 3600):02}:{int((start.total_seconds() % 3600) // 60):02}:{int(start.total_seconds() % 60):02}.{int(start.microseconds / 1000):03}"
        end_str = f"{int(end.total_seconds() // 3600):02}:{int((end.total_seconds() % 3600) // 60):02}:{int(end.total_seconds() % 60):02}.{int(end.microseconds / 1000):03}"

        vtt_content += f"{start_str} --> {end_str}\n{content}\n\n"

    vtt_path = filepath.rsplit('.', 1)[0] + ".vtt"
    with open(vtt_path, "w", encoding="utf-8-sig") as vtt_file:
        vtt_file.write(vtt_content)

    return vtt_path

def add_subtitles_to_video(video_path, vtt_path, style):
    output_path = video_path.rsplit('.', 1)[0] + "_subtitled.mp4"

    # Définir le style de sous-titres dans FFmpeg en fonction du choix de style
    if style == "style1":
        subtitle_filter = f"subtitles='{vtt_path}':force_style='Fontsize=24,Fontname=Arial,Bold=1'"
    elif style == "style2":
        subtitle_filter = f"subtitles='{vtt_path}':force_style='Fontsize=24,Fontname=Arial,Italic=1'"
    elif style == "style3":
        subtitle_filter = f"subtitles='{vtt_path}':force_style='Fontsize=24,Fontname=Arial,OutlineColour=&H80000000,Outline=2'"
    elif style == "youtube_shorts":
        # Style de sous-titres pour YouTube Shorts
        subtitle_filter = (
            f"subtitles='{vtt_path}':force_style='Fontsize=28,"
            "Fontname=Arial,Bold=1,PrimaryColour=&HFFFFFF&,"
            "OutlineColour=&H000000&,Outline=4,Shadow=0,"
            "Alignment=2,MarginV=30'"
        )
    else:
        # Style par défaut
        subtitle_filter = f"subtitles='{vtt_path}'"

    # Commande FFmpeg pour ajouter les sous-titres avec le style choisi
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", subtitle_filter,
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