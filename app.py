from flask import Flask, render_template, request, redirect, send_from_directory, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import whisper
import subprocess
from datetime import timedelta
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = whisper.load_model("medium")
process_status = {"progress": 0}  # Variable pour le suivi du statut

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global process_status
    process_status["progress"] = 0
    if 'file' not in request.files or 'language' not in request.form or 'style' not in request.form:
        return jsonify({"error": "Missing parameters"}), 400

    file = request.files['file']
    language = request.form['language']
    style = request.form['style']
    create_chapters = request.form.get('createChapters') == 'true'  # Nouvelle option

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Transcrire la vidéo et générer les sous-titres
        vtt_path = transcribe_video(filepath, language)
        process_status["progress"] = 70
        subtitled_video_path = add_subtitles_to_video(filepath, vtt_path, style)
        process_status["progress"] = 100

        # Générer le chapitrage si sélectionné
        chapters_text = ""
        if create_chapters:
            chapters_text = generate_chapters(filepath)

        # Obtenir les métadonnées de la vidéo traitée
        video_metadata = get_video_metadata(subtitled_video_path)

        return jsonify({
            "video_url": url_for('download_file', filename=os.path.basename(subtitled_video_path)),
            "metadata": video_metadata,
            "chapters": chapters_text
        })
    
@app.route('/status')
def status():
    global process_status
    return jsonify(process_status)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)    

def generate_chapters(filepath):
    """Génère un texte de chapitrage pour YouTube à partir des segments de transcription."""
    result = model.transcribe(filepath)
    segments = result["segments"]

    # Format du chapitrage
    chapters = "CHAPTERS\n\n"
    for i, segment in enumerate(segments):
        start = timedelta(seconds=segment["start"])
        minutes = int(start.total_seconds() // 60)
        seconds = int(start.total_seconds() % 60)
        timestamp = f"{minutes:02}:{seconds:02}"

        # Extrait le contenu et crée un titre
        content = segment["text"].strip()
        title = summarize_content(content)

        # Ajoute le titre et le timestamp au texte de chapitrage
        chapters += f"{timestamp} - {title}\n"

        # Arrête après 10 chapitres pour limiter la longueur
        if i >= 9:
            break

    return chapters

def summarize_content(content):
    """Crée un titre court basé sur le contenu de chaque segment."""
    words = content.split()
    return " ".join(words[:5]) + "..."  

def transcribe_video(filepath, language):
    result = model.transcribe(filepath, task="transcribe", language=language)

    segments = result["segments"]
    vtt_content = "WEBVTT\n\n"

    for segment in segments:
        start = timedelta(seconds=segment["start"])
        end = timedelta(seconds=segment["end"])
        content = segment["text"].strip()

        start_str = f"{int(start.total_seconds() // 3600):02}:{int((start.total_seconds() % 3600) // 60):02}:{int(start.total_seconds() % 60):02}.{int(start.microseconds / 1000):03}"
        end_str = f"{int(end.total_seconds() // 3600):02}:{int((end.total_seconds() % 3600) // 60):02}:{int(end.total_seconds() % 60):02}.{int(end.microseconds / 1000):03}"

        vtt_content += f"{start_str} --> {end_str}\n{content}\n\n"

    vtt_path = filepath.rsplit('.', 1)[0] + ".vtt"
    with open(vtt_path, "w", encoding="utf-8-sig") as vtt_file:
        vtt_file.write(vtt_content)

    return vtt_path

def add_subtitles_to_video(video_path, vtt_path, style):
    output_path = video_path.rsplit('.', 1)[0] + "_subtitled.mp4"

    if style == "youtube_shorts":
        subtitle_filter = (
            f"subtitles='{vtt_path}':force_style='Fontsize=22,"
            "Fontname=Arial,Bold=1,PrimaryColour=&HFFFFFF&,"
            "OutlineColour=&H000000&,Outline=4,Shadow=0,"
            "Alignment=2,MarginV=30'"
        )
    elif style == "minimalist":
        subtitle_filter = (
            f"subtitles='{vtt_path}':force_style='Fontsize=20,"
            "Fontname=Arial,Bold=1,PrimaryColour=&HFFFFFF&,"
            "BackColour=&H80000000,Outline=0,Shadow=0,"
            "Alignment=2,MarginV=30'"
        )
    elif style == "highlight":
        subtitle_filter = (
            f"subtitles='{vtt_path}':force_style='Fontsize=20,"
            "Fontname=Arial,Bold=1,PrimaryColour=&H000000&,"
            "BackColour=&HFFFF00&,Outline=0,Shadow=0,"
            "Alignment=2,MarginV=30'"
        )
    else:
        subtitle_filter = f"subtitles='{vtt_path}'"

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", subtitle_filter,
        "-c:a", "copy",
        output_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    return output_path

def get_video_metadata(filepath):
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        filepath
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    width, height, fps, duration = result.stdout.splitlines()
    size = round(os.path.getsize(filepath) / (1024 * 1024), 2)

    duration = int(float(duration))
    minutes, seconds = divmod(duration, 60)
    duration_str = f"{minutes}:{seconds:02}"

    return {
        "dimensions": f"{width}x{height}",
        "frame_rate": f"{fps.split('/')[0]} fps",
        "duration": duration_str,
        "size": f"{size} MB"
    }

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)