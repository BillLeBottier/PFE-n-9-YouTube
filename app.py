from flask import Flask, render_template, request, redirect, send_from_directory, url_for, jsonify
from werkzeug.utils import secure_filename
import os
from openai import OpenAI
from datetime import timedelta
import re
from collections import defaultdict, Counter
import time
from datetime import datetime
import logging
import requests
from dotenv import load_dotenv
import subprocess  

# Chargement des variables d'environnement
load_dotenv()

# Configuration d'OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configuration améliorée
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

process_status = {"progress": 0}  # Variable pour le suivi du statut

@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    global process_status
    process_status["progress"] = 0
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Aucun fichier envoyé"}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "Aucun fichier sélectionné"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Format de fichier non autorisé"}), 400

        # Sécurisation du nom de fichier avec timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(f"{timestamp}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Sauvegarde du fichier
        file.save(filepath)
        logger.info(f"Fichier sauvegardé: {filepath}")

        try:
            # Récupération des options
            language = request.form.get('language', 'fr')
            style = request.form.get('style', 'default')
            create_chapters = request.form.get('createChapters') == 'true'

            # Transcription
            process_status["progress"] = 30
            vtt_path = transcribe_video(filepath, language)
            
            # Ajout des sous-titres avec le style choisi
            process_status["progress"] = 70
            subtitled_video_path = add_subtitles_to_video(filepath, vtt_path, style)
            
            # Génération des chapitres si demandé
            chapters_text = ""
            if create_chapters:
                chapters_text = generate_chapters(filepath)

            # Métadonnées de la vidéo
            video_metadata = get_video_metadata(subtitled_video_path)
            process_status["progress"] = 100

            # Nettoyage des fichiers temporaires
            cleanup_temp_files(filepath, vtt_path)

            return jsonify({
                "video_url": url_for('download_file', filename=os.path.basename(subtitled_video_path)),
                "metadata": video_metadata,
                "chapters": chapters_text
            })

        except Exception as e:
            logger.error(f"Erreur lors du traitement: {str(e)}")
            cleanup_temp_files(filepath)
            return jsonify({"error": "Erreur lors du traitement de la vidéo"}), 500

    except Exception as e:
        logger.error(f"Erreur lors de l'upload: {str(e)}")
        return jsonify({"error": "Erreur lors de l'upload"}), 500

def cleanup_temp_files(*files):
    """Ne supprime plus les fichiers temporaires"""
    pass  # On ne fait plus rien

def cleanup_old_files(max_age_hours=24):
    """Nettoie les anciens fichiers du dossier uploads"""
    try:
        current_time = time.time()
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.getmtime(filepath) < current_time - (max_age_hours * 3600):
                os.remove(filepath)
                logger.info(f"Ancien fichier supprimé: {filepath}")
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des anciens fichiers: {str(e)}")

@app.route('/status')
def status():
    global process_status
    return jsonify(process_status)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(
        app.config['UPLOAD_FOLDER'], 
        filename,
        as_attachment=True  # Force le téléchargement
    )

@app.route('/generate-preview', methods=['POST'])
def generate_preview():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
        
    file = request.files['file']
    # Générer un court extrait de sous-titres (30 premières secondes par exemple)
    preview_vtt = generate_subtitle_preview(file)
    
    return jsonify({
        'previewVtt': preview_vtt
    })

def generate_subtitle_preview(video_file):
    # Logique pour générer un aperçu des sous-titres
    # Retourner l'URL du fichier VTT généré
    pass

def extract_keywords(text):
    """Extraction simplifiée des mots-clés en filtrant les mots courts et communs."""
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if len(word) > 3]  # Filtre les mots très courts

def summarize_content(content):
    """Génère un titre court pour un contenu de chapitre basé sur les mots les plus fréquents."""
    keywords = extract_keywords(content)
    counter = Counter(keywords)
    most_common = [word for word, _ in counter.most_common(3)]  # Les 3 mots les plus fréquents
    return " ".join(most_common).capitalize() + "..."  # Résumé avec les mots fréquents

def generate_chapters(filepath):
    """Génère des chapitres en utilisant l'API OpenAI Whisper"""
    try:
        with open(filepath, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                language="fr",
                response_format="verbose_json"
            )

        segments = transcript.segments
        
        # Initialisation des chapitres
        chapters = []
        current_chapter = {"start": None, "end": None, "content": ""}
        keywords = []
        
        for i, segment in enumerate(segments):
            # Extraction des mots-clés de chaque segment
            segment_keywords = extract_keywords(segment.text)
            
            # Comparaison des mots-clés pour détecter un changement de thème
            if current_chapter["content"]:
                common_keywords = set(segment_keywords) & set(keywords)
                # S'il y a peu de mots-clés en commun, on considère un changement de thème
                if len(common_keywords) < 2:
                    # Enregistre le chapitre courant
                    chapters.append({
                        "start": current_chapter["start"],
                        "end": timedelta(seconds=segment.start),
                        "title": summarize_content(current_chapter["content"])
                    })
                    # Redémarre un nouveau chapitre
                    current_chapter = {"start": timedelta(seconds=segment.start), "end": None, "content": ""}
                    keywords = segment_keywords  # Mots-clés pour le nouveau chapitre
                else:
                    keywords.extend(segment_keywords)
            else:
                # Début d'un nouveau chapitre
                current_chapter["start"] = timedelta(seconds=segment.start)
                keywords = segment_keywords
            
            # Ajoute le contenu actuel au chapitre en cours
            current_chapter["content"] += segment.text + " "
        
        # Ajoute le dernier chapitre
        if current_chapter["content"]:
            chapters.append({
                "start": current_chapter["start"],
                "end": timedelta(seconds=segments[-1].end),
                "title": summarize_content(current_chapter["content"])
            })

        # Formate les chapitres pour l'affichage
        chapters_text = "CHAPITRAGE\n\n"
        for chapter in chapters:
            start_str = f"{int(chapter['start'].total_seconds() // 60):02}:{int(chapter['start'].total_seconds() % 60):02}"
            chapters_text += f"{start_str} - {chapter['title']}\n"

        return chapters_text

    except Exception as e:
        logger.error(f"Erreur lors de la génération des chapitres: {str(e)}")
        raise

def transcribe_video(filepath, language):
    """Transcrit la vidéo en utilisant l'API OpenAI Whisper"""
    try:
        # Ouvrir le fichier audio
        with open(filepath, "rb") as audio_file:
            # Appel à l'API OpenAI avec la nouvelle syntaxe
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                language=language,
                response_format="verbose_json"
            )

        # Création du fichier VTT
        vtt_content = "WEBVTT\n\n"
        
        # Parcourir les segments de la transcription
        for segment in transcript.segments:
            start = timedelta(seconds=float(segment.start))
            end = timedelta(seconds=float(segment.end))
            content = segment.text.strip()
            
            start_str = f"{int(start.total_seconds() // 3600):02}:{int((start.total_seconds() % 3600) // 60):02}:{int(start.total_seconds() % 60):02}.{int(start.microseconds / 1000):03}"
            end_str = f"{int(end.total_seconds() // 3600):02}:{int((end.total_seconds() % 3600) // 60):02}:{int(end.total_seconds() % 60):02}.{int(end.microseconds / 1000):03}"
            
            vtt_content += f"{start_str} --> {end_str}\n{content}\n\n"

        # Sauvegarder le fichier VTT
        vtt_path = filepath.rsplit('.', 1)[0] + ".vtt"
        with open(vtt_path, "w", encoding="utf-8-sig") as vtt_file:
            vtt_file.write(vtt_content)

        return vtt_path

    except Exception as e:
        logger.error(f"Erreur lors de la transcription: {str(e)}")
        raise

def add_subtitles_to_video(video_path, vtt_path, style):
    output_path = video_path.rsplit('.', 1)[0] + "_subtitled.mp4"

    # Styles de sous-titres
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
    else:  # style par défaut
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

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "Fichier trop volumineux (max 500MB)"}), 413

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "Erreur interne du serveur"}), 500

@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Erreur lors de la suppression: {str(e)}")
        return jsonify({"success": False})

if __name__ == '__main__':
    # Création des dossiers nécessaires
    for folder in ['static', 'templates', 'uploads']:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # Nettoyage initial des anciens fichiers (optionnel maintenant)
    # cleanup_old_files()
    
    # Démarrage du serveur
    app.run(debug=True)