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
from google.cloud import secretmanager
from google.cloud import storage
import json

# Chargement des variables d'environnement
load_dotenv()

def access_secret_version(project_id, secret_id, version_id="latest"):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Configuration d'OpenAI
project_id = "still-catwalk-441910-u6"  # Votre ID de projet
OPENAI_API_KEY = access_secret_version(project_id, "openai-api-key")
client = OpenAI(api_key=OPENAI_API_KEY)

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

# Configuration Cloud Storage
CLOUD_STORAGE_BUCKET = "video-subtitles-storage-xyz"

process_status = {"progress": 0}  # Variable pour le suivi du statut

def get_storage_client():
    """Initialise le client Storage avec le compte de service"""
    try:
        # Récupérer la clé du compte de service depuis Secret Manager
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/video-service-key/versions/latest"
        response = client.access_secret_version(request={"name": name})
        credentials_dict = json.loads(response.payload.data.decode("UTF-8"))
        
        # Créer le client Storage avec ces credentials
        return storage.Client.from_service_account_info(credentials_dict)
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du client Storage: {str(e)}")
        raise

# Remplacer l'initialisation existante du client Storage par :
storage_client = get_storage_client()
bucket = storage_client.bucket(CLOUD_STORAGE_BUCKET)

def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
        logger.info("FFmpeg est installé")
        return True
    except FileNotFoundError:
        logger.error("FFmpeg n'est pas installé")
        return False

@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    global process_status
    process_status = {"progress": 0}
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Aucun fichier envoyé"}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "Aucun fichier sélectionné"}), 400

        # Sécurisation du nom de fichier avec timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(f"{timestamp}_{file.filename}")
        
        try:
            # Créer un fichier temporaire local
            logger.info("Début de la sauvegarde du fichier temporaire")
            temp_local_path = os.path.join('/tmp', filename)
            file.save(temp_local_path)
            logger.info(f"Fichier temporaire sauvegardé: {temp_local_path}")

            # Récupération des options
            language = request.form.get('language', 'fr')
            style = request.form.get('style', 'default')
            create_chapters = request.form.get('createChapters') == 'true'

            # Transcription
            logger.info("Début de la transcription")
            process_status["progress"] = 30
            vtt_path = transcribe_video(temp_local_path, language)
            logger.info(f"Transcription terminée, fichier VTT créé: {vtt_path}")
            
            # Upload du fichier VTT
            logger.info("Début de l'upload VTT")
            process_status["progress"] = 50
            vtt_filename = os.path.basename(vtt_path)
            vtt_blob = bucket.blob(f"vtt/{vtt_filename}")
            vtt_blob.upload_from_filename(vtt_path)
            logger.info("Upload VTT terminé")
            
            # Génération de l'URL signée pour VTT
            vtt_url = vtt_blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=30),
                method="GET"
            )
            logger.info("URL VTT générée")
            
            # Ajout des sous-titres
            logger.info("Début de l'ajout des sous-titres")
            process_status["progress"] = 70
            subtitled_video_path = add_subtitles_to_video(temp_local_path, vtt_path, style)
            logger.info(f"Sous-titres ajoutés, fichier créé: {subtitled_video_path}")
            
            # Upload de la vidéo sous-titrée
            logger.info("Début de l'upload vidéo")
            process_status["progress"] = 90
            output_filename = os.path.basename(subtitled_video_path)
            video_blob = bucket.blob(f"videos/{output_filename}")
            video_blob.upload_from_filename(subtitled_video_path)
            logger.info("Upload vidéo terminé")

            # Génération de l'URL signée pour la vidéo
            video_url = video_blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=30),
                method="GET"
            )
            logger.info("URL vidéo générée")

            # Obtenir les métadonnées AVANT de supprimer les fichiers
            try:
                video_metadata = get_video_metadata(subtitled_video_path)
            except Exception as e:
                logger.error(f"Erreur lors de l'obtention des métadonnées: {str(e)}")
                video_metadata = {
                    "dimensions": "N/A",
                    "frame_rate": "N/A",
                    "duration": "N/A",
                    "size": "N/A"
                }

            # Générer les chapitres si nécessaire
            chapters_text = ""
            if create_chapters:
                try:
                    chapters_text = generate_chapters(temp_local_path)
                except Exception as e:
                    logger.error(f"Erreur lors de la génération des chapitres: {str(e)}")
                    chapters_text = ""

            # Nettoyage des fichiers temporaires
            logger.info("Début du nettoyage des fichiers temporaires")
            for file_path in [temp_local_path, vtt_path, subtitled_video_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            logger.info("Nettoyage terminé")

            process_status["progress"] = 100
            return jsonify({
                "video_url": video_url,
                "vtt_url": vtt_url,
                "metadata": video_metadata,
                "chapters": chapters_text
            })

        except Exception as e:
            logger.error(f"Erreur détaillée lors du traitement: {str(e)}")
            process_status = {
                "progress": 0,
                "error": f"Erreur lors du traitement: {str(e)}"
            }
            return jsonify(process_status), 500

    except Exception as e:
        logger.error(f"Erreur lors de l'upload: {str(e)}")
        process_status = {
            "progress": 0,
            "error": f"Erreur lors de l'upload: {str(e)}"
        }
        return jsonify(process_status), 500

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
    try:
        global process_status
        logger.info(f"Status actuel: {process_status}")
        return jsonify(process_status)
    except Exception as e:
        logger.error(f"Erreur lors de la vérification du statut: {str(e)}")
        return jsonify({
            "progress": 0,
            "error": str(e)
        }), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        blob = bucket.blob(f"videos/{filename}")
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=30),
            method="GET"
        )
        return redirect(url)
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement: {str(e)}")
        return jsonify({"error": "Erreur lors du téléchargement"}), 500

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
        chunk_size = 25 * 1024 * 1024  # 25MB chunks
        
        with open(filepath, "rb") as audio_file:
            # Lire le fichier en chunks
            while True:
                chunk = audio_file.read(chunk_size)
                if not chunk:
                    break
                    
                # Appel à l'API OpenAI avec la nouvelle syntaxe
                transcript = client.audio.transcriptions.create(
                    file=("audio.mp4", chunk),
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
    try:
        command = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            filepath
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Erreur ffprobe: {result.stderr}")
            raise Exception("Erreur lors de l'obtention des métadonnées")

        output = result.stdout.strip().split('\n')
        if len(output) < 4:
            raise Exception("Sortie ffprobe incomplète")

        width, height, fps, duration = output[:4]
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
    except Exception as e:
        logger.error(f"Erreur dans get_video_metadata: {str(e)}")
        return {
            "dimensions": "N/A",
            "frame_rate": "N/A",
            "duration": "N/A",
            "size": "N/A"
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
        # Supprimer la vidéo
        video_blob = bucket.blob(f"videos/{filename}")
        video_blob.delete()
        
        # Supprimer le VTT associé
        vtt_filename = filename.rsplit('.', 1)[0] + ".vtt"
        vtt_blob = bucket.blob(f"vtt/{vtt_filename}")
        vtt_blob.delete()
        
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