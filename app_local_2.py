from flask import Flask, render_template, request, send_from_directory, url_for, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import openai
from datetime import datetime, timedelta
import logging
import subprocess
from dotenv import load_dotenv
import json
from typing import List, Dict
import shutil
import re
import io
import zipfile

# Chargement des variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Flask et des dossiers
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
OUTPUT_FOLDER = 'outputs/'
TEMP_FOLDER = 'temp/'

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TEMP_FOLDER, 'static']:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1GB max
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Configuration OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY n'est pas défini dans le fichier .env")
openai.api_key = OPENAI_API_KEY
client = openai

# Variable pour le suivi du statut
process_status = {"progress": 0}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_video_duration(video_path):
    """Retourne la durée de la vidéo en secondes via ffprobe."""
    command = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return float(result.stdout.strip())

def transcribe_video(filepath, language):
    """
    Extrait l'audio du fichier (ou segment) et réalise la transcription via OpenAI.
    Retourne le chemin du fichier VTT généré.
    """
    try:
        audio_path = filepath.rsplit('.', 1)[0] + "_audio.mp4"
        command = [
            "ffmpeg", "-i", filepath,
            "-vn", "-acodec", "copy",
            audio_path
        ]
        subprocess.run(command, check=True)
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                language=language,
                response_format="vtt"
            )
        vtt_path = os.path.join(OUTPUT_FOLDER, os.path.basename(filepath).rsplit('.', 1)[0] + "_Sous-titres.vtt")
        with open(vtt_path, "w", encoding="utf-8") as vtt_file:
            vtt_file.write(transcript)
        os.remove(audio_path)
        return vtt_path
    except Exception as e:
        logger.error(f"Erreur lors de la transcription: {str(e)}")
        raise

def add_subtitles_to_video(video_path, vtt_path, style):
    """
    Incruste le fichier VTT dans la vidéo selon un style donné.
    Retourne le chemin de la vidéo sous-titrée.
    """
    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(video_path).rsplit('.', 1)[0] + "_subtitled.mp4")
    
    if style == "youtube_shorts":
        subtitle_filter = (
            f"subtitles='{vtt_path}':force_style='Fontsize=20,Fontname=Franklin Gothic Medium Italic,"
            "Bold=1,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,Outline=2,Shadow=0,Alignment=2,MarginV=30'"
        )
    elif style == "minimalist":
        subtitle_filter = (
            f"subtitles='{vtt_path}':force_style='Fontsize=16,Fontname=Helvetica,Bold=0,PrimaryColour=&HFFFFFF&,"
            "OutlineColour=&H000000&,Outline=0,Shadow=0,Alignment=2,MarginV=30'"
        )
    elif style == "default":
        subtitle_filter = (
            f"subtitles='{vtt_path}':force_style='Fontsize=15,Fontname=Arial,Bold=1,PrimaryColour=&HFFFFFF&,"
            "OutlineColour=&H000000&,Outline=1,Shadow=0,Alignment=2,MarginV=30'"
        )
    else:
        subtitle_filter = f"subtitles='{vtt_path}'"

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", subtitle_filter,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(command, check=True)
    return output_path

def split_video(video_path, segment_duration=1200):
    """
    Découpe la vidéo en segments de 'segment_duration' secondes (20 minutes par défaut).
    Retourne la liste des chemins vers les segments.
    """
    segments_folder = os.path.join(TEMP_FOLDER, "segments")
    os.makedirs(segments_folder, exist_ok=True)
    output_pattern = os.path.join(segments_folder, "segment_%03d.mp4")
    command = [
        "ffmpeg", "-i", video_path,
        "-c", "copy",
        "-map", "0",
        "-segment_time", str(segment_duration),
        "-f", "segment",
        output_pattern
    ]
    subprocess.run(command, check=True)
    segments = sorted([os.path.join(segments_folder, f) for f in os.listdir(segments_folder) if f.endswith(".mp4")])
    return segments

def concat_videos(video_list, output_path):
    """
    Concatène plusieurs vidéos en une seule via le concat demuxer de ffmpeg.
    """
    list_file = os.path.join(TEMP_FOLDER, "concat_list.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for video in video_list:
            f.write(f"file '{os.path.abspath(video)}'\n")
    command = [
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path
    ]
    subprocess.run(command, check=True)
    os.remove(list_file)
    return output_path

def extract_text_from_vtt(vtt_path):
    """
    Extrait le texte brut du fichier VTT (sans timestamps).
    """
    try:
        with open(vtt_path, 'r', encoding='utf-8') as vtt_file:
            lines = vtt_file.readlines()
        transcript = []
        for line in lines:
            if '-->' not in line and line.strip():
                transcript.append(line.strip())
        return ' '.join(transcript)
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du texte du fichier VTT: {str(e)}")
        raise

def get_video_resolution(video_path):
    """Retourne la largeur et la hauteur d'une vidéo via ffprobe."""
    command = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    info = json.loads(result.stdout)
    if "streams" not in info or not info["streams"]:
        raise ValueError(f"Impossible de récupérer les flux vidéo pour {video_path}: {info}")
    width = info["streams"][0]["width"]
    height = info["streams"][0]["height"]
    return width, height

def extract_shorts_timestamps(vtt_path: str, shorts_count: int = 3, shorts_duration: int = 30) -> List[Dict]:
    """
    Analyse le fichier VTT et identifie des segments pour créer des shorts.
    Renvoie un tableau JSON avec le format exact :
    [
        {"start": "00:00:00", "end": "00:00:30", "description": "Description en français"}
    ]
    """
    try:
        with open(vtt_path, 'r', encoding='utf-8') as vtt_file:
            vtt_content = vtt_file.read()
        prompt = f"""
Analyse ce fichier de sous-titres VTT et identifie {shorts_count} segments intéressants pour créer des shorts en FRANCAIS.
Règles pour les segments :
- Durée entre {max(15, min(shorts_duration-5,55))} et {min(60, shorts_duration+5)} secondes
- Utilise les timestamps existants du VTT
- Termine un segment sur une fin de phrase
- Évite les chevauchements entre segments
Contenu VTT :
{vtt_content}
Renvoie uniquement un tableau JSON avec ce format exact, sans texte supplémentaire :
[
    {{
        "start": "00:00:00",
        "end": "00:00:30",
        "description": "Description en français"
    }}
]
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un expert en édition vidéo. Réponds uniquement avec un JSON valide."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        response_text = response.choices[0].message.content.strip()
        clean_response = response_text
        if '```' in clean_response:
            clean_response = clean_response.split('```')[1]
            if clean_response.startswith('json'):
                clean_response = '\n'.join(clean_response.split('\n')[1:])
            clean_response = clean_response.strip()
        segments = json.loads(clean_response)
        if not isinstance(segments, list):
            raise ValueError("La réponse n'est pas un tableau JSON valide")
        for segment in segments:
            if not all(key in segment for key in ['start', 'end', 'description']):
                raise ValueError("Format de segment invalide")
        return segments
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction des timestamps pour shorts: {str(e)}")
        return []

def create_short(video_path: str, start_time: str, end_time: str, index: int) -> str:
    """
    Crée un short au format vertical (9:16) sans bord noir.
    """
    try:
        output_filename = f"short_{index + 1}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        width, height = get_video_resolution(video_path)
        target_aspect = 9 / 16
        video_aspect = width / height
        if video_aspect > target_aspect:
            new_width = int(height * target_aspect)
            crop_x = (width - new_width) // 2
            crop_filter = f"crop={new_width}:{height}:{crop_x}:0"
        else:
            new_height = int(width / target_aspect)
            crop_y = (height - new_height) // 2
            crop_filter = f"crop={width}:{new_height}:0:{crop_y}"
        command = [
            "ffmpeg",
            "-i", video_path,
            "-ss", start_time,
            "-to", end_time,
            "-vf", crop_filter,
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-c:a", "aac", "-b:a", "128k",
            output_path
        ]
        subprocess.run(command, check=True)
        return output_filename
    except Exception as e:
        logger.error(f"Erreur lors de la création du short: {str(e)}")
        raise

def clean_output_folder():
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)


def generate_chapters(video_path: str, vtt_path: str) -> str:
    """
    Génère les chapitres de la vidéo au format YouTube en utilisant le fichier VTT
    Retourne une chaîne formatée pour YouTube (HH:MM:SS Titre du chapitre)
    """
    try:
        # Lecture du fichier VTT
        with open(vtt_path, 'r', encoding='utf-8') as vtt_file:
            vtt_content = vtt_file.read()

        # Prompt pour GPT avec le contenu VTT
        prompt = f"""
        Analyse ce fichier de sous-titres au format VTT et crée des chapitres pertinents en FRANCAIS.
        Le fichier contient déjà les timestamps exacts, utilise-les pour créer des chapitres cohérents.

        Règles pour les chapitres :
        - Utilise les timestamps existants du VTT
        - Format exact requis: "HH:MM:SS Titre du chapitre"  
        - Maximum 6-8 chapitres bien répartis sur la durée de la vidéo
        - Premier chapitre toujours à 0:00 Introduction
        - Titres courts et descriptifs (3-6 mots)
        - Un chapitre par ligne

        Contenu VTT :
        {vtt_content}

        Retourne uniquement les chapitres en FRANCAIS, un par ligne, sans texte supplémentaire.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un expert en création de chapitres YouTube qui sait analyser les fichiers VTT."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        chapters = response.choices[0].message.content.strip()
        return chapters

    except Exception as e:
        logger.error(f"Erreur lors de la génération des chapitres: {str(e)}")
        raise


#Résumé court

def generate_summary(video_transcript: str) -> str:
    """
    Génère un résumé global de la vidéo en à peu près 10 lignes.
    """
    try:
        # Créer le prompt pour générer un résumé global
        prompt = f"""
        Voici la transcription d'une vidéo. Résume-la en 5 à 10 lignes. 
        Reste concis et clair, en couvrant les points principaux de la vidéo.
        
        --- Transcription ---
        {video_transcript}

        Résumé :
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Utilisation de GPT-4 si disponible
            messages=[
                {"role": "system", "content": "Tu es un expert en résumé vidéo."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        summary = response.choices[0].message.content.strip()
        return summary

    except Exception as e:
        logger.error(f"Erreur lors de la génération du résumé: {str(e)}")
        raise


#Résumé narrratif 

def generate_personal_summary(video_transcript: str) -> str:
    """
    Génère un résumé de la vidéo à la première personne.
    """
    try:
        # Créer le prompt pour générer un résumé à la première personne
        prompt = f"""
        Voici la transcription d'une vidéo. Résume-la comme si la personne qui parle dans la vidéo se décrivait en première personne. 
        Raconte ce qu'elle explique et ce qui se passe dans la vidéo, avec une narration à la première personne.

        Transcription:
        {video_transcript}

        Résumé à la première personne :
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un expert en rédaction de résumés à la première personne."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        personal_summary = response.choices[0].message.content.strip()
        return personal_summary

    except Exception as e:
        logger.error(f"Erreur lors de la génération du résumé à la première personne: {str(e)}")
        raise


def generate_keywords(video_transcript: str) -> str:
    """
    Génère une liste de mots-clés pertinents de la vidéo.
    """
    try:
        # Créer le prompt pour générer des mots-clés
        prompt = f"""
        Voici la transcription d'une vidéo. Génère une liste de 5 à 10 mots-clés pertinents en FRANCAIS qui résument le contenu de cette vidéo.

        --- Transcription ---
        {video_transcript}

        Mots-clés :
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Utilisation de GPT-4 si disponible
            messages=[
                {"role": "system", "content": "Tu es un expert en extraction de mots-clés."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )

        keywords = response.choices[0].message.content.strip()
        return keywords

    except Exception as e:
        logger.error(f"Erreur lors de la génération des mots-clés: {str(e)}")
        raise


#textes brut pour les résumés 

def extract_text_from_vtt(vtt_path):
    """
    Extrait le texte brut du fichier .vtt (sans les timestamps et autres informations de formatage).
    """
    try:
        with open(vtt_path, 'r', encoding='utf-8') as vtt_file:
            vtt_content = vtt_file.read()

        # Suppression des timestamps et autres informations formatées (lignes avec des temps)
        transcript = []
        for line in vtt_content.splitlines():
            if '-->' not in line:  # Ignore the timestamp lines
                if line.strip():  # Ignore empty lines
                    transcript.append(line.strip())
        
        return ' '.join(transcript)  # Combine the lines into a single string

    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du texte du fichier VTT: {str(e)}")
        raise


###########################
# Fonctions de traitement #
###########################

def process_short_video(video_path, language, style, create_chapters):
    """
    Traite une vidéo courte (<=20 minutes) avec le pipeline initial.
    Retourne final_video, vtt_path, et éventuellement les contenus générés.
    """
    vtt_path = transcribe_video(video_path, language)
    logger.info("Transcription terminée")
    video_transcript = extract_text_from_vtt(vtt_path)
    logger.info("Texte brut extrait du VTT")
    final_video = add_subtitles_to_video(video_path, vtt_path, style)
    
    chapters_text = summary = personal_summary = keywords = None
    if create_chapters:
        chapters_text = generate_chapters(video_path, vtt_path)
        summary = generate_summary(video_transcript)
        personal_summary = generate_personal_summary(video_transcript)
        keywords = generate_keywords(video_transcript)
    return final_video, vtt_path, chapters_text, summary, personal_summary, keywords

def process_long_video(video_path, language, style, create_chapters):
    """
    Traite une vidéo longue (>20 minutes) en la découpant en segments de 20 minutes,
    traitant chaque segment individuellement (transcription + sous-titres) et concaténant le résultat.
    Pour la génération des contenus textuels, des messages indicatifs sont renvoyés.
    """
    segments = split_video(video_path, segment_duration=1200)
    logger.info(f"Vidéo découpée en {len(segments)} segments")
    processed_segments = []
    for i, segment in enumerate(segments):
        logger.info(f"Traitement du segment {i+1}/{len(segments)}: {segment}")
        segment_vtt = transcribe_video(segment, language)
        segment_subtitled = add_subtitles_to_video(segment, segment_vtt, style)
        processed_segments.append(segment_subtitled)
    final_video_path = os.path.join(OUTPUT_FOLDER, os.path.basename(video_path).rsplit('.', 1)[0] + "_final.mp4")
    concat_videos(processed_segments, final_video_path)
    # Pour les vidéos longues, nous renvoyons des messages indicatifs pour le contenu textuel
    chapters_text = summary = personal_summary = keywords = "Contenu généré sur segments"
    # Pour l'URL du VTT, on peut indiquer que les sous-titres sont gérés par segments
    return final_video_path, "Segments traités", chapters_text, summary, personal_summary, keywords

###########################
# Routes Flask          #
###########################

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global process_status
    process_status = {"progress": 0}
    clean_output_folder()

    try:
        if 'file' not in request.files:
            return jsonify({"error": "Aucun fichier envoyé"}), 400
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "Aucun fichier sélectionné"}), 400

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(f"{timestamp}_{file.filename}")
        temp_path = os.path.join(TEMP_FOLDER, filename)
        file.save(temp_path)
        logger.info(f"Fichier sauvegardé: {temp_path}")

        language = request.form.get('language', 'fr')
        style = request.form.get('style', 'default')
        create_chapters = request.form.get('createChapters') == 'true'
        create_shorts = request.form.get('createShorts') == 'true'

        duration = get_video_duration(temp_path)
        logger.info(f"Durée de la vidéo: {duration} secondes")

        if duration <= 1200:
            final_video, vtt_path, chapters_text, summary, personal_summary, keywords = process_short_video(temp_path, language, style, create_chapters)
        else:
            final_video, vtt_path, chapters_text, summary, personal_summary, keywords = process_long_video(temp_path, language, style, create_chapters)

        video_url = url_for('serve_file', filename=os.path.basename(final_video))
        vtt_url = url_for('serve_file', filename=os.path.basename(vtt_path)) if duration <= 1200 else "Segments traités"

        os.remove(temp_path)
        process_status["progress"] = 100

        response_data = {
            "video_url": video_url,
            "vtt_url": vtt_url
        }
        if create_chapters:
            response_data.update({
                "chapters": chapters_text,
                "summary": summary,
                "personal_summary": personal_summary,
                "keywords": keywords
            })

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        process_status = {"progress": 0, "error": f"Erreur: {str(e)}"}
        return jsonify(process_status), 500

@app.route('/files/<filename>')
def serve_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/status')
def get_status():
    return jsonify(process_status)

@app.route('/generate_shorts', methods=['POST'])
def generate_shorts():
    try:
        video_path = request.form.get('video_path')
        vtt_path = request.form.get('vtt_path')
        shorts_count = int(request.form.get('shorts_count', 3))
        shorts_duration = int(request.form.get('shorts_duration', 30))

        if not video_path or not vtt_path:
            return jsonify({"error": "Chemins vidéo et VTT requis"}), 400

        video_path = os.path.join(OUTPUT_FOLDER, os.path.basename(video_path))
        vtt_path = os.path.join(OUTPUT_FOLDER, os.path.basename(vtt_path))

        segments = extract_shorts_timestamps(vtt_path, shorts_count, shorts_duration)
        shorts_info = []
        for i, segment in enumerate(segments):
            output_filename = create_short(video_path, segment['start'], segment['end'], i)
            shorts_info.append({
                'url': url_for('serve_file', filename=output_filename),
                'description': segment['description'],
                'start': segment['start'],
                'end': segment['end']
            })

        return jsonify({"shorts": shorts_info})

    except Exception as e:
        logger.error(f"Erreur lors de la génération des shorts: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_zip', methods=['POST'])
def generate_zip():
    chapters = request.form.get('chapters')
    video_url = request.form.get('video_url')
    vtt_url = request.form.get('vtt_url')
    summary = request.form.get('summary')
    personal_summary = request.form.get('personal_summary')
    keywords = request.form.get('keywords')

    if not all([chapters, video_url, vtt_url, summary, personal_summary, keywords]):
        return jsonify({"error": "Tous les paramètres doivent être fournis."}), 400

    video_filename = video_url.split('/')[-1]
    vtt_filename = vtt_url.split('/')[-1]
    video_path = os.path.join(OUTPUT_FOLDER, video_filename)
    vtt_path = os.path.join(OUTPUT_FOLDER, vtt_filename)

    summary_content = (
        f"Chapitrage:\n{chapters}\n\n"
        f"Résumé global:\n{summary}\n\n"
        f"Résumé à la première personne:\n{personal_summary}\n\n"
        f"Mots-clés:\n{keywords}\n"
    )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        if os.path.exists(video_path):
            zip_file.write(video_path, arcname=video_filename)
        if os.path.exists(vtt_path):
            zip_file.write(vtt_path, arcname=vtt_filename)
        zip_file.writestr('summary.txt', summary_content)

    zip_filename = 'video_package.zip'
    zip_buffer.seek(0)
    logger.info(f"Sending file: {zip_filename}")
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=zip_filename
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)
