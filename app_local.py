from flask import Flask, render_template, request, send_from_directory, url_for, jsonify
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
import webvtt
import re

#import pour le fichier zipcr
import io
import zipfile
from flask import send_file

client = openai

# Chargement des variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Flask
app = Flask(__name__)

# Configuration des dossiers
UPLOAD_FOLDER = 'uploads/'
OUTPUT_FOLDER = 'outputs/'
TEMP_FOLDER = 'temp/'

# Création des dossiers nécessaires
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TEMP_FOLDER, 'static']:
    os.makedirs(folder, exist_ok=True)

# Configuration
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1gb max
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Configuration OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY n'est pas définie dans le fichier .env")
openai.api_key = OPENAI_API_KEY


# Variable pour le suivi du statut
process_status = {"progress": 0}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_video(filepath, language):
    try:
        # Extraction audio
        audio_path = filepath.rsplit('.', 1)[0] + "_audio.mp4"
        command = [
            "ffmpeg", "-i", filepath,
            "-vn", "-acodec", "copy",
            audio_path
        ]
        subprocess.run(command, check=True)

        # Transcription avec OpenAI
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                language=language,
                response_format="vtt"
            )

        # Création du fichier VTT
        vtt_path = os.path.join(OUTPUT_FOLDER, "Sous-titres.vtt")
        with open(vtt_path, "w", encoding="utf-8") as vtt_file:
            vtt_file.write(transcript)

        # Nettoyage
        os.remove(audio_path)
        return vtt_path

    except Exception as e:
        logger.error(f"Erreur lors de la transcription: {str(e)}")
        raise

#Convertir fichier vtt en .ass pour sous-titres dynamiques

def convert_vtt_to_ass(vtt_path, ass_path):
    def format_timestamp(vtt_timestamp):
        """Convertit un timestamp VTT en format ASS (h:mm:ss.cs)."""
        h, m, s = vtt_timestamp.split(":")
        s, ms = s.split(".")
        return f"{int(h)}:{m}:{s}.{ms[:2]}"  # On garde seulement deux chiffres après la virgule

    print(f"🔄 Conversion du VTT en ASS pour le karaoké : {vtt_path} → {ass_path}")

    try:
        with open(ass_path, "w", encoding="utf-8") as f:
            # En-tête du fichier ASS
            f.write("[Script Info]\n")
            f.write("Title: Karaoke Subtitles\n")
            f.write("ScriptType: v4.00+\n")
            f.write("PlayResX: 1920\n")
            f.write("PlayResY: 1080\n\n")

            # Définition du style Karaoké
            f.write("[V4+ Styles]\n")
            f.write("Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
            f.write("Style: Karaoke,Arial,60,&HFFFFFF,&H000000,&H000000,1,0,1,2,0,2,10,10,30,1\n\n")

            # Section des événements (dialogues)
            f.write("[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

            for caption in webvtt.read(vtt_path):
                start = format_timestamp(caption.start)
                end = format_timestamp(caption.end)
                text = caption.text.replace("\n", " ")

                words = text.split()
                if not words:
                    print(f"⚠️ Sous-titre vide ignoré ({caption.start} → {caption.end})")
                    continue

                # Calculer la durée totale
                start_seconds = sum(x * float(t) for x, t in zip([3600, 60, 1], caption.start.split(":")))
                end_seconds = sum(x * float(t) for x, t in zip([3600, 60, 1], caption.end.split(":")))
                total_duration = (end_seconds - start_seconds) * 100  # Convertir en centièmes de secondes

                if total_duration <= 0:
                    print(f"⚠️ Durée invalide ignorée ({caption.start} → {caption.end})")
                    continue

                # Durée par mot (répartition égale)
                word_duration = max(int(total_duration / len(words)), 1)

                # Texte karaoké avec timing progressif
                karaoke_text = "".join([f"{{\\K{word_duration}}}{word} " for word in words])

                # Ajouter la ligne dans le fichier ASS
                f.write(f"Dialogue: 0,{start},{end},Karaoke,,0,0,0,,{karaoke_text.strip()}\n")

        print(f"✅ Conversion terminée : {ass_path}")

    except Exception as e:
        print(f"❌ Erreur pendant la conversion : {e}")



def add_subtitles_to_video(video_path, vtt_path, style):
    
    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(video_path).rsplit('.', 1)[0] + "_subtitled.mp4")

    ass_path = vtt_path.replace(".vtt", ".ass")

    # Configuration des styles de sous-titres (inchangée)
    if style == "youtube_shorts":
        subtitle_filter = (
            f"subtitles='{vtt_path}':force_style='Fontsize=20,"
            "Fontname=Franklin Gothic Medium Italic,Bold=1,"
            "PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,"
            "Outline=2,Shadow=0,Alignment=2,MarginV=30'"
        )

    elif style == "minimalist":
        subtitle_filter = (
            f"subtitles='{vtt_path}':force_style='Fontsize=16,"
            "Fontname=Helvetica,Bold=0,PrimaryColour=&HFFFFFF&,"
            "OutlineColour=&H000000&,Outline=0,Shadow=0,"
            "Alignment=2,MarginV=30'"
        )

    elif style == "default":  # Style par défaut
        subtitle_filter = (
            f"subtitles='{vtt_path}':force_style='Fontsize=15,"
            "Fontname=Arial,Bold=1,PrimaryColour=&HFFFFFF&,"
            "OutlineColour=&H000000&,Outline=1,Shadow=0,"
            "Alignment=2,MarginV=30'"
        )
    
    elif style == "karaoke":
        convert_vtt_to_ass(vtt_path, ass_path)
        subtitle_filter = f"ass='{ass_path}'"  # Utiliser ASS pour un effet karaoké

    else:  # style par défaut
        subtitle_filter = f"subtitles='{vtt_path}'"

    # Nouvelle commande ffmpeg avec paramètres de qualité
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", subtitle_filter,
        "-c:v", "libx264",  # Utilisation du codec H.264
        "-preset", "slow",   # Meilleure compression mais plus lent
        "-crf", "18",       # Qualité très élevée (0-51, où 0 est sans perte)
        "-c:a", "copy",     # Copie l'audio sans ré-encodage
        output_path
    ]


    try:
        subprocess.run(command, check=True)
        print(f"✅ Vidéo avec sous-titres générée : {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur FFmpeg : {e}")

    return output_path

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

#Mots-clés 

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

#Ajout d'une fonction pour récupérer la résolution d'origine

def get_video_resolution(video_path):
    """Retourne la largeur et hauteur d'une vidéo en pixels."""
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    info = json.loads(result.stdout)
    
    width = info["streams"][0]["width"]
    height = info["streams"][0]["height"]
    
    return width, height


#Fonction d'extraction de timestamps pour la création de shorts intelligents


def extract_shorts_timestamps(vtt_path: str, shorts_count: int = 3, shorts_duration: int = 30) -> List[Dict]:

    """
    Analyse le fichier VTT pour identifier des segments intéressants pour des shorts
    """
    try:
        with open(vtt_path, 'r', encoding='utf-8') as vtt_file:
            vtt_content = vtt_file.read()

        prompt = f"""

        Analyse ce fichier de sous-titres VTT et identifie {shorts_count} segments intéressants pour créer des shorts en FRANCAIS.

        Règles pour les segments :
        - Durée entre {max(15, min(shorts_duration-5, 55))} et {min(60, shorts_duration+5)} secondes
        - Contenu accrocheur et autonome qui est compréhensible sans contexte
        - Utilise les timestamps existants du VTT
        - Termine un segment sur une fin de phrase
        - Évite les chevauchements entre segments


        Contenu VTT :
        {vtt_content}

        Renvoie uniquement un tableau JSON avec ce format exact, sans autre texte :
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

        # Récupérer la réponse et nettoyer
        response_text = response.choices[0].message.content.strip()
        
        # Nettoyer la réponse de tout formatage markdown
        clean_response = response_text
        if '```' in clean_response:
            # Extraire le contenu entre les backticks
            clean_response = clean_response.split('```')[1]
            if 'json' in clean_response.split('\n')[0]:
                clean_response = '\n'.join(clean_response.split('\n')[1:])
            clean_response = clean_response.strip()

        logger.info(f"Réponse nettoyée: {clean_response}")
        
        # Parser le JSON
        segments = json.loads(clean_response)
        
        # Vérifier la structure des données
        if not isinstance(segments, list):
            raise ValueError("La réponse n'est pas un tableau JSON valide")
            
        for segment in segments:
            if not all(key in segment for key in ['start', 'end', 'description']):
                raise ValueError("Format de segment invalide")

        return segments

    except json.JSONDecodeError as e:
        logger.error(f"Erreur de parsing JSON: {str(e)}")
        logger.error(f"Contenu reçu: {clean_response}")
        return []
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction des timestamps pour shorts: {str(e)}")
        return []


def clean_output_folder():
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)  # Supprime tout le dossier
    os.makedirs(OUTPUT_FOLDER)  # Le recrée vide



def create_short(video_path: str, start_time: str, end_time: str, index: int) -> str:
    """
    Crée un short au format vertical (1080x1920) sans bord noir et avec sous-titres adaptés.
    """
    try:
        temp_short_path = os.path.join(OUTPUT_FOLDER, f"temp_short_{index + 1}.mp4")
        output_filename = f"short_{index + 1}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # Récupérer la résolution d'origine
        width, height = get_video_resolution(video_path)

        # Calcul du crop pour un format 9:16
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

        # Créer d'abord le short sans sous-titres
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
            temp_short_path
        ]
        subprocess.run(command, check=True)

        # Extraire l'audio du short pour Whisper
        audio_path = os.path.join(OUTPUT_FOLDER, f"temp_audio_{index + 1}.mp4")
        command = [
            "ffmpeg", "-i", temp_short_path,
            "-vn", "-acodec", "copy",
            audio_path
        ]
        subprocess.run(command, check=True)

        # Transcription avec Whisper
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                language="fr",
                response_format="vtt"
            )

        # Sauvegarder les sous-titres temporaires
        temp_vtt_path = os.path.join(OUTPUT_FOLDER, f"temp_short_{index + 1}.vtt")
        with open(temp_vtt_path, "w", encoding="utf-8") as vtt_file:
            vtt_file.write(transcript)

        # Ajouter les sous-titres au short avec style adapté
        subtitle_filter = (
            f"subtitles='{temp_vtt_path}':force_style='Fontsize=20,"  # Taille réduite de 24 à 20
            "Fontname=Arial,Bold=1,PrimaryColour=&HFFFFFF&,"
            "OutlineColour=&H000000&,Outline=2,Shadow=0,"
            "Alignment=2,MarginV=30,LineSpacing=12,"  # Marge verticale réduite de 60 à 30
            "TextMaxPixels=400'"  # Garde la même limite de largeur
        )

        command = [
            "ffmpeg",
            "-i", temp_short_path,
            "-vf", subtitle_filter,
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-c:a", "copy",
            output_path
        ]
        subprocess.run(command, check=True)

        # Nettoyage des fichiers temporaires
        os.remove(temp_short_path)
        os.remove(audio_path)
        os.remove(temp_vtt_path)

        return output_filename

    except Exception as e:
        logger.error(f"Erreur lors de la création du short: {str(e)}")
        raise



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global process_status
    process_status = {"progress": 0}
    
    # Nettoyage du dossier output avant de traiter un nouveau fichier
    clean_output_folder()

    try:
        if 'file' not in request.files:
            return jsonify({"error": "Aucun fichier envoyé"}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "Aucun fichier sélectionné"}), 400

        # Sécurisation du nom de fichier avec timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamped_filename = secure_filename(f"{timestamp}_{file.filename}")
        
        # Sauvegarde temporaire avec le nom horodaté
        temp_path = os.path.join(TEMP_FOLDER, timestamped_filename)
        file.save(temp_path)
        logger.info(f"Fichier sauvegardé avec timestamp: {timestamped_filename}")

        # Options
        language = request.form.get('language', 'fr')
        style = request.form.get('style', 'default')
        create_chapters = request.form.get('createChapters') == 'true'
        create_shorts = request.form.get('createShorts') == 'true'

        # Transcription (toujours nécessaire)
        process_status["progress"] = 30
        vtt_path = transcribe_video(temp_path, language)
        logger.info("Transcription terminée")

        # Extraire le texte brut du fichier VTT
        video_transcript = extract_text_from_vtt(vtt_path)
        logger.info("Texte brut extrait du fichier VTT")

        # Initialisation des variables
        chapters_text = ""
        summary = ""
        personal_summary = ""
        keywords = ""

        # Si chapitrage est demandé
        if create_chapters:
            process_status["progress"] = 45
            try:
                chapters_text = generate_chapters(temp_path, vtt_path)
                logger.info("Chapitres générés")
                
                # Générer le résumé global
                summary = generate_summary(video_transcript)
                logger.info("Résumé global généré")
                
                # Générer le résumé à la première personne
                personal_summary = generate_personal_summary(video_transcript)
                logger.info("Résumé à la première personne généré")
                
                # Générer les mots-clés
                keywords = generate_keywords(video_transcript)
                logger.info("Mots-clés générés")
            except Exception as e:
                logger.error(f"Erreur lors de la génération des contenus: {str(e)}")

        # Ajout des sous-titres
        process_status["progress"] = 60
        output_path = add_subtitles_to_video(temp_path, vtt_path, style)
        logger.info("Sous-titres ajoutés")

        # Génération des URLs locales
        video_url = url_for('serve_file', filename=os.path.basename(output_path))
        vtt_url = url_for('serve_file', filename=os.path.basename(vtt_path))

        # Ne pas supprimer le fichier temporaire si on doit générer des shorts
        if not create_shorts:
            os.remove(temp_path)
        
        process_status["progress"] = 100

        # Retourner le nom du fichier avec timestamp
        response_data = {
            "video_url": video_url,
            "vtt_url": vtt_url,
            "original_filename": timestamped_filename  # Utiliser le nom avec timestamp
        }

        if create_chapters:
            response_data.update({
                "chapters": chapters_text,
                "summary": summary,
                "personal_summary": personal_summary,
                "keywords": keywords
            })

        logger.info(f"Nom du fichier renvoyé au frontend: {timestamped_filename}")
        return jsonify(response_data)

    except Exception as e:
        if 'temp_path' in locals():
            os.remove(temp_path)
        logger.error(f"Erreur: {str(e)}")
        process_status = {
            "progress": 0,
            "error": f"Erreur: {str(e)}"
        }
        return jsonify(process_status), 500

@app.route('/files/<filename>')
def serve_file(filename):
    """Sert les fichiers depuis le dossier OUTPUT_FOLDER"""
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/status')
def get_status():
    """Retourne le statut actuel du traitement"""
    return jsonify(process_status)

@app.route('/generate_shorts', methods=['POST'])
def generate_shorts():

    try:
        # Récupérer les paramètres
        original_filename = request.form.get('original_video_filename')
        vtt_path = request.form.get('vtt_path')
        shorts_count = int(request.form.get('shorts_count', 3))
        shorts_duration = int(request.form.get('shorts_duration', 30))


        # Logs de debugging
        logger.info(f"Paramètres reçus:")
        logger.info(f"- original_filename: {original_filename}")
        logger.info(f"- vtt_path: {vtt_path}")
        logger.info(f"- shorts_count: {shorts_count}")
        logger.info(f"- shorts_duration: {shorts_duration}")

        # Construire le chemin complet
        original_video_path = os.path.join(TEMP_FOLDER, original_filename)
        logger.info(f"Chemin complet de la vidéo: {original_video_path}")
        logger.info(f"Le fichier existe? {os.path.exists(original_video_path)}")

        if not os.path.exists(original_video_path):
            error_msg = f"Vidéo originale non trouvée: {original_video_path}"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400

        # Vérifier que le fichier VTT existe aussi
        vtt_full_path = os.path.join(OUTPUT_FOLDER, os.path.basename(vtt_path))
        logger.info(f"Chemin complet du VTT: {vtt_full_path}")
        logger.info(f"Le fichier VTT existe? {os.path.exists(vtt_full_path)}")

        if not os.path.exists(vtt_full_path):
            error_msg = f"Fichier VTT non trouvé: {vtt_full_path}"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400

        # Extraction des segments
        segments = extract_shorts_timestamps(vtt_full_path, shorts_count, shorts_duration)
        logger.info(f"Segments extraits: {segments}")



        # Création des shorts
        shorts_info = []
        for i, segment in enumerate(segments):
            logger.info(f"Création du short {i+1}/{len(segments)}")
            output_filename = create_short(
                original_video_path,
                segment['start'],
                segment['end'],
                i
            )
            logger.info(f"Short créé: {output_filename}")
            
            shorts_info.append({
                'url': url_for('serve_file', filename=output_filename),
                'description': segment['description'],
                'start': segment['start'],
                'end': segment['end']
            })

        return jsonify({
            "shorts": shorts_info
        })

    except Exception as e:
        error_msg = f"Erreur lors de la génération des shorts: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)  # Ceci affichera le stack trace complet
        return jsonify({"error": error_msg}), 500
    


@app.route('/generate_zip', methods=['POST'])
def generate_zip():
    # Lire les données envoyées avec le formulaire
    chapters = request.form.get('chapters')
    video_url = request.form.get('video_url')
    vtt_url = request.form.get('vtt_url')
    summary = request.form.get('summary')
    personal_summary = request.form.get('personal_summary')
    keywords = request.form.get('keywords')

    # Vérifier si toutes les données nécessaires sont présentes
    if not all([chapters, video_url, vtt_url, summary, personal_summary, keywords]):
        return jsonify({"error": "Tous les paramètres doivent être fournis."}), 400

    # Extraire les noms de fichiers à partir des URLs
    video_filename = video_url.split('/')[-1]
    vtt_filename = vtt_url.split('/')[-1]

    # Construire les chemins vers les fichiers
    video_path = os.path.join(OUTPUT_FOLDER, video_filename)
    vtt_path = os.path.join(OUTPUT_FOLDER, vtt_filename)

    # Créer le contenu du fichier texte (résumé)
    summary_content = (
        f"Chapitrage:\n{chapters}\n\n"
        f"Résumé global:\n{summary}\n\n"
        f"Résumé à la première personne:\n{personal_summary}\n\n"
        f"Mots-clés:\n{keywords}\n"
    )

    # Créer le fichier ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Ajouter la vidéo sous-titrée
        if os.path.exists(video_path):
            zip_file.write(video_path, arcname=video_filename)
        # Ajouter le fichier .vtt
        if os.path.exists(vtt_path):
            zip_file.write(vtt_path, arcname=vtt_filename)
        # Ajouter le fichier texte contenant les résumés et mots-clés
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