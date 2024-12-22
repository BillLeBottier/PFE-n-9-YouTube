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

def add_subtitles_to_video(video_path, vtt_path, style):
    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(video_path).rsplit('.', 1)[0] + "_subtitled.mp4")

    # Configuration des styles de sous-titres (inchangée)
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

    subprocess.run(command, check=True)
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
        Analyse ce fichier de sous-titres au format VTT et crée des chapitres pertinents.
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

        Retourne uniquement les chapitres, un par ligne, sans texte supplémentaire.
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
        Voici la transcription d'une vidéo. Génère une liste de 5 à 10 mots-clés pertinents qui résument le contenu de cette vidéo.

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

def extract_shorts_timestamps(vtt_path: str) -> List[Dict]:
    """
    Analyse le fichier VTT pour identifier des segments intéressants pour des shorts
    """
    try:
        with open(vtt_path, 'r', encoding='utf-8') as vtt_file:
            vtt_content = vtt_file.read()

        prompt = f"""
        Analyse ce fichier de sous-titres VTT et identifie 3 à 5 segments intéressants pour créer des shorts.

        Règles pour les segments :
        - Durée entre 15 et 45 secondes
        - Contenu accrocheur et autonome (compréhensible sans contexte)
        - Utilise les timestamps existants du VTT
        - Évite les segments qui coupent une phrase au milieu

        Contenu VTT :
        {vtt_content}

        Renvoie uniquement un tableau JSON avec ce format exact, sans autre texte :
        [
            {{
                "start": "00:00:00",
                "end": "00:00:30",
                "description": "Description"
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

def create_short(video_path: str, start_time: str, end_time: str, index: int) -> str:
    """
    Crée un court segment vidéo à partir des timestamps donnés
    """
    try:
        output_filename = f"short_{index + 1}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        command = [
            "ffmpeg",
            "-i", video_path,
            "-ss", start_time,
            "-to", end_time,
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-c:a", "copy",
            output_path
        ]

        subprocess.run(command, check=True)
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
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Aucun fichier envoyé"}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "Aucun fichier sélectionné"}), 400

        # Sécurisation du nom de fichier
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(f"{timestamp}_{file.filename}")
        
        # Sauvegarde temporaire
        temp_path = os.path.join(TEMP_FOLDER, filename)
        file.save(temp_path)
        logger.info(f"Fichier sauvegardé: {temp_path}")

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

        # Nettoyage
        os.remove(temp_path)
        
        process_status["progress"] = 100

        # Retourner uniquement les données demandées
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
    """
    Endpoint pour générer les shorts à partir d'une vidéo
    """
    try:
        video_path = request.form.get('video_path')
        vtt_path = request.form.get('vtt_path')

        if not video_path or not vtt_path:
            return jsonify({"error": "Chemins vidéo et VTT requis"}), 400

        # Chemin complet des fichiers
        video_path = os.path.join(OUTPUT_FOLDER, os.path.basename(video_path))
        vtt_path = os.path.join(OUTPUT_FOLDER, os.path.basename(vtt_path))

        # Extraction des segments intéressants
        segments = extract_shorts_timestamps(vtt_path)

        # Création des shorts
        shorts_info = []
        for i, segment in enumerate(segments):
            output_filename = create_short(
                video_path, 
                segment['start'], 
                segment['end'], 
                i
            )
            
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
        logger.error(f"Erreur lors de la génération des shorts: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)