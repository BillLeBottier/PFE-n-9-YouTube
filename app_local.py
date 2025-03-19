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

def transcribe_video(filepath, language, word_limit=None, min_duration=1.0):
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

        print(transcript)

        # Création du fichier VTT
        vtt_path = os.path.join(OUTPUT_FOLDER, "Sous-titres.vtt")
        
        # ✅ Si word_limit est défini, on segmente, sinon on garde la transcription telle quelle
        if word_limit:
            segmented_subtitles = segment_vtt(transcript, word_limit, min_duration)
            with open(vtt_path, "w", encoding="utf-8") as vtt_file:
                vtt_file.write(segmented_subtitles)
        else:
            with open(vtt_path, "w", encoding="utf-8") as vtt_file:
                vtt_file.write(transcript)

        # Nettoyage
        os.remove(audio_path)
        return vtt_path

    except Exception as e:
        logger.error(f"Erreur lors de la transcription: {str(e)}")
        raise

#Fonction si l'option est activé pour le nombre de mots/sous-titres

def transcribe_video_with_limit(vtt_path, word_limit):
    """Crée un fichier VTT segmenté en fonction du nombre de mots par sous-titre."""
    with open(vtt_path, "r", encoding="utf-8") as f:
        vtt_content = f.read()

    segmented_subtitles = segment_vtt(vtt_content, word_limit)

    vtt_custom_path = vtt_path.replace(".vtt", "_custom.vtt")
    with open(vtt_custom_path, "w", encoding="utf-8") as vtt_file:
        vtt_file.write(segmented_subtitles)

    return vtt_custom_path

#Fonction pour nombre de mots par sous-titres

def parse_time(timestamp):
    """Convertit un timestamp VTT en secondes (float). Gère les erreurs de format."""
    try:
        parts = re.split('[:.]', timestamp)
        
        if len(parts) == 4:  # Format attendu hh:mm:ss.sss
            h, m, s, ms = parts
        elif len(parts) == 3:  # Cas où il manque les millisecondes
            h, m, s = parts
            ms = "000"  # On met 000 par défaut
        else:
            raise ValueError(f"Format de timestamp invalide: {timestamp}")

        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    
    except Exception as e:
        logger.error(f"Erreur dans parse_time() avec {timestamp}: {e}")
        return 0  # Retourne 0 en cas d'erreur


def segment_vtt(vtt_content, word_limit, min_duration=1.0):
    """Fusionne tous les sous-titres, puis segmente en blocs de `word_limit` mots avec des timestamps alignés."""
    
    captions = webvtt.read_buffer(io.StringIO(vtt_content))
    
    # 🔄 Étape 1 : Fusionner tout le texte en une seule ligne et garder les timestamps
    all_words = []
    word_timestamps = []

    for caption in captions:
        words = caption.text.split()
        if words:  # Ignorer les sous-titres vides
            all_words.extend(words)
            word_timestamps.extend([caption.start] * len(words))  # Associer chaque mot à son timestamp

    # Vérification des tailles pour éviter les erreurs
    if not all_words or not word_timestamps:
        logger.error("❌ Erreur: Aucun texte ou timestamp trouvé.")
        return "WEBVTT\n\n"

    # 🔄 Étape 2 : Découper en segments de `word_limit` mots avec des timestamps valides
    new_captions = []
    previous_end_time = 0  # Initialiser l'heure de fin du dernier sous-titre

    start_idx = 0
    while start_idx < len(all_words):
        chunk = all_words[start_idx:start_idx + word_limit]  # Prend `word_limit` mots
        if not chunk:
            break
        
        # 🔍 Déterminer le bon timestamp avec sécurité
        start_time = parse_time(word_timestamps[start_idx])  # Timestamp du premier mot du segment
        
        # Si ce n'est pas le premier sous-titre, s'assurer qu'il commence après la fin du précédent
        start_time = max(start_time, previous_end_time)

        end_idx = min(len(word_timestamps) - 1, start_idx + word_limit - 1)  # Sécuriser l'accès
        end_time = parse_time(word_timestamps[end_idx])  # Timestamp du dernier mot du segment

        # 🚀 Correction : Éviter les sous-titres de durée trop courte
        if end_time - start_time < float(min_duration):
            end_time = start_time + float(min_duration)  # Ajoute la durée minimale spécifiée

        # 📝 Ajouter au fichier VTT
        start_vtt_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02}.{int((start_time % 1) * 1000):03}"
        end_vtt_str = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02}.{int((end_time % 1) * 1000):03}"

        new_captions.append(f"{start_vtt_str} --> {end_vtt_str}\n{' '.join(chunk)}\n")

        # 🔄 Mise à jour de `previous_end_time` pour que la prochaine ligne commence après la fin de celle-ci
        previous_end_time = end_time

        start_idx += word_limit  # Avancer au prochain segment

    logger.info(f"✅ Segmentation stricte en blocs de {word_limit} mots effectuée")
    return "WEBVTT\n\n" + "\n".join(new_captions)


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
    
    logger.info(f"🎬 Sous-titres ajoutés à la vidéo: {vtt_path}")
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
            "OutlineColour=&H000000&,Outline=0,Shadow=1,"
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
        Analyse ce fichier de sous-titres au format VTT et crée des chapitres pertinents en FRANCAIS 
        pour une vidéo YouTube. Les chapitres doivent aider les spectateurs à naviguer facilement dans 
        la vidéo et à comprendre sa structure.
        Le fichier contient déjà les timestamps exacts, utilise-les pour créer des chapitres cohérents.

        Règles pour les chapitres :
        Utilise les timestamps existants du VTT pour créer des chapitres cohérents.
        Format exact requis : 'HH:MM:SS Titre du chapitre'.
        Maximum 6 à 8 chapitres bien répartis sur la durée de la vidéo.
        Le premier chapitre doit toujours être à '00:00:00 Introduction'.
        Le dernier chapitre doit toujours être avant le dernier timestamp

        Les titres des chapitres doivent être :
        Courts (3 à 6 mots maximum).
        Descriptifs et informatifs.
        En français courant, sans jargon technique inutile.
        Accrocheurs pour inciter les spectateurs à cliquer sur le chapitre.
        

        Les chapitres doivent être placés à des moments clés de la vidéo, comme les transitions entre les sujets, les démonstrations, ou les conclusions.

        Les chapitres doivent refléter fidèlement le contenu de la vidéo. Évite de créer des chapitres qui ne correspondent pas au sujet traité.

        Contenu VTT :
        {vtt_content}

        Retourne uniquement les chapitres en FRANCAIS, un par ligne, sans texte supplémentaire. 
        Assure-toi que le format est strictement respecté : 
        'HH:MM:SS Titre du chapitre'. 
        Si tu détectes des problèmes dans le fichier VTT (comme des timestamps manquants ou incohérents), 
        signale-le avant de générer les chapitres.
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
        Tu es un expert en rédaction de descriptions YouTube. Analyse la transcription de cette vidéo et rédige une description optimisée pour YouTube en FRANCAIS.
        La description doit être concise, engageante, et inclure des mots-clés pertinents pour améliorer le référencement et le SEO.

        Règles pour la description :
        N'utilise aucun formatage Markdown (comme les astérisques pour le gras).
        Une seule ligne vide entre chaque section.
        Fait des sauts à la ligne inteligent et que tu codera par <br>

        Structure de la description :
        Introduction accrocheuse (1-2 lignes) : Résume l'essentiel de la vidéo de manière engageante. Utilise des phrases courtes et percutantes.
        Détails du contenu (3-5 lignes) : Décris les points clés de la vidéo en utilisant des paragraphes courts et des listes si nécessaire.
        Appels à l'action (1-2 lignes) : Encourage les spectateurs à s'abonner, liker la vidéo, ou laisser un commentaire.
        Liens utiles : Ajoute des liens vers les réseaux sociaux, le site web, ou d'autres vidéos pertinentes.
        Mots-clés et hashtags : Inclus 3 à 5 mots-clés pertinents et 2 à 3 hashtags pour améliorer le référencement.

        Retourne uniquement la description YouTube en FRANCAIS, sans commentaire supplémentaire.
        
        --- Transcription ---
        {video_transcript}

        Résumé :
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Utilisation de GPT-4 si disponible
            messages=[
                {"role": "system", "content": "Tu es un expert en rédaction de descriptions YouTube."},
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
        Tu es un assistant qui aide les créateurs de vidéos à se souvenir du contenu de leurs vidéos sans avoir à les revoir. 
        Résume la transcription de cette vidéo en utilisant une narration à la première personne, comme si le créateur décrivait lui-même ce qu'il a expliqué dans la vidéo.
        Règles pour le résumé :
        Utilise un ton naturel et professionnel, comme si le créateur parlait directement à un collègue ou à un ami.
        Structure le résumé en points clés, avec une phrase d'introduction et une conclusion.
        Chaque point clé doit correspondre à une section ou une idée principale de la vidéo.
        Inclus des détails spécifiques mentionnés dans la vidéo, comme des chiffres, des exemples, ou des citations importantes.

       

        Le résumé doit tenir en 8 à 10 lignes maximum, tout en couvrant tous les points clés de la vidéo.

        Exemple de résumé :
        'Dans cette vidéo, j'explique comment optimiser une chaîne YouTube en 2024. 
        Je commence par les 3 étapes clés pour augmenter l'engagement : 
        1) créer des miniatures accrocheuses, 
        2) utiliser des titres percutants, 
        et 3) interagir avec les commentaires. 
        Ensuite, je partage mes astuces pour monétiser une chaîne rapidement, en insistant sur l'importance de la régularité. 
        Enfin, je donne des exemples concrets de créateurs qui ont réussi en appliquant ces méthodes.'

        Transcription de la vidéo :
        {video_transcript}

        Retourne uniquement le résumé à la première personne, sans commentaire supplémentaire.
        
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un expert en rédaction de résumés"},
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
        Tu es un expert en référencement YouTube. 
        Analyse la transcription de cette vidéo et génère une liste de mots-clés pertinents en FRANCAIS pour améliorer la visibilité de la vidéo sur YouTube.
        Règles pour les mots-clés :
        La liste doit inclure 5 à 10 mots-clés pertinents en FRANCAIS.
        Inclus un mélange de mots-clés courts (1-2 mots) et d'expressions longues (3-5 mots).
        Les mots-clés doivent être spécifiques au contenu de la vidéo, sans termes génériques ou vagues.
        Les mots-clés doivent refléter les habitudes de recherche des utilisateurs sur YouTube.

        Exemples de mots-clés bien formatés :
        Marketing digital
        Stratégies de contenu
        Tendances marketing 2024
        Comment utiliser ChatGPT pour le marketing
        
        Transcription de la vidéo :
        {video_transcript}

        Retourne uniquement une liste de mots-clés en FRANCAIS, séparés par des virgules, sans commentaire supplémentaire.

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



def create_short(video_path: str, start_time: str, end_time: str, index: int, word_limit: int = None, language: str = "fr") -> str:
    """
    Crée un short au format vertical (1080x1920) sans bord noir et avec sous-titres adaptés.
    Permet de limiter le nombre de mots par ligne de sous-titre si word_limit est spécifié.
    Utilise la langue spécifiée pour la transcription.
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
                language=language,
                response_format="vtt"
            )

        # Sauvegarder les sous-titres temporaires
        temp_vtt_path = os.path.join(OUTPUT_FOLDER, f"temp_short_{index + 1}.vtt")
        
        # Si word_limit est spécifié, segmenter les sous-titres
        if word_limit and isinstance(word_limit, int) and word_limit > 0:
            segmented_subtitles = segment_vtt(transcript, word_limit)
            with open(temp_vtt_path, "w", encoding="utf-8") as vtt_file:
                vtt_file.write(segmented_subtitles)
            logger.info(f"✅ Sous-titres du short {index + 1} segmentés avec {word_limit} mots par ligne")
        else:
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
        
        # ✅ Correction : Vérifier que word_limit n'est pas vide avant de le convertir
        word_limit = request.form.get('wordLimit')
        word_limit = int(word_limit) if word_limit and word_limit.strip().isdigit() and int(word_limit) > 0 else None
        
        # Valeur par défaut pour la durée minimale des sous-titres
        min_sub_duration = 1.0

        # Transcription (toujours nécessaire)
        process_status["progress"] = 30
        vtt_path = transcribe_video(temp_path, language, word_limit, min_sub_duration)  # Fichier VTT original (toujours utilisé)

        # ✅ Toujours créer un fichier custom, mais s'il n'y a pas d'option, on copie le VTT normal
        vtt_custom_path = None  # Ne pas créer `Sous-titres_custom.vtt` par défaut

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

        # ✅ Utiliser le fichier VTT généré
        final_vtt_path = vtt_path
        logger.info("📌 Utilisation du fichier VTT généré.")

        output_path = add_subtitles_to_video(temp_path, final_vtt_path, style)

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
            "original_filename": timestamped_filename,  # Utiliser le nom avec timestamp
            "language": language  # Ajouter la langue sélectionnée dans la réponse
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
        return jsonify({"error": str(e)}), 500


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
        
        # Récupérer le paramètre word_limit
        word_limit = request.form.get('word_limit')
        word_limit = int(word_limit) if word_limit and word_limit.strip().isdigit() and int(word_limit) > 0 else None
        
        # Récupérer le paramètre language (même langue que celle choisie initialement)
        language = request.form.get('language', 'fr')
        
        # Logs de debugging
        logger.info(f"Paramètres reçus:")
        logger.info(f"- original_filename: {original_filename}")
        logger.info(f"- vtt_path: {vtt_path}")
        logger.info(f"- shorts_count: {shorts_count}")
        logger.info(f"- shorts_duration: {shorts_duration}")
        logger.info(f"- word_limit: {word_limit}")
        logger.info(f"- language: {language}")

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
                i,
                word_limit=word_limit,
                language=language
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
    


@app.route('/features')
def features():
    """ Page des fonctionnalités de l'application """
    return render_template('features.html')
 
@app.route('/downloads')
def downloads():
    """ Page des téléchargements où l'utilisateur voit ses vidéos générées """
    files = []
    
    # Vérifier si le dossier outputs existe
    if os.path.exists(OUTPUT_FOLDER):
        # Récupérer tous les fichiers MP4 (vidéos et shorts)
        mp4_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith('.mp4')]
        
        # Créer une liste d'objets contenant les informations sur chaque fichier
        for file in mp4_files:
            file_path = os.path.join(OUTPUT_FOLDER, file)
            url = url_for('serve_file', filename=file)
            
            # Détermine s'il s'agit d'un short ou d'une vidéo complète
            is_short = file.startswith('short_')
            
            # Obtenir la date de création du fichier
            creation_time = os.path.getctime(file_path)
            creation_date = datetime.fromtimestamp(creation_time).strftime('%d/%m/%Y %H:%M')
            
            # Obtenir la taille du fichier en Mo
            size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
            
            files.append({
                'name': file,
                'url': url,
                'is_short': is_short,
                'date': creation_date,
                'size': size_mb
            })
        
        # Trier les fichiers par date de création (les plus récents d'abord)
        files.sort(key=lambda x: x['date'], reverse=True)
    
    return render_template('downloads.html', files=files)

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