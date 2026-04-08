import os
import uuid
from google import genai
from flask import Flask, render_template, request, jsonify, session
import speech_recognition as sr
import numpy as np
import librosa
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
from pydub import AudioSegment
from deepface import DeepFace
import cv2
import base64
import PyPDF2
import docx
import spacy
import random
from datetime import datetime
import json
from dotenv import load_dotenv
from supabase import create_client, Client
from groq import Groq

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.secret_key = os.getenv("FLASK_SECRET_KEY", "cyberverse-dev-secret-key")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

conversation_history = []
current_interview_type = None
extracted_skills = []

tech_question_count = 0
tech_score = 0
MAX_TECH_QUESTIONS = 2
tech_questions_history = []
tech_answers_history = []
tech_feedback_history = []
tech_marks_history = []

hr_question_count = 0
hr_score = 0
MAX_HR_QUESTIONS = 2
hr_emotions_history = []
hr_soft_skills_history = []

hr_questions = [
    "Tell me about a time you worked in a team. How did you contribute?",
    "Describe a challenging situation at work and how you handled it.",
    "What motivates you to perform well in your job?",
    "How do you handle stress and pressure in the workplace?",
    "Where do you see yourself in five years?",
    "Tell me about a time you failed. How did you deal with it?",
    "What is your greatest strength and how have you used it in a professional setting?",
    "How do you prioritize your tasks when you have multiple deadlines?"
]

hr_question_index = 0


def generate_with_groq(prompt):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return "Can you explain the concept of OOP in Python?"


def gemini_mark_hr_answer(answer_text):
    try:
        prompt = (
            f"Mark the following HR interview answer on an INTEGER scale from 0 to 10 "
            f"based on clarity, relevance, professionalism, confidence, and communication quality.\n\n"
            f"Answer: '{answer_text}'\n\n"
            f"Return only one integer: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, or 10."
        )
        response_text = generate_with_groq(prompt)
        mark = int(response_text.strip().split()[0])

        if mark < 0:
            mark = 0
        elif mark > 10:
            mark = 10

        return mark
    except Exception as e:
        logger.error(f"Error in marking HR answer: {e}")
        return 4


def gemini_mark_answer(answer_text):
    try:
        prompt = (
            f"Mark the following technical interview answer on an INTEGER scale from 0 to 10 "
            f"based on accuracy, relevance, clarity, and completeness.\n\n"
            f"Answer: '{answer_text}'\n\n"
            f"Return only one integer: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, or 10."
        )
        response_text = generate_with_groq(prompt)
        mark = int(response_text.strip().split()[0])

        if mark < 0:
            mark = 0
        elif mark > 10:
            mark = 10

        return mark
    except Exception as e:
        logger.error(f"Error in marking technical answer: {e}")
        return 4


def generate_tech_answer_feedback(question, answer_text, mark):
    try:
        prompt = (
            f"You are a technical interviewer.\n"
            f"Question: {question}\n"
            f"Candidate Answer: {answer_text}\n"
            f"Score Given: {mark}/10\n\n"
            f"Write short feedback in 2-3 lines in simple English.\n"
            f"Include:\n"
            f"1. what was good\n"
            f"2. what was missing or weak\n"
            f"3. one improvement suggestion\n"
            f"Keep it concise and interview-style."
        )
        feedback = generate_with_groq(prompt)
        return feedback.strip()
    except Exception as e:
        logger.error(f"Error generating tech answer feedback: {e}")
        return "Your answer had some relevant points, but it needed more clarity, technical accuracy, and completeness."


def generate_tech_overall_summary(questions, answers, feedbacks, score, max_score):
    try:
        combined_qna = []
        for i in range(min(len(questions), len(answers), len(feedbacks))):
            combined_qna.append(
                f"Q{i+1}: {questions[i]}\n"
                f"A{i+1}: {answers[i]}\n"
                f"Feedback{i+1}: {feedbacks[i]}"
            )

        combined_text = "\n\n".join(combined_qna)

        prompt = (
            f"You are generating a final technical interview review.\n"
            f"Final Score: {score}/{max_score}\n\n"
            f"Interview Data:\n{combined_text}\n\n"
            f"Write a short overall summary in simple English for a student.\n"
            f"Cover:\n"
            f"- overall performance\n"
            f"- strengths\n"
            f"- weak areas\n"
            f"- what to improve next\n"
            f"Return only one paragraph."
        )
        summary = generate_with_groq(prompt)
        return summary.strip()
    except Exception as e:
        logger.error(f"Error generating tech overall summary: {e}")
        if score >= max_score * 0.8:
            return "You performed well in the technical interview and showed good understanding of the main concepts. Keep improving answer depth and explanation style to make your responses even stronger."
        elif score >= max_score * 0.5:
            return "You showed a basic understanding of technical concepts, but your answers need more clarity, detail, and correctness in some places. Practice explaining concepts step by step with examples."
        else:
            return "Your technical interview performance shows that you need more preparation in core concepts and clearer explanation. Focus on fundamentals, revise key topics, and practice answering technical questions aloud."


def build_tech_profile_report(profile):
    tech_score_value = float(profile.get("tech_score", 0))
    tech_max_score_value = float(profile.get("tech_max_score", 20))

    tech_questions = profile.get("tech_questions_history", [])
    if isinstance(tech_questions, str):
        tech_questions = json.loads(tech_questions) if tech_questions else []

    tech_answers = profile.get("tech_answers_history", [])
    if isinstance(tech_answers, str):
        tech_answers = json.loads(tech_answers) if tech_answers else []

    tech_feedback = profile.get("tech_feedback_history", [])
    if isinstance(tech_feedback, str):
        tech_feedback = json.loads(tech_feedback) if tech_feedback else []

    tech_marks = profile.get("tech_marks_history", [])
    if isinstance(tech_marks, str):
        tech_marks = json.loads(tech_marks) if tech_marks else []

    strengths = []
    improvements = []

    if tech_score_value >= tech_max_score_value * 0.8:
        strengths.append("You showed a strong understanding of the technical topics asked.")
        strengths.append("Your overall technical performance was good and above average.")
    elif tech_score_value >= tech_max_score_value * 0.5:
        strengths.append("You showed basic technical understanding in several areas.")
        improvements.append("Try to make your answers more detailed and technically precise.")
    else:
        improvements.append("You need stronger preparation in technical fundamentals.")
        improvements.append("Practice explaining concepts clearly with correct terminology.")

    if tech_marks:
        high_marks = [m for m in tech_marks if m >= 7]
        low_marks = [m for m in tech_marks if m <= 4]

        if high_marks:
            strengths.append("Some of your answers were relevant and showed good conceptual knowledge.")
        if low_marks:
            improvements.append("A few answers lacked completeness or correctness and need more revision.")

    if not strengths:
        strengths.append("You attempted the interview sincerely and showed willingness to answer technical questions.")

    if not improvements:
        improvements.append("Keep practicing more structured and example-based technical answers.")

    topic_coverage = (
        f"You answered {len(tech_answers)} technical question(s) in this interview. "
        f"Your final score was {tech_score_value} out of {tech_max_score_value}."
    )

    overall_summary = generate_tech_overall_summary(
        tech_questions,
        tech_answers,
        tech_feedback,
        tech_score_value,
        tech_max_score_value
    )

    return {
        "score": tech_score_value,
        "max_score": tech_max_score_value,
        "topic_coverage": topic_coverage,
        "strengths": strengths,
        "improvements": improvements,
        "question_feedback": tech_feedback if tech_feedback else ["No detailed technical feedback available yet."],
        "overall_summary": overall_summary
    }


def detect_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if result and isinstance(result, list) and len(result) > 0:
            emotions = {k: float(v) for k, v in result[0]['emotion'].items()}
            dominant_emotion = result[0]['dominant_emotion']
            logger.info(f"Detected emotions: {emotions}, Dominant: {dominant_emotion}")
            return emotions, dominant_emotion
        else:
            logger.warning("No emotions detected in frame")
            return {}, None
    except Exception as e:
        logger.error(f"Error in emotion detection: {e}")
        return {}, None


def capture_frame(image_data):
    try:
        if "," in image_data:
            _, encoded = image_data.split(",", 1)
            binary_data = base64.b64decode(encoded)
        else:
            binary_data = base64.b64decode(image_data)
        temp_path = "temp_frame.jpg"
        with open(temp_path, "wb") as f:
            f.write(binary_data)
        frame = cv2.imread(temp_path)
        if frame is None:
            logger.error("Failed to load frame from temp file")
            return None
        return frame
    except Exception as e:
        logger.error(f"Error capturing frame: {e}")
        return None


def generate_tech_question(response=None):
    global extracted_skills, conversation_history, tech_question_count
    try:
        context = "\n".join([f"{entry['role']}: {entry['text']}" for entry in conversation_history])

        if extracted_skills and len(extracted_skills) > 0:
            skill = random.choice(extracted_skills)
            if not response:
                prompt = (
                    f"Given the conversation context:\n{context}\n"
                    f"Ask a basic technical interview question about {skill} that requires more than a one-word answer. "
                    f"But don't ask questions that are too big and don't ask any question that requires code submission."
                )
            else:
                prompt = (
                    f"Given the conversation context:\n{context}\n"
                    f"Based on the response: '{response}', ask a follow-up technical question about {skill} that builds on the previous answer, but not too big."
                )
        else:
            if not response:
                prompt = (
                    f"Given the conversation context:\n{context}\n"
                    "Ask a basic technical interview question about Python that requires more than a one-word answer."
                )
            else:
                prompt = (
                    f"Given the conversation context:\n{context}\n"
                    f"Based on the response: '{response}', ask a follow-up technical question about Python that builds on the previous answer."
                )

        response_text = generate_with_groq(prompt)
        question = response_text.strip()
        logger.info(f"Generated tech question: {question}")
        return question
    except Exception as e:
        logger.error(f"Error generating tech question: {e}")
        return "What is the difference between a list and a tuple in Python?"


def generate_hr_question():
    global hr_question_index, hr_questions, conversation_history
    context = "\n".join([f"{entry['role']}: {entry['text']}" for entry in conversation_history])
    try:
        if len(conversation_history) > 1:
            prompt = (
                f"Given the conversation context:\n{context}\n"
                "Ask a follow-up HR interview question that builds on the previous response. Return only the question."
            )
            response_text = generate_with_groq(prompt)
            question = response_text.strip()
        else:
            question = hr_questions[hr_question_index]
            hr_question_index = (hr_question_index + 1) % len(hr_questions)
        logger.info(f"Generated HR question: {question}")
        return question
    except Exception as e:
        logger.error(f"Error generating HR question: {e}")
        question = hr_questions[hr_question_index]
        hr_question_index = (hr_question_index + 1) % len(hr_questions)
        return question


def analyze_speech(audio_file):
    try:
        y, sr_value = librosa.load(audio_file)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr_value)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        energy = np.mean(librosa.feature.rms(y=y))
        return pitch_mean, energy
    except Exception as e:
        logger.error(f"Error in speech analysis: {e}")
        return 0, 0


def analyze_soft_skills(text, pitch, energy, emotions=None):
    try:
        sentiment = analyzer.polarity_scores(text)
        confidence = "High" if pitch > 100 else "Low"
        enthusiasm = "High" if energy > 0.1 else "Low"
        positivity = sentiment['compound']
        emotion_feedback = f"Dominant Emotion: {max(emotions, key=emotions.get)}" if emotions else "No emotions detected"
        return {
            "confidence": confidence,
            "enthusiasm": enthusiasm,
            "positivity": positivity,
            "emotion_feedback": emotion_feedback,
            "emotions": emotions or {}
        }
    except Exception as e:
        logger.error(f"Error in soft skills analysis: {e}")
        return {
            "confidence": "Unknown",
            "enthusiasm": "Unknown",
            "positivity": 0.0,
            "emotion_feedback": "Unknown",
            "emotions": {}
        }


def convert_to_wav(input_file, output_file="response.wav"):
    try:
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav")
        logger.info(f"Audio converted to WAV: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        return None


def extract_text(file):
    try:
        text = ""
        if file.filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        elif file.filename.endswith('.docx'):
            doc = docx.Document(file)
            for paragraph in doc.paragraphs:
                para_text = paragraph.text or ""
                text += para_text + "\n"
        else:
            logger.error("Unsupported file format")
            return ""
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""


def extract_skills(text):
    global extracted_skills
    try:
        skills_list = [
            "python", "java", "c", "c++", "javascript", "sql", "html", "css",
            "ruby", "php", "go", "rust", "typescript", "kotlin", "swift",
            "scala", "r", "perl", "matlab", "bash", "powershell",
            "flask", "django", "spring", "react", "angular", "vue.js", "node.js",
            "express", "laravel", "rails", "aspnet", "svelte",
            "android", "ios", "flutter", "xamarin", "react native",
            "tensorflow", "pytorch", "scikit-learn", "keras", "pandas", "numpy",
            "opencv", "theano", "caffe", "mxnet",
            "hadoop", "spark", "kafka", "flink", "airflow", "tableau", "power bi",
            "dask", "apache hive", "apache pig",
            "mongodb", "postgresql", "mysql", "oracle", "sqlite", "cassandra",
            "redis", "elasticsearch", "mariadb", "firebase",
            "aws", "azure", "google cloud", "ibm cloud", "oracle cloud", "heroku",
            "digitalocean", "linode",
            "docker", "kubernetes", "jenkins", "ansible", "terraform", "chef",
            "puppet", "circleci", "travis ci", "github actions", "gitlab ci",
            "bitbucket pipelines",
            "git", "svn", "mercurial", "perforce",
            "apache", "nginx", "tomcat", "iis", "haproxy", "traefik", "dns",
            "dhcp", "iptables", "wireguard",
            "selenium", "junit", "pytest", "mocha", "jest", "cypress", "postman",
            "soapui",
            "linux", "windows server", "macos", "Ubuntu", "centos", "redhat",
            "vim", "emacs", "vscode", "intellij", "eclipse", "grafana",
            "prometheus", "loki", "jaeger", "rabbitmq", "celery", "gunicorn",
            "supervisor", "logstash", "kibana", "splunk",
            "metasploit", "nmap", "wireshark", "burp suite", "owasp zap",
            "nessus", "qualys",
            "arduino", "raspberry pi", "esp32", "stm32", "zigbee", "mqtt",
            "graphql", "rest", "soap", "websocket", "grpc", "protobuf",
            "webpack", "babel", "eslint", "prettier", "rollup"
        ]
        doc = nlp(text.lower())
        extracted_skills = set()

        for i in range(len(doc)):
            for skill in skills_list:
                skill_tokens = skill.split()
                if len(skill_tokens) == 1:
                    if doc[i].text == skill:
                        extracted_skills.add(skill)
                else:
                    window = doc[i:i + len(skill_tokens)]
                    if all(t.text == skill_tokens[j] for j, t in enumerate(window)) and len(window) == len(skill_tokens):
                        extracted_skills.add(skill)

        extracted_skills = list(extracted_skills)
        logger.info(f"Extracted skills: {extracted_skills} (Count: {len(extracted_skills)})")
        return extracted_skills
    except Exception as e:
        logger.error(f"Error extracting skills: {e}")
        return []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/sign_up', methods=['POST'])
def sign_up_route():
    try:
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            return jsonify({"error": "Missing email or password"}), 400

        user_id = str(uuid.uuid4())

        existing_user = supabase.table("users").select("email").eq("email", email).execute()
        if existing_user.data:
            return jsonify({"error": "Email already exists"}), 400

        user_data = {
            "user_id": user_id,
            "email": email,
            "created_at": datetime.utcnow().isoformat()
        }
        user_response = supabase.table("users").insert(user_data).execute()
        if not user_response.data:
            return jsonify({"error": "Failed to insert user into database"}), 500

        profile_data = {
            "user_id": user_id,
            "tech_score": 0,
            "tech_max_score": 20,
            "tech_questions_history": [],
            "tech_answers_history": [],
            "tech_feedback_history": [],
            "tech_marks_history": [],
            "hr_score": 0,
            "hr_max_score": 20,
            "hr_emotions": [],
            "hr_soft_skills": [],
            "last_updated": datetime.utcnow().isoformat()
        }
        profile_response = supabase.table("profiles").insert(profile_data).execute()
        if not profile_response.data:
            return jsonify({"error": "Failed to create user profile"}), 500

        session["user_id"] = user_id
        logger.info(f"User signed up with user_id: {user_id}")
        return jsonify({"success": True, "user_id": user_id})
    except Exception as e:
        logger.error(f"Error in sign-up route: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/sign_in', methods=['POST'])
def sign_in_route():
    try:
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            return jsonify({"error": "Missing email or password"}), 400

        user = supabase.table("users").select("user_id").eq("email", email).execute()
        if not user.data:
            return jsonify({"error": "Invalid email or password"}), 400

        user_id = user.data[0]["user_id"]
        session["user_id"] = user_id
        logger.info(f"User signed in with user_id: {user_id}")
        return jsonify({"success": True, "user_id": user_id})
    except Exception as e:
        logger.error(f"Error in sign-in route: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/logout', methods=['POST'])
def logout_route():
    session.pop("user_id", None)
    return jsonify({"success": True})


@app.route('/start_interview', methods=['POST'])
def start_interview():
    global conversation_history, current_interview_type
    global tech_question_count, tech_score, tech_questions_history, tech_answers_history, tech_feedback_history, tech_marks_history
    global hr_question_count, hr_score, hr_emotions_history, hr_soft_skills_history

    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "User not authenticated"}), 401

        interview_type = request.json.get('type')
        if not interview_type:
            return jsonify({"error": "Interview type not provided"}), 400

        conversation_history = []
        current_interview_type = interview_type

        if interview_type == "tech":
            tech_question_count = 0
            tech_score = 0
            tech_questions_history = []
            tech_answers_history = []
            tech_feedback_history = []
            tech_marks_history = []

            question = generate_tech_question()
            tech_questions_history.append(question)
        else:
            hr_question_count = 0
            hr_score = 0
            hr_emotions_history = []
            hr_soft_skills_history = []
            question = generate_hr_question()

        conversation_history.append({"role": "interviewer", "text": question})
        result = {"question": question}

        logger.info(f"Start interview response: {result}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in start_interview: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/submit_response', methods=['POST'])
def submit_response():
    global conversation_history, current_interview_type
    global tech_question_count, tech_score, tech_questions_history, tech_answers_history, tech_feedback_history, tech_marks_history
    global hr_question_count, hr_score, hr_emotions_history, hr_soft_skills_history

    try:
        result = {}

        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "User not authenticated"}), 401

        interview_type = request.form.get('type') or current_interview_type
        if not interview_type:
            return jsonify({"error": "Interview type not provided"}), 400

        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        temp_audio_path = "temp_audio.webm"
        audio_file.save(temp_audio_path)

        audio_path = convert_to_wav(temp_audio_path)
        if not audio_path:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return jsonify({"error": "Failed to convert audio"}), 500

        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)

        try:
            response_text = recognizer.recognize_google(audio)
        except Exception:
            response_text = "Could not understand audio."

        logger.info(f"User answer (transcribed): {response_text}")
        conversation_history.append({"role": "user", "text": response_text})

        pitch, energy = analyze_speech(audio_path)
        image_data = request.form.get('image_data')
        emotions, dominant_emotion = None, None

        if image_data:
            frame = capture_frame(image_data)
            if frame is not None:
                emotions, dominant_emotion = detect_emotion(frame)
            else:
                logger.warning("No frame captured from image data")

        soft_skills = analyze_soft_skills(response_text, pitch, energy, emotions)

        if interview_type == "tech":
            current_question = ""
            for entry in reversed(conversation_history[:-1]):
                if entry["role"] == "interviewer":
                    current_question = entry["text"]
                    break

            mark = gemini_mark_answer(response_text)
            feedback_text = generate_tech_answer_feedback(current_question, response_text, mark)

            tech_score += mark
            tech_question_count += 1

            tech_answers_history.append(response_text)
            tech_feedback_history.append(feedback_text)
            tech_marks_history.append(mark)

            if tech_question_count >= MAX_TECH_QUESTIONS:
                final_message = f"Tech Interview Completed. Your score is {int(tech_score)} out of 20. Check your profile for a detailed report."
                conversation_history.append({"role": "interviewer", "text": final_message})

                update_data = {
                    "tech_score": int(tech_score),
                    "tech_questions_history": tech_questions_history,
                    "tech_answers_history": tech_answers_history,
                    "tech_feedback_history": tech_feedback_history,
                    "tech_marks_history": tech_marks_history,
                    "last_updated": datetime.utcnow().isoformat()
                }

                logger.info(f"Attempting to update tech profile for user_id: {user_id}")
                update_response = supabase.table("profiles").update(update_data).eq("user_id", user_id).execute()
                logger.info(f"Tech update response: {update_response.data}")

                result = {
                    "question": final_message,
                    "completed": True,
                    "score": int(tech_score),
                    "max_score": 20
                }
            else:
                next_question = generate_tech_question(response_text)
                tech_questions_history.append(next_question)
                conversation_history.append({"role": "interviewer", "text": next_question})

                result = {
                    "question": next_question,
                    "completed": False,
                    "latest_feedback": feedback_text
                }

        else:
            mark = gemini_mark_hr_answer(response_text)
            hr_score += mark
            hr_question_count += 1
            hr_emotions_history.append(emotions if emotions else {})
            hr_soft_skills_history.append(soft_skills)

            if hr_question_count >= MAX_HR_QUESTIONS:
                final_message = f"HR Interview Completed. Your score is {int(hr_score)} out of 20. Check your profile for a detailed report."
                conversation_history.append({"role": "interviewer", "text": final_message})

                update_data = {
                    "hr_score": int(hr_score),
                    "hr_emotions": hr_emotions_history,
                    "hr_soft_skills": hr_soft_skills_history,
                    "last_updated": datetime.utcnow().isoformat()
                }
                logger.info(f"Attempting to update HR profile for user_id: {user_id}")
                update_response = supabase.table("profiles").update(update_data).eq("user_id", user_id).execute()
                logger.info(f"HR update response: {update_response.data}")

                result = {
                    "question": final_message,
                    "completed": True,
                    "score": int(hr_score),
                    "max_score": 20
                }
            else:
                next_question = generate_hr_question()
                conversation_history.append({"role": "interviewer", "text": next_question})

                result = {
                    "question": next_question,
                    "completed": False,
                    "emotions": emotions if emotions else {},
                    "dominant_emotion": dominant_emotion if dominant_emotion else "None"
                }

        logger.info(f"Submit response: {result}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in submit_response: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists("temp_audio.webm"):
            os.remove("temp_audio.webm")
        if os.path.exists("response.wav"):
            os.remove("response.wav")


@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    global extracted_skills
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "User not authenticated"}), 401

        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({"error": "No selected file"}), 400

        if file:
            text = extract_text(file)
            if not text:
                return jsonify({"error": "Failed to extract text from resume"}), 500

            skills = extract_skills(text)
            extracted_skills = skills
            logger.info(f"Extracted skills: {extracted_skills}")
            return jsonify({"skills": extracted_skills})

    except Exception as e:
        logger.error(f"Error in upload_resume: {e}")
        return jsonify({"error": str(e)}), 400


@app.route('/profile', methods=['GET'])
def profile():
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "User not authenticated"}), 401

        profile_data = supabase.table("profiles").select("*").eq("user_id", user_id).execute()
        if not profile_data.data:
            return jsonify({"error": "Profile not found"}), 404

        profile = profile_data.data[0]
        report = build_tech_profile_report(profile)
        logger.info(f"Generated tech profile report for user_id {user_id}: {report}")
        return jsonify(report)

    except Exception as e:
        logger.error(f"Error in profile route: {e}")
        return jsonify({"error": "Unable to generate tech report"}), 500


@app.route('/hr_profile', methods=['GET'])
def hr_profile():
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "User not authenticated"}), 401

        profile_data = supabase.table("profiles").select("*").eq("user_id", user_id).execute()
        if not profile_data.data:
            return jsonify({"error": "Profile not found"}), 404

        profile = profile_data.data[0]
        hr_score_value = float(profile["hr_score"])
        max_hr_score = profile["hr_max_score"]

        hr_emotions_history_local = profile["hr_emotions"]
        if isinstance(hr_emotions_history_local, str):
            hr_emotions_history_local = json.loads(hr_emotions_history_local) if hr_emotions_history_local else []

        hr_soft_skills_history_local = profile["hr_soft_skills"]
        if isinstance(hr_soft_skills_history_local, str):
            hr_soft_skills_history_local = json.loads(hr_soft_skills_history_local) if hr_soft_skills_history_local else []

        avg_emotions_description = ""
        avg_emotions = {}

        if hr_emotions_history_local:
            valid_emotion_entries = [emotion_dict for emotion_dict in hr_emotions_history_local if emotion_dict]
            if valid_emotion_entries:
                for emotion_dict in valid_emotion_entries:
                    for emotion, value in emotion_dict.items():
                        avg_emotions[emotion] = avg_emotions.get(emotion, 0) + value

                for emotion in avg_emotions:
                    avg_emotions[emotion] /= len(valid_emotion_entries)

                prominent_emotions = sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True)[:2]
                if prominent_emotions:
                    primary_emotion, primary_value = prominent_emotions[0]
                    if primary_value > 40:
                        avg_emotions_description += f"You came across as quite {primary_emotion} during the interview. "
                    elif primary_value > 20:
                        avg_emotions_description += f"You showed some {primary_emotion} vibes at times. "
                    else:
                        avg_emotions_description += f"You stayed pretty balanced, with a hint of {primary_emotion}. "

                    if len(prominent_emotions) > 1:
                        secondary_emotion, secondary_value = prominent_emotions[1]
                        if secondary_value > 20:
                            avg_emotions_description += f"There was also a touch of {secondary_emotion} in your responses."
            else:
                avg_emotions_description = "Your emotions were pretty neutral throughout—nice and steady!"
        else:
            avg_emotions_description = "No emotional data available from your interview."

        confidence_count = {"High": 0, "Low": 0}
        enthusiasm_count = {"High": 0, "Low": 0}
        avg_positivity = 0

        for skills in hr_soft_skills_history_local:
            confidence_count[skills.get("confidence", "Low")] += 1
            enthusiasm_count[skills.get("enthusiasm", "Low")] += 1
            avg_positivity += skills.get("positivity", 0)

        avg_positivity /= len(hr_soft_skills_history_local) if hr_soft_skills_history_local else 1

        confidence_description = (
            "You sounded confident most of the time—great job keeping your voice steady!"
            if confidence_count["High"] >= confidence_count["Low"]
            else "You seemed a bit hesitant at times; try speaking up a little more next time."
        )

        enthusiasm_description = (
            "Your energy was infectious—you really brought some enthusiasm to the table!"
            if enthusiasm_count["High"] >= enthusiasm_count["Low"]
            else "You could perk up a bit; adding some energy might make your answers pop more."
        )

        positivity_description = (
            "Your responses had a nice positive vibe—very uplifting!"
            if avg_positivity > 0.2
            else "You were fairly neutral; maybe sprinkle in some positivity to shine brighter!"
            if avg_positivity >= -0.2
            else "Things felt a bit downbeat; try focusing on the brighter side in your answers."
        )

        feedback = []
        if hr_score_value < max_hr_score * 0.7:
            feedback.append("Your answers could use a bit more clarity and polish—try structuring them with a clear start, middle, and end.")
        if confidence_count["Low"] > confidence_count["High"]:
            feedback.append("You might want to practice speaking with more confidence; a louder, steady tone can make a big difference.")
        if enthusiasm_count["Low"] > enthusiasm_count["High"]:
            feedback.append("Bring some more enthusiasm to your voice—varying your tone can show you’re engaged and excited.")
        if avg_positivity < 0:
            feedback.append("Try to keep a positive spin on things—it helps leave a great impression!")
        if "angry" in avg_emotions and avg_emotions["angry"] > 20:
            feedback.append("You seemed a bit frustrated at times; staying calm and composed could help you come across even better.")

        overall_summary = f"Overall, you scored {hr_score_value} out of {max_hr_score}, which is "
        if hr_score_value >= max_hr_score * 0.9:
            overall_summary += "fantastic—you’re really shining in these interviews! "
        elif hr_score_value >= max_hr_score * 0.7:
            overall_summary += "solid—you’re doing well with room to polish a few things. "
        else:
            overall_summary += "a good start—there’s definitely potential to build on! "

        areas_to_improve = []
        if hr_score_value < max_hr_score * 0.9:
            if confidence_count["Low"] > confidence_count["High"] or enthusiasm_count["Low"] > enthusiasm_count["High"]:
                areas_to_improve.append("working on your delivery—confidence and enthusiasm can really elevate your presence")
            if avg_positivity < 0.2:
                areas_to_improve.append("adding a bit more positivity to your tone—it can make you more memorable")
            if hr_score_value < max_hr_score * 0.7:
                areas_to_improve.append("structuring your answers more clearly—think about giving concise examples with impact")
            if not areas_to_improve:
                areas_to_improve.append("fine-tuning small details to push your performance to the next level")

        overall_summary += "To improve, focus on " + " and ".join(areas_to_improve) + ". Keep practicing, and you’ll get even stronger!"

        report = {
            "score": hr_score_value,
            "max_score": max_hr_score,
            "emotions": avg_emotions_description,
            "confidence": confidence_description,
            "enthusiasm": enthusiasm_description,
            "positivity": positivity_description,
            "feedback": feedback if feedback else ["You’re doing great—keep it up with consistent practice!"],
            "overall_summary": overall_summary
        }

        logger.info(f"HR profile report: {report}")
        return jsonify(report)

    except Exception as e:
        logger.error(f"Error generating HR profile: {e}")
        return jsonify({"error": "Unable to generate report"}), 500


if __name__ == '__main__':
    app.run(debug=True)