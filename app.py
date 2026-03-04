from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import cv2
import numpy as np
import os
from flask_cors import CORS
import psycopg2

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Use Environment Variable for Secret Key if available
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'vison_scan_secret_9988')

# Database Configuration (Supabase Pooler)
DBDetails = {
    'host': 'aws-1-ap-south-1.pooler.supabase.com',
    'port': '6543',
    'database': 'postgres',
    'password': '@Visionscan31', 
    'user': 'postgres.wdbbvrooovmlnbohavxq'
}

def get_db_connection():
    return psycopg2.connect(**DBDetails)

# --- GENDER DETECTION SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENDER_PROTOTXT = os.path.join(BASE_DIR, "models", "deploy_gender.prototxt")
GENDER_MODEL = os.path.join(BASE_DIR, "models", "gender_net.caffemodel")
# IMPORTANT: Put this XML in your /models folder!
FACE_CASCADE_PATH = os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml")

# Global variables for models (Initialized as None)
gender_net = None
face_cascade = None
GENDER_LIST = ['Male', 'Female', 'Transgender']

def load_ai_models():
    """Lazy loads models only when needed to prevent Vercel boot-up crashes."""
    global gender_net, face_cascade
    if gender_net is None:
        if not os.path.exists(GENDER_MODEL):
            raise FileNotFoundError(f"Model file missing at: {GENDER_MODEL}")
        gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTOTXT, GENDER_MODEL)
    
    if face_cascade is None:
        if not os.path.exists(FACE_CASCADE_PATH):
            # Fallback if the file is missing in /models
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        else:
            face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    con = None
    try:
        data = request.get_json()
        con = get_db_connection()
        cur = con.cursor()
        cur.execute(
            'INSERT INTO visionscan(name, phonenumber, age, gender) VALUES (%s, %s, %s, %s)',
            (data.get('fullname'), data.get('phonenumber'), data.get('age'), data.get('gender'))
        )
        con.commit()
        cur.close()
        return jsonify({"Success": "Signup Successful"}), 201
    except Exception as e:
        return jsonify({"Error": str(e)}), 500
    finally:
        if con: con.close()

@app.route('/login', methods=['POST'])
def login():
    con = None
    try:
        data = request.get_json()
        name = data.get('fullname')
        phonenumber = data.get('phonenumber')

        con = get_db_connection()
        cur = con.cursor()
        cur.execute("SELECT phonenumber FROM visionscan WHERE name = %s", (name,))
        result = cur.fetchone()
        cur.close()

        if result and str(result[0]) == str(phonenumber):
            session['user'] = name
            return jsonify({"Success": "Login Successful", "redirect": "/camera"}), 200
        
        return jsonify({"Error": "Invalid credentials"}), 401
    except Exception as e:
        return jsonify({"Error": str(e)}), 500
    finally:
        if con: con.close()

@app.route('/camera')
def camera_page():
    if 'user' not in session:
        return redirect(url_for('home'))
    return render_template('camera.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/scan', methods=['POST'])
def scan():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if 'frame' not in request.files:
        return jsonify({"error": "No image received"}), 400
    
    try:
        # 1. Load models inside the request context
        load_ai_models()
        
        # 2. Process image
        file = request.files['frame']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None: 
            return jsonify({"error": "Invalid format"}), 400
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        detected_gender = "Unknown"
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = image[y:y + h, x:x + w]
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.426, 87.768, 114.895), False)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            detected_gender = GENDER_LIST[gender_preds[0].argmax()]
        
        return jsonify({"gender": detected_gender})
    except Exception as e:
        return jsonify({"error": f"Scan failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
