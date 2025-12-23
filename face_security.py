import cv2
import face_recognition
import numpy as np
import pickle
import os
from datetime import datetime
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, render_template, Response, jsonify, request
import threading

app = Flask(__name__)

# Configuration
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
DATABASE_FILE = "face_encodings.pkl"
LOG_FILE = "access_log.csv"
ALERT_EMAIL = "security@example.com"  # Change to your email
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "your_email@gmail.com"  # Change this
SMTP_PASSWORD = "your_app_password"  # Change this

# Global variables
known_face_encodings = []
known_face_names = []
camera = None
access_log = []
alert_sent = False


def load_known_faces():
    """Load known faces from database"""
    global known_face_encodings, known_face_names

    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'rb') as f:
            data = pickle.load(f)
            known_face_encodings = data['encodings']
            known_face_names = data['names']
        print(f"Loaded {len(known_face_names)} known faces")
    else:
        print("No face database found. Starting fresh.")


def save_face_encoding(name, encoding):
    """Save a new face encoding to database"""
    known_face_names.append(name)
    known_face_encodings.append(encoding)

    data = {
        'names': known_face_names,
        'encodings': known_face_encodings
    }

    with open(DATABASE_FILE, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved face for: {name}")


def log_access(name, status, confidence=0):
    """Log access attempts"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp},{name},{status},{confidence:.2f}"
    access_log.append(log_entry)

    # Save to file
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry + "\n")

    print(f"Access {status}: {name} at {timestamp}")


def send_alert(unknown_person_image=None):
    """Send email alert for unauthorized access"""
    global alert_sent

    if alert_sent:
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USERNAME
        msg['To'] = ALERT_EMAIL
        msg['Subject'] = "SECURITY ALERT: Unauthorized Access Detected"

        body = f"""
        Security Alert!

        Unauthorized access detected at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        Check the security system immediately.

        Location: Main Entrance
        System: Face Security System
        """

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()

        alert_sent = True
        print("Alert email sent!")

    except Exception as e:
        print(f"Failed to send alert: {e}")


def recognize_face(frame):
    """Recognize faces in the frame"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        name = "Unknown"
        confidence = 0
        color = (0, 0, 255)  # Red for unknown

        if len(known_face_encodings) > 0:
            # Compare with known faces
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(distances)

            if distances[best_match_index] < 0.6:  # Threshold for recognition
                name = known_face_names[best_match_index]
                confidence = (1 - distances[best_match_index]) * 100
                color = (0, 255, 0)  # Green for authorized

                if name != "Unknown":
                    log_access(name, "GRANTED", confidence)
            else:
                # Save unknown face
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unknown_face_path = os.path.join(UNKNOWN_FACES_DIR, f"unknown_{timestamp}.jpg")
                cv2.imwrite(unknown_face_path, frame)

                log_access("Unknown Person", "DENIED", 0)

                # Send alert in background
                alert_thread = threading.Thread(target=send_alert, args=(unknown_face_path,))
                alert_thread.daemon = True
                alert_thread.start()

        # Draw rectangle and label
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, f"{name} ({confidence:.1f}%)",
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        results.append({
            'name': name,
            'confidence': confidence,
            'location': face_location,
            'color': color
        })

    return frame, results


def generate_frames():
    """Generate video frames for streaming"""
    global camera

    if camera is None:
        camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Process frame for face recognition
        processed_frame, _ = recognize_face(frame)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def get_access_logs(limit=50):
    """Get recent access logs"""
    if not os.path.exists(LOG_FILE):
        return []

    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()[-limit:]

    logs = []
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 4:
            logs.append({
                'timestamp': parts[0],
                'name': parts[1],
                'status': parts[2],
                'confidence': parts[3]
            })

    return logs[::-1]  # Reverse to show newest first


# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/logs')
def get_logs():
    limit = request.args.get('limit', 50, type=int)
    logs = get_access_logs(limit)
    return jsonify(logs)


@app.route('/api/stats')
def get_stats():
    stats = {
        'known_faces': len(known_face_names),
        'total_logs': len(access_log) if access_log else 0,
        'system_status': 'Active',
        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return jsonify(stats)


@app.route('/api/add_face', methods=['POST'])
def add_face():
    try:
        data = request.json
        name = data.get('name')

        if not name:
            return jsonify({'error': 'Name is required'}), 400

        # Capture current frame and add to known faces
        if camera is not None:
            success, frame = camera.read()
            if success:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_frame)

                if len(face_encodings) > 0:
                    save_face_encoding(name, face_encodings[0])
                    return jsonify({'success': True, 'message': f'Face added for {name}'})

        return jsonify({'error': 'No face detected'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def init_system():
    """Initialize the security system"""
    # Create directories if they don't exist
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)

    # Load known faces
    load_known_faces()

    print("Security System Initialized")
    print(f"Known faces: {len(known_face_names)}")


if __name__ == '__main__':
    init_system()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)