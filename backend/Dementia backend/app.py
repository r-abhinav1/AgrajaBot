from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from scripts.pipeline_classifier import DementiaPipeline
import warnings
import subprocess

# Firebase Admin / Firestore (optional)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except Exception:
    FIREBASE_AVAILABLE = False

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

pipeline = DementiaPipeline(model_path=os.path.join(os.path.dirname(__file__), 'models', 'CNN_final.pth'), debug=False)

# Initialize Firestore if available and credentials file exists
db = None
if FIREBASE_AVAILABLE:
    try:
        cred_path = os.path.join(os.path.dirname(__file__), 'firebase_credentials.json')
        if os.path.isfile(cred_path):
            if not firebase_admin._apps:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
            db = firestore.client()
        else:
            print("[WARN] firebase_credentials.json not found. Firestore disabled.")
    except Exception as e:
        db = None
        print(f"[WARN] Firebase initialization failed: {e}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/test',methods=['POST'])
def test():
    return "working"

@app.route('/dementia', methods=['POST'])
def test_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'no selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'file type not allowed'}), 400
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # If .m4a, convert to 16kHz mono WAV via ffmpeg CLI to avoid pydub subprocess issues
    classify_path = save_path
    temp_converted = None
    if filename.lower().endswith('.m4a'):
        base = os.path.splitext(filename)[0]
        temp_converted = os.path.join(app.config['UPLOAD_FOLDER'], f"{base}.wav")
        ffmpeg_bin = os.environ.get('FFMPEG_BIN', 'ffmpeg')
        cmd = [
            ffmpeg_bin,
            '-y',
            '-i', save_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            temp_converted,
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                return jsonify({
                    'success': False,
                    'filename': filename,
                    'error': 'ffmpeg conversion failed',
                    'stderr': res.stderr
                }), 422
            classify_path = temp_converted
        except FileNotFoundError:
            return jsonify({
                'success': False,
                'filename': filename,
                'error': 'ffmpeg not found. Set FFMPEG_BIN env var or install ffmpeg.'
            }), 422
    # Optional fields for Firestore logging
    userdocid = request.form.get('userDocId') or request.form.get('userdocid')
    agrajadocid = request.form.get('agrajaDocId') or request.form.get('agrajadocid')

    # Run classification
    try:
        result = pipeline.classify(classify_path)
        print(result)

        # Firestore writes (best-effort, non-blocking on failure)
        dementia_doc_id = None
        if db is not None and agrajadocid:
            try:
                # Determine agraja document path
                if userdocid:
                    agraja_ref = db.document(f"users/{userdocid}/agraja/{agrajadocid}")
                else:
                    agraja_ref = db.document(f"agraja/{agrajadocid}")

                # Create a new dementia log document
                dementia_ref = db.collection('dementia').document()
                dementia_data = {
                    'agrajaDocId': agrajadocid,
                    'timestamp': firestore.SERVER_TIMESTAMP,
                    'confidenceScore': float(result.get('confidence') or 0.0),
                    'classification': result.get('category')
                }
                if userdocid:
                    # Store userDocId as a DocumentReference to users/{userDocId}
                    user_ref = db.document(f"users/{userdocid}")
                    dementia_data['userDocId'] = user_ref
                dementia_ref.set(dementia_data)
                dementia_doc_id = dementia_ref.id

                # Update agraja document last status
                agraja_ref.set({
                    'last_dementia_status': result.get('category'),
                    'last_confidence_score': float(result.get('confidence') or 0.0)
                }, merge=True)
            except Exception as fe:
                print(f"[WARN] Firestore write failed: {fe}")

        return jsonify({
            'success': True,
            'filename': filename,
            'confidence': result.get('confidence'),
            'category': result.get('category'),
            'dementiaDocId': dementia_doc_id
        }), 200
    except RuntimeError as e:
        # Common case: ffmpeg missing for .m4a files
        message = str(e)
        status = 422 if 'ffmpeg' in message.lower() else 500
        return jsonify({
            'success': False,
            'filename': filename,
            'error': message
        }), status
    except Exception as e:
        # Generic failure
        return jsonify({
            'success': False,
            'filename': filename,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run("0.0.0.0",debug=True)
