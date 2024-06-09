import os
import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
from database import DatabaseManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# 데이터베이스 경로 설정 및 로드
db_path = 'face_embeddings.db'
db_manager = DatabaseManager(db_path)
names, known_embeddings = db_manager.load_embeddings()

# MTCNN 모델 로드 (얼굴 검출)
mtcnn = MTCNN(keep_all=True)  # 여러 얼굴을 검출하도록 설정

# InceptionResnetV1 모델 로드 (FaceNet)
model = InceptionResnetV1(pretrained='vggface2').eval()

# 얼굴 임베딩 함수
def get_embedding(model, face):
    face = face.unsqueeze(0)
    with torch.no_grad():
        embedding = model(face)
    embedding = embedding[0].cpu().numpy().flatten()
    # Normalize embedding
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('image')
def handle_image(data_image):
    print("Received image data")  # 디버깅 로그 추가
    # 이미지 데이터를 바이트 배열로 변환
    try:
        image_data = base64.b64decode(data_image)
        image_data = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("Image decoded and converted to RGB")  # 디버깅 로그 추가
        boxes, _ = mtcnn.detect(img_rgb)
        print(f"Detected boxes: {boxes}")  # 디버깅 로그 추가

        responses = []

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = img_rgb[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face = cv2.resize(face, (160, 160))
                face = torch.tensor(face).permute(2, 0, 1).float() / 255.0
                embedding = get_embedding(model, face)
                print(f"Face embedding: {embedding}")  # 디버깅 로그 추가

                min_dist = float('inf')
                label = 'Unknown'
                color = 'red'
                accuracy = 0
                for i, known_embedding in enumerate(known_embeddings):
                    dist = cosine(embedding, known_embedding)  # 1차원 배열로 변환
                    if dist < min_dist:
                        min_dist = dist
                        label = names[i]
                        accuracy = (1 - min_dist) * 100  # 유사도를 백분율로 변환

                print(f"Min distance: {min_dist}, Label: {label}, Accuracy: {accuracy}")  # 디버깅 로그 추가

                if accuracy < 80:  # 정확도가 80% 미만이면 Unknown으로 설정
                    label = 'Unknown'
                    color = 'red'
                else:
                    color = 'green'

                responses.append({'label': label, 'box': [x1, y1, x2, y2], 'color': color, 'accuracy': accuracy})

        if not responses:
            responses.append({'label': 'No face detected'})

        emit('response_back', {'faces': responses})
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == '__main__':
    socketio.run(app, debug=True)
