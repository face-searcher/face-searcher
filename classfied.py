import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import sqlite3
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# 얼굴 임베딩 생성기 클래스
class FaceEmbeddingGenerator:
    def __init__(self):
        self.detector = FaceEmbeddingGenerator.FaceDetector()
        self.preprocessor = FaceEmbeddingGenerator.FacePreprocessor()
        self.embedder = FaceEmbeddingGenerator.FaceEmbedder()
    
    class FaceDetector:
        def __init__(self):
            self.mtcnn = MTCNN(keep_all=False, device='cuda:0' if torch.cuda.is_available() else 'cpu')
        
        def detect(self, img):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes, _ = self.mtcnn.detect(img_rgb)
            return boxes

    class FacePreprocessor:
        @staticmethod
        def preprocess(face):
            face = cv2.resize(face, (160, 160))
            face = torch.tensor(face).permute(2, 0, 1).float() / 255.0
            return face

    class FaceEmbedder:
        def __init__(self):
            self.model = InceptionResnetV1(pretrained='vggface2').eval()

        def get_embedding(self, face):
            face = face.unsqueeze(0)
            with torch.no_grad():
                embedding = self.model(face)
            return embedding[0].cpu().numpy()
    
    def process_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None
        boxes = self.detector.detect(img)
        if boxes is None:
            return None
        x1, y1, x2, y2 = map(int, boxes[0])
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            return None
        face = self.preprocessor.preprocess(face)
        return self.embedder.get_embedding(face)

# 데이터베이스 관리 클래스
class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.initializer = DatabaseManager.DatabaseInitializer(db_path)
        self.initializer.init_db()
    
    class DatabaseInitializer:
        def __init__(self, db_path):
            self.db_path = db_path
        
        def init_db(self):
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('''
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id INTEGER PRIMARY KEY,
                        embedding BLOB NOT NULL,
                        label TEXT NOT NULL
                    )
                ''')
                c.execute('''
                    CREATE TABLE IF NOT EXISTS model_storage (
                        id INTEGER PRIMARY KEY,
                        model BLOB
                    )
                ''')
                conn.commit()
    
    class EmbeddingHandler:
        def __init__(self, db_path):
            self.db_path = db_path
        
        def save_embeddings(self, embeddings):
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.executemany('''
                    INSERT INTO embeddings (embedding, label) VALUES (?, ?)
                ''', embeddings)
                conn.commit()
        
        def load_embeddings(self):
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('SELECT embedding, label FROM embeddings')
                rows = c.fetchall()
            embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in rows]
            labels = [row[1] for row in rows]
            return np.array(embeddings), np.array(labels)

    class ModelHandler:
        def __init__(self, db_path):
            self.db_path = db_path
        
        def save_model(self, model):
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                model_blob = pickle.dumps(model)
                c.execute('DELETE FROM model_storage')
                c.execute('INSERT INTO model_storage (model) VALUES (?)', (model_blob,))
                conn.commit()
        
        def load_model(self):
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('SELECT model FROM model_storage')
                model_blob = c.fetchone()[0]
            return pickle.loads(model_blob)

    def save_embeddings(self, embeddings):
        handler = self.EmbeddingHandler(self.db_path)
        handler.save_embeddings(embeddings)
    
    def load_embeddings(self):
        handler = self.EmbeddingHandler(self.db_path)
        return handler.load_embeddings()
    
    def save_model(self, model):
        handler = self.ModelHandler(self.db_path)
        handler.save_model(model)
    
    def load_model(self):
        handler = self.ModelHandler(self.db_path)
        return handler.load_model()

# 얼굴 인식 클래스
class FaceRecognizer:
    def __init__(self, db_manager, threshold=0.5):
        self.db_manager = db_manager
        self.threshold = threshold
        self.detector = FaceRecognizer.FaceDetector()
        self.preprocessor = FaceRecognizer.FacePreprocessor()
        self.embedder = FaceRecognizer.FaceEmbedder()
        self.knn = None
    
    class FaceDetector:
        def __init__(self):
            self.mtcnn = MTCNN(keep_all=False, device='cuda:0' if torch.cuda.is_available() else 'cpu')
        
        def detect(self, img):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes, _ = self.mtcnn.detect(img_rgb)
            return boxes

    class FacePreprocessor:
        @staticmethod
        def preprocess(face):
            face = cv2.resize(face, (160, 160))
            face = torch.tensor(face).permute(2, 0, 1).float() / 255.0
            return face

    class FaceEmbedder:
        def __init__(self):
            self.model = InceptionResnetV1(pretrained='vggface2').eval()

        def get_embedding(self, face):
            face = face.unsqueeze(0)
            with torch.no_grad():
                embedding = self.model(face)
            return embedding[0].cpu().numpy()
    
    class Prediction:
        def __init__(self, knn, threshold):
            self.knn = knn
            self.threshold = threshold
        
        def predict(self, embedding):
            prediction = self.knn.predict([embedding])
            label = prediction[0]
            distances, _ = self.knn.kneighbors([embedding], n_neighbors=1, return_distance=True)
            closest_dist = distances[0][0]
            if closest_dist > self.threshold:
                label = 'Unknown'
            accuracy = (1 - closest_dist) * 100
            return label, accuracy
    
    def load_model(self):
        self.knn = self.db_manager.load_model()
    
    def process_frame(self, frame):
        boxes = self.detector.detect(frame)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                face = self.preprocessor.preprocess(face)
                embedding = self.embedder.get_embedding(face)
                predictor = FaceRecognizer.Prediction(self.knn, self.threshold)
                label, accuracy = predictor.predict(embedding)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, f'{label} ({accuracy:.2f}%)', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return frame
    
    def run_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("오류 : 웹캠 열기 실패")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("오류 : 이미지 캡쳐 실패")
                break
            frame = self.process_frame(frame)
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# 메인 실행 부분
if __name__ == "__main__":
    folder_path = os.path.join(os.path.expanduser("~"), 'Desktop/face-searcher')
    db_path = os.path.join(folder_path, 'face_embeddings.db')
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"폴더 경로를 찾을 수 없습니다: {folder_path}")
    
    face_gen = FaceEmbeddingGenerator()
    db_manager = DatabaseManager(db_path)
    
    dataset_path = os.path.join(folder_path, 'facesDataset')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"데이터셋 경로를 찾을 수 없습니다: {dataset_path}")
    
    # 임베딩 생성 및 저장
    persons = os.listdir(dataset_path)
    embeddings = []
    for person_name in tqdm(persons, desc="사람 처리 중", unit="명"):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue
        images = os.listdir(person_path)
        for image_name in tqdm(images, desc=f"{person_name}의 이미지 처리 중", unit="장", leave=False):
            image_path = os.path.join(person_path, image_name)
            embedding = face_gen.process_image(image_path)
            if embedding is not None:
                embeddings.append((embedding.tobytes(), person_name))
            if len(embeddings) > 100:
                db_manager.save_embeddings(embeddings)
                embeddings = []
    if embeddings:
        db_manager.save_embeddings(embeddings)
    
    # 데이터 로드 및 k-NN 학습
    data, labels = db_manager.load_embeddings()
    data = data / np.linalg.norm(data, axis=1, keepdims=True)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    db_manager.save_model(knn)
    
    # 성능 평가
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # 실시간 얼굴 인식
    recognizer = FaceRecognizer(db_manager)
    recognizer.load_model()
    recognizer.run_webcam()
