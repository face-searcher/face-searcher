import os
import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
import sqlite3

# MTCNN 모델 로드 (얼굴 검출)
mtcnn = MTCNN(keep_all=False)

# InceptionResnetV1 모델 로드 (FaceNet)
model = InceptionResnetV1(pretrained='vggface2').eval()

# 얼굴 임베딩 함수
def get_embedding(model, face):
    face = face.unsqueeze(0)
    with torch.no_grad():
        embedding = model(face)
    return embedding[0].cpu().numpy()

# 데이터셋 경로 설정 (바탕화면의 faces 폴더)
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
dataset_path = os.path.join(desktop_path, 'faces')
print(f"데이터셋 경로: {dataset_path}")

# 경로가 존재하는지 확인하고, 없으면 생성
if not os.path.exists(dataset_path):
    print(f"경로가 존재하지 않습니다: {dataset_path}")
    os.makedirs(dataset_path)
    print(f"경로를 생성했습니다: {dataset_path}")

# 데이터베이스 연결
conn = sqlite3.connect('faces.db')
c = conn.cursor()

# 테이블 생성
c.execute('''
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    embedding BLOB
)
''')

embedding_list = []
label_list = []

# 데이터셋에서 얼굴 이미지 로드 및 임베딩 생성
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        img = cv2.imread(image_path)
        
        # 이미지 로드 실패 시 건너뛰기
        if img is None:
            print(f"이미지를 읽을 수 없습니다: {image_path}")
            continue

        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(f"이미지를 변환할 수 없습니다: {image_path}")
            continue
        
        box, _ = mtcnn.detect(img_rgb)

        if box is not None:
            x1, y1, x2, y2 = map(int, box[0])
            face = img_rgb[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face = cv2.resize(face, (160, 160))
            face = torch.tensor(face).permute(2, 0, 1).float() / 255.0

            embedding = get_embedding(model, face)
            embedding_list.append(embedding)
            label_list.append(person_name)

            # 데이터베이스에 임베딩과 레이블 저장
            c.execute('INSERT INTO faces (name, embedding) VALUES (?, ?)', (person_name, embedding.tobytes()))


# 변경사항 저장 및 연결 종료
conn.commit()
conn.close()


# # 임베딩 및 레이블 배열로 변환
# embeddings = np.array(embedding_list)
# labels = np.array(label_list)

# # npz 파일로 저장
# np.savez('trainEmbeds.npz', x=embeddings, y=labels)

# # npz 파일에서 알려진 얼굴 임베딩 및 레이블 로드
# data = np.load('trainEmbeds.npz')
# known_embeddings = data['x']
# labels = data['y']

# # 웹캠 초기화
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     boxes, _ = mtcnn.detect(frame_rgb)

#     if boxes is not None:
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box)
#             face = frame_rgb[y1:y2, x1:x2]
#             if face.size == 0:
#                 continue
#             face = cv2.resize(face, (160, 160))
#             face = torch.tensor(face).permute(2, 0, 1).float() / 255.0

#             embedding = get_embedding(model, face)

#             min_dist = float('inf')
#             label = 'Unknown'
#             for i, known_embedding in enumerate(known_embeddings):
#                 dist = cosine(embedding, known_embedding)
#                 if dist < min_dist:
#                     min_dist = dist
#                     label = labels[i]

#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(frame, str(label), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#     cv2.imshow('Face Recognition', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
