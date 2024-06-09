import sqlite3
import numpy as np
import pickle

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
    
    class EmbeddingHandler:
        def __init__(self, db_path):
            self.db_path = db_path
        
        def load_embeddings(self):
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('SELECT embedding, label FROM embeddings')
                rows = c.fetchall()
            embeddings = [np.frombuffer(row[0], dtype=np.float32).flatten() for row in rows]
            labels = [row[1] for row in rows]
            embeddings = np.array(embeddings)
            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return labels, embeddings

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

    def load_embeddings(self):
        handler = self.EmbeddingHandler(self.db_path)
        return handler.load_embeddings()
    
    def save_model(self, model):
        handler = self.ModelHandler(self.db_path)
        handler.save_model(model)
    
    def load_model(self):
        handler = self.ModelHandler(self.db_path)
        return handler.load_model()
