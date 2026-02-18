import os
import time
import random
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "q_quora.csv"
MODEL_PATH = BASE_DIR / "models" / "doc2vec_q_quora_queries.model"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_queries_from_csv(filepath):
    df = pd.read_csv(filepath)
    queries = df.iloc[:, 3].dropna().tolist()
    return queries


def preprocess_query(query):
    return simple_preprocess(str(query), deacc=True)

def create_tagged_documents(queries):
    tagged_docs = []
    for i, query in enumerate(queries):
        tokens = preprocess_query(query)
        tag = f'QUERY_{i}'
        tagged_docs.append(TaggedDocument(words=tokens, tags=[tag]))
    return tagged_docs

queries = load_queries_from_csv(DATA_PATH)
tagged_documents = create_tagged_documents(queries)


model = Doc2Vec(vector_size=300, window=5, epochs=5, dm=0, dbow_words=1)
model.build_vocab(tagged_documents)

start_time = time.time()

model.train(
    tagged_documents,
    total_examples=model.corpus_count,
    epochs=model.epochs
)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")
print(f"Model has {len(model.dv)} document vectors")

model.save(str(MODEL_PATH))