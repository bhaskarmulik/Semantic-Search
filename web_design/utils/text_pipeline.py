import pickle
from sentence_transformers import SentenceTransformer
import torch
from pickle import load

def get_dense_embeddings(text):
    '''
    This function takes in text and returns the embeddings of the text
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('D:/My projects/RAG-based Semantic Search/web_design/utils/multi-qa-mpnet-base-dot-v1').to(device)
    return model.encode(text, convert_to_tensor=True).tolist()

def get_sparse_embeddings(text):
    '''
    This function takes in text and returns the embeddings of the text
    '''
    bm25 = load(open('D:/My projects/RAG-based Semantic Search/web_design/utils/sparse_encoder.pkl', 'rb'))
    return bm25.encode_documents(text)
    