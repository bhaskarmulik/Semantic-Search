#First lets import the necessary libraries
from ast import List
import numpy as np
import pickle as pkl
import torch
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
#Now we will define a class that initializes the text pipeline

class TextPipeline(SentenceTransformer):

    '''
    This class is used to get the text embeddings from the SentenceTransformer model
    For this, we first load the model ,tokenizer and vectorizer
    We then tokenize the text and then pass through the TF-IDF vectorizer to get the sparse vector
    Shape of sparse vector is (num_of_vectors_passed, num_of_features = 3348)
    We then get the sentence embeddings from the model
    Shape of sentence embeddings is (num_of_vectors_passed, 768)
    '''


    def __init__(self, path_to_tokenizer, path_to_model, path_to_vectorizer):
        super().__init__()
        self.tokenizer = self.load_tokenizer(path_to_tokenizer)     #initialize the tokenizer
        self.model = self.load_model(path_to_model)            #initialize the model
        self.vectorizer = pkl.load(open(path_to_vectorizer, 'rb'))

    def load_tokenizer(self, path_to_tokenizer):
        return Tokenizer.from_file(path=path_to_tokenizer)
    
    def load_model(self, path_to_model):
        return SentenceTransformer(path_to_model)
    
    def sentence_embeddings(self, input_text):
        return self.model.encode(input_text)
    
    def tokenized_text(self, input_text):
        return self.tokenizer.tokenize(input_text)
    
    def tfidf_embeddings(self, tokens):
        vec =  self.vectorizer.transform(tokens).toarray()
        return torch.tensor(vec).float()
    
    def text_to_vec(self, input_text):
        tfidf_vec = self.tfidf_embeddings(self.tokenized_text(input_text))
        sentence_vec = self.sentence_embeddings(input_text)
        return tfidf_vec, sentence_vec
    
