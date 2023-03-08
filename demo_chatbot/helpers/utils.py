from helpers.constant import *
from typing import List, Dict
import pickle
import os

path_saved_model = os.path.join(os.getcwd(),SAVED_MODEL_PATH)

def preprocess_text(text):
    return text

def load_model():
    loaded_vectorizer = pickle.load(open(os.path.join(path_saved_model,'chatbot_tfidf_vectorizer.pkl'),'rb'))
    loaded_model = pickle.load(open(os.path.join(path_saved_model,'chatbot_model.pkl'),'rb'))
    return loaded_model,loaded_vectorizer

def vectorizer(vectorizer):
    return vectorizer

def inference(model, vectorizer, text) -> dict:
    if not model:
        # dummy value if model not exist
        inference_result = [0,0,0,0.2,0.8]
    else:
        x = vectorizer.transform([text])
        inference_result = model.predict_proba(x)[0]

    max_proba = inference_result.max()
    index_result = inference_result.argmax(axis=0)
    
    max_label = MAP_INDEX_TO_LABEL[index_result]
    
    return {'max_proba' : max_proba, 'max_label' : max_label} # ex {'max_proba' : 0.8, 'max_label' : 'ACCOUNT' }


def is_above_threshold(inference_result : dict) -> bool:
    if inference_result['max_proba'] > THRESHOLD[inference_result['max_label']]:
        return True
    else:
        return False
    
def exit_chat(text : str) -> bool:
    if text == EXIT_CHAT:
        return True
    else:
        return False

def short_chat_rule(text : str) -> bool:
    if len(text.split()) < 4:
        return True
    else:
        return False 