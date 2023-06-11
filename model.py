from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import pandas as pd
import tensorflow as tf
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def stemSentence(sentence):
    porter = PorterStemmer()
    token_words = word_tokenize(sentence)
    stem_sentence = [porter.stem(word) for word in token_words]
    return ' '.join(stem_sentence)

def classify(product):
    if not hasattr(classify, "model"):
        model = tf.keras.models.load_model('my_model.h5')
        classify.model = model 
    else:
        model = classify.model
    if not hasattr(classify, "tokenizer"):
        tokenizer = Tokenizer()
        data = pd.read_csv('data3.csv')
        data['nama'] = data['nama'].apply(lambda text: stemSentence(text))
        tokenizer.fit_on_texts(data['nama'])
        classify.tokenizer = tokenizer
    else: 
        tokenizer = classify.tokenizer        
    text = stemSentence(product)
    text = tokenizer.texts_to_sequences([text])
    padded_input_sequence = pad_sequences(text, maxlen=25)
    predictions = model.predict(padded_input_sequence)
    predicted_class = predictions.argmax(axis=-1)[0]
    print(predicted_class)
    return predicted_class

