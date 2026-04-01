# step 1: Import Lbraries and load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

#pip install pipreqs


## Load the IMDB datasets word index
word_index = imdb.get_word_index()
reverse_word_index={value: key for key ,value in word_index.items()}


# Load the pre_trained model with Relu activation
# model=load_model('model.h5')
model=load_model('clean_model.h5')
# model.summary()


# Step 2: helper Fnctions
# fucntion to decode reviews
def decode_review(encoded_review): #we get the word here
  return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])


# Fucntion to preprocess user input # form word to number
def preprocess_text(text):
  # Define the maximum vocabulary size based on the model's embedding layer (from error message)
  words = text.lower().split()
  encoded_review = [word_index.get(word, 2) + 3 for word in words]
  padded_review = pad_sequences([encoded_review], maxlen=500)
  return padded_review



### predicition Function
def predict_sentiment(review):
  preprocess_input=preprocess_text(review)

  prediction=model.predict(preprocess_input)

  sentiment = 'Positive' if prediction[0][0] >0.5 else 'Negative'

  return sentiment,prediction[0][0]





# step 4 : user input and prediciton
# example review for prediciton
# example_review = "This movie was fantastic! The acting was grea and he plot was thrilling."

# sentiment,score=predict_sentiment(example_review)


# print(f"Rview: {example_review}")
# print(f'Sentimnt: {sentiment}')
# print(f'Predicion Score: {score}')

## streamlit app
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as psoitive or negative.')


# user Input

user_input=st.text_area('Movie Review')

if st.button('Classify'):
   preprocess_input=preprocess_text(user_input)

   prediction=model.predict(preprocess_input)

   sentiment = 'Positive' if prediction[0][0] >0.5 else 'Negative'

   # Display the result
   st.write(f'SEntiment: {sentiment}')
   st.write(f'Prediction SCore: {prediction[0][0]}')
else :
   st.write('Please enter a movie review')   

