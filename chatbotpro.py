import nltk
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# # import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()

# Download required NLTK data
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('punkt_tab')

data = pd.read_csv('Mental_Health_FAQ.csv').drop("Question_ID", axis=1)




def preprocess_text(text):

  sentences = nltk.sent_tokenize(text)


  preprocessed_sentences = []
  for sentence in sentences:
      tokens = [lemmatizer.lemmatize(word.lower())for word in nltk.word_tokenize(sentence) if word.isalnum()]


      preprocessed_sentence = ' '.join(tokens)
      preprocessed_sentences.append(preprocessed_sentence)
  
  return ' '.join(preprocessed_sentences)

data['Tokenized Questions'] = data['Questions'].apply(preprocess_text)
# data

xtrain = data['Tokenized Questions'].to_list()
xtrain

tfidf_vectorizer = TfidfVectorizer()
corpus = tfidf_vectorizer.fit_transform(xtrain)


def responder(text_input):
  patient_input_processed = preprocess_text(text_input)
  vectorized_patient_input = tfidf_vectorizer.transform([patient_input_processed])
  similarity_score = cosine_similarity(vectorized_patient_input, corpus)
  argument_maximum = similarity_score.argmax()
  print(data['Answers'].iloc[argument_maximum])


bot_greeting = ["Hello and welcome! I'm here to support you. Whether you need a listening ear or helpful resources, you're not alone. How can I assist you today?",
                "Hi there! I'm glad you're here. Taking the first step toward mental well-being is important, and I'm here to help in any way I can. How are you feeling today?",
                "Welcome! Your well-being matters and this is a safe space for you. Let me know how I can support you on your journey.",
                "Hey! I'm here to listen and provide guidance. Whether you're feeling overwhelmed or just need someone to talk to, you're in the right place.What do you have on your mind?"
]

bot_farewell= ["Take care of yourself! Remember, you're not alone, and support is always here when you need it. Wishing you peace and strength until next time."
              "It was great talking with you. Keep prioritizing your well-being, and don’t hesitate to reach out whenever you need support. Take care! 🌿"
              "You're doing great! Keep going, and remember that small steps make a big difference. I’m here whenever you need to chat. Stay well! 😊"
              "Goodbye for now! Remember to breathe, take breaks, and seek support whenever you need it. You’ve got this! See you soon. 🌼"]


human_greeting= ['hi','hello', 'good day', 'hey', 'hola' ]

human_exit = ['thank you', 'thanks', 'bye', 'goodbye', 'quit']


import random
random_greeting = random.choice(bot_greeting)
random_farewell = random.choice(bot_farewell)

#------------------------STEAMLIT--------------------------

st.markdown("<h1 style = 'color: #DD5746; text-align: center; font-size: 60px; font-family: Sans serif'>MENTAL HEALTH FACILITY CHATBOT</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FFC470; text-align: center; font-family: Serif '>BUILT by LOLA</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html = True)

st.header('Project Background Information', divider = True)
st.write('A mental health chatbot is an AI-powered program designed to provide emotional support, coping strategies, and mental health resources to users. It serves as a first point of contact, offering a safe and non-judgmental space for individuals to express their feelings. The chatbot can assist with stress management, mindfulness exercises, and crisis support while guiding users toward professional help when needed. Available 24/7, it enhances accessibility to mental health care, ensuring that support is always within reach.')

st.markdown("<br>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html = True)
col1, col2 = st.columns(2)
col2.image('pngwing.com.png')

userPrompt = st.chat_input('Please, ask Your Question')
if userPrompt:
   col1.chat_message("ai").write(userPrompt)

   userPrompt = userPrompt.lower()
   if userPrompt in human_greeting:
      col1.chat_message("human").write(random_greeting)
   elif userPrompt in human_exit:
      col1.chat_message("human").write(random_farewell)
   else:
      proUserinput = preprocess_text(userPrompt)
      vect_user = tfidf_vectorizer.transform([proUserinput])
      similarity_scores = cosine_similarity(vect_user, corpus)
      most_similar_index = np.argmax(similarity_scores)
      col1.chat_message("human").write(data['Answers'].iloc[most_similar_index])