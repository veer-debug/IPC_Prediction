from flask import Flask,render_template,redirect,request,session
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

app=Flask(__name__)

@app.route('/')
def predict_legal_section_and_punishment(user_input,):
    nlp = spacy.load("en_core_web_sm")   
    data = pd.read_csv('../IPC_Prediction/ipc_sections.csv')
    data['Entities'] = data['Offense'].apply(lambda x: ' '.join([ent.text.lower() for ent in nlp(x).ents]))
    flattened_data = data[['Section', 'Punishment', 'Entities']].explode('Entities')


    label_encoder = LabelEncoder()
    flattened_data['Section_Label'] = label_encoder.fit_transform(flattened_data['Section'])

    flattened_data['Entities'] = flattened_data['Entities'].fillna('')
    tfidf_vectorizer = TfidfVectorizer(lowercase=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(flattened_data['Entities'])
    classifier = SVC(kernel='linear')
    
    classifier.fit(tfidf_matrix, flattened_data['Section_Label'])
    user_entities = ' '.join([ent.text.lower() for ent in nlp(user_input).ents])
    user_tfidf = tfidf_vectorizer.transform([user_entities])
    predicted_label = classifier.predict(user_tfidf)
    predicted_section = label_encoder.inverse_transform(predicted_label)
    punishment = data[data['Section'] == predicted_section[0]]['Punishment'].iloc[0]

    return user_input,predicted_section[0], punishment



        