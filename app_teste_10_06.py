import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import warnings
#import lightgbm
from lightgbm import LGBMClassifier
import lightgbm as ltb
import plotly.express as px
import plotly.graph_objects as go

from PIL import Image

# Page setting
st.set_page_config(page_title='Dashboard Scoring Credit - Prêt à dépenser ',  layout='wide')


@st.cache(ttl=60*5, suppress_st_warning = True)

#telecharger la base de données et le modèle entrainé:

def load_train_dataset():
          train_dataset = joblib.load('train_dataset.pkl')   
          train_dataset['SK_ID_CURR'] = train_dataset.index        
          return train_dataset

def load_model():
          model_lgbm = joblib.load('lgbm_model_trained.pkl')
          return model_lgbm

def get_traindataset(id):
          id = int(id)
          X = train_dataset[train_dataset['SK_ID_CURR'] == id]
          return X

def calculate_probability(train_dataset):
           probability = lgbm_model.predict_proba(X)[:,1]
           return probability 
def calculate_target(train_dataset):
           target_predit = lgbm.model.predict(X)
           return target_predit
          
                    

#header
t1, t2 = st.columns((0.07,1)) 


img =Image.open('Logo_pad.PNG')
graphique_shap_importance = Image.open('Shap_importance.png')
df = joblib.load('df_complet.pkl')
                    
t1.image(img, width = 120) #logo
t2.title("Dashboard Scoring Credit ") # Titre du dashboard 

# Filtre pour choisir le client: 
#df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(str) # transformer l'ID en string 
id_client = st.selectbox('Selectionnez un Id client', df.index, help = 'Choisissez un seul id client')

id = int(id)
X = train_dataset[train_dataset['SK_ID_CURR'] == id]
probability = lgbm_model.predict_proba(X)[:,1]
st.write('Probabilité de defaut de paiement:', str(round(probability *100)) +'%')
                  
#affichage de la prédiction
#chaine = '**profil:**' + str(type_client) 
#st.markdown(chaine)

#chaine2 = '**Probabilité de defaut de paiement:** {}%'.format(round(probability*100)
#st.markdown(chaine2)
