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

def load_test_dataset():
          test_dataset = joblib.load('test_dataset_complet.pkl')   
          test_dataset['SK_ID_CURR'] = test_dataset.index        
          return train_dataset

def load_model():
          model_lgbm = joblib.load('lgbm_model_trained.pkl')
          return model_lgbm

def get_test_dataset(id):
          id = int(id)
          X = test_dataset[test_dataset['SK_ID_CURR'] == id]
          return X

def calculate_probability(test_dataset):
           probability = lgbm_model.predict_proba(X)[:,1]
           return probability 
def calculate_target(test_dataset):
           target_predit = lgbm.model.predict(X)
           return target_predit
          
                    

#header
t1, t2 = st.columns((0.07,1)) 


img =Image.open('Logo_pad.PNG')
graphique_shap_importance = Image.open('Shap_importance.png')
df = joblib.load('dataset_test_complet.pkl')
                    
t1.image(img, width = 120) #logo
t2.title("Dashboard Scoring Credit ") # Titre du dashboard 

# Filtre pour choisir le client: 
#df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(str) # transformer l'ID en string 
id_client = st.selectbox('Selectionnez un Id client', df.index, help = 'Choisissez un seul id client')

lgbm_model = joblib.load('lgbm_model_trained.pkl')
test_set = joblib.load('test_set.pkl')   
probability = lgbm_model.predict_proba(test_set) # executer le prédict sur le jeu de données inédit pour nous données la probabilité de défaut de paiement

probability = pd.DataFrame(probability, columns= ["0", "1"], index= df.index)
probability["id_client"] = probability.index
probability = probability[["id_client", "1"]]

prob = probability[(probability["id_client"] ==id_client) & (probability["1"])]
prob = prob[['1']]

chaine = '**Risque de défault :**' + str(prob) + '% de risque de défaut'
st.markdown(chaine)

#type de client:
dt = probability
dt["type_de_client"] = "p"
dt["type_de_client"] = np.where((dt['1']>= 0.48), "client à risque", dt['type_de_client'])
dt['type_de_client'] = np.where((dt['1']<0.48), "client peu risqué", dt['type_de_client'])
type_de_client = dt[(dt['id_client']==id_client) & (dt['type_de_client'])]
type_de_client = type_de_client['type_de_client']
          
chaine2 = '**type de client :**' + str(type_client)
st.markdown(chaine2)


                                 
df['type_de_client'] = dt['type_de_client']

df["id"] = df.index

# PARTIE GRAPHIQUE 

g1, g2, g3 = st.columns((1,1,1))


# 1er graph:
g1.subheader("Ranking des features importances avec SHAP ") # Titre du dashboard 
g1.image(graphique_shap_importance, width = 500) #graphique de features importance avec Shap


# 2eme graph:

import plotly.express as px
import plotly.graph_objects as go

fig1 = px.box(df, x= "type_de_client", y= "PAYMENT_RATE", title= f'Payment Rate par Target')
fig1.update_traces(marker_color='#264653')
fig2 =  px.scatter(x = df['type_de_client'][(df["id"] == id_client)], y = df['PAYMENT_RATE'][(df["id"] == id_client)])
fig2.update_traces(marker_color= 'red')
fig3 = go.Figure(data=fig1.data + fig2.data)
#fig3.update_layout(title_text="Payement Rate par rapport au type de client ",title_x=0,margin= dict(l=5,r=5,b=10,t=30), yaxis_title=None, xaxis_title=None)
g2.subheader("Payment Rate x Client_Type")
g2.plotly_chart(fig3, use_container_width=False)

# 3eme graph  

fig4 = px.box(df, x= "CODE_GENDER", y= "DAYS_BIRTH")
fig4.update_traces(marker_color='#264653')
fig5 =  px.scatter(x = df['CODE_GENDER'][(df["id"] == id_client)], y = df['DAYS_BIRTH'][(df["id"] == id_client)])
fig5.update_traces(marker_color= 'red')
fig6 = go.Figure(data=fig4.data + fig5.data)
g3.subheader("DAYS BIRTH x GENDER_CODE")
g3.plotly_chart(fig6, use_container_width=True)

# 2ème ligne de graphs:

l1, l2, l3 = st.columns((1,1,1))

#4ème graph : 
fig7 = px.box(df, x= "NAME_FAMILY_STATUS_Married", y= "EXT_SOURCE_2")
fig7.update_traces(marker_color='#264653')
fig8 =  px.scatter(x = df['NAME_FAMILY_STATUS_Married'][(df["id"] == id_client)], y = df['EXT_SOURCE_2'][(df["id"] == id_client)])
fig8.update_traces(marker_color= 'red')
fig9 = go.Figure(data=fig7.data + fig8.data)
l1.subheader("EXT_SOURCE_2 x Marital Situation (maried)")
l1.plotly_chart(fig9, use_container_width=True)

#5ème graph : 
fig10 = px.box(df, x= "NAME_EDUCATION_TYPE", y= "EXT_SOURCE_2")
fig10.update_traces(marker_color='#264653')
fig11 =  px.scatter(x = df['NAME_EDUCATION_TYPE'][(df["id"] == id_client)], y = df['EXT_SOURCE_2'][(df["id"] == id_client)])
fig11.update_traces(marker_color= 'red')
fig12 = go.Figure(data=fig10.data + fig11.data)
l2.subheader("EXT_SOURCE_2 x Education Type")
l2.plotly_chart(fig12, use_container_width=True)

#6ème graph : 
fig10 = px.box(df, x= "NAME_EDUCATION_TYPE", y= "EXT_SOURCE_3")
fig10.update_traces(marker_color='#264653')
fig11 =  px.scatter(x = df['NAME_EDUCATION_TYPE'][(df["id"] == id_client)], y = df['EXT_SOURCE_3'][(df["id"] == id_client)])
fig11.update_traces(marker_color= 'red')
fig12 = go.Figure(data=fig10.data + fig11.data)
l3.subheader("EXT_SOURCE_3 x Education Type")
l3.plotly_chart(fig12, use_container_width=True)


#ajout informations concernant variables:

k1, k2, k3 = st.columns((1,1,1))
k1.subheader("Explication des variables")

'''
\n\
* **RANKING de Features Importance avec SHAP** : les valeurs de Shap sont représentées pour chaque variable dans leur ordre d’importance, chaque point représente une valeur de Shap, les points rouges représentent des valeurs élevées de la variable et les points bleus des valeurs basses de la variable
* **EXT_SOURCE_2, EXT_SOURCE_3** : Score Normalisé - Source Externe \n\
* **CLIENT TYPE** : Les clients avec une probabilité de défaut de paiement supérieur à 48% sont considerés des clients à risque et ceux avec une probabilité sont considerés clients peu risqués \n\
* **GENDER_CODE** : M - Masculin / F - Feminin \n\
* **Status Marital: Marié(e)** : valeur moyenne pour l'ensemble des clients en défaut\n\
* **DAYS_BIRTH** : Age du client (en jours) au moment de la demande de crédit\n\n\
'''
