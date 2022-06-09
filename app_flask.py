from flask import Flask, render_template, jsonify, request, flash, redirect, url_for
from flask import Flask
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

app = Flask(__name__)

#load model
lgbm_model = joblib.load ('C:/Users/kathl/Desktop/Projet7_OP/github/lgbm_model.pkl')

#load_df
path_df = 'C:/Users/kathl/Desktop/Projet7_OP/github/trainset_final.csv'
dataframe = pd.read_csv(path_df)
path_df_final = 'C:/Users/kathl/Desktop/Projet7_OP/github/trainset_final.csv'
df_final = pd.read_csv(path_df_final)


@app.route('/')
def hello ():
    return "Depoyement de la prédiction des probabilités de défaut de paiement des crédit"
    
@app.route('/predict', methods =['POST','GET'])

def predict(ID, dataframe):
    '''Renvoie la prediction a partir du modele ainsi que les probabilites d\'appartenance à chaque classe'''
    ID = int(ID)
    X = dataframe[dataframe['SK_ID_CURR'] == ID]
    X = X.drop(['SK_ID_CURR'], axis = 1)
    print('prediction shape X: ', X.shape)  
    
    prediction = lgbm_model.predict(X)
    proba = lgbm.model.predict_proba(X)[:,1]
    output = (proba)
    return prediction(ID, conditions(output))
    
    def conditions (output):
       if output >= float(0.48):
        return 'Client à risque. \n La probabilité de risque de défaut de paiement du crédit est de {}'.format(output)
       if output < float(0.48):
        return  'Client peu risqué. \n La La probabilité de risque de défaut de paiement du crédit est de {}'.format(output)  
        
    if __name__ == '__main__':
       app.run(debug=True)  
    
# Changer directory python: cd /d C:\Users\kathl\Desktop\Projet7_OP\github

# pour éxecuter à nouveau: http://127.0.0.1:5000 (Press CTRL+C to quit) 