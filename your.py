import streamlit as st
from firebase_admin import firestore
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import polars as pl
import csv
import pandas as pd
import numpy as np
import math
import re
import random
#from ydata_profiling import ProfileReport
#import seaborn as sns

#Entrenamiento del modelo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay, roc_curve, roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

#Visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

  
def app():
    #db=firestore.client()
    user_timestamp=pd.read_csv('userid-timestamp-artid-artname-traid-traname.tsv',sep='\t')
    user_profile = pd.read_csv('userid-profile.tsv',sep='\t')

    try:
        st.title('Historico de rating del usuario: '+st.session_state['username'] )

        #Dibuja tabla principal
        #st.dataframe(user_timestamp, use_container_width=True)
        

      

      

        #dibuja df usuarios
        #st.dataframe(user_profile, use_container_width=True)

        
        #user_timestamp.isna().sum()
        #user_timestamp['artist_id'].nunique()
        #user_timestamp['track_id'].nunique()        
        #user_timestamp.dropna(subset=['artist_id', 'track_id'], inplace = True)
        #user_timestamp.shape
        #user_profile.isna().sum()
        #user_timestamp.dropna(subset=['artist_id', 'track_id'], inplace = True)
        
        #st.text('Tabla usuarios: ')
        user_profile.rename(columns = {'#id':'user_id'}, inplace = True)
        #st.dataframe(user_profile, use_container_width=True)

        #st.text('Tabla canciones: ')
        #st.dataframe(user_timestamp, use_container_width=True)

        #st.text('Tabla combinada: ')
        
        user_timestamp.rename(columns = {'user':'user_id',"artist_id.1":"track_id"}, inplace = True)
        #user_timestamp['artist_id'].nunique()
  

        user_profile.isna().sum()
        user_timestamp.isna().sum()
        
        #user_timestamp.dropna(subset=['artist_id', 'track_id'], inplace = True)
        #st.text('Tabla user_timestamp: ')
        #st.dataframe(user_timestamp, use_container_width=True)

        uput = pd.merge(user_profile, user_timestamp, on='user_id')
        
        #dibuja rating
        #st.text('Tabla rating 1: ')
        #st.dataframe(uput, use_container_width=True)

        count = uput.groupby(['user_id', 'track_id'], as_index=False).agg({'timestamp': 'count'}).rename(columns={'timestamp': 'count'})
        #uput = uput[['user_id', 'gender', 'age', 'country', 'artist_id', 'artis', 'track_id', 'song']].drop_duplicates()

        #st.text('Tabla rating 2: ')
        #st.dataframe(uput, use_container_width=True)

        uput = pd.merge(uput, count, on=['user_id', 'track_id'])
        uput = pd.merge(user_profile, user_timestamp, on='user_id')

        #st.text('Tabla rating 3: ')
        #st.dataframe(uput, use_container_width=True)

        count = uput.groupby(['user_id', 'track_id'], as_index=False).agg({'timestamp': 'count'}).rename(columns={'timestamp': 'count'})
        uput = uput[['user_id', 'gender', 'age', 'country', 'artist_id', 'artist', 'track_id', 'song']].drop_duplicates()
        uput = pd.merge(uput, count, on=['user_id', 'track_id'])

        #st.text('Tabla rating 4: ')
        #st.dataframe(uput, use_container_width=True)

        scaler = MinMaxScaler()
        df_intermedio = pd.DataFrame()
        for i in uput['user_id'].unique():
            uput_df = uput[uput['user_id'] == i]
            uput_df['rating'] = scaler.fit_transform(uput_df[['count']])
            df_intermedio=pd.concat([df_intermedio,uput_df], ignore_index=True)

        df_intermedio=df_intermedio[df_intermedio['user_id']==st.session_state['username']].sort_values(by=['rating'], ascending=False)

        #dibuja rating
        df_filter=df_intermedio[['artist','song','count','rating']]
        st.dataframe(df_filter, use_container_width=True)
        
        #Función para cargar modelo y predecir




        #Dibuja tabla fitlrada de usuario
        #st.dataframe(df_intermedio, use_container_width=True)
        
        st.sidebar.header('User Input Parameters')
        
        #funcion para poner los parametros en el sidebar
        #def user_input_parameters():
        #    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
        #    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
        #    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
        #    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
        #    data = {'sepal_length': sepal_length,
        #            'sepal_width': sepal_width,
        #            'petal_length': petal_length,
        #            'petal_width': petal_width,
        #            }
        #    features = pd.DataFrame(data, index=[0])
        #    return features

        #df = user_input_parameters()

        #st.write(df)
        #st.text(' resultados',df) 
        #print('Finalizando')

        #st.write(user_timestamp)



        #funcion para clasificar las plantas 
        def classify(num):
            if num == 0:
                return 'Metal'
            elif num == 1:
                return 'Rock'
            else:
                return 'Reggae'
            
        if st.button('Recommend songs'):
                if 1 == 1:
                    st.success(classify(1))
                
                
    except:
        if st.session_state.username=='':
            st.text('Please Login first')        