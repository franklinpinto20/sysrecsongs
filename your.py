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
        
        if(df_intermedio.empty):
            addInfo=st.session_state['displayName']
            #st.text('Detalles: '+addInfo)

            resto=addInfo
            gender=resto.split("-",1)[0]
            resto=resto.split("-",1)[1]
            age=resto.split("-",1)[0]
            resto=resto.split("-",1)[1]
            country=resto.split("-",1)[0]
            resto=resto.split("-",1)[1]
            fdate=resto.split("-",1)[0]
            resto=resto.split("-",1)[1]
            musicType=resto.split("-",1)[0]
            df_intermedio=uput[uput['gender']=='m']
            df_intermedio=df_intermedio[df_intermedio['country']==country].sort_values(by=['count'], ascending=False)
            df_intermedio['rating']=1
        else:
            #dibuja rating
        
            #st.title('df_filter')
    
            df_filter=df_intermedio[['artist','song','count','rating']]
            st.dataframe(df_filter, use_container_width=True)
        
        #Función para cargar modelo y predecir
       
       #Entrenamiento del modelo
       # from sklearn.model_selection import train_test_split
      

        #Librerías extras
        import itertools
        from surprise.dump import dump, load
        from sklearn.metrics.pairwise import cosine_similarity
        from surprise import Dataset
        from surprise import Reader
        from surprise import KNNBasic, KNNWithMeans
       
        df_intermedio['rating'] = df_intermedio['rating']*4
        # Ajustar los valores mayores que 5 a 5
        df_intermedio.loc[df_intermedio['rating'] > 4, 'rating'] = 4
        df_intermedio['rating_'] = df_intermedio['rating']+1
      
        #st.dataframe(df_intermedio, use_container_width=True)
        
        #df_final = df_intermedio.sample(500)
        
        train, test = train_test_split(df_intermedio, test_size=0.2)
        
        df_intermedio['rating'] = df_intermedio['rating']*4
        
              
         # Ajustar los valores mayores que 5 a 5
        train_a = train
        test_a = test
        
        def save_model(algo):
            dump(file_name='user_user_model', algo=algo)

        def train_model(ratings_df):
            test_a['rating_'] = (test_a['rating'] + 1)
            reader = Reader(rating_scale=(1, ratings_df['rating_'].max()))
            train_set = Dataset.load_from_df(ratings_df[['user_id', 'track_id', 'rating_']], reader).build_full_trainset()
            algo_user = KNNBasic(k=10, min_k=1,sim_options={'name': 'pearson','user_based': True})
            algo_user.fit(train_set)
            save_model(algo_user)
            return algo_user
        
        
        train_model(train_a)


        def get_user_predictions(user_id, songs_list, size_predictions):
            algo = load('user_user_model')[1]
            # reader = Reader(rating_scale=(1, ratings_df['playcount'].max()))
            # test_set = Dataset.load_from_df(ratings_df[['user_id', 'track_id', 'playcount']], reader).build_full_trainset()
            test_set = [(user_id, song, 0) for song in songs_list]
            predictions=algo.test(test_set)
            print(predictions)

            user_predictions=list(filter(lambda x: x[0]==user_id,predictions))
            #Ordenamos de mayor a menor estimación de relevancia
            user_predictions.sort(key=lambda x : x.est, reverse=True)
            #tomamos las 10 primeras predicciones
            user_predictions=user_predictions[0:size_predictions]
            #Se convierte a dataframe
            df_predictions = pd.DataFrame.from_records(list(map(lambda x: (x.iid, x.est) , user_predictions)))
            #Lo unimos con el dataframe de películas
            return df_predictions
        #st.title('Lista canciones' )
        dfSongs=user_timestamp.groupby(['track_id','song'], as_index=False).agg({'timestamp': 'count'})
        #dfSongs=  user_timestamp.groupby(('track_id','song'))
        #dfSongs=dfSongs[['track_id', 'song']].drop_duplicates()
        dfSongs = pd.DataFrame(dfSongs, columns=["track_id", "song"])
        #st.dataframe(dfSongs, use_container_width=True)
         
        st.title('Recomendaciones automáticas' )
    
    
    
        df_predictions=get_user_predictions(st.session_state['username'], dfSongs['track_id'] , 10)
        df_predictions = df_predictions.rename(columns={0: 'track_id'})
        df_predictions = df_predictions.rename(columns={1: 'punctuation'})
        
      
        df_predictions['song'] = df_predictions['track_id'].map(dfSongs.set_index('track_id')['song'])
        
       
        df_predictions=df_predictions[['song','punctuation']]
        
        #df_predictions = df_predictions.groupby(['song', 'punctuation'], as_index=False)
        
        st.dataframe(df_predictions, use_container_width=True)
        
        
        #Dibuja tabla fitlrada de usuario
        #st.dataframe(df_intermedio, use_container_width=True)
        
        
        


        #funcion para clasificar las 
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
