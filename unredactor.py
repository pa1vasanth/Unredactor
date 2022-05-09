import os
import sys
import pandas as pd
import numpy as np
import spacy
import nltk
from nltk import sent_tokenize,word_tokenize
from sklearn.feature_extraction import DictVectorizer
#from nestor import keyword as kex
from nltk.corpus import stopwords
import warnings
from sklearn import svm
warnings.filterwarnings('ignore')
nlp = spacy.load('en_core_web_md')
from sklearn.metrics import precision_score,f1_score,recall_score,accuracy_score
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

def dataextract():
    url="https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv"
    colnames=['user','type','Name','Sentence']
    data=pd.read_table(url,sep='\t',names=colnames,header=None,error_bad_lines=False)
    df=data.dropna().reset_index(drop=True)
    return df


def cleaning(data):
    stop_words = set(stopwords.words('english'))
    special_char=[',',':',';','?','>','<','/']
    
    for i in range(len(data)):
        st=data['Sentence'].iloc[i]
        
        words=word_tokenize(st)
        dd=[]
        for j in range(len(words)):
            if (words[j]  not in stop_words) and (words[j] not in special_char ):
                dd.append(words[j])

        data['Sentence'].iloc[i]=' '.join(dd)
    return data

def vectorizer(data):
    countvect=CountVectorizer()
    vectors=countvect.fit_transform(data['Sentence'])
    df=pd.DataFrame(vectors.toarray())
    d=[data,df]
    f_df=pd.concat(d,axis=1)
    return f_df

def filtering(data):
    
    df=data[(data['type']=='training')|(data['type']=='validation')]
    test_df=data[(data['type']=='testing')]
    
    return df,test_df

def train_features(C_data):
    features=[]
    f=[]
    sia = SentimentIntensityAnalyzer()
    data=C_data['Sentence']
    for i in range(len(data)):
        st=data.iloc[i]
        dicti={}
        senti=sia.polarity_scores(st)
        words=word_tokenize(st)
        for j in range(len(words)):
            if u"\u2588" in words[j]:
                nam_len=len(words[j])
        
        dicti['namelen']=nam_len
        dicti['sentcount']=len(sent_tokenize(st))
        dicti['wordcount']=len(word_tokenize(st))
        dicti['charcount']=len(st)
        dicti['positivity']=senti['pos']
        features.append(dicti) 
    features_df=pd.DataFrame(features)
    d=[C_data,features_df]
    f_df=pd.concat(d,axis=1)
    return f_df





if __name__ == '__main__':
    data=dataextract()
    df=cleaning(data)
    vect_df=vectorizer(df)
    feature_df=train_features(vect_df)
    train,test=filtering(feature_df)
    y=train['Name']
    x=train.drop(['user','type' ,'Name','Sentence'],axis=1) 
    model=RandomForestClassifier(random_state=25345)
    clf=model.fit(x,y)
    test_features=test.drop(['user','type' ,'Name','Sentence'],axis=1)
    Predicted_names=clf.predict(test_features)
    print(Predicted_names)    
    print("Precision: ", precision_score(test['Name'].values.tolist(),Predicted_names,average='micro'))
    print("F1-score: ", f1_score(test['Name'].values.tolist(),Predicted_names,average='macro'))
    print("Recall: " ,recall_score(test['Name'].values.tolist(),Predicted_names,average='macro'))




    
