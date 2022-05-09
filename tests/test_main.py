import pytest
import os
import sys
sys.path.append('..')
import nltk 
import sklearn
import pandas as pd
import unredactor

def test_dataextract():
    data=unredactor.dataextract()
    if data is not None:
        assert True

def test_cleaning():
    data=unredactor.dataextract()
    cleaned=unredactor.cleaning(data)
    if cleaned is not None:
        assert True

def test_vectorizer():
    data=unredactor.dataextract()
    cleaned=unredactor.cleaning(data)
    vect=unredactor.vectorizer(cleaned)
    if vect is not None:
        assert True

def test_filtering():
    data=unredactor.dataextract()
    train,test=unredactor.filtering(data)

    if train is not None and test is not None:
        assert True


def test_train_features():
    data=unredactor.dataextract()
    
    features=unredactor.train_features(data)

    if features is not None:
        assert True




