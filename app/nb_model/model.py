import pandas
import pickle as pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from dateutil.parser import parse
import numpy as np
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory , ArrayDictionary , StopWordRemover

factory = StopWordRemoverFactory()
a = list(factory.get_stop_words())
if "di" in a: a.remove("di")
if "adalah" in a: a.remove("adalah")    
dictionary = ArrayDictionary(a)
stopwordId = StopWordRemover(dictionary)

sf= StemmerFactory()
stemmerId = sf.create_stemmer() 

def date_detection(doc,fuzzy=True):
    try: 
        parse(doc, fuzzy=fuzzy)
        return True

    except ValueError:
        return False
    except :
        return False
    
def all_caps_detection(doc):
    return (len([word for word in doc if word.isupper()]) > 0)

def contain_digits_detection(doc):
    return any(c.isdigit() for c in doc)

def karacter_detection(doc,char=':'):
    return char in doc
    
def place_detection(doc,char='di'):
    return char in doc
    
def more_than_n_term_detection(doc,n=18):
    return (len(doc) > n)

def text_to_vector(doc):
    return np.array([
            date_detection(doc),
            all_caps_detection(doc),
            contain_digits_detection(doc),
            karacter_detection(doc),
            place_detection(doc),
            more_than_n_term_detection(doc)])+ 0

def tokenization(doc):
    doc = re.sub('[^a-zA-Z]', ' ' ,doc)
    doc = " ".join(doc.split())
    doc = doc.lower()
    
    doc = stopwordId.remove(doc)    
    doc = stemmerId.stem(doc)
    
    return doc.split(" ")

def training():
    df_train = pandas.read_csv("./static/datatrain.csv",index_col=False)
    df_train.label = df_train.label.replace('nama ','nama')
    X = [text_to_vector(i) for i in df_train["text"]]
    clf = GaussianNB()    
    clf.fit(X, df_train["label"].astype(str))

    with open('./static/clf.pk', 'wb') as fin:
        pickle.dump(clf, fin)

def testing(x=None,y=None):
    with open('./static/clf.pk', 'rb') as f:
        clf = pickle.load(f)
    if x is None and y is None:
        df = pandas.read_csv("./static/datatest.csv",index_col=False)
        df.label = df.label.replace('nama ','nama')

        X= [text_to_vector(i) for i in df["text"]]
        y_pred = clf.predict(X)

        acc = accuracy_score(df["label"], y_pred)
        return acc , y_pred
    else:
        X = [text_to_vector(i) for i in [x]]
        y_pred = clf.predict(X)
        acc = accuracy_score([y], y_pred)
        return acc , y_pred

