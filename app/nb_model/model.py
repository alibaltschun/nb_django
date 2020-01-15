import pandas
import pickle as pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from dateutil.parser import parse
import numpy as np
    
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

def akronim_detection(doc):
    upercase = [word for word in doc.split(" ") if word.isupper()]
    return len([word for word in upercase if (len(word)>1 and len(word)<5)]) > 0

def karacter_detection(doc,char=':'):
    return char in doc
    

def more_than_n_term_detection(doc,n=18):
    return (len(doc) > n)

def text_to_vector(doc):
    return np.array([
            date_detection(doc),
            all_caps_detection(doc),
            contain_digits_detection(doc),
            akronim_detection(doc),
            karacter_detection(doc),
            more_than_n_term_detection(doc)])+ 0

def training():
    df_train = pandas.read_csv("./static/datatrain.csv",index_col=False)
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

        df_val = df.loc[df['is_valid'] == True]
        df_test = df.loc[df['is_valid'] == False]

        X_val = [text_to_vector(i) for i in df_val["text"]]
        X_test =  [text_to_vector(i) for i in df_test["text"]]

        y_pred_val = clf.predict(X_val)
        y_pred_test = clf.predict(X_test)

        acc_val = accuracy_score(df_val["label"], y_pred_val)
        acc_test = accuracy_score(df_test["label"], y_pred_test)

        return [acc_val,acc_test] , [y_pred_val,y_pred_test]
    else:
        X = [text_to_vector(i) for i in [x]]
        y_pred = clf.predict(X)
        acc = accuracy_score([y], y_pred)
        return acc , y_pred

