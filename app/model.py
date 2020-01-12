import pandas
import pickle as pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

def preprocessingTokenization(doc):
    return doc.split(" ")

def preprocessingVectorizer(arr):
    with open('./static/vectorizer.pk', 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer.transform(arr).toarray()

def training():
    df_train = pandas.read_csv("./static/datatrain.csv",index_col=False)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df_train["text"]).toarray()
    clf = GaussianNB()    
    clf.fit(X, df_train["label"])

    with open('./static/clf.pk', 'wb') as fin:
        pickle.dump(clf, fin)
    with open('./static/vectorizer.pk', 'wb') as fin:
        pickle.dump(vectorizer, fin)

def testing(x=None,y=None):
    with open('./static/clf.pk', 'rb') as f:
        clf = pickle.load(f)
    with open('./static/vectorizer.pk', 'rb') as f:
        vectorizer = pickle.load(f) 
    if x is None and y is None:
        df = pandas.read_csv("./static/datatest.csv",index_col=False)

        df_val = df.loc[df['is_valid'] == True]
        df_test = df.loc[df['is_valid'] == False]

        X_val = vectorizer.transform(df_val["text"]).toarray()
        X_test = vectorizer.transform(df_test["text"]).toarray()

        y_pred_val = clf.predict(X_val)
        y_pred_test = clf.predict(X_test)

        acc_val = accuracy_score(df_val["label"], y_pred_val)
        acc_test = accuracy_score(df_test["label"], y_pred_test)

        return [acc_val,acc_test] , [y_pred_val,y_pred_test]
    else:
        X = vectorizer.transform([x]).toarray()
        y_pred = clf.predict(X)
        acc = accuracy_score([y], y_pred)
        return acc , y_pred
