import pandas as pd
from pandas._libs.internals import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm._libsvm import fit

if __name__ == '__main__':
    col_names = ['index','obj','pred','sub','val']
    data = pd.read_csv("train.csv",header=None,  names=col_names)
    data =data.iloc[1:, :]
    #data=data.drop('index',1)
    #data.reindex(data.index)
    print(data.head())



    le = preprocessing.LabelEncoder()
    data['index'] = le.fit_transform(data['index'])
    data['obj'] = le.fit_transform(data['obj'])
    data['pred'] = le.fit_transform(data['pred'])
    data['sub'] = le.fit_transform(data['sub'])
    print(data.head())




    # # split dataset in features and target variable
    feature_cols = ['index','obj','pred','sub']
    X = data[feature_cols]  # Features
    Y = data['val']  # Target variable

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) #0.8, 12
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)

    print(Y_pred)

    # Returns a NumPy Array
    # Predict for One Observation (image)
    #logreg.predict(X_test[0].reshape(1, -1))
    #predictions = logreg.predict(X_test)
    #(predictions)

    #print(logreg.predict(X_test))

    #ogreg.predict(X_test)
    # Use score method to get accuracy of model
    score = logreg.score(X_test, Y_test)
    print(score)

    cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)
    print(cnf_matrix)

#klasyfikacja baysowska
