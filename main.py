import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    col_names = ['index','obj','pred','sub','val']
    data = pd.read_csv("train.csv",header=None,  names=col_names)
    data =data.iloc[1:, :]
    data=data.drop('index',1)
    #data.reindex(data.index)
    #print(data.head())

    le = preprocessing.LabelEncoder()
    le.fit(['index','obj','pred','sub'])
    list(le.classes_)
    le.transform(['index','obj','pred','sub'])
    print(le.head())


    # # split dataset in features and target variable
    # feature_cols = ['obj','pred','sub']
    # X = data[feature_cols]  # Features
    # y = data.val  # Target variable
    #
    #
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    # logreg = LogisticRegression()
    # logreg.fit(X_train, y_train)
    # y_pred = logreg.predict(X_test)
    #
    # cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    # print(cnf_matrix)

