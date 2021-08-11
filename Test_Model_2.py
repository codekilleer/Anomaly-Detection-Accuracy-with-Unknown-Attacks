import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from scipy.io import arff
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt


def get_data(s):
    data = arff.loadarff(s)
    df = pd.DataFrame(data[0])
    return df

def load_model(filepath):
    return keras.models.load_model(filepath)

def get_encoder_file(path):
    pkl_file = open(path, 'rb')
    le_departure = pickle.load(pkl_file)
    pkl_file.close()
    return le_departure

def encode_data(data):
    en = get_encoder_file("Model_Data_2/encoderf.pkl")
    return data.apply(en.fit_transform)


def help_label_encode(data):
    en = OneHotEncoder()
    return en.fit_transform(data).toarray()


def encodelabels(data):
    i = 0
    ret = [0 if str(x)[2:-1]=='normal' else 1 for x in data]
    ret = pd.DataFrame(ret)
    ret = help_label_encode(ret)
    return pd.DataFrame(ret)

def seperate_encode_data(data):
    data.columns = [x for x in range(len(data.columns))]
    catogory = pd.DataFrame({})
    notcatogory = pd.DataFrame({})
    for x in range(len(data.columns)):
        if type(data.iloc[0, x]) is str or type(data.iloc[0, x]) is bytes:
            catogory[x] = data.iloc[:, x]
        else:
            notcatogory[x] = data.iloc[:, x]

    catogory = encode_data(catogory)
    data = pd.concat([catogory, notcatogory], axis='columns')
    return data


def scalevalues(data):
    return StandardScaler().fit_transform(data)

def plot_confusion_matrix(cm,classes,
                          title='Confusion matrix'):
    sns.heatmap(cm, square=True, annot=True, cbar=False)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted value')
    plt.ylabel('True value')
    plt.title(title)
    plt.plot(figsize=(12000,12000))
    plt.show()


model_file_path="Model_Data_2/Anomaly_Model.h5"
data_file_path="NSL-KDD/KDDTest+.arff"
feature_list=[1, 2, 4, 5, 7, 8, 9, 12, 15, 16, 20, 21, 22, 23, 25, 28, 31, 32, 33, 37, 40, 41]

if __name__=="__main__":
    model=load_model(model_file_path)
    test_data=get_data(data_file_path)

    x_test = test_data.iloc[:, feature_list]
    y_test = test_data.iloc[:, -1]

    x_test = seperate_encode_data(x_test)
    y_test = encodelabels(y_test)

    x_test = scalevalues(x_test)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    print(model.summary())

    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.9)
    #rint(y_test)
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print("Done doing confusion matrix")

    plot_confusion_matrix(cm, ["Normal",'Anomaly'], title='Confusion matrix, without normalization')
    print(accuracy_score(y_test, y_pred))
    print(cm)
    print(classification_report(y_test, y_pred))


