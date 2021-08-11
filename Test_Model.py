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



def load_model(filepath):
    return keras.models.load_model(filepath)

def get_data(s):
    ret = []
    with open(s, "r") as filestream:
        for x in filestream:
            ret.append(x.split(","))

    return pd.DataFrame(ret)

def get_encoder_file(path):
    pkl_file = open(path, 'rb')
    le_departure = pickle.load(pkl_file)
    pkl_file.close()
    return le_departure

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

def encode_data(data):
    en = get_encoder_file("Model_Data/encoderf.pkl")
    return data.apply(en.fit_transform)


def seperate_encode_data(data):
    data.columns = [x for x in range(len(data.columns))]
    catogory = pd.DataFrame({})
    notcatogory = pd.DataFrame({})
    for x in range(len(data.columns)):
        if type(data.iloc[0, x]) is str:
            catogory[x] = data.iloc[:, x]
        else:
            notcatogory[x] = data.iloc[:, x]

    catogory = encode_data(catogory)
    data = pd.concat([catogory, notcatogory], axis='columns')
    return data


def help_label_encode(data):
    en = OneHotEncoder()
    return en.fit_transform(data).toarray()


def encodelabels(data):
    i = 0
    ret = []
    for x in data:
        x.lower()
        if x in A1:
            ret.append(1)
        elif x in A2:
            ret.append(2)
        elif x in A3:
            ret.append(3)
        elif x in A4:
            ret.append(4)
        elif x == 'normal':
            ret.append(0)
        else:
            ret.append(5)
        i += 1
    ret = pd.DataFrame(ret)
    ret = help_label_encode(ret)
    return pd.DataFrame(ret)

A1 = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'mailbomb', 'processtable', 'udpstorm', 'apache2', 'worm']
A2 = ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint']
A3 = ['guess_password', 'ftp_write', 'imap', 'phf', 'multihop', 'Warezmaster', 'xlock', 'xsnoop', 'snmpguess',
      'snmpgetattack', 'httptunnel', 'sendmail', 'named']
A4 = ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps']


model_file_path="Model_Data/Anomaly_Model.h5"
data_file_path="NSL-KDD/KDDTest+.txt"
feature_list=[1, 2, 4, 5, 7, 8, 9, 12, 15, 16, 20, 21, 22, 23, 25, 28, 31, 32, 33, 37, 40, 41]

if __name__=="__main__":
    model=load_model(model_file_path)
    test_data=get_data(data_file_path)

    x_test = test_data.iloc[:, feature_list]
    y_test = test_data.iloc[:, -2]

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

    plot_confusion_matrix(cm, [0,1,2,3,4,5], title='Confusion matrix, without normalization')
    print(accuracy_score(y_test, y_pred))
    print(cm)
    print(classification_report(y_test, y_pred))


