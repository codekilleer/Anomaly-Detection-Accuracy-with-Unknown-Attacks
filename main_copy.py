import os
import pickle
import shutil
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras import regularizers
from kerastuner.tuners import RandomSearch
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

        ## As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()



def save_encoder_file(efile, filename):
    # dump(efile, open(filename+'.pkl', 'wb'))
    output = open(filename + '.pkl', 'wb')
    pickle.dump(efile, output)
    output.close()


def getdata(s):
    ret = []
    with open(s, "r") as filestream:
        for x in filestream:
            ret.append(x.split(","))

    return pd.DataFrame(ret)


def scalevalues(data):
    return StandardScaler().fit_transform(data)


def get_model(hp):
    model = Sequential()
    model.add(layers.Dense(units=32, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.0001),
                           activation='relu', input_dim=len(x_train[0])))
    model.add(Dropout(.05))
    for i in range(hp.Int('num_layers', 4, 8)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               kernel_regularizer=regularizers.l2(0.0001),
                               kernel_initializer='uniform',
                               activation='relu'))
        model.add(keras.layers.Dropout(rate=hp.Choice('drop_rate' + str(i), [0.05, 0.1, 0.15, 0.2])))
    model.add(layers.Dense(units=6, kernel_initializer='uniform', activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix'):
    sns.heatmap(cm, square=True, annot=True, cbar=False)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted value')
    plt.ylabel('True value')
    plt.title(title)
    plt.plot(figsize=(12000, 12000))
    plt.show()


def save_model(file, filename):
    if not os.path.isdir("Model_Data"):
        os.mkdir("Model_Data")
    file.save("Model_Data"+filename+".h5")


def save_featured_used(ar):
    if not os.path.isdir("Model_Data"):
        os.mkdir("Model_Data")
    with open("Model_Data/feature_list.fe", "wb") as fh:
        pickle.dump(ar, fh)


def model_folder_check():
    if os.path.isdir("Model_Run_Data"):
        shutil.rmtree("Model_Run_Data")


def save_RandomSearch_model(ob, path):
    joblib.dump(ob, path)


def encode_data(data):
    en = LabelEncoder()
    save_encoder_file(en, "Model_Data/encoderf")
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


def run_model(modelname,batch, epo, x_train, y_train, x_validate, y_validate):
    model = RandomSearch(get_model,
                         objective='val_accuracy',
                         max_trials=2,
                         executions_per_trial=1,
                         directory=modelname+'Model_Run_Data',
                         project_name='batch' + str(batch) + 'Epoch' + str(epo))
    history = model.search(x=x_train, y=y_train,
                           epochs=epo,
                           validation_data=(x_validate, y_validate))
    return model.get_best_models()[0], history


def subsetting_data(x,y):
    x_A = []
    y_A = []
    x_B = []
    y_B = []
    x_C = []
    y_C = []
    x_onetest = []
    y_onetest = []
    x_n=[]
    y_n=[]
    i=0
    for a in y.values:
        if a[1] == 1 or a[2]==1:
            x_A.append(x[i, :])
            y_A.append(a)
        if a[1] == 1 or a[3]==1:
            x_B.append(x[i, :])
            y_B.append(a)
        if a[2] == 1 or a[4]==1:
            x_C.append(x[i, :])
            y_C.append(a)
        if a[0] == 1 or a[5]==1:
            x_n.append(x[i, :])
            y_n.append(a)
        if a[3] == 1 or a[4]==1:
            x_onetest.append(x[i, :])
            y_onetest.append(a)
    return x_A, y_A, x_B, y_B, x_C, y_C, x_n, y_n, x_onetest, y_onetest




epoch = [50]
batch_size = [64]

A1 = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'mailbomb', 'processtable', 'udpstorm', 'apache2', 'worm']
A2 = ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint']
A3 = ['guess_password', 'ftp_write', 'imap', 'phf', 'multihop', 'Warezmaster', 'xlock', 'xsnoop', 'snmpguess',
      'snmpgetattack', 'httptunnel', 'sendmail', 'named']
A4 = ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps']

train_file_path = 'NSL-KDD/KDDTrain+.txt'
validate_file_path = 'NSL-KDD/KDDTrain+_20Percent.txt'
test_file_path = 'NSL-KDD/KDDTest+.txt'

training_data = getdata(train_file_path)
validate_data = getdata(validate_file_path)
test_data = getdata(test_file_path)

feature_list = [1, 2, 4, 5, 7, 8, 9, 12, 15, 16, 20, 21, 22, 23, 25, 28, 31, 32, 33, 37, 40, 41]

x_train = training_data.iloc[:, feature_list]
y_train = training_data.iloc[:, -2]
x_validate = validate_data.iloc[:, feature_list]
y_validate = validate_data.iloc[:, -2]
x_test = test_data.iloc[:, feature_list]
y_test = test_data.iloc[:, -2]


if __name__ == "__main__":
    model_folder_check()
    save_featured_used(feature_list)

    x_train = seperate_encode_data(x_train)
    y_train = encodelabels(y_train)
    x_validate = seperate_encode_data(x_validate)
    y_validate = encodelabels(y_validate)
    x_test = seperate_encode_data(x_test)
    y_test = encodelabels(y_test)

    x_train = scalevalues(x_train)
    x_validate = scalevalues(x_validate)
    x_test = scalevalues(x_test)

    x_subsetA, y_subsetA, x_subsetB, y_subsetB, x_subsetC, y_subsetC, x_subsetn, y_subsetn, x_onetest, y_onetest = subsetting_data(x_train, y_train)
    x_subsetA = np.array(x_subsetA)
    y_subsetA = np.array(y_subsetA)
    x_subsetB = np.array(x_subsetB)
    y_subsetB = np.array(y_subsetB)
    x_subsetC = np.array(x_subsetC)
    y_subsetC = np.array(y_subsetC)
    x_subsetn = np.array(x_subsetn)
    y_subsetn = np.array(y_subsetn)
    x_onetest = np.array(x_onetest)
    y_onetest = np.array(y_onetest)

    x_subsetAval, y_subsetAval, x_subsetBval, y_subsetBval, x_subsetCval, y_subsetCval, x_subsetnval, y_subsetnval,waste1,waste2 = subsetting_data(x_validate,y_validate)
    x_subsetAval = np.array(x_subsetAval)
    y_subsetAval = np.array(y_subsetAval)
    x_subsetBval = np.array(x_subsetBval)
    y_subsetBval = np.array(y_subsetBval)
    x_subsetCval = np.array(x_subsetCval)
    y_subsetCval = np.array(y_subsetCval)
    x_subsetnval = np.array(x_subsetnval)
    y_subsetnval = np.array(y_subsetnval)


    x_test = np.array(x_test)
    y_test = np.array(y_test)

    print(x_train.shape, y_train.shape, x_validate.shape, y_validate.shape)
    # print(y_train)
    all_scores = []
    all_models = []
    all_batchsize = []
    all_epoch = []
    for batch in batch_size:
        for epo in epoch:
            #pname='batch:' + str(batch) + 'Epoch:' + str(epo)
            all_batchsize.append(batch)
            all_epoch.append(epo)

            print("Training model for batch:", batch, " and epoch", epo)
            model_one, history = run_model("ONE", batch, epo, np.vstack([x_subsetA, x_subsetn]),
                                          np.vstack([y_subsetA, y_subsetn]),
                                          np.vstack([x_subsetAval, x_subsetnval]),
                                          np.vstack([y_subsetAval, y_subsetnval]))
            model_four_one, history = run_model("four_one", batch, epo, np.vstack([x_subsetB, x_subsetn]),
                                          np.vstack([y_subsetB, y_subsetn]),
                                          np.vstack([x_subsetBval, x_subsetnval]),
                                          np.vstack([y_subsetBval, y_subsetnval]))
            model_four_two, history = run_model("four_two", batch, epo, np.vstack([x_subsetC, x_subsetn]),
                                                np.vstack([y_subsetC, y_subsetn]),
                                                np.vstack([x_subsetCval, x_subsetnval]),
                                                np.vstack([y_subsetCval, y_subsetnval]))

            all_models.append(model_one)
            all_models.append(model_four_one)
            all_models.append(model_four_two)



    #savinf models
    for x in range(len(all_models)):
        print("For model"+str(x))
        m=all_models[x]
        save_model(m,"model"+str(x))

        print("Summary\n",m.summary())
        cm=None
        cr=None
        tt=None
        if x==0:
            y_pred = m.predict(x_onetest)
            y_pred = (y_pred > 0.9)
            tt = accuracy_score(y_onetest, y_pred)
            cm = confusion_matrix(y_onetest.argmax(axis=1), y_pred.argmax(axis=1))
            cr = classification_report(y_onetest, y_pred)
        else:
            y_pred = m.predict(x_test)
            y_pred = (y_pred > 0.9)
            tt = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
            cr=classification_report(y_test, y_pred)

        print("Accuracy of model" +str(x) + " -" + str(tt * 100))
        print(cm)
        print(cr)
        plot_confusion_matrix(cm, [0, 1, 2, 3, 4, 5], title='Confusion matrix, without normalization')
