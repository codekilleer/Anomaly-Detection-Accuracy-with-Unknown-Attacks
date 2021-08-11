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
from scipy.io import arff
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
    data = arff.loadarff(s)
    df = pd.DataFrame(data[0])
    return df


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
    model.add(layers.Dense(units=2, kernel_initializer='uniform', activation='softmax'))
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
    if not os.path.isdir("Model_Data_2"):
        os.mkdir("Model_Data_2")
    file.save(filename)


def save_featured_used(ar):
    if not os.path.isdir("Model_Data_2"):
        os.mkdir("Model_Data_2")
    with open("Model_Data_2/feature_list.fe", "wb") as fh:
        pickle.dump(ar, fh)


def model_folder_check():
    if os.path.isdir("Model_Run_Data_2"):
        shutil.rmtree("Model_Run_Data_2")


def save_RandomSearch_model(ob, path):
    joblib.dump(ob, path)


def encode_data(data):
    en = LabelEncoder()
    save_encoder_file(en, "Model_Data_2/encoderf")
    return data.apply(en.fit_transform)


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


def help_label_encode(data):
    en = OneHotEncoder()
    return en.fit_transform(data).toarray()


def encodelabels(data):
    i = 0
    ret = [0 if str(x)[2:-1]=='normal' else 1 for x in data]
    ret = pd.DataFrame(ret)
    ret = help_label_encode(ret)
    return pd.DataFrame(ret)


epoch = [200]
batch_size = [64]

#A1 = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'mailbomb', 'processtable', 'udpstorm', 'apache2', 'worm']
#A2 = ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint']
#A3 = ['guess_password', 'ftp_write', 'imap', 'phf', 'multihop', 'Warezmaster', 'xlock', 'xsnoop', 'snmpguess',
#     'snmpgetattack', 'httptunnel', 'sendmail', 'named']
#A4 = ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps']

train_file_path = 'NSL-KDD/KDDTrain+.arff'
validate_file_path = 'NSL-KDD/KDDTrain+_20Percent.arff'
test_file_path = 'NSL-KDD/KDDTest+.arff'

training_data = getdata(train_file_path)
validate_data = getdata(validate_file_path)
test_data = getdata(test_file_path)

feature_list = [1, 2, 4, 5, 7, 8, 9, 12, 15, 16, 20, 21, 22, 23, 25, 28, 31, 32, 33, 37, 40, 41]

x_train = training_data.iloc[:, feature_list]
y_train = training_data.iloc[:, -1]
x_validate = validate_data.iloc[:, feature_list]
y_validate = validate_data.iloc[:, -1]
x_test = test_data.iloc[:, feature_list]
y_test = test_data.iloc[:, -1]


if __name__ == "__main__":
    model_folder_check()
    save_featured_used(feature_list)
    print(y_train)
    x_train = seperate_encode_data(x_train)
    y_train = encodelabels(y_train)
    x_validate = seperate_encode_data(x_validate)
    y_validate = encodelabels(y_validate)
    x_test = seperate_encode_data(x_test)
    y_test = encodelabels(y_test)

    print(x_test)

    x_train = scalevalues(x_train)
    x_validate = scalevalues(x_validate)
    x_test = scalevalues(x_test)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_validate = np.array(x_validate)
    y_validate = np.array(y_validate)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    print(y_train)

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
            model = RandomSearch(get_model,
                                 objective='val_accuracy',
                                 max_trials=1,
                                 executions_per_trial=1,
                                 directory='Model_Run_Data_2',
                                 project_name='batch' + str(batch) + 'Epoch' + str(epo))
            history = model.search(x=x_train, y=y_train,
                                   epochs=epo,
                                   validation_data=(x_validate, y_validate))
            # history = model.fit(x_train,y_train, batch_size = batch, epochs = epo)
            # loss, accuracy = model.evaluate(x_test, y_test)
            print("DoneDone")
            all_models.append(model)

            y_pred = model.get_best_models()[0].predict(x_test)
            y_pred = (y_pred > 0.9)

            cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
            tt = accuracy_score(y_test, y_pred)
            all_scores.append(tt)

            print('Print the loss and the accuracy of the model on the dataset for ', batch, epo)
            # print("sumary", model.results_summary())
            print('Best Model summary', (model.get_best_models()[0].summary()))

            print("Accuracy: ", tt * 100)

            # print("Done doing confusion matrix")
            print(cm)
            print(classification_report(y_test, y_pred))
            # print(x_test[0])
            plot_confusion_matrix(cm, [0, 1], title='Confusion matrix, without normalization')
            # plot_history(history)

    print("Model with the max Accuracy", max(all_scores), " Average score: ", sum(all_scores) / len(all_scores))
    print("Parameters, batch size:", all_batchsize[all_scores.index(max(all_scores))], " Epoch:",
          all_epoch[all_scores.index(max(all_scores))])
    save_model(all_models[all_scores.index(max(all_scores))].get_best_models()[0],
               "Model_Data_2/Anomaly_Model.h5")
    save_RandomSearch_model(all_models[all_scores.index(max(all_scores))],
                            "Model_Data_2/Random_search_best_model.pkl")