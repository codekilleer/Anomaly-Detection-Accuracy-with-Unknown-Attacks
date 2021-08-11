
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt

def getdata(s):
    ret = []
    with open(s, "r") as filestream:
        for x in filestream:
            ret.append(x.split(","))

    return pd.DataFrame(ret)


def encode_data(data):
    en = LabelEncoder()
    #save_encoder_file(en, "Model_Data/encoderf")
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


def scalevalues(data):
    return StandardScaler().fit_transform(data)


def expand_categories(values):
    result = []
    s = values.value_counts()
    t = float(len(values))
    for v in s.index:
        result.append("{}:{}%".format(v, round(100 * (s[v] / t), 2)))
    return "[{}]".format(",".join(result))


def analyze(df):
    print()
    cols = df.columns.values
    total = float(len(df))

    print("{} rows".format(int(total)))
    for col in cols:
        uniques = df[col].unique()
        unique_count = len(uniques)
        if unique_count > 100:
            print("** {}:{} ({}%)".format(col, unique_count, int(((unique_count) / total) * 100)))
        else:
            print("** {}:{}".format(col, expand_categories(df[col])))
            expand_categories(df[col])


A1 = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'mailbomb', 'processtable', 'udpstorm', 'apache2', 'worm']
A2 = ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint']
A3 = ['guess_password', 'ftp_write', 'imap', 'phf', 'multihop', 'Warezmaster', 'xlock', 'xsnoop', 'snmpguess',
      'snmpgetattack', 'httptunnel', 'sendmail', 'named']
A4 = ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps']
feature_list = [1, 2, 4, 5, 7, 8, 9, 12, 15, 16, 20, 21, 22, 23, 25, 28, 31, 32, 33, 37, 40, 41]
#feature_list = [x for x in range(1,42)]

data= getdata("NSL-KDD/KDDTrain+.txt")
data = data.iloc[:, feature_list]
data=seperate_encode_data(data)
print(data)
#data = scalevalues(data)

#data=pd.DataFrame(data)

analyze(data)

corrMatrix = data.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()
