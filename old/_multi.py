import os
import glob
import json
import re
import random
import time
import pickle
import pymongo as mongo
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import tensorflow.keras.backend as kb
from tensorflow.keras.models import load_model

pd.set_option("display.max_columns", 500)


class data_process():
    def __init__(self, mongo_database):
        self.data_path = self.data_path = './data'
        self.csv_path = glob.glob(os.path.join(self.data_path, '*'))
        self.columns = []
        self.labels = ['BENIGN', 'Bot', 'Dos', 'DDoS', 'FTP-Patator', 'PortScan', 'SSH-Patator', 'Web Attack']
        self.train_data = np.array([])
        self.train_label = np.array([])
        self.calibration_data = np.array([])
        self.calibration_label = np.array([])
        self.test_data = np.array([])
        self.test_label = np.array([])
        self.robustC_data = np.array([])
        self.credibilityC_data = np.array([])
        self.credibilityC_label = np.array([])
        self.robustC_label = np.array([])
        self.data_scale = None

        # mongo part
        client = mongo.MongoClient()
        db = client[mongo_database]
        self.table_test_normal = db['test_normal']
        self.table_train_normal = db['train_normal']
        self.table_calibration_normal = db['calibration_normal']
        self.table_robustC_normal = db['robustC_normal']
        self.table_credibilityC = db['credibilityC']

    def _data_processing(self):
        df = pd.read_csv(self.csv_path[0], header=0, nrows=100000)
        # read 5 csv file
        for i in range(1, len(self.csv_path)):
            df_tem = pd.read_csv(self.csv_path[i], header=0, nrows=100000)
            df = df.append(df_tem)

        # make the sequence of data random
        df = df.sample(frac=0.5, random_state=1)
        # here to make columns ' name more concise
        temp = df[' Label'].value_counts(normalize=True)
        print(temp)
        df[' Protocol'] = df[' Protocol'].astype(int)
        # use one-hot coding the ['Protocol'] and data_label
        df = pd.get_dummies(df, columns=[' Protocol', ' Label'], prefix=[' Protocol', ' Label'])

        self.columns = df.columns.to_list()
        print('data \' s columns:')
        print(self.columns)

        max_nums = []
        min_nums = []
        for i in range(len(self.columns)):
            max_nums.append(df[self.columns[i]].max())
            min_nums.append(df[self.columns[i]].min())

        _df = df[(df[' Label_Bot'] == 1) | (df[' Label_Web Attack'] == 1)]
        left_columns = [' Label_BENIGN', ' Label_Dos', ' Label_FTP-Patator', ' Label_PortScan', ' Label_SSH-Patator']
        for i in range(len(left_columns)):
            temp_df = df[df[left_columns[i]] == 1].head(1)
            temp_df.iloc[:, :83] = 0
            _df = _df.append(temp_df)

        # or df[' Label_Web Attack'] == 1
        df = df[(df[' Label_Bot'] == 0) & (df[' Label_Web Attack'] == 0)]

        # df_rob = df[df[' Label_Dos'] == 1]
        # df.drop(index=df[df[' Label_Dos'] == 1], inplace=True)
        # true_labels = [' Label_' + i for i in self.labels]
        # temp_2 = np.unique(np.array(_df[true_labels], dtype=int), axis=0)

        # label_kinds = len(temp_2.index)
        # for i in range(df_rob.shape[0] - 4):
        #     df_rob.iloc[i, -7:] = temp_2.index[random.randint(0, label_kinds - 1)]
        # for i in range(int(df_rob.shape[0]/100) - 4):
        #     df_rob.iloc[i, -7:] = temp_2.index[random.randint(0, label_kinds - 1)]
        #     df_rob.iloc[i*100:(i+1)*100, 0:20] = df_rob.iloc[i*100+ 3:(i+1)*100 + 3, 0:20]
        #     df_rob.iloc[i*100:(i+1)*100, 20:40] = df_rob.iloc[i*100+ 2:(i+1)*100 + 2, 20:40]
        #     df_rob.iloc[i*100:(i+1)*100, 40:65] = df_rob.iloc[i*100+ 1:(i+1)*100+ 1, 40:65]
        #     df_rob.iloc[i*100:(i+1)*100, 65:83] = df_rob.iloc[i*100+ 4:(i+1)*100 + 4, 65:83]
        #     print('%d have done'%i)

        # normal dataset
        # data_rob = mm.fit_transform(df_rob)
        data = np.array(df)
        data_credibility = np.array(_df)
        for i in range(data.shape[1] - 7 -3):
            data[:, i] = (data[:, i] - min_nums[i]) / (max_nums[i] - min_nums[i]) if (max_nums[i] - min_nums[
                i]) != 0 else 0
            data_credibility[:, i] = (data_credibility[:, i] - min_nums[i]) / (max_nums[i] - min_nums[i]) if (max_nums[i] - min_nums[i]) != 0 else 0
        # data_rob = mm.fit_transform(df_rob)
        self.train_data = data[:80000]
        self.calibration_data = data[85000:105000]
        self.test_data = data[-30000:]
        self.credibilityC_data = data_credibility
        # self.robustC_data = data_rob[-20000:]

    def get_traindata(self):
        if self.train_data.shape[0] == 0:
            self._mongo2np(self.table_train_normal)
        print(self.train_data.shape)
        print(self.train_label.shape)
        return self.train_data, self.train_label

    def get_calibrationdata(self):
        if self.calibration_data.shape[0] == 0:
            self._mongo2np(self.table_calibration_normal)
        print(self.calibration_data.shape)
        print(self.calibration_label.shape)
        return self.calibration_data, self.calibration_label

    def get_testdata(self):
        if self.test_data.shape[0] == 0:
            self._mongo2np(self.table_test_normal)
        print(self.test_data.shape)
        print(self.test_label.shape)
        return self.test_data, self.test_label

    def get_robustCdata(self):
        if self.robustC_data.shape[0] == 0:
            self._mongo2np(self.table_robustC_normal)
        print(self.robustC_data.shape)
        print(self.robustC_label.shape)
        return self.robustC_data, self.robustC_label

    def get_credilityCdata(self):
        if self.credibilityC_data.shape[0] == 0:
            self._mongo2np(self.table_credibilityC)
        print(self.credibilityC_data.shape)
        print(self.credibilityC_label.shape)
        return self.credibilityC_data, self.credibilityC_label

    def get_data_scale(self):
        if self.data_scale == None:
            data_labels = pd.DataFrame().astype(int)
            data_labels = data_labels.append(pd.DataFrame(self.train_label))
            data_labels = data_labels.append(pd.DataFrame(self.calibration_label))
            data_labels = data_labels.append(pd.DataFrame(self.test_label))
            data_labels = data_labels.append(pd.DataFrame(self.robustC_label))
            data_labels = data_labels.append(pd.DataFrame(self.credibilityC_label))

            data_labels = np.array(data_labels)
            temp = np.flipud(np.unique(data_labels, axis=0))

            result = [0 for i in range(len(data_labels[0]))]
            for i in range(len(temp)):
                for j in data_labels:
                    if (j == temp[i]).all():
                        result[i] = result[i] + 1
            result_sum = sum(result)
            for i in range(len(result)):
                result[i] = result[i] / result_sum
            self.data_scale = result

        return self.data_scale

    def get_label_classes(self):
        return self.labels

    def _save2mongo(self, nplist, table):
        # because we had clean the data, we can insert them into database directly
        data = []
        for i in nplist:
            data_in = dict(zip(self.columns, i))
            data.append(data_in)
        table.drop()
        table.insert_many(data)

    def _mongo2np(self, table):
        temp_data = []
        temp_labels = []
        label_titles = list(table.find()[0].keys())[-7:]
        # label_titles = list(table.find()[0].keys())[-6:]
        for hit in table.find():
            hit.pop('_id')
            data = []
            label = []
            label = [hit.pop(title) for title in label_titles]
            data = [v for k, v in hit.items()]
            temp_labels.append(label)
            temp_data.append(data)
        temp_labels = np.array(temp_labels)
        temp_data = np.array(temp_data)
        if table == self.table_train_normal:
            self.train_data = temp_data
            self.train_label = temp_labels
        elif table == self.table_test_normal:
            self.test_data = temp_data
            self.test_label = temp_labels
        elif table == self.table_calibration_normal:
            self.calibration_data = temp_data
            self.calibration_label = temp_labels
        elif table == self.table_robustC_normal:
            self.robustC_data = temp_data
            self.robustC_label = temp_labels
        elif table == self.table_credibilityC:
            self.credibilityC_data = temp_data
            self.credibilityC_label = temp_labels

    def data_sort(self, file_path):
        data_path = [
            'F:/tools/AWScli/data_set/GeneratedLabelledFlows/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
            'F:/tools/AWScli/data_set/GeneratedLabelledFlows/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
            'F:/tools/AWScli/data_set/GeneratedLabelledFlows/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
            'F:/tools/AWScli/data_set/GeneratedLabelledFlows/Friday-WorkingHours-Morning.pcap_ISCX.csv',
            'F:/tools/AWScli/data_set/GeneratedLabelledFlows/Wednesday-workingHours.pcap_ISCX.csv',
            'F:/tools/AWScli/data_set/GeneratedLabelledFlows/Tuesday-WorkingHours.pcap_ISCX.csv',
            'F:/tools/AWScli/data_set/GeneratedLabelledFlows/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv']
        df = pd.read_csv(data_path[0], header=0)
        for i in range(1, len(data_path)):
            if i == 2:
                df_tem = pd.read_csv(data_path[i], header=0, encoding='unicode_escape', low_memory=False)
            else:
                df_tem = pd.read_csv(data_path[i], header=0)
            df = df.append(df_tem)
        df = df.drop(columns=['Flow ID', ' Timestamp'])
        df = df.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
        df = df.drop(columns=[' Fwd Header Length.1'])
        df = df.sample(frac=1)

        print(df[' Label'].value_counts(normalize=True))
        a = input()

        def repl(string):
            matchobj = re.match(r'.+', string, re.M | re.I)
            result = re.findall(r'(?:[0-9]{1,3})', matchobj.group(), re.M | re.I)
            return int(result[0]) * 256 + int(result[1])

        df[' Destination IP'] = df[' Destination IP'].apply(lambda x: repl(x))
        df[' Source IP'] = df[' Source IP'].apply(lambda x: repl(x))

        df[' Label'].replace(['DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest'], 'Dos', inplace=True)
        df[' Label'].replace(
            ['Web Attack  Brute Force', 'Web Attack  XSS', 'Web Attack  Sql Injection', 'Infiltration',
             'Heartbleed'], 'Web Attack', inplace=True)
        temp = df[' Label'].value_counts(normalize=True)

        for i in temp.index:
            df_temp = df.loc[df[' Label'] == i]
            df_temp.to_csv(file_path + '/CIC_IDS_2017_' + i + '.csv', index=False)

    def run(self):
        print('start data pre-sort')
        self.data_sort('./data')
        # print('start data process')
        # self._data_processing()
        # self._save2mongo(self.test_data, self.table_test_normal)
        # self._save2mongo(self.train_data, self.table_train_normal)
        # self._save2mongo(self.calibration_data, self.table_calibration_normal)
        # # self._save2mongo(self.robustC_data, self.table_robustC_normal)
        # self._save2mongo(self.credibilityC_data, self.table_credibilityC)
        # print('data process is over')
        # self._mongo2np(self.table_calibration_normal)
        # self._mongo2np(self.table_train_normal_normal)
        # self._mongo2np(self.table_test_normal)


class model_build():
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    @property
    def model_simple(self):
        model = tf.keras.models.Sequential(name='simple')
        model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(self.input_dim,)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(self.output_dim, activation=tf.keras.activations.softmax))
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.categorical_accuracy])
        model.summary()

        return model

    @property
    def model_gru(self):
        model = tf.keras.Sequential(name='gru')
        model.add(tf.keras.layers.GRU(128, input_shape=(1, self.input_dim), return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.GRU(128, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.GRU(128, return_sequences=False))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(self.output_dim, activation=tf.keras.activations.softmax))

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.categorical_accuracy])
        model.summary()

        return model

    @property
    def model_lstm(self):
        model = tf.keras.models.Sequential(name='lstm')
        model.add(tf.keras.layers.LSTM(128, input_dim=self.input_dim, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(128, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(128, return_sequences=False))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(self.output_dim, activation=tf.keras.activations.sigmoid))

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.categorical_accuracy])
        model.summary()

        return model

    @property
    def model_cnn_gru(self):
        """
        cnn_gru　模型
        """
        model = tf.keras.models.Sequential(name='cnn_gru')
        model.add(tf.keras.layers.Conv1D(128, 1, activation='relu', input_shape=(1, self.input_dim)))  # input=(None, 1, 68)
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Conv1D(64, 1, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Conv1D(32, 1, activation='relu'))
        # model.add(tf.keras.layers.GRU(32, input_shape=(None, 32), return_sequences=True))
        model.add(tf.keras.layers.GRU(64, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.GRU(64, return_sequences=False))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(self.output_dim, activation=tf.keras.activations.softmax))
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.categorical_accuracy])
        model.summary()

        return model

    @property
    def model_cnn_lstm(self):
        model = tf.keras.models.Sequential(name='cnn_lstm')
        model.add(tf.keras.layers.Conv1D(128, 1, activation='relu', input_shape=(None, self.input_dim)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Conv1D(64, 1, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Conv1D(32, 1, activation='relu'))
        model.add(tf.keras.layers.LSTM(64, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(64, return_sequences=False))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(self.output_dim, activation=tf.keras.activations.softmax))
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.categorical_accuracy])

        model.summary()

        return model

    @property
    def model_knn(self):
        knn = KNeighborsClassifier(n_neighbors=3)
        return knn

class DataRecord:
    """
    该类包含的信息是
    model: 模型的model 信息，该模型是.h5 类型数据
           是tensorflow用于保存模型的特定格式
           该类型的数据保存了模型的包括模型结构，规模大小,
           模型各个层的权重，参数信息
    history: 保存了模型训练过程中的损失和精度所有值，
             该信息需要保存下来为后期的绘制模型训练图提供数据
    summary: 模型的结构信息
    epochs: 模型的迭代次数信息
    """
    model: str
    history: str
    summary: str
    epochs: int


class model_train():
    def __init__(self, model, name):
        self.data = data_process('Dknn_dataset_1')
        self.epochs = 5
        self.choose_model = model
        self.model_choice = name
        self.save_dir = f'./history_more_{self.epochs}'

    def model_fit(self):
        model = self.choose_model
        print('start load data...')
        train_data, train_labels = self.data.get_traindata()
        if self.model_choice != 'simple':
            train_data = np.expand_dims(train_data, axis=1)

        print('start to train...')
        if self.model_choice == 'knn':
            history = model.fit(train_data, train_labels)
            with open('./history_' + self.epochs + '/knn/model.h5', 'wb') as f:
                pickle.dump(history, f)

        else:
            history = model.fit(train_data, train_labels,
                                epochs=self.epochs, batch_size=256,
                                validation_split=0.4, shuffle=True)
            data_record = DataRecord()
            data_record.model = model
            data_record.summary = model.to_yaml()
            data_record.history = history
            data_record.epochs = self.epochs
            # data_record.times = i
            self.result_save(data_record)

    def result_save(self, data_record):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        # name = time.strftime('%mm%dd_%Hh%Mm%Ss', time.localtime(time.time()))
        path = os.path.join(self.save_dir, f'{self.model_choice}')
        if not os.path.exists(path):
            os.mkdir(path)
        data_record.model.save(os.path.join(path, 'model.h5'))
        with open(os.path.join(path, 'history.json'), 'w') as f:
            # print(data_record.history.history)
            data = {}
            for key, value in data_record.history.history.items():
                data[key] = [float(i) for i in value]
            f.write(json.dumps(data))
        with open(os.path.join(path, 'summary_epochs.txt'), 'w') as f:
            f.write(data_record.summary)
            f.write('\n\n')
            f.write('epochs:')
            f.write(str(data_record.epochs))

    def run(self):
        print('start model trainning')
        start_time = time.strftime('%Y-%m-%d: %H-%M-%S', time.localtime())
        start = time.time()
        self.model_fit()
        end = time.time()
        end_time = time.strftime('%Y-%m-%d: %H-%M-%S', time.localtime())
        print('start time ---------', start_time)
        print('end time -----------', end_time)
        print(f'cost time ---------:{end - start}')


class DataSet(object):
    data: list
    labels: list


class ShowResults(object):
    def __init__(self):
        self.save_dir = os.path.join('./history_more_5')
        self.data = data_process('Dknn_dataset_1')
        self.epochs = 5
        self.eval_result = None

    def get_files_path(self, flag=''):
        paths = []
        for _dir in os.listdir(self.save_dir):
            _dir_path = os.path.join(self.save_dir, _dir)
            for file in os.listdir(_dir_path):
                if file.endswith(flag):
                    paths.append(os.path.join(self.save_dir, _dir, file))
        return paths

    def roc(self, predictions, labels, model_name):
        # predictions_index = np.argmax(predictions, axis=1)
        # for i in range(len(predictions)):
        #     predictions[i] = [1 if j == predictions_index[i] else 0 for j in range(len(predictions[i]))]
        # print('------------------------------')
        # print(predictions[:20])
        # print(np.unique(testlabels, axis=0))
        # a = input()
        # f1_results = metrics.f1_score(testlabels, predictions, average='macro')
        # acc_results = metrics.f1_score(testlabels, predictions, average='micro')
        # pre_results = metrics.precision_score(testlabels, predictions, average='macro')
        # recall_results = metrics.recall_score(testlabels, predictions, average='macro')
        # print(f'f1-measure  score{f1_results}')
        # print(f'accuracy score {acc_results}')
        # print(f'precision score　{pre_results}')
        # print(f'recall score {recall_results}')
        print(f'model {model_name}----')
        fpr, tpr, threshold = metrics.roc_curve(labels.ravel(), predictions.ravel())
        print(fpr)
        print(tpr)
        print(threshold)
        x_labels = fpr
        y_labels = tpr
        return x_labels, y_labels

    def metric_infos(self, predictions, labels, model_name):
        # predictions = np.nan_to_num(predictions)
        labels = np.nan_to_num(labels)
        auc_results = metrics.roc_auc_score(y_true=labels, y_score=predictions)
        predictions_index = np.argmax(predictions, axis=1)
        for i in range(len(predictions)):
            predictions[i] = [1 if j == predictions_index[i] else 0 for j in range(len(predictions[i]))]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        label_kinds = np.flipud(np.unique(labels, axis=0))
        self.eval_result = [{} for i in label_kinds]
        for i in range(len(label_kinds)):
            for j in range(len(predictions)):
                if (predictions[j] == labels[j]).all():
                    if (predictions[j] == label_kinds[i]).all():
                        tn += 1
                    else:
                        tp += 1
                else:
                    if (predictions[j] == label_kinds[i]).all():
                        fn += 1
                    else:
                        fp += 1
            self.eval_result[i]['acc'] = (tp + tn) / (tp + tn + fp + fn)
            self.eval_result[i]['dr'] = tp / (tp + fn)
            self.eval_result[i]['far'] = fp / (tn + fp)
            self.eval_result[i]['precision'] = tp / (tp + fp)
            self.eval_result[i]['recall'] = tp / (tp + fn)
            self.eval_result[i]['f1_measure'] = 2 * tp / (2 * tp + fp + fn)

        acc = 0
        dr = 0
        far = 0
        precision = 0
        recall = 0
        f1_measure = 0
        data_scale = self.data.get_data_scale()
        for i in range(len(data_scale)):
            acc = acc + self.eval_result[i]['acc'] * data_scale[i]
            dr = dr + self.eval_result[i]['dr'] * data_scale[i]
            far = far + self.eval_result[i]['far'] * data_scale[i]
            precision = precision + self.eval_result[i]['precision'] * data_scale[i]
            recall = recall + self.eval_result[i]['recall'] * data_scale[i]
            f1_measure = f1_measure + self.eval_result[i]['f1_measure'] * data_scale[i]

        print(f'version-network for{model_name}-------------')
        print(f'f1-measure  score: {f1_measure}')
        print(f'accuracy score: {acc}')
        print(f'detect rate score: {dr}')
        print(f'far score: {far}')
        print(f'precision score:　{precision}')
        print(f'recall score: {recall}')
        print(f'auc score {auc_results}')

    def metric_info_everylabelclasses(self):
        label_classes = self.data.get_label_classes()
        testdata, labels = self.data.get_testdata()
        for model_path in self.get_files_path('h5'):
            model_name = model_path.split('/')[-2]
            start = time.time()
            model = tf.keras.models.load_model(model_path)
            if 'simple' not in model_path:
                predictions = model.predict(np.expand_dims(testdata, axis=1))
            else:
                predictions = model.predict(testdata)
            if predictions.shape[1] == 1:
                predictions = np.squeeze(predictions)
            for i in range(1, len(label_classes)):
                fpr, tpr, _ = metrics.roc_curve(labels[:, i], predictions[:, i])
                plt.plot(fpr, tpr, label=label_classes[i])
            plt.title(model_name)
            plt.legend()
            plt.show()

    def show_models_metrics(self):
        print('start load data...')
        testdata, testlabels = self.data.get_robustCdata()
        for model_path in self.get_files_path('h5'):
            model_name = model_path.split('/')[-2]
            start = time.time()
            if model_name == 'knn':
                with open('./history_' + self.epochs + '/knn/model.h5', 'wb') as f:
                    model = pickle.load(f)
            else:
                model = tf.keras.models.load_model(model_path)
            if 'simple' not in model_path:
                predictions = model.predict(np.expand_dims(testdata, axis=1))
            else:
                predictions = model.predict(testdata)
            if predictions.shape[1] == 1:
                predictions = np.squeeze(predictions)
            # x_labels, y_labels = self.roc(predictions, testlabels, model_name)
            # plt.plot(x_labels, y_labels, label=model_name)
            self.metric_infos(predictions, testlabels, model_name)

            end = time.time()
            print(f'cost time: {end - start}')
            print('---------------------------------')
        # plt.title('general')
        # plt.legend()
        # plt.show()

    def run(self):
        self.show_models_metrics()
        # self.metric_info_everylabelclasses()


class dknn():
    def __init__(self):
        self.model_dir = os.path.join('./history_more_5')
        self.model = None
        self.model_name = None
        self.data = data_process('Dknn_dataset_1')
        self.k = 50

    def _load_model(self, model_name):
        flag = '.h5'
        paths = []
        for _dir in os.listdir(self.model_dir):
            _dir_path = os.path.join(self.model_dir, _dir)
            for file in os.listdir(_dir_path):
                if file.endswith(flag):
                    paths.append(os.path.join(self.model_dir, _dir, file))

        for model_path in paths:
            if model_name == model_path.split('/')[-2]:
                if model_name == 'knn':
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                else:
                    self.model = load_model(model_path)
                    self.model_name = model_name
                    self.model.summary()

    def _create_adversarial_pattern(self, input_image, input_label):
        loss_object = tf.keras.losses.CategoricalCrossentropy()

        input_image = tf.convert_to_tensor(input_image)

        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = self.model(input_image)
            loss = loss_object(input_label, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, input_image)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        return signed_grad

    def _kneighbors(self, k, x, knns, function):
        _result = function(x)
        result = []
        for i in range(len(_result)):
            result.append(knns[i].kneighbors(np.squeeze(_result[i]), n_neighbors=k, return_distance=True))
        #     temp = []
        #     for z in range(len(_result[i])):
        #         distances = []
        #         for n in range(len(layers_output[i])):
        #             distances.append(np.linalg.norm(_result[i][z][0] - layers_output[i][n][0]))
        #         nearest = np.argsort(distances)
        #         temp.append([i for i in nearest[:k]])
        #     result.append(temp)
        return result

    def _calibrate(self, knns, X, y, k, data_label, function):
        """compute calibration value and return [[number of data with different label]]"""
        _result = self._kneighbors(k, X, knns, function)
        layer_number = len(_result)
        # get a_value of input x, layer by layer.
        result = [[] for n in range(len(_result[0][1]))]
        # interval = []
        for i in range(len(_result)):
            # max = 0
            # interval.append([])
            for z in range(len(_result[i][1])):
                a_value = 0
                for j in range(len(_result[i][1][z])):
                    if (data_label[_result[i][1][z][j]] != y[z]).any():
                        # print('######%d layer:here data_label[%d] != y[%d], distance = %f' % (i, j, z, _result[i][0][z][j]))
                        # while _result[i][0][z][j] > max:
                        #     interval[i].append(0)
                        #     max = max + 0.2
                        # for m in range(int(max/0.2)):
                        #     if m * 0.2 <= _result[i][0][z][j] < (m + 1) * 0.2:
                        #         interval[i][m] += 1
                        #         break
                        a_value = a_value + 1 * _result[i][0][z][j] ** (-1)
                result[z].append(a_value)
        # print(interval)
        # a = input()
        return result

    def _Nonconformity(self, knns, data, k, data_label, function):
        """compute every a_value and return [[[number of data which are different from all kinds of its label]]]"""
        _result = self._kneighbors(k, data, knns, function)
        # get every value in label
        label_kinds = np.flipud(np.unique(data_label, axis=0))
        # print(label_kinds)
        # get a_value of input x, layer by layer.
        result = [[[] for l in range(len(label_kinds))] for n in range(len(_result[0][1]))]
        for i in range(len(_result)):
            for z in range(len(_result[i][1])):
                for x in range(len(label_kinds)):
                    a_value = 0
                    for j in range(len(_result[i][1][z])):
                        if (data_label[_result[i][1][z][j]] != label_kinds[x]).any():
                            #
                            a_value = a_value + 1 * _result[i][0][z][j] ** (-1)
                    result[z][x].append(a_value)
        return result

    def _compute_pvalue(self, calibrations, a_values):
        """compute p_values"""
        result = []
        cali_len = len(calibrations)
        for i in a_values:
            _result = []
            for ii in i:
                count = 0
                for j in calibrations:
                    cali_value = sum(j)
                    a_value = sum(ii)
                    if a_value <= cali_value:
                        count = count + 1
                _result.append(count / cali_len)
            result.append(_result)
        return result

    def __com_confidence(self, predictions, test_label):
        b = np.argmax(predictions, axis=1)
        z = 0
        _prd_conf = []
        for i in b:
            _prd_conf.append((i, predictions[z][i]))
            z = z + 1
        interval = [[] for i in range(10)]
        for i in range(len(_prd_conf)):
            if _prd_conf[i][1] < 0.1:
                interval[0].append((i, _prd_conf[i][0]))
            elif 0.2 > _prd_conf[i][1] >= 0.1:
                interval[1].append((i, _prd_conf[i][0]))
            elif 0.3 > _prd_conf[i][1] >= 0.2:
                interval[2].append((i, _prd_conf[i][0]))
            elif 0.4 > _prd_conf[i][1] >= 0.3:
                interval[3].append((i, _prd_conf[i][0]))
            elif 0.5 > _prd_conf[i][1] >= 0.4:
                interval[4].append((i, _prd_conf[i][0]))
            elif 0.6 > _prd_conf[i][1] >= 0.5:
                interval[5].append((i, _prd_conf[i][0]))
            elif 0.7 > _prd_conf[i][1] >= 0.6:
                interval[6].append((i, _prd_conf[i][0]))
            elif 0.8 > _prd_conf[i][1] >= 0.7:
                interval[7].append((i, _prd_conf[i][0]))
            elif 0.9 > _prd_conf[i][1] >= 0.8:
                interval[8].append((i, _prd_conf[i][0]))
            elif 1 >= _prd_conf[i][1] >= 0.9:
                interval[9].append((i, _prd_conf[i][0]))

        rate_right = [0 for i in range(10)]
        for i in range(len(interval)):
            num = len(interval[i])
            if num == 0:
                continue
            num_right = 0
            for j in interval[i]:
                if j[1] == np.argmax(test_label[j[0]]):
                    num_right = num_right + 1
            rate_right[i] = num_right / num
        data_num = len(test_label)
        data_confidence_scale = [len(i) / data_num for i in interval]
        return data_confidence_scale, rate_right

    def __com_dknn_confidence(self, predictions, test_label):
        b = np.argmax(predictions, axis=1)
        z = 0
        _prd_conf = []
        for i in b:
            _prediction = list(predictions[z])
            # print('z = %d--------------------------' % z)
            # print('origin max : %f' % (_prediction[i]))
            _prediction.pop(i)
            # print('now max : %f' % max(_prediction))
            # print('z : %d, confidence: %f' % (z, 1 - max(_prediction)))
            _prd_conf.append((i, 1 - max(_prediction)))
            z = z + 1

        interval = [[] for i in range(10)]
        for i in range(len(_prd_conf)):
            if _prd_conf[i][1] < 0.1:
                interval[0].append((i, _prd_conf[i][0]))
            elif 0.2 > _prd_conf[i][1] >= 0.1:
                interval[1].append((i, _prd_conf[i][0]))
            elif 0.3 > _prd_conf[i][1] >= 0.2:
                interval[2].append((i, _prd_conf[i][0]))
            elif 0.4 > _prd_conf[i][1] >= 0.3:
                interval[3].append((i, _prd_conf[i][0]))
            elif 0.5 > _prd_conf[i][1] >= 0.4:
                interval[4].append((i, _prd_conf[i][0]))
            elif 0.6 > _prd_conf[i][1] >= 0.5:
                interval[5].append((i, _prd_conf[i][0]))
            elif 0.7 > _prd_conf[i][1] >= 0.6:
                interval[6].append((i, _prd_conf[i][0]))
            elif 0.8 > _prd_conf[i][1] >= 0.7:
                interval[7].append((i, _prd_conf[i][0]))
            elif 0.9 > _prd_conf[i][1] >= 0.8:
                interval[8].append((i, _prd_conf[i][0]))
            elif 1 > _prd_conf[i][1] >= 0.9:
                interval[9].append((i, _prd_conf[i][0]))

        rate_right = [0 for i in range(10)]
        for i in range(len(interval)):
            num = len(interval[i])
            if num == 0:
                continue
            num_right = 0
            for j in interval[i]:
                if j[1] == np.argmax(test_label[j[0]]):
                    num_right = num_right + 1
            rate_right[i] = num_right / num
        data_num = len(test_label)
        data_confidence_scale = [len(i) / data_num for i in interval]
        return data_confidence_scale, rate_right

    def __dknn_model_score(self, p_values, test_labels, log_file):
        label_shape = test_labels.shape[-1]
        b = np.argmax(p_values, axis=1)
        # print(p_values[:20])
        # print(b[:20])
        # a = input()
        predictions = []
        for i in range(len(b)):
            predictions.append([1 if j == b[i] else 0 for j in range(label_shape)])
        predictions = np.array(predictions)

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        label_kinds = np.flipud(np.unique(test_labels, axis=0))
        eval_result = [{} for i in label_kinds]
        for i in range(len(label_kinds)):
            for j in range(len(predictions)):
                if (predictions[j] == test_labels[j]).all():
                    if (predictions[j] == label_kinds[i]).all():
                        tn += 1
                    else:
                        tp += 1
                else:
                    if (predictions[j] == label_kinds[i]).all():
                        fn += 1
                    else:
                        fp += 1
            eval_result[i]['acc'] = (tp + tn) / (tp + tn + fp + fn)
            eval_result[i]['dr'] = tp / (tp + fn) if tp + fn != 0 else -1
            eval_result[i]['far'] = fp / (tn + fp) if tn + fp != 0 else -1
            eval_result[i]['precision'] = tp / (tp + fp) if tp + fp != 0 else -1
            eval_result[i]['recall'] = tp / (tp + fn) if tp + fn != 0 else -1
            eval_result[i]['f1_measure'] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn != 0 else -1

        acc = 0
        dr = 0
        far = 0
        precision = 0
        recall = 0
        f1_measure = 0
        data_scale = self.data.get_data_scale()
        for i in range(len(data_scale)):
            acc = acc + eval_result[i]['acc'] * data_scale[i]
            dr = dr + eval_result[i]['dr'] * data_scale[i]
            far = far + eval_result[i]['far'] * data_scale[i]
            precision = precision + eval_result[i]['precision'] * data_scale[i]
            recall = recall + eval_result[i]['recall'] * data_scale[i]
            f1_measure = f1_measure + eval_result[i]['f1_measure'] * data_scale[i]

        f1_results = metrics.f1_score(test_labels, predictions, average='macro')
        acc_results = metrics.f1_score(test_labels, predictions, average='micro')
        pre_results = metrics.precision_score(test_labels, predictions, average='macro')
        recall_results = metrics.recall_score(test_labels, predictions, average='macro')
        auc_results = metrics.roc_auc_score(y_true=test_labels, y_score=np.array(p_values))
        print('version-tensorflow-------------------------------------', file=log_file)
        print(f'f1-measure  score   {f1_results}', file=log_file)
        print(f'accuracy score      {acc_results}', file=log_file)
        print(f'precision score     {pre_results}', file=log_file)
        print(f'recall score        {recall_results}', file=log_file)
        print('version-network----------------------------------------', file=log_file)
        print(f'f1-measure  score:  {f1_measure}', file=log_file)
        print(f'accuracy score:     {acc}', file=log_file)
        print(f'detect rate score:  {dr}', file=log_file)
        print(f'far score:          {far}', file=log_file)
        print(f'precision score:    {precision}', file=log_file)
        print(f'recall score:       {recall}', file=log_file)
        print('auc socre----------------------------------------------', file=log_file)
        print(f'auc score           {auc_results}', file=log_file)

        fpr, tpr, _ = metrics.roc_curve(test_labels.ravel(), np.array(p_values).ravel())

        return fpr, tpr, auc_results

    def __credibility_plot(self, data_scale, data_Rrate, title):
        fig = plt.figure(figsize=(10, 10))
        X = [i * 0.1 + 0.05 for i in range(10)]
        ax1 = fig.add_subplot()
        ax1.set_ylim([0, 1.1])
        ax1.bar(X, data_Rrate, width=0.08, color='yellow')
        ax1.set_ylabel('right prediction rate', fontsize='15')
        for x, y in zip(X, data_Rrate):
            plt.text(x, y+0.005, '%.4f' % y, ha='center', va='bottom', fontsize=12)
        ax2 = ax1.twinx()
        ax2.set_ylim([0, 1.1])
        ax2.plot(X, data_scale, 'o-', ms=10, color='red')
        ax2.set_ylabel('data_scale', fontsize='15', color='red')
        ax2.set_xlabel("credibility region", fontsize='15')  # 横坐标名字
        plt.title((self.model_name + '_%d ' + title) % self.k)
        plt.savefig('./history/roc_credibilityCheck/' + self.model_name + '_%d ' % self.k + title + '.png')

    def __save_dknn(self, save_dir, data, data_name):
        with open(save_dir + '\\DkNN-' + self.model_name + '\\' + data_name + '.h5', mode='wb') as f:
            pickle.dump(data, f, 2)

    def __load_dknn(self, save_dir, data_name):
        with open(save_dir + '\\DkNN-' + self.model_name+ '\\' + data_name + '.h5', mode='rb') as f:
            return pickle.load(f)


    def run_first(self):
        # load train, clibration, test data and labels
        train_data, train_labels = self.data.get_traindata()
        # train_data = train_data[60000:]  # [60000:] ,smaller dataset for quicker train
        # train_labels = train_labels[60000:]
        calibration_data, calibration_labels = self.data.get_calibrationdata()
        # calibration_data = calibration_data[:10000]
        # calibration_labels = calibration_labels[:10000]
        test_data, test_labels = self.data.get_testdata()
        # test_data = test_data[10000:20000]
        # test_labels = test_labels[10000:20000]
        # write program information into file
        log = open('./history/log.txt', 'a')
        print(f'\n###data recorded at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}######', file=log, flush=True)
        # load models
        model_list = ['simple', 'lstm', 'cnn_lstm', 'cnn_gru', 'gru']   #
        for i in model_list:
            print(f'start DkNN analyzing on {i}')
            self.k = 50
            self._load_model(i)
            if self.model_name != 'simple' and test_data.shape[1] != 1:
                test_data = np.expand_dims(test_data, axis=1)

            start = time.time()
            predictions = self.model.predict(test_data)
            end = time.time()
            predictions = np.squeeze(predictions)   # cancel some meaningless dimmension
            dnn_fpr, dnn_tpr, dnn_auc = self.__dknn_model_score(predictions, test_labels, log)
            print(f'predicting cost time :{end-start}(ModelName: {self.model_name})', file=log)

            # # get robust check data
            # robust_test_data = self._create_adversarial_pattern(test_data, test_labels)
            # robust_test_data = test_data + robust_test_data
            # robust_dnn_result = self.model.predict(robust_test_data)
            # robust_dnn_result = np.squeeze(robust_dnn_result)

            print('dnn confidence accuracy:(ModelName: %s, K: %d)' % (self.model_name, self.k), file=log)
            dnn_result = self.__com_confidence(predictions, test_labels)
            # dnn_robust_result = self.__com_confidence(robust_dnn_result, test_labels)
            print(dnn_result, file=log, flush=True)
            # print('-' * 15, file=log)
            # print(dnn_robust_result, file=log, flush=True)

            self.__credibility_plot(dnn_result[0], dnn_result[1], 'normal')
            # self.__credibility_plot(dnn_robust_result[0], dnn_robust_result[1], 'robustCheck')

            # get all layers outputs
            if self.model_name != 'simple' and train_data.shape[1] != 1:
                train_data = np.expand_dims(train_data, axis=1)
            # get_cnngru_alllayer_outputs = kb.function([self.model.layers[0].input],
            #                                     [self.model.layers[0].output, self.model.layers[2].output,
            #                                      self.model.layers[4].output, self.model.layers[5].output,
            #                                      self.model.layers[7].output, self.model.layers[9].output,
            #                                      self.model.layers[10].output])
            get_alllayer_outputs = kb.function([self.model.layers[0].input],
                                               [l.output for l in self.model.layers[0:] if not isinstance(l, tf.keras.layers.Dropout)])
            # print debug information into ./history/log.txt
            print('start get layers output process(ModelName: %s)' % self.model_name, file=log)
            all_layers_output = get_alllayer_outputs(train_data)

            knns = []
            for z in range(len(all_layers_output)):
                knn = NearestNeighbors(n_neighbors=self.k, radius=1.0, p=2)
                if len(all_layers_output[z].shape) > 2:
                    knn.fit(np.squeeze(all_layers_output[z], axis=1))
                    knns.append(knn)
                else:
                    knn.fit(all_layers_output[z])
                    knns.append(knn)
            self.__save_dknn('.\\history', knns, 'knns')

            for j in range(2):
                print('start calibration process(ModelName: %s, K: %d)' % (self.model_name, self.k), file=log, flush=True)
                # load calibration data, and compute calibration_values
                start = time.time()
                if self.model_name != 'simple' and calibration_data.shape[1] != 1:
                    calibration_data = np.expand_dims(calibration_data, axis=1)
                result_cli = self._calibrate(knns, calibration_data, calibration_labels, self.k,
                                             train_labels, get_alllayer_outputs)
                end = time.time()
                print(f'training costs time :{end-start}(dknn ModelName: {self.model_name})', file=log)
                self.__save_dknn('.\\history', result_cli, 'calibration')

                # zero_num = 0
                # for i in result_cli:
                #     if np.all(i) == False:
                #         zero_num += 1
                # zero_rate = zero_num / len(result_cli)
                # print('K == %d zero_rate in result_cli is %f' % (self.k, zero_rate))
                print(result_cli[:20], file=log, flush=True)
                print('start test process(ModelName: %s, K: %d)' % (self.model_name, self.k), file=log)
                # load test data, and compute calibration_values
                if self.model_name != 'simple' and test_data.shape[1] != 1:
                    test_data = np.expand_dims(test_data, axis=1)
                start = time.time()
                result_test = self._Nonconformity(knns, test_data,
                                                  self.k, np.append(train_labels, test_labels,axis=0), get_alllayer_outputs)
                print(result_test[:20], file=log, flush=True)

                # compute the p_value
                print('start compute p_value(ModelName: %s, K: %d)' % (self.model_name, self.k), file=log, flush=True)
                p_values = self._compute_pvalue(result_cli, result_test)
                end = time.time()
                print(f'predicting costs time :{end-start}(dknn ModelName: {self.model_name})', file=log)
                print(p_values[:20], file=log)

                # # get robustness information
                # result_robust_test = self._Nonconformity(knns, robust_test_data,
                #                                          self.k, np.append(train_labels, test_labels, axis=0),
                #                                          get_alllayer_outputs)
                # p_values_robust = self._compute_pvalue(result_cli, result_robust_test)

                print('dknn--dnn acccuracy comparation:(ModelName: %s, K: %d)' % (self.model_name, self.k), file=log)
                dknn_fpr, dknn_tpr ,dknn_auc= self.__dknn_model_score(p_values, test_labels, log)
                plt.clf()
                plt.cla()
                plt.plot((0, 1), (0, 1), ls='--')
                plt.plot(dknn_fpr, dknn_tpr, label='dknn' + self.model_name + 'auc: %.4f' % dknn_auc)
                plt.plot(dnn_fpr, dnn_tpr, label=self.model_name + 'auc: %.4f' % dnn_auc)
                plt.title(self.model_name + '_k=%d' % self.k)
                plt.legend()
                plt.savefig('./history/roc/'+self.model_name + '_%d' % self.k + '.png')

                print('dknn confidence accuracy:(ModelName: %s, K: %d)' % (self.model_name, self.k), file=log)
                dknn_result = self.__com_confidence(p_values, test_labels)
                # dknn_robust_result = self.__com_confidence(p_values_robust, test_labels)
                print(dknn_result, file=log)
                # print('-' * 15, file=log)
                # print(dknn_robust_result, file=log, flush=True)

                # self.__credibility_plot(dknn_result[0], dknn_result[1], 'Dknn_normal')
                # self.__credibility_plot(dknn_robust_result[0], dknn_robust_result[1], 'Dknn_robustCheck')

                #increase K
                self.k = self.k + 50

        print('program finished', file=log)

    def run(self):
        # load train, clibration, test data and labels
        # train_data, train_labels = self.data.get_traindata()
        # train_data = train_data[60000:]  # [60000:] ,smaller dataset for quicker train
        # train_labels = train_labels[60000:]
        # calibration_data, calibration_labels = self.data.get_calibrationdata()
        # calibration_data = calibration_data[:10000]
        # calibration_labels = calibration_labels[:10000]
        test_data, test_labels = self.data.get_testdata()
        # test_data = test_data[10000:20000]
        # test_labels = test_labels[10000:20000]
        # write program information into file
        log = open('./history/log.txt', 'a')
        print(f'\n###data recorded at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}######', file=log,
              flush=True)
        # load models
        model_list = ['simple', 'lstm', 'cnn_lstm', 'cnn_gru', 'gru']  #
        for i in model_list:
            print(f'start DkNN analyzing on {i}')
            self.k = 50
            self._load_model(i)
            if self.model_name != 'simple' and test_data.shape[1] != 1:
                test_data = np.expand_dims(test_data, axis=1)

            start = time.time()
            predictions = self.model.predict(test_data)
            end = time.time()
            predictions = np.squeeze(predictions)  # cancel some meaningless dimmension
            dnn_fpr, dnn_tpr, dnn_auc = self.__dknn_model_score(predictions, test_labels, log)
            print(f'predicting cost time :{end - start}(ModelName: {self.model_name})', file=log)

            # # get robust check data
            # robust_test_data = self._create_adversarial_pattern(test_data, test_labels)
            # robust_test_data = test_data + robust_test_data
            # robust_dnn_result = self.model.predict(robust_test_data)
            # robust_dnn_result = np.squeeze(robust_dnn_result)

            print('dnn confidence accuracy:(ModelName: %s, K: %d)' % (self.model_name, self.k), file=log)
            dnn_result = self.__com_confidence(predictions, test_labels)
            # dnn_robust_result = self.__com_confidence(robust_dnn_result, test_labels)
            print(dnn_result, file=log, flush=True)
            # print('-' * 15, file=log)
            # print(dnn_robust_result, file=log, flush=True)

            self.__credibility_plot(dnn_result[0], dnn_result[1], 'normal')
            # self.__credibility_plot(dnn_robust_result[0], dnn_robust_result[1], 'robustCheck')

            # get all layers outputs
            if self.model_name != 'simple' and train_data.shape[1] != 1:
                train_data = np.expand_dims(train_data, axis=1)
            # get_cnngru_alllayer_outputs = kb.function([self.model.layers[0].input],
            #                                     [self.model.layers[0].output, self.model.layers[2].output,
            #                                      self.model.layers[4].output, self.model.layers[5].output,
            #                                      self.model.layers[7].output, self.model.layers[9].output,
            #                                      self.model.layers[10].output])
            get_alllayer_outputs = kb.function([self.model.layers[0].input],
                                               [l.output for l in self.model.layers[0:] if
                                                not isinstance(l, tf.keras.layers.Dropout)])
            # print debug information into ./history/log.txt
            # print('start get layers output process(ModelName: %s)' % self.model_name, file=log)
            # all_layers_output = get_alllayer_outputs(train_data)

            # knns = []
            # for z in range(len(all_layers_output)):
            #     knn = NearestNeighbors(n_neighbors=self.k, radius=1.0, p=2)
            #     if len(all_layers_output[z].shape) > 2:
            #         knn.fit(np.squeeze(all_layers_output[z], axis=1))
            #         knns.append(knn)
            #     else:
            #         knn.fit(all_layers_output[z])
            #         knns.append(knn)
            knns = self.__load_dknn('.\\history', 'knns')

            for j in range(2):
                # print('start calibration process(ModelName: %s, K: %d)' % (self.model_name, self.k), file=log,
                #       flush=True)
                # # load calibration data, and compute calibration_values
                # start = time.time()
                # if self.model_name != 'simple' and calibration_data.shape[1] != 1:
                #     calibration_data = np.expand_dims(calibration_data, axis=1)
                # result_cli = self._calibrate(knns, calibration_data, calibration_labels, self.k,
                #                              train_labels, get_alllayer_outputs)
                # end = time.time()
                # print(f'training costs time :{end - start}(dknn ModelName: {self.model_name})', file=log)
                result_cli = self.__load_dknn('.\\history', 'calibration')

                # zero_num = 0
                # for i in result_cli:
                #     if np.all(i) == False:
                #         zero_num += 1
                # zero_rate = zero_num / len(result_cli)
                # print('K == %d zero_rate in result_cli is %f' % (self.k, zero_rate))
                # print(result_cli[:20], file=log, flush=True)
                print('start test process(ModelName: %s, K: %d)' % (self.model_name, self.k), file=log)
                # load test data, and compute calibration_values
                if self.model_name != 'simple' and test_data.shape[1] != 1:
                    test_data = np.expand_dims(test_data, axis=1)
                start = time.time()
                result_test = self._Nonconformity(knns, test_data,
                                                  self.k, np.append(train_labels, test_labels, axis=0),
                                                  get_alllayer_outputs)
                print(result_test[:20], file=log, flush=True)

                # compute the p_value
                print('start compute p_value(ModelName: %s, K: %d)' % (self.model_name, self.k), file=log, flush=True)
                p_values = self._compute_pvalue(result_cli, result_test)
                end = time.time()
                print(f'predicting costs time :{end - start}(dknn ModelName: {self.model_name})', file=log)
                print(p_values[:20], file=log)

                # # get robustness information
                # result_robust_test = self._Nonconformity(knns, robust_test_data,
                #                                          self.k, np.append(train_labels, test_labels, axis=0),
                #                                          get_alllayer_outputs)
                # p_values_robust = self._compute_pvalue(result_cli, result_robust_test)

                print('dknn--dnn acccuracy comparation:(ModelName: %s, K: %d)' % (self.model_name, self.k), file=log)
                dknn_fpr, dknn_tpr, dknn_auc = self.__dknn_model_score(p_values, test_labels, log)
                plt.clf()
                plt.cla()
                plt.plot((0, 1), (0, 1), ls='--')
                plt.plot(dknn_fpr, dknn_tpr, label='dknn' + self.model_name + 'auc: %.4f' % dknn_auc)
                plt.plot(dnn_fpr, dnn_tpr, label=self.model_name + 'auc: %.4f' % dnn_auc)
                plt.title(self.model_name + '_k=%d' % self.k)
                plt.legend()
                plt.savefig('./history/roc/' + self.model_name + '_%d' % self.k + '.png')

                print('dknn confidence accuracy:(ModelName: %s, K: %d)' % (self.model_name, self.k), file=log)
                dknn_result = self.__com_confidence(p_values, test_labels)
                # dknn_robust_result = self.__com_confidence(p_values_robust, test_labels)
                print(dknn_result, file=log)
                # print('-' * 15, file=log)
                # print(dknn_robust_result, file=log, flush=True)

                # self.__credibility_plot(dknn_result[0], dknn_result[1], 'Dknn_normal')
                # self.__credibility_plot(dknn_robust_result[0], dknn_robust_result[1], 'Dknn_robustCheck')

                # increase K
                self.k = self.k + 50

        print('program finished', file=log)

if __name__ == '__main__':
    data_process('Dknn_dataset_1').run()

    # models_build = model_build(83, 7)
    # model_train(models_build.model_simple, 'simple').run()
    # model_train(models_build.model_gru, 'gru').run()
    # model_train(models_build.model_lstm, 'lstm').run()
    # model_train(models_build.model_cnn_gru, 'cnn_gru').run()
    # model_train(models_build.model_cnn_lstm, 'cnn_lstm').run()
    #
    # dknn_model = dknn()
    # dknn_model.run()
