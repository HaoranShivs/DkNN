import os
import glob
import json
import re
import random
import time
import pickle
import pandas as pd
import numpy as np

#import matplotlib.pyplot as plt
#import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

# from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
#import tensorflow.keras.backend as kb
#from tensorflow.keras.models import load_model


pd.set_option("display.max_columns", 20)


class data_process():
    def __init__(self, rawdata_path='./', data_path='./'):    # data_path is the path of the data that has been classified
        self.data_path = data_path
        self.rawdata_path = rawdata_path
        self.csv_path = glob.glob(os.path.join(self.data_path, '*'))
        self.columns = []
        self.labels = ['BENIGN', 'Bot', 'Dos', 'DDoS', 'FTP-Patator', 'PortScan', 'SSH-Patator', 'Web Attack']
        # self.train_data = np.array([])
        # self.train_label = np.array([])
        # self.calibration_data = np.array([])
        # self.calibration_label = np.array([])
        # self.test_data = np.array([])
        # self.test_label = np.array([])
        # self.credibilityC_data = np.array([])
        # self.credibilityC_label = np.array([])
        # self.robustC_data = np.array([])
        # self.robustC_label = np.array([])
        self.data_scale = None


    def _data_processing(self):
        df = pd.read_csv(self.csv_path[0], header=0, nrows=50000)
        # read 5 csv file
        for i in range(1, len(self.csv_path)):
            df_tem = pd.read_csv(self.csv_path[i], header=0, nrows=50000)
            df = df.append(df_tem)

        # make the sequence of data random
        df = df.sample(frac=0.9, random_state=1)
        # here to make columns ' name more concise
        temp = df[' Label'].value_counts(normalize=True)
        print(temp)
        a = input()
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

        # for i in range(len(left_columns)):
        #     temp_df = df[df[left_columns[i]] == 1].head(1)
        #     temp_df.iloc[:, :83] = 0
        #     _df = _df.append(temp_df)

        # # normal dataset
        # data = np.array(df)
        # for i in range(data.shape[1] - 7 -3):
        #     data[:, i] = (data[:, i] - min_nums[i]) / (max_nums[i] - min_nums[i]) if (max_nums[i] - min_nums[
        #         i]) != 0 else 0
        # data = pd.DataFrame(data, columns=self.columns)
        # print(data.shape)

        # self.train_data = data[:80000]
        # self.train_data.to_csv(self.data_path + '/../prepared_data/CIC_IDS_2017_traindata' + '.csv', index=False)
        # self.calibration_data = data[85000:105000]
        # self.calibration_data.to_csv(self.data_path + '/../prepared_data/CIC_IDS_2017_calibrationdata' + '.csv', index=False)
        # self.test_data = data[-30000:]
        # self.test_data.to_csv(self.data_path + '/../prepared_data/CIC_IDS_2017_testdata' + '.csv', index=False)

        # to make data to measure credibility
        _df = df[(df[' Label_Bot'] == 1) | (df[' Label_Web Attack'] == 1)]
        # left_columns = [' Label_BENIGN', ' Label_Dos', ' Label_FTP-Patator', ' Label_PortScan', ' Label_SSH-Patator']
        df = df[(df[' Label_Bot'] == 0) & (df[' Label_Web Attack'] == 0)]

        data = np.array(df)
        data_credibility = np.array(_df)
        for i in range(data.shape[1] - 7 -3):
            data[:, i] = (data[:, i] - min_nums[i]) / (max_nums[i] - min_nums[i]) if (max_nums[i] - min_nums[
                i]) != 0 else 0
            data_credibility[:, i] = (data_credibility[:, i] - min_nums[i]) / (max_nums[i] - min_nums[i]) if (max_nums[i] - min_nums[i]) != 0 else 0
        data = pd.DataFrame(data, columns=self.columns)
        data_credibility = pd.DataFrame(data_credibility, columns=self.columns)
        self.credibilityT_data = data[:80000]
        self.credibilityT_data.to_csv(self.data_path + '/../prepared_data/CIC_IDS_2017_credibilityTdata' + '.csv', index=False)
        self.credibilityC_data = data_credibility[-20000:]
        self.credibilityC_data.to_csv(self.data_path + '/../prepared_data/CIC_IDS_2017_credibilityCdata' + '.csv', index=False)
        self.credibility_cali_data = data[90000:110000]
        self.credibility_cali_data.to_csv(self.data_path + '/../prepared_data/CIC_IDS_2017_credibility_cali_data' + '.csv', index=False)

    def get_traindata(self, traindata_path):
        train_data = pd.read_csv(traindata_path, header=0)
        train_label = train_data.iloc[:,-8:]
        train_data = train_data.drop(train_data.columns[-8:], axis=1)
        train_data = np.array(train_data)
        train_label = np.array(train_label)
        print(f'train data shape is{train_data.shape}\ntrain label shape is{train_label.shape}')
        return train_data, train_label

    def get_calibrationdata(self, calibrationdata_path):
        calibration_data = pd.read_csv(calibrationdata_path, header=0)
        calibration_label = calibration_data.iloc[:,-8:]
        calibration_data = calibration_data.drop(calibration_data.columns[-8:], axis=1)
        calibration_data = np.array(calibration_data)
        calibration_label = np.array(calibration_label)
        print(f'calibration data shape is{calibration_data.shape}\ncalibration label shape is{calibration_label.shape}')
        return calibration_data, calibration_label

    def get_credibilityTdata(self, credibilityTdata_path):
        credibilityT_data = pd.read_csv(credibilityTdata_path, header=0)
        credibilityT_label = credibilityT_data.iloc[:,-8:]
        credibilityT_data = credibilityT_data.drop(credibilityT_data.columns[-8:], axis=1)
        credibilityT_data = np.array(credibilityT_data)
        credibilityT_label = np.array(credibilityT_label)
        print(f'credibilityT data shape is{credibilityT_data.shape}\ncredibilityT label shape is{credibilityT_label.shape}')
        return credibilityT_data, credibilityT_label

    def get_credibilityTdata(self, credibility_cali_data_path):
        credibility_cali_data = pd.read_csv(credibility_cali_data_path, header=0)
        credibility_cali_label = credibility_cali_data.iloc[:,-8:]
        credibility_cali_data = credibility_cali_data.drop(credibility_cali_data.columns[-8:], axis=1)
        credibility_cali_data = np.array(credibility_cali_data)
        credibility_cali_label = np.array(credibility_cali_label)
        print(f'credibilityT data shape is{credibility_cali_data.shape}\ncredibilityT label shape is{credibility_cali_label.shape}')
        return credibility_cali_data, credibility_cali_label

    def get_testdata(self, testdata_path):
        test_data = pd.read_csv(testdata_path, header=0)
        test_label = test_data.iloc[:,-8:]
        test_data = test_data.drop(test_data.columns[-8:], axis=1)
        test_data = np.array(test_data)
        test_label = np.array(test_label)
        print(f'test data shape is{test_data.shape}\ntest label shape is{test_label.shape}')
        return test_data, test_label

    def get_credibilityCdata(self, credibilityCdata_path):
        credibilityC_data = pd.read_csv(credibilityCdata_path, header=0)
        credibilityC_label = credibilityC_data.iloc[:,-8:]
        credibilityC_data = credibilityC_data.drop(credibilityC_data.columns[-8:], axis=1)
        credibilityC_data = np.array(credibilityC_data)
        credibilityC_label = np.array(credibilityC_label)
        print(f'credibilityC data shape is{credibilityC_data.shape}\ncredibilityC label shape is{credibilityC_label.shape}')
        return credibilityC_data, credibilityC_label

    # def get_robustCdata(self):
    #     if self.robustC_data.shape[0] == 0:
    #         self._mongo2np(self.table_robustC_normal)
    #     print(self.robustC_data.shape)
    #     print(self.robustC_label.shape)
    #     return self.robustC_data, self.robustC_label

    def get_data_scale(self, trainlabels, label_sample):
        if self.data_scale == None:
            data_labels = np.array(trainlabels)

            result = [0 for i in range(len(data_labels[0]))]
            for i in range(len(label_sample)):
                for j in data_labels:
                    if (j == label_sample[i]).all():
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

    def data_sort(self):
        rawdata = glob.glob(os.path.join(self.rawdata_path, '*'))
        df = pd.read_csv(rawdata[0], header=0)
        print(rawdata)
        for i in range(1, len(rawdata)):
            if i == 2:
                df_tem = pd.read_csv(rawdata[i], header=0, encoding='unicode_escape', low_memory=False)
            else:
                df_tem = pd.read_csv(rawdata[i], header=0)
            df = df.append(df_tem)
        df = df.drop(columns=['Flow ID', ' Timestamp'])
        df = df.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
        df = df.drop(columns=[' Fwd Header Length.1'])
        df = df.sample(frac=1)
        print(df)

        print(df[' Label'].value_counts(normalize=True))

        def repl(string):
            matchobj = re.match(r'.+', string, re.M | re.I)
            result = re.findall(r'(?:[0-9]{1,3})', matchobj.group(), re.M | re.I)
            return int(result[0]) * 256 + int(result[1])

        df[' Destination IP'] = df[' Destination IP'].apply(lambda x: repl(x))
        df[' Source IP'] = df[' Source IP'].apply(lambda x: repl(x))

        df[' Label'].replace(['DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest'], 'Dos', inplace=True)
        df[' Label'].replace(
            ['Web Attack � Brute Force', 'Web Attack � XSS', 'Web Attack � Sql Injection', 'Infiltration',
             'Heartbleed'], 'Web Attack', inplace=True)
        temp = df[' Label'].value_counts(normalize=True)

        for i in temp.index:
            df_temp = df.loc[df[' Label'] == i]
            df_temp.to_csv(self.data_path + '/CIC_IDS_2017_' + i + '.csv', index=False)

    def run(self):
        # print('start data pre-sort')
        # self.data_sort()
        print('start data process')
        self._data_processing()
        # self._save2mongo(self.test_data, self.table_test_normal)
        # self._save2mongo(self.train_data, self.table_train_normal)
        # self._save2mongo(self.calibration_data, self.table_calibration_normal)
        # # self._save2mongo(self.robustC_data, self.table_robustC_normal)
        # self._save2mongo(self.credibilityC_data, self.table_credibilityC)
        # print('data process is over')
        # self._mongo2np(self.table_calibration_normal)
        # self._mongo2np(self.table_train_normal_normal)
        # self._mongo2np(self.table_test_normal)

if __name__ == '__main__':
    data = data_process('F:/tools/AWScli/data_set/GeneratedLabelledFlows', './data')
    data.run()
    # train_data = data.get_traindata('F:/学校事务/学习Python/16272215-龚正阳-网络工程/16272215-龚正阳-源码2/STR-IDS_DKNN/DkNN/prepared_data/CIC_IDS_2017_traindata.csv')
    # test_data = data.get_testdata('F:/学校事务/学习Python/16272215-龚正阳-网络工程/16272215-龚正阳-源码2/STR-IDS_DKNN/DkNN/prepared_data/CIC_IDS_2017_testdata.csv')