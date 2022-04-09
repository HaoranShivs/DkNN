import os
import json
import time
import pickle
import numpy as np
import tensorflow as tf
from dataprocess import data_process

class model_build():
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    @property
    def model_simple(self):
        model = tf.keras.models.Sequential(name='simple')
        model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(self.input_dim,), name='dense1'))
        model.add(tf.keras.layers.Dropout(0.2, name='drop1'))
        model.add(tf.keras.layers.Dense(128, activation='relu', name='dense2'))
        model.add(tf.keras.layers.Dropout(0.2, name='drop2'))
        model.add(tf.keras.layers.Dense(128, activation='relu', name='dense3'))
        model.add(tf.keras.layers.Dropout(0.2, name='drop3'))
        model.add(tf.keras.layers.Dense(self.output_dim, activation=tf.keras.activations.softmax, name='dense4'))
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.categorical_accuracy])
        model.summary()

        return model

    @property
    def model_gru(self):
        model = tf.keras.Sequential(name='gru')
        model.add(tf.keras.layers.GRU(128, input_shape=(1, self.input_dim), return_sequences=True, name='gru1'))
        model.add(tf.keras.layers.Dropout(0.2, name='drop1'))
        model.add(tf.keras.layers.GRU(128, return_sequences=True, name='gru2'))
        model.add(tf.keras.layers.Dropout(0.2, name='drop2'))
        model.add(tf.keras.layers.GRU(128, return_sequences=False, name='gru3'))
        model.add(tf.keras.layers.Dropout(0.2, name='drop3'))
        model.add(tf.keras.layers.Dense(self.output_dim, activation=tf.keras.activations.softmax, name='dense1'))

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.categorical_accuracy])
        model.summary()

        return model

    @property
    def model_lstm(self):
        model = tf.keras.models.Sequential(name='lstm')
        model.add(tf.keras.layers.LSTM(128, input_dim=self.input_dim, return_sequences=True, name='lstm1'))
        model.add(tf.keras.layers.Dropout(0.2, name='drop1'))
        model.add(tf.keras.layers.LSTM(128, return_sequences=True, name='lstm2'))
        model.add(tf.keras.layers.Dropout(0.2, name='drop2'))
        model.add(tf.keras.layers.LSTM(128, return_sequences=False, name='lstm3'))
        model.add(tf.keras.layers.Dropout(0.2, name='drop3'))
        model.add(tf.keras.layers.Dense(self.output_dim, activation=tf.keras.activations.sigmoid, name='dense1'))

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.categorical_accuracy])
        model.summary()

        return model

    @property
    def model_cnn_gru(self):
        """
        cnn_gru模型
        """
        model = tf.keras.models.Sequential(name='cnn_gru')
        model.add(tf.keras.layers.Conv1D(128, 1, activation='relu', input_shape=(1, self.input_dim), name='conv1d1'))  # input=(None, 1, 68)
        model.add(tf.keras.layers.Dropout(0.2, name='drop1'))
        model.add(tf.keras.layers.Conv1D(96, 1, activation='relu', name='conv1d2'))
        model.add(tf.keras.layers.Dropout(0.2, name='drop2'))
        model.add(tf.keras.layers.Conv1D(128, 1, activation='relu', name='conv1d3'))
        # model.add(tf.keras.layers.GRU(32, input_shape=(None, 32), return_sequences=True))
        model.add(tf.keras.layers.GRU(128, return_sequences=True, name='gru1'))
        model.add(tf.keras.layers.Dropout(0.2, name='drop3'))
        model.add(tf.keras.layers.GRU(128, return_sequences=False, name='gru2'))
        model.add(tf.keras.layers.Dropout(0.2, name='drop4'))
        model.add(tf.keras.layers.Dense(64, activation='relu', name='dense1'))
        model.add(tf.keras.layers.Dense(self.output_dim, activation=tf.keras.activations.softmax, name='dense2'))
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.categorical_accuracy])
        model.summary()

        return model

    @property
    def model_cnn_lstm(self):
        model = tf.keras.models.Sequential(name='cnn_lstm')
        model.add(tf.keras.layers.Conv1D(128, 1, activation='relu', input_shape=(None, self.input_dim), name='conv1d1'))
        model.add(tf.keras.layers.Dropout(0.2, name='drop1'))
        model.add(tf.keras.layers.Conv1D(96, 1, activation='relu', name='conv1d2'))
        model.add(tf.keras.layers.Dropout(0.2, name='drop2'))
        model.add(tf.keras.layers.Conv1D(128, 1, activation='relu', name='conv1d3'))
        model.add(tf.keras.layers.LSTM(128, return_sequences=True, name='lstm1'))
        model.add(tf.keras.layers.Dropout(0.2, name='drop3'))
        model.add(tf.keras.layers.LSTM(128, return_sequences=False, name='lstm2'))
        model.add(tf.keras.layers.Dropout(0.2, name='drop4'))
        model.add(tf.keras.layers.Dense(64, activation='relu', name='dense1'))
        model.add(tf.keras.layers.Dense(self.output_dim, activation=tf.keras.activations.softmax, name='dense2'))
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.categorical_accuracy])

        model.summary()

        return model


class DataRecord:
    """
    该类包含的信息是
    model: 模型的model 信息,该模型是.h5 类型数据
           是tensorflow用于保存模型的特定格式
           该类型的数据保存了模型的包括模型结构,规模大小,
           模型各个层的权重,参数信息
    history: 保存了模型训练过程中的损失和精度所有值,
             该信息需要保存下来为后期的绘制模型训练图提供数据
    summary: 模型的结构信息
    epochs: 模型的迭代次数信息
    """
    model: str
    history: str
    summary: str
    epochs: int


class model_train():
    def __init__(self, model, name, traindata_path):
        self.data = data_process()
        self.data_path = traindata_path
        self.epochs = 5
        self.choose_model = model
        self.model_choice = name
        self.save_dir = f'F:/models_trained_for_DkNN/history_more_{self.epochs}'

    def model_fit(self):
        model = self.choose_model
        print('start load data...')
        train_data, train_labels = self.data.get_traindata(self.data_path)
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
            data_record.summary = model.to_json()
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
        data_record.model.save_weights(os.path.join(path, 'model_weights.h5'), save_format='h5',overwrite=True)
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

if __name__ == '__main__':
    traindata_path = 'F:/学校事务/学习Python/16272215-龚正阳-网络工程/16272215-龚正阳-源码2/STR-IDS_DKNN/DkNN/prepared_data/CIC_IDS_2017_traindata.csv'
    models_build = model_build(83, 8)
    model_train(models_build.model_simple, 'simple', traindata_path).run()
    model_train(models_build.model_gru, 'gru', traindata_path).run()
    model_train(models_build.model_lstm, 'lstm', traindata_path).run()
    model_train(models_build.model_cnn_gru, 'cnn_gru', traindata_path).run()
    model_train(models_build.model_cnn_lstm, 'cnn_lstm', traindata_path).run()
