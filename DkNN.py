
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class kNN_layer(tf.keras.layers.Layer):
    def __init__(self, kpoints):
        super(kNN_layer, self).__init__()
        self.kpoints = kpoints

    def call(self, inputs, kNTree, label_sample, train_label):     # kNTree means kNeighborTree
        _result = kNTree.kneighbors(np.array(inputs), n_neighbors=self.kpoints, return_distance=True)
        nonconformity_matrix = []
        for j in range(_result[1].shape[0]):
            nonconformity = []
            for i in label_sample:
                _nonconformity = 0
                for z in range(_result[1][j].shape[0]):
                    if (train_label[_result[1][j][z]] == i).all():
                        pass
                    else:
                        _nonconformity = _nonconformity + (1/_result[0][j][z] if _result[0][j][z] != 0 else 0)
                nonconformity.append(_nonconformity)
            nonconformity_matrix.append(nonconformity)
        return  tf.convert_to_tensor(nonconformity_matrix, dtype=tf.float32)

class DkNN_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(DkNN_layer, self).__init__()

    def call(self, nonconformity, label_sample, cali_nonconformity):
        cali_num = cali_nonconformity.shape[0]
        p_value_matrix = []
        for z in range(nonconformity.shape[0]):
            p_value = []
            for i in  range(label_sample.shape[0]):
                _nonconformity = 0
                for j in nonconformity[z, :,i]:
                    _nonconformity = _nonconformity + j
                cali_count = 0
                for j in cali_nonconformity:
                    if j >= _nonconformity:
                        cali_count = cali_count + 1
                _p_value = cali_count / cali_num
                p_value.append(_p_value)
            p_value_matrix.append(p_value)
        return tf.convert_to_tensor(p_value_matrix, dtype=tf.float32)

class DkNN_simple(tf.keras.Model):
    def __init__(self, kpoints, label_sample, kNTree, train_label, cali_nonconformity, input_dim=83, output_dim=8):
        super(DkNN_simple, self).__init__(name='')
        self.label_sample, self.kNTree, self.train_label, self.cali_nonconformity = label_sample, kNTree, train_label, cali_nonconformity

        self.kNN_layer0 = kNN_layer(kpoints)
        
        self.input_dense1 = tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,), name='dense1')
        self.dropout2 = tf.keras.layers.Dropout(0.2, )
        self.kNN_layer3 = kNN_layer(kpoints)

        self.dense4 = tf.keras.layers.Dense(128, activation='relu', name='dense2')
        self.dropout5 = tf.keras.layers.Dropout(0.2)
        self.kNN_layer6 = kNN_layer(kpoints)

        self.dense7 = tf.keras.layers.Dense(128, activation='relu', name='dense3')
        self.dropout8 = tf.keras.layers.Dropout(0.2)
        self.kNN_layer9 = kNN_layer(kpoints)

        self.output_dense10 = tf.keras.layers.Dense(output_dim, activation=tf.keras.activations.softmax, name='dense4')
        self.kNN_layer11 = kNN_layer(kpoints)

        self.DkNN_layer12 = DkNN_layer()

    def call(self, input_tensor, training=False):
        knn0 = self.kNN_layer0(input_tensor, self.kNTree[0], self.label_sample, self.train_label)

        x = self.input_dense1(input_tensor)
        x = self.dropout2(x)
        knn1 = self.kNN_layer3(x, self.kNTree[1], self.label_sample, self.train_label)

        x = self.dense4(x)
        x = self.dropout5(x)
        knn2 = self.kNN_layer6(x, self.kNTree[2], self.label_sample, self.train_label)

        x = self.dense7(x)
        x = self.dropout8(x)
        knn3 = self.kNN_layer9(x, self.kNTree[3], self.label_sample, self.train_label)

        x = self.output_dense10(x)
        knn4 = self.kNN_layer11(x, self.kNTree[4], self.label_sample, self.train_label)

        nonconformity = tf.concat([tf.expand_dims(knn0, axis=1),tf.expand_dims(knn1, axis=1), tf.expand_dims(knn2, axis=1), tf.expand_dims(knn3, axis=1), tf.expand_dims(knn4, axis=1)], 1)
        return self.DkNN_layer12(nonconformity, self.label_sample, self.cali_nonconformity)


class DkNN_gru(tf.keras.Model):
    def __init__(self, kpoints, label_sample, kNTree, train_label, cali_nonconformity, input_dim=83, output_dim=8):
        super(DkNN_gru, self).__init__(name='')
        self.label_sample, self.kNTree, self.train_label, self.cali_nonconformity = label_sample, kNTree, train_label, cali_nonconformity

        self.kNN_layer0 = kNN_layer(kpoints)
        
        self.input_gru1 = tf.keras.layers.GRU(128, input_shape=(1, input_dim), return_sequences=True, name='gru1')
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.kNN_layer3 = kNN_layer(kpoints)

        self.gru4 = tf.keras.layers.GRU(128, return_sequences=True, name='gru2')
        self.dropout5 = tf.keras.layers.Dropout(0.2)
        self.kNN_layer6 = kNN_layer(kpoints)

        self.gru7 = tf.keras.layers.GRU(128, return_sequences=False, name='gru3')
        self.dropout8 = tf.keras.layers.Dropout(0.2)
        self.kNN_layer9 = kNN_layer(kpoints)

        self.output_dense10 = tf.keras.layers.Dense(output_dim, activation=tf.keras.activations.softmax, name='dense1')
        self.kNN_layer11 = kNN_layer(kpoints)

        self.DkNN_layer12 = DkNN_layer()

    def call(self, input_tensor, training=False):
        knn0 = self.kNN_layer0(np.squeeze(input_tensor, axis=1), self.kNTree[0], self.label_sample, self.train_label)

        x = self.input_gru1(input_tensor)
        x = self.dropout2(x)
        knn1 = self.kNN_layer3(tf.squeeze(x, axis=1), self.kNTree[1], self.label_sample, self.train_label)

        x = self.gru4(x)
        x = self.dropout5(x)
        knn2 = self.kNN_layer6(tf.squeeze(x, axis=1), self.kNTree[2], self.label_sample, self.train_label)

        x = self.gru7(x)
        x = self.dropout8(x)
        knn3 = self.kNN_layer9(x, self.kNTree[3], self.label_sample, self.train_label)
        
        x = self.output_dense10(x)
        knn4 = self.kNN_layer11(x, self.kNTree[4], self.label_sample, self.train_label)

        nonconformity = tf.concat([tf.expand_dims(knn0, axis=1),tf.expand_dims(knn1, axis=1), tf.expand_dims(knn2, axis=1), tf.expand_dims(knn3, axis=1), tf.expand_dims(knn4, axis=1)], 1)

        return self.DkNN_layer12(nonconformity, self.label_sample, self.cali_nonconformity)

class DkNN_lstm(tf.keras.Model):
    def __init__(self, kpoints, label_sample, kNTree, train_label, cali_nonconformity, input_dim=83, output_dim=8):
        super(DkNN_lstm, self).__init__(name='')
        self.label_sample, self.kNTree, self.train_label, self.cali_nonconformity = label_sample, kNTree, train_label, cali_nonconformity

        self.kNN_layer0 = kNN_layer(kpoints)
        
        self.input_lstm1 = tf.keras.layers.LSTM(128, input_dim=input_dim, return_sequences=True, name='lstm1')
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.kNN_layer3 = kNN_layer(kpoints)

        self.lstm4 = tf.keras.layers.LSTM(128, return_sequences=True, name='lstm2')
        self.dropout5 = tf.keras.layers.Dropout(0.2)
        self.kNN_layer6 = kNN_layer(kpoints)

        self.lstm7 = tf.keras.layers.LSTM(128, return_sequences=False, name='lstm3')
        self.dropout8 = tf.keras.layers.Dropout(0.2)
        self.kNN_layer9 = kNN_layer(kpoints)

        self.output_dense10 = tf.keras.layers.Dense(output_dim, activation=tf.keras.activations.softmax, name='dense1')
        self.kNN_layer11 = kNN_layer(kpoints)

        self.DkNN_layer12 = DkNN_layer()

    def call(self, input_tensor, training=False):
        knn0 = self.kNN_layer0(np.squeeze(input_tensor, axis=1), self.kNTree[0], self.label_sample, self.train_label)

        x = self.input_lstm1(input_tensor)
        x = self.dropout2(x)
        knn1 = self.kNN_layer3(tf.squeeze(x, axis=1), self.kNTree[1], self.label_sample, self.train_label)

        x = self.lstm4(x)
        x = self.dropout5(x)
        knn2 = self.kNN_layer6(tf.squeeze(x, axis=1), self.kNTree[2], self.label_sample, self.train_label)

        x = self.lstm7(x)
        x = self.dropout8(x)
        knn3 = self.kNN_layer9(x, self.kNTree[3], self.label_sample, self.train_label)
        

        x = self.output_dense10(x)
        knn4 = self.kNN_layer11(x, self.kNTree[4], self.label_sample, self.train_label)

        nonconformity = tf.concat([tf.expand_dims(knn0, axis=1),tf.expand_dims(knn1, axis=1), tf.expand_dims(knn2, axis=1), tf.expand_dims(knn3, axis=1), tf.expand_dims(knn4, axis=1)], 1)

        return self.DkNN_layer12(nonconformity, self.label_sample, self.cali_nonconformity)


class DkNN_cnn_gru(tf.keras.Model):
    def __init__(self, kpoints, label_sample, kNTree, train_label, cali_nonconformity, input_dim=83, output_dim=8):
        super(DkNN_cnn_gru, self).__init__(name='')
        self.label_sample, self.kNTree, self.train_label, self.cali_nonconformity = label_sample, kNTree, train_label, cali_nonconformity
 
        self.kNN_layer0 = kNN_layer(kpoints)
        
        self.input_conv1d1 = tf.keras.layers.Conv1D(128, 1, activation='relu', input_shape=(None, input_dim), name='conv1d1')
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.kNN_layer3 = kNN_layer(kpoints)

        self.conv1d4 = tf.keras.layers.Conv1D(96, 1, activation='relu', name='conv1d2')
        self.dropout5 = tf.keras.layers.Dropout(0.2)
        self.kNN_layer6 = kNN_layer(kpoints)

        self.conv1d7 = tf.keras.layers.Conv1D(128, 1, activation='relu', name='conv1d3')
        self.dropout8 = tf.keras.layers.Dropout(0.2)
        self.kNN_layer9 = kNN_layer(kpoints)

        self.gru10 = tf.keras.layers.GRU(128, return_sequences=True, name='gru1')
        self.dropout11 = tf.keras.layers.Dropout(0.2)
        self.kNN_layer12 = kNN_layer(kpoints)

        self.gru13 = tf.keras.layers.GRU(128, return_sequences=False, name='gru2')
        self.dropout14 = tf.keras.layers.Dropout(0.2)
        self.kNN_layer15 = kNN_layer(kpoints)

        self.dense16 = tf.keras.layers.Dense(64, activation='relu', name='dense1')
        self.kNN_layer17 = kNN_layer(kpoints)

        self.output_dense18 = tf.keras.layers.Dense(output_dim, activation=tf.keras.activations.softmax, name='dense2')
        self.kNN_layer19 = kNN_layer(kpoints)

        self.DkNN_layer20 = DkNN_layer()

    def call(self, input_tensor, training=False):
        knn0 = self.kNN_layer0(np.squeeze(input_tensor, axis=1), self.kNTree[0], self.label_sample, self.train_label)

        x = self.input_conv1d1(input_tensor)
        x = self.dropout2(x)
        knn1 = self.kNN_layer3(tf.squeeze(x, axis=1), self.kNTree[1], self.label_sample, self.train_label)

        x = self.conv1d4(x)
        x = self.dropout5(x)
        knn2 = self.kNN_layer6(tf.squeeze(x, axis=1), self.kNTree[2], self.label_sample, self.train_label)

        x = self.conv1d7(x)
        x = self.dropout8(x)
        knn3 = self.kNN_layer9(tf.squeeze(x, axis=1), self.kNTree[3], self.label_sample, self.train_label)

        x = self.gru10(x)
        x = self.dropout11(x)
        knn4 = self.kNN_layer12(tf.squeeze(x, axis=1), self.kNTree[4], self.label_sample, self.train_label)

        x = self.gru13(x)
        x = self.dropout14(x)
        knn5 = self.kNN_layer15(x, self.kNTree[5], self.label_sample, self.train_label)

        x = self.dense16(x)
        knn6 = self.kNN_layer17(x, self.kNTree[6], self.label_sample, self.train_label)

        x = self.output_dense18(x)
        knn7 = self.kNN_layer19(x, self.kNTree[7], self.label_sample, self.train_label)

        nonconformity = tf.concat([tf.expand_dims(knn0, axis=1),tf.expand_dims(knn1, axis=1), tf.expand_dims(knn2, axis=1), tf.expand_dims(knn3, axis=1), tf.expand_dims(knn4, axis=1), tf.expand_dims(knn5, axis=1), tf.expand_dims(knn6, axis=1), tf.expand_dims(knn7, axis=1)], 1)

        return self.DkNN_layer20(nonconformity, self.label_sample, self.cali_nonconformity)


class DkNN_cnn_lstm(tf.keras.Model):
    def __init__(self, kpoints, label_sample, kNTree, train_label, cali_nonconformity, input_dim=83, output_dim=8):
        super(DkNN_cnn_lstm, self).__init__(name='')
        self.label_sample, self.kNTree, self.train_label, self.cali_nonconformity = label_sample, kNTree, train_label, cali_nonconformity
 
        self.kNN_layer0 = kNN_layer(kpoints)
        
        self.input_conv1d1 = tf.keras.layers.Conv1D(128, 1, activation='relu', input_shape=(None, input_dim), name='conv1d1')
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.kNN_layer3 = kNN_layer(kpoints)

        self.conv1d4 = tf.keras.layers.Conv1D(96, 1, activation='relu', name='conv1d2')
        self.dropout5 = tf.keras.layers.Dropout(0.2)
        self.kNN_layer6 = kNN_layer(kpoints)

        self.conv1d7 = tf.keras.layers.Conv1D(128, 1, activation='relu', name='conv1d3')
        self.dropout8 = tf.keras.layers.Dropout(0.2)
        self.kNN_layer9 = kNN_layer(kpoints)

        self.lstm10 = tf.keras.layers.LSTM(128, return_sequences=True, name='lstm1')
        self.dropout11 = tf.keras.layers.Dropout(0.2)
        self.kNN_layer12 = kNN_layer(kpoints)

        self.lstm13 = tf.keras.layers.LSTM(128, return_sequences=False, name='lstm2')
        self.dropout14 = tf.keras.layers.Dropout(0.2)
        self.kNN_layer15 = kNN_layer(kpoints)

        self.dense16 = tf.keras.layers.Dense(64, activation='relu', name='dense1')
        self.kNN_layer17 = kNN_layer(kpoints)

        self.output_dense18 = tf.keras.layers.Dense(output_dim, activation=tf.keras.activations.softmax, name='dense2')
        self.kNN_layer19 = kNN_layer(kpoints)

        self.DkNN_layer20 = DkNN_layer()

    def call(self, input_tensor, training=False):
        knn0 = self.kNN_layer0(np.squeeze(input_tensor, axis=1), self.kNTree[0], self.label_sample, self.train_label)

        x = self.input_conv1d1(input_tensor)
        x = self.dropout2(x)
        knn1 = self.kNN_layer3(tf.squeeze(x, axis=1), self.kNTree[1], self.label_sample, self.train_label)

        x = self.conv1d4(x)
        x = self.dropout5(x)
        knn2 = self.kNN_layer6(tf.squeeze(x, axis=1), self.kNTree[2], self.label_sample, self.train_label)

        x = self.conv1d7(x)
        x = self.dropout8(x)
        knn3 = self.kNN_layer9(tf.squeeze(x, axis=1), self.kNTree[3], self.label_sample, self.train_label)

        x = self.lstm10(x)
        x = self.dropout11(x)
        knn4 = self.kNN_layer12(tf.squeeze(x, axis=1), self.kNTree[4], self.label_sample, self.train_label)

        x = self.lstm13(x)
        x = self.dropout14(x)
        knn5 = self.kNN_layer15(x, self.kNTree[5], self.label_sample, self.train_label)

        x = self.dense16(x)
        knn6 = self.kNN_layer17(x, self.kNTree[6], self.label_sample, self.train_label)

        x = self.output_dense18(x)
        knn7 = self.kNN_layer19(x, self.kNTree[7], self.label_sample, self.train_label)

        nonconformity = tf.concat([tf.expand_dims(knn0, axis=1),tf.expand_dims(knn1, axis=1), tf.expand_dims(knn2, axis=1), tf.expand_dims(knn3, axis=1), tf.expand_dims(knn4, axis=1), tf.expand_dims(knn5, axis=1), tf.expand_dims(knn6, axis=1), tf.expand_dims(knn7, axis=1)], 1)

        return self.DkNN_layer20(nonconformity, self.label_sample, self.cali_nonconformity)



        