from models import model_build
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kb
from dataprocess import data_process
import pickle


if __name__ == '__main__':
    input_dim = 83
    output_dim = 8
    model_list = ['simple', 'gru', 'lstm', 'cnn_gru', 'cnn_lstm']
    models = model_build(input_dim, output_dim)
    # model_file = 'F:/models_trained_for_DkNN/history_more_5'
    model_file = 'F:/models_trained_for_DkNN/history_more_credibility_5'
    data = data_process()
    # cali_data, cali_label = data.get_calibrationdata('F:/学校事务/学习Python/16272215-龚正阳-网络工程/16272215-龚正阳-源码2/STR-IDS_DKNN/DkNN/prepared_data/CIC_IDS_2017_calibrationdata.csv')
    # train_data, train_label = data.get_traindata('F:/学校事务/学习Python/16272215-龚正阳-网络工程/16272215-龚正阳-源码2/STR-IDS_DKNN/DkNN/prepared_data/CIC_IDS_2017_traindata.csv')
    cali_data, cali_label = data.get_calibrationdata('F:/学校事务/学习Python/16272215-龚正阳-网络工程/16272215-龚正阳-源码2/STR-IDS_DKNN/DkNN/prepared_data/CIC_IDS_2017_credibility_cali_data.csv')
    train_data, train_label = data.get_traindata('F:/学校事务/学习Python/16272215-龚正阳-网络工程/16272215-龚正阳-源码2/STR-IDS_DKNN/DkNN/prepared_data/CIC_IDS_2017_credibilityTdata.csv')
    k = 100
    for i in model_list:
        model_path = model_file + '/' + i +'/model_weights.h5'

        if i != 'simple' and len(cali_data.shape) == 2:
            cali_data = np.expand_dims(cali_data, axis=1)

        if i == 'simple':
            model = models.model_simple
        elif i == 'gru':
            model = models.model_gru
        elif i == 'lstm':
            model = models.model_lstm
        elif i == 'cnn_gru':
            model = models.model_cnn_gru
        elif i == 'cnn_lstm':
            model = models.model_cnn_lstm
        else:
            raise('Program Error! Check the \'model list\'.')
        
        model.load_weights(model_path, by_name=True)
        get_alllayer_outputs = kb.function([model.layers[0].input],
                                                [l.output for l in model.layers[0:] if not isinstance(l, tf.keras.layers.Dropout)])
        if i != 'simple' and len(cali_data.shape) == 2:
            cali_data = np.expand_dims(cali_data, axis=1)
        all_layers_output = get_alllayer_outputs(cali_data)

        with open(model_file + '/' + i +'/knn.pickle', 'rb') as f:
            knn = pickle.load(f)
        
        _kneighbors = knn[0].kneighbors(np.squeeze(cali_data), n_neighbors=k, return_distance=True)

        cali_nonconformity = []
        for x in range(_kneighbors[1].shape[0]):
            _cali_nonconformity = 0
            for z in range(len(_kneighbors[1][x])):
                if (train_label[_kneighbors[1][x][z]] == cali_label[x]).all():
                    pass
                else:
                    _cali_nonconformity = _cali_nonconformity + (1 / _kneighbors[0][x][z] if _kneighbors[0][x][z] != 0 else 0)
            cali_nonconformity.append(_cali_nonconformity)
        for j in range(1, len(knn)):
            _kneighbors = knn[j].kneighbors(np.squeeze(all_layers_output[j-1]), n_neighbors=k, return_distance=True)
            for x in range(len(_kneighbors[1])):
                _cali_nonconformity = 0
                for z in range(len(_kneighbors[1][x])):
                    if (train_label[_kneighbors[1][x][z]] == cali_label[x]).all():
                        pass
                    else:
                        _cali_nonconformity = _cali_nonconformity + (1 / _kneighbors[0][x][z] if _kneighbors[0][x][z] != 0 else 0)
                cali_nonconformity[x] = cali_nonconformity[x] + _cali_nonconformity
        with open(model_path + '/../cali_nonconformity.pickle', mode='wb') as f:
            pickle.dump(cali_nonconformity, f)
        print(i + ' model cali_nonconformity computing finished')
        
        
