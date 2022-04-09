
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import tensorflow.keras.backend as kb
from dataprocess  import data_process
import pickle
from models import model_build

if __name__ == '__main__':
    data = data_process()
    train_data, train_label = data.get_traindata('F:/学校事务/学习Python/16272215-龚正阳-网络工程/16272215-龚正阳-源码2/STR-IDS_DKNN/DkNN/prepared_data/CIC_IDS_2017_traindata.csv')
    # train_data, train_label = data.get_credibilityTdata('F:/学校事务/学习Python/16272215-龚正阳-网络工程/16272215-龚正阳-源码2/STR-IDS_DKNN/DkNN/prepared_data/CIC_IDS_2017_credibilityTdata.csv')
    
    model_file = 'F:/models_trained_for_DkNN/history_more_5'
    # model_file = 'F:/models_trained_for_DkNN/history_more_credibility_5'
    input_dim = 83
    output_dim = 8
    model_list = ['simple', 'gru', 'lstm', 'cnn_gru', 'cnn_lstm']
    models = model_build(input_dim, output_dim)

    for i in model_list:
        model_path = model_file + '/' + i +'/model_weights.h5'

        if i != 'simple' and len(train_data.shape) == 2:
            test_data = np.expand_dims(train_data, axis=1)

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

        knn_trained_list = []
        if i != 'simple':
            train_data = np.expand_dims(train_data, axis=1)
        get_alllayer_outputs = kb.function([model.layers[0].input],
                                                [l.output for l in model.layers[0:] if not isinstance(l, tf.keras.layers.Dropout)])
        all_layers_output = get_alllayer_outputs(train_data)
        train_data = np.squeeze(train_data)
        _knn = KNeighborsClassifier(3, p=2)
        _knn.fit(train_data, train_label)
        knn_trained_list.append(_knn)
        for j in all_layers_output:
            _knn = KNeighborsClassifier(3, p=2)
            _knn.fit(np.squeeze(j), train_label)
            knn_trained_list.append(_knn)
        with open(model_path + '/../knn.pickle', mode='wb') as f:
            pickle.dump(knn_trained_list, f)
        print(i + ' model kneighborTree building finished')


        
        
    
    

