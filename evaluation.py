from ast import Raise
import time
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics
from dataprocess import data_process
from DkNN import DkNN_simple, DkNN_gru, DkNN_lstm, DkNN_cnn_gru, DkNN_cnn_lstm
from models import model_build

def model_score(predictions, label_sample, test_labels, data_scale, model_name, k, log_file):

    label_shape = test_labels.shape[-1]
    b = np.argmax(predictions, axis=1)
    _predictions = []
    for i in range(len(b)):
        _predictions.append([1 if j == b[i] else 0 for j in range(label_shape)])
    _predictions = np.array(_predictions)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    eval_result = [{} for i in label_sample]
    for i in range(len(label_sample)):
        for j in range(len(_predictions)):
            if (_predictions[j] == test_labels[j]).all():
                if (_predictions[j] == label_sample[i]).all():
                    tn += 1
                else:
                    tp += 1
            else:
                if (_predictions[j] == label_sample[i]).all():
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
    for i in range(len(data_scale)):
        acc = acc + eval_result[i]['acc'] * data_scale[i]
        dr = dr + eval_result[i]['dr'] * data_scale[i]
        far = far + eval_result[i]['far'] * data_scale[i]
        precision = precision + eval_result[i]['precision'] * data_scale[i]
        recall = recall + eval_result[i]['recall'] * data_scale[i]
        f1_measure = f1_measure + eval_result[i]['f1_measure'] * data_scale[i]

    f1_results = metrics.f1_score(test_labels, _predictions, average='macro')
    acc_results = metrics.f1_score(test_labels, _predictions, average='micro')
    pre_results = metrics.precision_score(test_labels, _predictions, average='macro')
    recall_results = metrics.recall_score(test_labels, _predictions, average='macro')
    auc_results = metrics.roc_auc_score(y_true=test_labels, y_score=np.array(predictions))
    
    if k == 0:
        print(f'basic messure of {model_name}, k:{k}', file=log_file)
    else:
        print(f'basic messure of DkNN+{model_name}, k:{k}', file=log_file)
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
    print(f'auc score           {auc_results}', file=log_file, flush=True)

    fpr, tpr, _ = metrics.roc_curve(test_labels.ravel(), np.array(predictions).ravel())

    return fpr, tpr, auc_results

def confidence(predictions, test_label, model_name, k, log_file):
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
        if k == 0:
            print('dnn confidence result:(ModelName: %s, K: %d)' % (model_name, k), file=log_file)
        else:
            print('dknn confidence result:(ModelName: DkNN+%s, K: %d)' % (model_name, k), file=log_file)
        print(f'confidence scale:   {data_confidence_scale}', file=log)
        print(f'confidence accuracy:{rate_right}', file=log_file, flush=True)
        return data_confidence_scale, rate_right

def roc_plot(model_name, k, dknn_info, dnn_info, save_path):
    dknn_fpr, dknn_tpr, dknn_auc = dknn_info
    dnn_fpr, dnn_tpr, dnn_auc = dnn_info
    plt.clf()
    plt.cla()
    plt.plot((0, 1), (0, 1), ls='--')
    plt.plot(dknn_fpr, dknn_tpr, label='dknn' + model_name + '  auc: %.4f' % dknn_auc)
    plt.plot(dnn_fpr, dnn_tpr, label=model_name + '  auc: %.4f' % dnn_auc)
    plt.title(model_name + '_k=%d' % k)
    plt.legend()
    plt.savefig(save_path + '/' + model_name + '/' + 'roc_%d' % k + '.png')

def credibility_plot(data_scale, data_Rrate, title, model_name, k, save_path):
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
        if k == 0:
            plt.title((model_name + ' ' + title))
        else:
            plt.title((model_name + ' k:%d ' + title) % k)
        plt.savefig(save_path + '/' + model_name + '/' +  'credibility_%d ' % k + title + '.png')

if __name__ == '__main__':
    data_path = 'F:/学校事务/学习Python/16272215-龚正阳-网络工程/16272215-龚正阳-源码2/STR-IDS_DKNN/DkNN/prepared_data'
    normal_model_path = 'F:/models_trained_for_DkNN/history_more_5'
    log = open('F:/models_trained_for_DkNN/history_more_5/log.txt', 'a')
    print(f'\n#####data recorded at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}######', file=log, flush=True)

    data = data_process()
    _, train_label = data.get_traindata(data_path + '/CIC_IDS_2017_traindata.csv')
    test_data, test_label = data.get_testdata(data_path + '/CIC_IDS_2017_testdata.csv')
    label_sample = np.identity(8)
    data_scale = data.get_data_scale(train_label)

    input_dim = 83
    output_dim = 8
    model_list = ['simple', 'gru', 'lstm', 'cnn_gru', 'cnn_lstm']
    models = model_build(input_dim, output_dim)
    k = 100

    tf.config.run_functions_eagerly(True)  # necessary for the function in the DkNN model.

    for m in model_list:
        with open(normal_model_path+ '/' + m + '/knn.pickle', 'rb') as f:
            kNTree = pickle.load(f)
        with open(normal_model_path+ '/' + m + '/cali_nonconformity.pickle', 'rb') as f:
            cali_nonconformity = pickle.load(f)
        cali_nonconformity = np.array(cali_nonconformity)

        if m != 'simple' and len(test_data.shape) == 2:
            test_data = np.expand_dims(test_data, axis=1)

        if m == 'simple':
            model = models.model_simple
            model_dknn = DkNN_simple(100, label_sample, kNTree, train_label, cali_nonconformity)
            model_dknn(test_data[:1])
        elif m == 'gru':
            model = models.model_gru
            model_dknn = DkNN_gru(100, label_sample, kNTree, train_label, cali_nonconformity)
            model_dknn(test_data[:1])
        elif m == 'lstm':
            model = models.model_lstm
            model_dknn = DkNN_lstm(100, label_sample, kNTree, train_label, cali_nonconformity)
            model_dknn(test_data[:1])
        elif m == 'cnn_gru':
            model = models.model_cnn_gru
            model_dknn = DkNN_cnn_gru(100, label_sample, kNTree, train_label, cali_nonconformity)
            model_dknn(test_data[:1])
        elif m == 'cnn_lstm':
            model = models.model_cnn_lstm
            model_dknn = DkNN_cnn_lstm(100, label_sample, kNTree, train_label, cali_nonconformity)
            model_dknn(test_data[:1])
        else:
            Raise('Program Error! Check the \'model list\'.')
        model.load_weights(normal_model_path + '/'+ m + '/model_weights.h5', by_name=True)
        model_dknn.load_weights(normal_model_path + '/'+ m + '/model_weights.h5', by_name=True)

        predictions = model.predict(test_data)
        # knn_fpr, knn_tpr ,knn_auc = model_score(predictions, label_sample, test_label[:200], train_label, log)
        # credibility_plot =
        dknn_predictions = model_dknn.predict(test_data)
        with open(normal_model_path+ '/' + m + '/normal_model_prediction.pickle', 'rb') as f:
            predictions = pickle.load(f)
        with open(normal_model_path+ '/' + m + '/dknn_model_prediction.pickle', 'rb') as f:
            dknn_predictions = pickle.load(f)

        dnn_fpr, dnn_tpr, dnn_auc = model_score(predictions, label_sample.tolist(), test_label, data_scale, m, 0, log)
        dknn_fpr, dknn_tpr, dknn_auc = model_score(dknn_predictions, label_sample.tolist(), test_label, data_scale, m, k, log)

        print(dnn_fpr, dnn_tpr, dnn_auc)
        print(dknn_fpr, dknn_tpr, dknn_auc)

        dnn_confidence = confidence(predictions, test_label, m, 0, log)
        dknn_confidence = confidence(dknn_predictions, test_label, m, k, log)

        roc_plot(m, k, (dknn_fpr, dknn_tpr, dknn_auc), (dnn_fpr, dnn_tpr, dnn_auc), normal_model_path)

        credibility_plot(dnn_confidence[0], dnn_confidence[1], 'normal', m, 0, normal_model_path)
        credibility_plot(dknn_confidence[0], dknn_confidence[1], 'normal', m, k, normal_model_path)







                

        
        

    



