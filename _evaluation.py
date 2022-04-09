import time
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics
from dataprocess import data_process
import tensorflow.keras.backend as kb
from models import model_build

def k_neighbors(k, test_data, knns, function):
    _result = function(test_data)
    result = []
    for i in range(len(_result) + 1):
        if i == 0:
            result.append(knns[i].kneighbors(np.squeeze(test_data), n_neighbors=k, return_distance=True))
        else:
            result.append(knns[i].kneighbors(np.squeeze(_result[i-1]), n_neighbors=k, return_distance=True))
    #     temp = []
    #     for z in range(len(_result[i])):
    #         distances = []
    #         for n in range(len(layers_output[i])):
    #             distances.append(np.linalg.norm(_result[i][z][0] - layers_output[i][n][0]))
    #         nearest = np.argsort(distances)
    #         temp.append([i for i in nearest[:k]])
    #     result.append(temp)
    return result

def Nonconformity(knns, test_data, k, label_sample, train_label, function):
    """compute every a_value and return [[[number of data which are different from all kinds of its label]]]"""
    _result = k_neighbors(k, test_data, knns, function)
    # get a_value of input x, layer by layer.
    result = [[[] for l in range(len(label_sample))] for n in range(len(_result[0][1]))]
    for i in range(len(_result)):
        for z in range(len(_result[i][1])):
            for x in range(len(label_sample)):
                a_value = 0
                for j in range(len(_result[i][1][z])):
                    if (train_label[_result[i][1][z][j]] == label_sample[x]).all():
                        pass
                    else:
                        a_value = a_value + (1 / _result[i][0][z][j]) if _result[i][0][z][j] != 0 else 0
                result[z][x].append(a_value)
    return result

def compute_p_value(calibrations, a_values):
    """compute p_values"""
    result = []
    cali_len = len(calibrations)
    for i in a_values:
        _result = []
        for ii in i:
            count = 0
            a_value = sum(ii)
            for j in calibrations:
                if a_value <= j:
                    count = count + 1
            _result.append(count / cali_len)
        result.append(_result)
    return result

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
    try:
        auc_results = metrics.roc_auc_score(y_true=test_labels, y_score=np.array(predictions))
    except ValueError:
        auc_results = 0
    
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
    plt.figure(figsize=(6.4, 4.8))
    plt.plot((0, 1), (0, 1), ls='--')
    plt.plot(dknn_fpr, dknn_tpr, label='dknn' + model_name + '  auc: %.4f' % dknn_auc)
    plt.plot(dnn_fpr, dnn_tpr, label=model_name + '  auc: %.4f' % dnn_auc)
    plt.title(model_name + '_k=%d' % k)
    plt.legend()
    plt.savefig(save_path + '/' + model_name + '/' + 'roc_%d' % k + '.png')

def credibility_plot(data_scale, data_Rrate, title, model_name, k, save_path):
        fig = plt.figure(figsize=(6.4, 6.4))
        X = [i * 0.1 + 0.05 for i in range(10)]
        ax1 = fig.add_subplot()
        ax1.set_ylim([0, 1.1])
        ax1.bar(X, data_Rrate, width=0.08, color='yellow')
        ax1.set_ylabel('right prediction rate', fontsize='15')
        for x, y in zip(X, data_Rrate):
            plt.text(x, y+0.005, '%.3f' % y, ha='center', va='bottom', fontsize=12)
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

def create_adversarial_pattern(model, input_image, input_label):
        loss_object = tf.keras.losses.CategoricalCrossentropy()

        input_image = tf.convert_to_tensor(input_image)

        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = model(input_image)
            loss = loss_object(input_label, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, input_image)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        return signed_grad

def credibility_plot_Compare(title, model_list, performance, k, save_path):
    X = [i * 0.1 + 0.05 for i in range(10)]

    if k == 0:
        label_name_pre = ''
        save_name_suffix = ''
        plot_title = 'dnn data scale'
    else:
        label_name_pre = 'dknn_'
        save_name_suffix = '_k=%d'%k
        plot_title = 'dknn data scale with k=%d'%k

    plt.clf()
    plt.cla()
    plt.figure(figsize=(6.4, 4.8))
    for m in model_list:
        if m == 'simple':
            data_scale = performance['simple'][1][0]
        elif m == 'gru':
            data_scale = performance['gru'][1][0]
        elif m == 'lstm':
            data_scale = performance['lstm'][1][0]
        elif m == 'cnn_gru':
            data_scale = performance['cnn_gru'][1][0]
        elif m == 'cnn_lstm':
            data_scale = performance['cnn_lstm'][1][0]
        else:
            raise('Program Error! Check the \'model list\'.')
        plt.plot(X, data_scale, label=label_name_pre + m)
    plt.title(plot_title)
    plt.legend()
    plt.savefig(save_path + '/' + title + save_name_suffix + '.png')

if __name__ == '__main__':
    data_path = 'F:/学校事务/学习Python/16272215-龚正阳-网络工程/16272215-龚正阳-源码2/STR-IDS_DKNN/DkNN/prepared_data'
    # normal_model_path = 'F:/models_trained_for_DkNN/history_more_credibility_5'
    # log = open('F:/models_trained_for_DkNN/history_more_credibility_5/log.txt', 'a')
    normal_model_path = 'F:/models_trained_for_DkNN/history_more_5'
    log = open('F:/models_trained_for_DkNN/history_more_5/log.txt', 'a')
    print(f'\n#####data recorded at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}######', file=log, flush=True)

    data = data_process()
    _, train_label = data.get_traindata(data_path + '/CIC_IDS_2017_traindata.csv')
    test_data, test_label = data.get_testdata(data_path + '/CIC_IDS_2017_testdata.csv')
    # _, train_label = data.get_traindata(data_path + '/CIC_IDS_2017_credibilityTdata.csv')
    # test_data, test_label = data.get_testdata(data_path + '/CIC_IDS_2017_credibilityCdata.csv')
    label_sample = np.identity(8)
    data_scale = data.get_data_scale(train_label, label_sample)

    input_dim = 83
    output_dim = 8
    model_list = ['simple', 'gru', 'lstm', 'cnn_gru', 'cnn_lstm']
    models = model_build(input_dim, output_dim)
    k = 100

    # tf.config.run_functions_eagerly(True)  # necessary for the function in the DkNN model.

    # for m in model_list:
    #     with open(normal_model_path+ '/' + m + '/knn.pickle', 'rb') as f:
    #         kNTree = pickle.load(f)
    #     with open(normal_model_path+ '/' + m + '/cali_nonconformity.pickle', 'rb') as f:
    #         cali_nonconformity = pickle.load(f)
    #     cali_nonconformity = np.array(cali_nonconformity)

    #     if m != 'simple' and len(test_data.shape) == 2:
    #         test_data = np.expand_dims(test_data, axis=1)

    #     if m == 'simple':
    #         model = models.model_simple
    #     elif m == 'gru':
    #         model = models.model_gru
    #     elif m == 'lstm':
    #         model = models.model_lstm
    #     elif m == 'cnn_gru':
    #         model = models.model_cnn_gru
    #     elif m == 'cnn_lstm':
    #         model = models.model_cnn_lstm
    #     else:
    #         raise('Program Error! Check the \'model list\'.')
        
    #     model.load_weights(normal_model_path + '/'+ m + '/model_weights.h5', by_name=True)

    #     # test_data = create_adversarial_pattern(model, test_data, test_label)

    #     start = time.time()
    #     predictions = model.predict(test_data)
    #     end = time.time()
    #     print(f'predicting cost time :{end-start}(ModelName: {m})', file=log)

    #     # get the prediction of the dknn
    #     get_alllayer_outputs = kb.function([model.layers[0].input], [l.output for l in model.layers[0:] if not isinstance(l, tf.keras.layers.Dropout)])

    #     start = time.time()
    #     result_test = Nonconformity(kNTree, test_data, k, label_sample, train_label, get_alllayer_outputs)

    #     with open(normal_model_path + '/' + m + '/cali_nonconformity_test.pickle', mode='wb') as f:
    #         pickle.dump(result_test, f, 2)

    #     # with open(normal_model_path + '/' + m + '/cali_nonconformity_test.pickle', mode='rb') as f:
    #     #     result_test = pickle.load(f)

    #     # compute the p_value
    #     print('start compute p_value(ModelName: %s, K: %d)' % (m, k), file=log, flush=True)
    #     p_values = compute_p_value(cali_nonconformity.tolist(), result_test)
    #     a = input()
    #     end = time.time()
    #     print(f'predicting costs time :{end-start}(dknn ModelName: {m})', file=log)

    #     # evaluate the model and dknn model
    #     dnn_fpr, dnn_tpr, dnn_auc = model_score(predictions, label_sample.tolist(), test_label, data_scale, m, 0, log)
    #     dknn_fpr, dknn_tpr, dknn_auc = model_score(p_values, label_sample.tolist(), test_label, data_scale, m, k, log)

    #     dnn_confidence = confidence(predictions, test_label, m, 0, log)
    #     dknn_confidence = confidence(p_values, test_label, m, k, log)

    #     roc_plot(m, k, (dknn_fpr, dknn_tpr, dknn_auc), (dnn_fpr, dnn_tpr, dnn_auc), normal_model_path)

    #     credibility_plot(dnn_confidence[0], dnn_confidence[1], 'normal', m, 0, normal_model_path)
    #     credibility_plot(dknn_confidence[0], dknn_confidence[1], 'normal', m, k, normal_model_path)

    dnn_plot_info = {}
    dknn_plot_info = {}
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
        elif m == 'gru':
            model = models.model_gru
        elif m == 'lstm':
            model = models.model_lstm
        elif m == 'cnn_gru':
            model = models.model_cnn_gru
        elif m == 'cnn_lstm':
            model = models.model_cnn_lstm
        else:
            raise('Program Error! Check the \'model list\'.')
        
        model.load_weights(normal_model_path + '/'+ m + '/model_weights.h5', by_name=True)

        test_data = create_adversarial_pattern(model, test_data, test_label)

        start = time.time()
        predictions = model.predict(test_data)
        end = time.time()
        print(f'predicting cost time :{end-start}(ModelName: {m})', file=log)

        # get the prediction of the dknn
        get_alllayer_outputs = kb.function([model.layers[0].input], [l.output for l in model.layers[0:] if not isinstance(l, tf.keras.layers.Dropout)])

        start = time.time()
        result_test = Nonconformity(kNTree, test_data, k, label_sample, train_label, get_alllayer_outputs)

        # with open(normal_model_path + '/' + m + '/cali_nonconformity_test.pickle', mode='wb') as f:
        #     pickle.dump(result_test, f)

        with open(normal_model_path + '/' + m + '/cali_nonconformity_robust.pickle', mode='wb') as f:
            pickle.dump(result_test, f)

        # with open(normal_model_path + '/' + m + '/cali_nonconformity_test.pickle', mode='rb') as f:
        #     result_test = pickle.load(f)

        # compute the p_value
        print('start compute p_value(ModelName: %s, K: %d)' % (m, k))
        p_values = compute_p_value(cali_nonconformity.tolist(), result_test)
        print(f'predicting costs time :{end-start}(dknn ModelName: {m})', file=log)

        # evaluate the model and dknn model
        dnn_fpr, dnn_tpr, dnn_auc = model_score(predictions, label_sample.tolist(), test_label, data_scale, m, 0, log)
        dknn_fpr, dknn_tpr, dknn_auc = model_score(p_values, label_sample.tolist(), test_label, data_scale, m, k, log)

        dnn_confidence = confidence(predictions, test_label, m, 0, log)
        dknn_confidence = confidence(p_values, test_label, m, k, log)

        dnn_performance = ((dnn_fpr, dnn_tpr, dnn_auc), dnn_confidence)
        dknn_performance = ((dknn_fpr, dknn_tpr, dknn_auc), dknn_confidence)
        with open(normal_model_path + '/' + m + '/dnn_performance_robust.pickle', mode='wb') as f:
            pickle.dump(dnn_performance, f)
        with open(normal_model_path + '/' + m + '/dknn_performance_robust.pickle', mode='wb') as f:
            pickle.dump(dknn_performance, f)

        # with open(normal_model_path + '/' + m + '/dnn_performance.pickle', mode='rb') as f:
        #     dnn_performance = pickle.load(f)
        # with open(normal_model_path + '/' + m + '/dknn_performance.pickle', mode='rb') as f:
        #     dknn_performance = pickle.load(f)

        # roc_plot(m, k, dknn_performance[0], dnn_performance[0], normal_model_path)

        if m == 'simple':
            dnn_plot_info['simple'] = dnn_performance
            dknn_plot_info['simple'] = dknn_performance
        elif m == 'gru':
            dnn_plot_info['gru'] = dnn_performance
            dknn_plot_info['gru'] = dknn_performance
        elif m == 'lstm':
            dnn_plot_info['lstm'] = dnn_performance
            dknn_plot_info['lstm'] = dknn_performance
        elif m == 'cnn_gru':
            dnn_plot_info['cnn_gru'] = dnn_performance
            dknn_plot_info['cnn_gru'] = dknn_performance
        elif m == 'cnn_lstm':
            dnn_plot_info['cnn_lstm'] = dnn_performance
            dknn_plot_info['cnn_lstm'] = dknn_performance
        else:
            raise('Program Error! Check the \'model list\'.')
    
    credibility_plot_Compare('robustnessCheck', model_list,  dnn_plot_info, 0, normal_model_path)
    credibility_plot_Compare('robustnessCheck', model_list,  dknn_plot_info, k, normal_model_path)