from json import load
import os
import numpy as np
import pandas as pd

pd.options.display.max_rows = 50

from torch.utils.data import DataLoader, Dataset


class UNSW_NB15_BASE(Dataset):

    def __init__(self, root_dir, mode='Train'):
        assert mode == 'Train' or mode == 'Test'
        self.mode = mode
        self.data_array = None
        self.original_columns_name = [
            'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
            'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
            'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat',
            'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
            'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label'
        ]
        self.columns_keep_idx = [
            3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
        ]
        self.to_num_column_idx = [0, 2, 4, 5, 13, 47]
        self.discrete_column = [3, 4, 5, 13, 35, 38, 41, 47, 48]
        self.to_num_column_dic = {
            4: {
                b'tcp': 0,
                b'udp': 1,
                b'unas': 2,
                b'arp': 3,
                b'ospf': 4,
                b'sctp': 5,
                b'icmp': 6,
                b'any': 7,
                b'gre': 8,
                b'rsvp': 9,
                b'ipv6': 10,
                b'pim': 11,
                b'swipe': 12,
                b'sun-nd': 13,
                b'mobile': 14,
                b'sep': 15,
                b'ib': 16,
                b'idpr': 17,
                b'idpr-cmtp': 18,
                b'idrp': 19,
                b'iatp': 20,
                b'ifmp': 21,
                b'i-nlsp': 22,
                b'igp': 23,
                b'il': 24,
                b'ipv6-route': 25,
                b'ip': 26,
                b'isis': 27,
                b'ipx-n-ip': 28,
                b'ipcomp': 29,
                b'ipcv': 30,
                b'ipip': 31,
                b'iplt': 32,
                b'ipnip': 33,
                b'ippc': 34,
                b'irtp': 35,
                b'ipv6-frag': 36,
                b'ipv6-no': 37,
                b'ipv6-opts': 38,
                b'hmp': 39,
                b'zero': 40,
                b'gmtp': 41,
                b'cpnx': 42,
                b'a/n': 43,
                b'aes-sp3-d': 44,
                b'argus': 45,
                b'aris': 46,
                b'ax.25': 47,
                b'bbn-rcc': 48,
                b'bna': 49,
                b'br-sat-mon': 50,
                b'cbt': 51,
                b'cftp': 52,
                b'chaos': 53,
                b'compaq-peer': 54,
                b'cphb': 55,
                b'crtp': 56,
                b'ggp': 57,
                b'crudp': 58,
                b'dcn': 59,
                b'ddp': 60,
                b'ddx': 61,
                b'dgp': 62,
                b'egp': 63,
                b'eigrp': 64,
                b'emcon': 65,
                b'encap': 66,
                b'etherip': 67,
                b'fc': 68,
                b'iso-tp4': 69,
                b'fire': 70,
                b'iso-ip': 71,
                b'leaf-2': 72,
                b'kryptolan': 73,
                b'sccopmce': 74,
                b'sdrp': 75,
                b'secure-vmtp': 76,
                b'skip': 77,
                b'sm': 78,
                b'smp': 79,
                b'snp': 80,
                b'sprite-rpc': 81,
                b'sps': 82,
                b'srp': 83,
                b'st2': 84,
                b'stp': 85,
                b'tcf': 86,
                b'tlsp': 87,
                b'tp++': 88,
                b'trunk-1': 89,
                b'trunk-2': 90,
                b'ttp': 91,
                b'uti': 92,
                b'vines': 93,
                b'visa': 94,
                b'vmtp': 95,
                b'vrrp': 96,
                b'wb-expak': 97,
                b'wb-mon': 98,
                b'wsn': 99,
                b'xnet': 100,
                b'xns-idp': 101,
                b'l2tp': 102,
                b'scps': 103,
                b'sat-mon': 104,
                b'sat-expak': 105,
                b'larp': 106,
                b'leaf-1': 107,
                b'xtp': 108,
                b'merit-inp': 109,
                b'mfe-nsp': 110,
                b'mhrp': 111,
                b'micp': 112,
                b'mtp': 113,
                b'mux': 114,
                b'narp': 115,
                b'netblt': 116,
                b'nsfnet-igp': 117,
                b'nvp': 118,
                b'pgm': 119,
                b'pipe': 120,
                b'pnni': 121,
                b'pri-enc': 122,
                b'prm': 123,
                b'ptp': 124,
                b'pup': 125,
                b'pvp': 126,
                b'qnx': 127,
                b'rdp': 128,
                b'rvd': 129,
                b'3pc': 130,
                b'igmp': 131,
                b'udt': 132,
                b'rtp': 133,
                b'esp': 134
            },
            5: {
                b'FIN': 0,
                b'CON': 1,
                b'INT': 2,
                b'REQ': 3,
                b'RST': 4,
                b'ECO': 5,
                b'CLO': 6,
                b'URH': 7,
                b'ACC': 8,
                b'PAR': 9,
                b'TST': 10,
                b'ECR': 11,
                b'no': 12,
                b'URN': 13,
                b'MAS': 14,
                b'TXD': 15
            },
            13: {
                b'-': 0,
                b'dns': 1,
                b'http': 2,
                b'ftp-data': 3,
                b'smtp': 4,
                b'ftp': 5,
                b'ssh': 6,
                b'pop3': 7,
                b'dhcp': 8,
                b'ssl': 9,
                b'snmp': 10,
                b'radius': 11,
                b'irc': 12
            },
            47: {
                b'Normal': 0,
                b'Generic': 1,
                b'Exploits': 2,
                b'Fuzzers': 3,
                b'DoS': 4,
                b'Reconnaissance': 5,
                b'Analysis': 6,
                b'Backdoors': 7,
                b'Shellcode': 8,
                b'Worms': 9
            }
        }
        self.columns_name = [
            'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload',
            'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
            'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports',
            'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm',
            'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat'
        ]
        self.label_category = {
            b'Normal': 0,
            b'Generic': 1,
            b'Exploits': 2,
            b'Fuzzers': 3,
            b'DoS': 4,
            b'Reconnaissance': 5,
            b'Analysis': 6,
            b'Backdoors': 7,
            b'Shellcode': 8,
            b'Worms': 9
        }
        self.dport_use = [53, 80, 6881, 5190, 111, 25, 0, 143, 22, 21, 179, 520, 445, 514]
        self.data_num = 0
        self.data_feature = None
        self.data_label = None
        self.feature_length = 0

        self.raw_data_dir_list = [root_dir + '/raw_dataset/' + i for i in os.listdir(root_dir + '/raw_dataset/')]
        if not os.path.isdir(root_dir + '/prepared_data'):
            os.makedirs(root_dir + '/prepared_data')
        self.save_sorted_data_dir = root_dir + '/sorted_data'
        if not os.path.isdir(self.save_sorted_data_dir):
            os.makedirs(self.save_sorted_data_dir)
        self.processed_train_data_dir = root_dir + '/prepared_data/unsw_nb15_trainset.csv'
        self.processed_test_data_dir = root_dir + '/prepared_data/unsw_nb15_testset.csv'
        self.rebalanced_train_data_dir = root_dir + '/prepared_data/unsw_nb15_rebalanced_trainset.csv'
        self.rebalanced_test_data_dir = root_dir + '/prepared_data/unsw_nb15_rebalanced_testset.csv'

    def load_raw_data(self):

        def del_unicode_start(x):
            if '\xef\xbb\xbf' in x:
                return x[3:]
            else:
                return x

        def null_string(x):
            if x == '':
                return '0'
            if x == ' ':
                return '0'
            else:
                return x

        def garbled(x):
            if x == '0x000b':
                return '11'
            elif x == '0x000c':
                return '12'
            elif x == '0xc0a8':
                return '49320'
            elif x == '0x20205321':
                return '0'
            elif x == '0xcc09':
                return '52233'
            elif x == '-':
                return '0'
            else:
                return x

        def label(x):
            if x == '':
                return 'Normal'
            elif x == ' Fuzzers ' or x == ' Fuzzers':
                return 'Fuzzers'
            elif x == 'Reconnaissanc':
                return 'Reconnaissance'
            elif x == ' Reconnaissance ' or x == ' Reconnaissan':
                return 'Reconnaissance'
            elif x == 'Backdoor':
                return 'Backdoors'
            elif x == ' Shellcode ':
                return 'Shellcode'
            else:
                return x

        self.data_array = np.loadtxt(self.raw_data_dir_list[0],
                                     dtype=np.string_,
                                     delimiter=',',
                                     encoding='Latin-1',
                                     converters={
                                         0: del_unicode_start,
                                         1: garbled,
                                         3: garbled,
                                         37: null_string,
                                         38: null_string,
                                         39: null_string,
                                         47: label
                                     })
        for i in range(1, len(self.raw_data_dir_list)):
            self.data_array = np.vstack((self.data_array,
                                         np.loadtxt(self.raw_data_dir_list[i],
                                                    dtype=np.string_,
                                                    delimiter=',',
                                                    encoding='Latin-1',
                                                    converters={
                                                        0: del_unicode_start,
                                                        1: garbled,
                                                        3: garbled,
                                                        37: null_string,
                                                        38: null_string,
                                                        39: null_string,
                                                        47: label
                                                    })))
        self.data_num = self.data_array.shape[0]
        self.__unsw_nb15_label_process()
        print('load raw data over')

    def label_normal(self):
        indices = self.data_array[:, 48] == b'0'
        self.data_array[indices, 47] = b'Normal'

    def unsw_nb_numerical(self):

        def string_to_num(data_array, column_index):
            unique_protocol, times = np.unique(data_array[:, column_index], axis=0, return_counts=True)
            times_sort_idx = np.argsort(times, axis=0)  # descending
            times_sort_idx = times_sort_idx[::-1]  # indescending
            unique_protocol = unique_protocol[times_sort_idx]  # make unique_protocol indescending on times
            temp = data_array[:, column_index]
            protocol_key, protocol_value = [], []  # [keys], [values]
            for i in range(unique_protocol.shape[0]):
                temp[temp == unique_protocol[i]] = i
                protocol_key.append(unique_protocol[i])
                protocol_value.append(i)
            return protocol_key, protocol_value

        to_num_column_dic = []
        for i in self.to_num_column_idx:
            key, value = string_to_num(self.data_array, i)
            to_num_column_dic.append(dict(zip(key, value)))

        self.to_num_column_dic = dict(zip(self.to_num_column_idx, to_num_column_dic))

        self.data_array = self.data_array.astype(np.float32)

    def normalize(self):
        column_idx = [i for i in range(len(self.original_columns_name))]
        continuous_column_idx = [i for i in column_idx if i not in self.discrete_column]

        target_part = self.data_array[:, continuous_column_idx]
        column_max = np.max(target_part, axis=0, keepdims=True)
        column_min = np.min(target_part, axis=0, keepdims=True)

        _range = column_max - column_min
        __range = np.where(_range == 0, 1e-5, _range)

        self.data_array[:, continuous_column_idx] = (self.data_array[:, continuous_column_idx] - column_min) / (__range)
        self.data_array[:, continuous_column_idx] = np.where(_range == 0, 0, self.data_array[:, continuous_column_idx])

    def disorder(self):
        np.random.shuffle(self.data_array)

    '''def Uniform_data_by_label_type(self):
        label = self.label_category.keys()
        num = int(self.data_num / len(label))
        array_tuple = ()
        for i in label:
            indices = self.label_category[i]
            data = self.data_array[self.data_array[:, 47] == indices]

            if data.shape[0] > num:
                data_array = data[:num]
            else:
                lack_num = num - data.shape[0]
                remainder = lack_num % data.shape[0]
                cycle = (lack_num - remainder) / data.shape[0]
                if cycle >= 1:
                    data_array = np.vstack((data, data))
                    for j in range(1, int(cycle)):
                        data_array = np.vstack((data_array, data))
                    data_array = np.vstack((data_array, data[:remainder]))
                else:
                    data_array = np.vstack((data, data[:remainder]))
            array_tuple = array_tuple + (data_array, )
        self.data_array = np.vstack(array_tuple)

        # save to files
        np.random.shuffle(self.data_array)
        self.save_data((self.rebalanced_train_data_dir, self.rebalanced_test_data_dir))'''

    def data_process(self):
        self.label_normal()
        self.unsw_nb_numerical()
        self.normalize()
        self.del_redundant_columns()
        self.disorder()

    def one_hot(self):
        df = pd.DataFrame(self.data_array, columns=self.columns_name)

        df['dsport'] = df['dsport'].astype(int)
        df['proto'] = df['proto'].astype(int)
        df['state'] = df['state'].astype(int)
        df['service'] = df['service'].astype(int)
        df['attack_cat'] = df['attack_cat'].astype(int)

        Dport_replace = lambda x: x if x in self.dport_use else 0
        df['dsport'] = df['dsport'].apply(Dport_replace)

        df = pd.get_dummies(df, columns=['dsport', 'proto', 'state', 'service', 'attack_cat'], prefix=['dsport', 'proto', 'state', 'service', 'attack_cat'])

        # make the sequence of data random
        df = df.sample(frac=1, random_state=1)

        columns = df.columns.to_list()
        print(columns)
        self.columns_name = columns
        
        self.data_array = df.to_numpy(dtype=np.float32)

    def save_sorted_data(self):
        labels_name = self.label_category.keys()
        labels = np.eye(len(labels_name))
        print(self.label_category, labels_name)
        for i in labels_name:
            label = labels[self.label_category[i]]
            label_index = len(labels_name)
            indices = []
            for j in range(self.data_array.shape[0]):
                if (self.data_array[j, -label_index:] == label).all():
                    indices.append(j)
            data = self.data_array[indices]
            file_path = self.save_sorted_data_dir + '/UNSW_NB15_' + i.decode()
            np.save(file_path, data)
            print(label, self.label_category[i])

    def separate_feature_label(self):
        label_length = len(self.label_category)
        feature_length = self.data_array.shape[1] - label_length
        self.data_feature = self.data_array[:, :feature_length].copy()
        self.data_label = self.data_array[:, feature_length:].copy()
        self.feature_length = feature_length
        self.data_array = None

    def load_sorted_data(self):
        sorted_data_dir_list = [self.save_sorted_data_dir + '/' + i for i in os.listdir(self.save_sorted_data_dir)]

        data_array = np.load(sorted_data_dir_list[0])
        # print(sorted_data_dir_list[0])
        for i in range(1, len(sorted_data_dir_list)):
            # print(sorted_data_dir_list[i])
            _data_array = np.load(sorted_data_dir_list[i])
            data_array = np.vstack((data_array, _data_array))
        
        np.random.shuffle(data_array)

        self.data_array = data_array
        self.data_num = self.data_array.shape[0]
        data_array = None
        # label_length = 1
        # feature_length = data_array.shape[1] - label_length
        # self.data_feature = data_array[:, :feature_length].copy()
        # self.data_label = data_array[:, feature_length:].copy()
        # self.feature_length = feature_length
        # data_array = None
        print('load sorted data over')

    def load_sorted_data_rebalanced(self):
        sorted_data_dir_list = [self.save_sorted_data_dir + '/' + i for i in os.listdir(self.save_sorted_data_dir)]
        need_num_everykind = 50000

        _data_array = np.load(sorted_data_dir_list[0])
        data_array = _data_array
        row_num = _data_array.shape[0]
        still_need_num = need_num_everykind - row_num
        if still_need_num >= 0:
            times, remainder = int(need_num_everykind / row_num), need_num_everykind % row_num
            for i in range(times - 1):
                data_array = np.concatenate((data_array, _data_array), axis=0)
            data_array = np.concatenate((data_array, _data_array[:remainder]), axis=0)
        else:
            data_array = _data_array[:need_num_everykind]

        for i in range(1, len(sorted_data_dir_list)):
            _data_array = np.load(sorted_data_dir_list[i])
            row_num = _data_array.shape[0]
            times, remainder = int(need_num_everykind / row_num), need_num_everykind % row_num
            for i in range(times):
                data_array = np.concatenate((data_array, _data_array), axis=0)
            data_array = np.concatenate((data_array, _data_array[:remainder]), axis=0)
        np.random.shuffle(data_array)

        self.data_array = data_array
        self.data_num = self.data_array.shape[0]
        data_array = None
        # label_length = 1
        # feature_length = data_array.shape[1] - label_length
        # self.data_feature = data_array[:, :feature_length].copy()
        # self.data_label = data_array[:, feature_length:].copy()
        # self.feature_length = feature_length
        # self.data_num = self.data_feature.shape[0]
        # data_array = None
        print('load rebalanced data over')

    def del_redundant_columns(self):
        self.data_array = self.data_array[:, self.columns_keep_idx]

    def __unsw_nb15_label_process(self):
        index = self.data_array[:, 47] == b'Reconnaissanc'
        self.data_array[index, 47] = b'Reconnaissance'
        # index = self.data_array[:,47] == b' Reconnaissance'
        # self.data_array[index,47] = b'Reconnaissance'

    def __getitem__(self, index):
        return self.data_feature[index], self.data_label[index]

    def __len__(self):
        return self.data_num

    def save_data(self, save_dirs=None):
        num = int(self.data_num * 0.6)
        if save_dirs == None:
            np.savez_compressed(self.processed_train_data_dir, data_feature=self.data_feature[:num], data_label =self.data_label[:num])
            np.savez_compressed(self.processed_test_data_dir, data_feature=self.data_feature[num:], data_label =self.data_label[num:])
        else:
            np.savez_compressed(save_dirs[0], data_feature=self.data_feature[:num], data_label =self.data_label[:num])
            np.savez_compressed(save_dirs[1], data_feature=self.data_feature[num:], data_label =self.data_label[num:])

    def load_processed_data(self):
        if self.mode == 'Train':
            load_data = np.load(self.processed_train_data_dir + '.npz')
        if self.mode == 'Test':
            load_data = np.load(self.processed_test_data_dir + '.npz')
        self.data_feature, self.data_label = load_data['data_feature'], load_data['data_label']
        self.data_num = self.data_feature.shape[0]
        feature_length = self.data_feature.shape[1]
        self.feature_length = feature_length

    def load_rebalanced_data(self):
        if self.mode == 'Train':
            load_data = np.load(self.rebalanced_train_data_dir + '.npz')
        if self.mode == 'Test':
            load_data = np.load(self.rebalanced_test_data_dir + '.npz')
        self.data_feature, self.data_label = load_data['data_feature'], load_data['data_label']
        self.data_num = self.data_feature.shape[0]
        feature_length = self.data_feature.shape[1]
        self.feature_length = feature_length
        data_array = None


class UNSW_NB15_DataLoader(DataLoader):

    def __init__(self, root_dir, batch_size=1, mode='Train', rebalanced=False):
        assert mode == 'Train' or mode == 'Test'
        self.data = UNSW_NB15_BASE(root_dir, mode)
        if rebalanced:
            self.data.load_rebalanced_data()
        else:
            self.data.load_processed_data()
        super().__init__(self.data, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == '__main__':
    dataset = UNSW_NB15_BASE('E:/DataSets/UNSW-NB15 - CSV Files')
    # dataset = UNSW_NB15_DataLoader('E:/DataSets/UNSW-NB15 - CSV Files', 128, mode='Train', rebalanced=True)
    # dataset.load_raw_data()
    # dataset.data_process()
    # dataset.one_hot()
    # dataset.save_sorted_data()
    # print(dataset.data_array.shape)
    dataset.load_sorted_data()
    # dataset.load_sorted_data_rebalanced()
    print(np.unique(dataset.data_array[:,-10:], axis=0))
    dataset.separate_feature_label()
    # print(dataset.data_feature.shape, dataset.data_label.shape)
    dataset.save_data()
    # dataset.save_data((dataset.rebalanced_train_data_dir, dataset.rebalanced_test_data_dir))
    # # dataset.Uniform_data_by_label_type()
    # # print(dataset.to_num_column_dic)
    # dataset.load_processed_data()
    # print(dataset.data_feature.shape)
    # dataset.load_rebalanced_data()
    # # dataset.image_by_features()
    # # dataset.save_image_data()
    # # dataset.load_rebalanced_data()
    # # print(np.unique(dataset.data_label))
    # # dataset.load_sorted_data_rebalanced()
    # # cnt = 0
    # # for (x, y) in dataset:
    # #     if cnt > 10:
    # #         break
    # #     print(x,x.shape)
    # #     cnt += 1
    # for i in range(10):
    #     x,y = dataset[i]
    #     print(x.shape,y.shape)
