import numpy as np

from torch.utils.data import DataLoader, Dataset


class KDD_CUP_99_Base(Dataset):

    def __init__(self, root_dir, mode='Train'):
        assert mode == 'Train' or mode == 'Test'
        self.mode = mode
        self.data_array = None
        self.columns_name = [
            'duration', 'protocol', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_hot_login', 'is_guest_login', 'count', 'srv_count',
            'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate'
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
        ]
        self.to_num_column_idx = [1, 2, 3, 41]
        self.discrete_column = [1, 2, 3, 11, 13, 14, 20, 21, 41]
        self.data_num = 50000
        self.to_num_column_dic = {
            1: {
                b'icmp': 0,
                b'tcp': 1,
                b'udp': 2
            },
            2: {
                b'ecr_i': 0,
                b'private': 1,
                b'http': 2,
                b'smtp': 3,
                b'other': 4,
                b'domain_u': 5,
                b'ftp_data': 6,
                b'eco_i': 7,
                b'finger': 8,
                b'urp_i': 9,
                b'ftp': 10,
                b'telnet': 11,
                b'ntp_u': 12,
                b'auth': 13,
                b'pop_3': 14,
                b'time': 15,
                b'domain': 16,
                b'Z39_50': 17,
                b'gopher': 18,
                b'mtp': 19,
                b'ssh': 20,
                b'whois': 21,
                b'remote_job': 22,
                b'rje': 23,
                b'link': 24,
                b'imap4': 25,
                b'ctf': 26,
                b'name': 27,
                b'supdup': 28,
                b'echo': 29,
                b'discard': 30,
                b'nntp': 31,
                b'uucp_path': 32,
                b'systat': 33,
                b'sunrpc': 34,
                b'daytime': 35,
                b'netstat': 36,
                b'pop_2': 37,
                b'netbios_ssn': 38,
                b'netbios_ns': 39,
                b'vmnet': 40,
                b'iso_tsap': 41,
                b'netbios_dgm': 42,
                b'sql_net': 43,
                b'shell': 44,
                b'csnet_ns': 45,
                b'klogin': 46,
                b'hostnames': 47,
                b'bgp': 48,
                b'exec': 49,
                b'login': 50,
                b'printer': 51,
                b'http_443': 52,
                b'efs': 53,
                b'uucp': 54,
                b'ldap': 55,
                b'kshell': 56,
                b'nnsp': 57,
                b'courier': 58,
                b'IRC': 59,
                b'urh_i': 60,
                b'X11': 61,
                b'tim_i': 62,
                b'red_i': 63,
                b'pm_dump': 64,
                b'tftp_u': 65,
                b'aol': 66,
                b'http_8001': 67,
                b'harvest': 68,
                b'http_2784': 69
            },
            3: {
                b'SF': 0,
                b'S0': 1,
                b'REJ': 2,
                b'RSTR': 3,
                b'RSTO': 4,
                b'SH': 5,
                b'S1': 6,
                b'S2': 7,
                b'RSTOS0': 8,
                b'OTH': 9,
                b'S3': 10
            },
            41: {
                b'smurf.': 0,
                b'neptune.': 1,
                b'normal.': 2,
                b'satan.': 3,
                b'ipsweep.': 4,
                b'portsweep.': 5,
                b'nmap.': 6,
                b'back.': 7,
                b'warezclient.': 8,
                b'teardrop.': 9,
                b'pod.': 10,
                b'guess_passwd.': 11,
                b'buffer_overflow.': 12,
                b'land.': 13,
                b'warezmaster.': 14,
                b'imap.': 15,
                b'rootkit.': 16,
                b'loadmodule.': 17,
                b'ftp_write.': 18,
                b'multihop.': 19,
                b'phf.': 20,
                b'perl.': 21,
                b'spy.': 22
            }
        }
        self.label_category = {
            'NORMAL': [2],
            'PROBE': [3, 4, 5, 6],
            'DOS': [0, 1, 7, 9, 10, 13],
            'U2R': [12, 16, 17, 21],
            'R2L': [8, 11, 14, 15, 18, 19, 20, 22]
        }

        self.load_data_dir = root_dir + '/kddcup.data.corrected'
        self.processed_train_data_dir = root_dir + '/kddcup_trainset.csv'
        self.processed_test_data_dir = root_dir + '/kddcup_testset.csv'

    def normalize(self):
        column_idx = [i for i in range(42)]
        continuous_column_idx = [i for i in column_idx if i not in self.discrete_column]

        target_part = self.data_array[:, continuous_column_idx]
        column_max = np.max(target_part, axis=0, keepdims=True)
        column_min = np.min(target_part, axis=0, keepdims=True)

        _range = column_max - column_min
        __range = np.where(_range == 0, 1e-5, _range)

        self.data_array[:, continuous_column_idx] = (self.data_array[:, continuous_column_idx] - column_min) / (__range)
        self.data_array[:, continuous_column_idx] = np.where(_range == 0, 0, self.data_array[:, continuous_column_idx])

    def kdd_cup99_numerical(self):
        '''column [1,2,3,11,13,14,20,21,41] are discrete, need to separated
           columns need to be converted to num :[1,2,3,41]
        '''

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
        self.data_array = self.data_array.astype(np.float32)

        self.to_num_column_dic = dict(zip(self.to_num_column_idx, to_num_column_dic))

    def disorder(self):
        np.random.shuffle(self.data_array)

    def __if_in(self, array, list):
        array = array[:, -1]
        temp = array == list[0]
        for i in range(1, len(list)):
            temp = temp + (array == list[i])
        return temp

    def get_category_data(self, category):
        indices = self.label_category[category]
        select = self.__if_in(self.data_array, indices)

        self.data_array = self.data_array[select]

    def Uniform_data_by_label_type(self):
        label = self.label_category.keys()
        num = int(self.data_num / len(label))
        print('totol data_num:%d, every_kind_num:%d' % (self.data_num, num))
        array_tuple = ()
        for i in label:
            indices = self.label_category[i]
            select = self.__if_in(self.data_array, indices)
            data = self.data_array[select]

            if data.shape[0] > num:
                data_array = data[:num]
            else:
                lack_num = num - data.shape[0]
                remainder = lack_num % data.shape[0]
                cycle = (lack_num - remainder) / data.shape[0]
                if cycle >= 1:
                    data_array = np.vstack((data, data))
                    for i in range(1, int(cycle)):
                        data_array = np.vstack((data_array, data))
                    data_array = np.vstack((data_array, data[:remainder]))
                else:
                    data_array = np.vstack((data, data[:remainder]))

            array_tuple = array_tuple + (data_array, )
        self.data_array = np.vstack(array_tuple)

    def data_process(self):
        self.kdd_cup99_numerical()
        self.normalize()
        self.disorder()
        # self.Uniform_data_by_label_type()

    def load_raw_data(self):
        self.data_array = np.loadtxt(self.load_data_dir, dtype=np.string_, delimiter=',')
        self.data_num = self.data_array.shape[0]

    def load_processed_data(self):
        if self.mode == 'Train':
            self.data_array = np.loadtxt(self.processed_train_data_dir, dtype=np.float32, delimiter=',')
        if self.mode == 'Test':
            self.data_array = np.loadtxt(self.processed_test_data_dir, dtype=np.float32, delimiter=',')
        self.data_num = self.data_array.shape[0]

    def __getitem__(self, index):
        return self.data_array[index, :41], self.data_array[index, 41]

    def __len__(self):
        return self.data_array.shape[0]

    def save_data(self):
        num = int(self.data_num * 0.6)
        np.savetxt(self.processed_train_data_dir, self.data_array[:num, :], delimiter=',')
        np.savetxt(self.processed_test_data_dir, self.data_array[num:, :], delimiter=',')


class KDD_CUP_99_DataLoader(DataLoader):

    def __init__(self, root_dir, batch_size=1, mode='Train', category=None):
        assert mode == 'Train' or mode == 'Test'
        self.data = KDD_CUP_99_Base(root_dir, mode)
        self.data.load_processed_data()
        if category != None and category in self.data.label_category.keys():
            self.data.get_category_data(category)
        super().__init__(self.data, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == '__main__':
    DATA_DIR = 'E:/DataSets/kddcup.data'
    dataset = KDD_CUP_99_Base(DATA_DIR)
    dataset.load_raw_data()
    # dataset.load_processed_data()
    dataset.data_process()
    # dataset.get_category_data('U2R')
    dataset.save_data()
    print(dataset.__len__())
    for i in range(10):
        print(dataset[i])
