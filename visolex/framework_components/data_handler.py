import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from copy import deepcopy
from ast import literal_eval
from itertools import chain
from utils import add_special_token
from framework_components.aligned_tokenizer import aligned_tokenize


class DataHandler:
    # This module is responsible for feeding the data to teacher/student
    # If teacher is applied, then student gets the teacher-labeled data instead of ground-truth labels
    def __init__(self, args, tokenizer, logger=None):
        self.args = args
        self.logger = logger
        self.tokenizer = tokenizer
        self.datasets = {}
        self.seed = args.seed
        np.random.seed(self.seed)

    def load_dataset(self, method='train'):
        dataset = WSDataset(self.args, method=method, tokenizer=self.tokenizer, logger=self.logger)
        self.datasets[method] = dataset
        return dataset

    def create_pseudodataset(self, wsdataset):
        dataset = PseudoDataset(self.args, wsdataset, self.logger)
        return dataset


class WSDataset(Dataset):
    # WSDataset: Dataset for Weak Supervision.
    def __init__(self, args, method, tokenizer, logger=None):
        super(WSDataset, self).__init__()
        self.args = args
        self.seed = args.seed
        self.method = method
        self.tokenizer = tokenizer
        self.lower_case = args.lower_case
        self.datapath = os.path.join(args.datapath, "{}.csv".format(self.method))
        self.logger = logger
        self.remove_accents = args.remove_accents
        self.rm_accent_ratio = args.rm_accent_ratio
        self.data = {}
        self.no_accent_data = {}
        self.load_dataset()
        self.num_labels = len(self.tokenizer)
    
    def preprocess(self, data, strip_accents=False):
        preprocessed_dataset = aligned_tokenize(
            data=data, 
            tokenizer=self.tokenizer,
            method=self.method,
            lower_case=self.lower_case,
            rm_accent_ratio=self.rm_accent_ratio,
            strip_accents=strip_accents,
        )
        return preprocessed_dataset

    def load_dataset(self):
        if self.method == "unlabeled":
            converters = {'input': literal_eval, 'regrex_rule': literal_eval, 'dict_rule': literal_eval}
            data = pd.read_csv(self.datapath, converters=converters)
            data.columns = ['id', 'original', 'input', 'rule_01', 'rule_02']
            data = data[['id', 'original', 'input', 'rule_01', 'rule_02']]
        else:
            converters = {'input': literal_eval, 'output': literal_eval, 'regrex_rule': literal_eval, 'dict_rule': literal_eval}
            data = pd.read_csv(self.datapath, converters=converters)
            data.columns = ['id', 'original', 'normalized', 'input', 'output', 'rule_01', 'rule_02']
        data = data.dropna(ignore_index=True)

        self.logger.info("Pre-processing {} data for student...".format(self.method))
        self.data = self.preprocess(data) # dictionary: column - list of values (each_sentence)
        if self.remove_accents and self.method != 'unlabeled':
            self.logger.info("Removing accents of {} data for student...".format(self.method))
            self.no_accent_data = self.preprocess(data, strip_accents=True)
            new_dict = {}
            for key, values in self.no_accent_data.items():
                new_dict[key] = values + self.data[key]
            self.no_accent_data = new_dict

    def __len__(self):
        return len(self.data['id'])

    def __getitem__(self, item):
        ret = {
            'id': self.data['id'][item],
            'input': self.data['input'][item],
            'output': self.data['output'][item] if 'output' in self.data else None,
            'input_ids': self.data['input_ids'][item],
            'output_ids': self.data['output_ids'][item] if 'output' in self.data else None,
            'align_index': self.data['align_index'][item],
            'weak_labels': self.data['weak_labels'][item],
            'sent_len': self.data['sent_len'][item],
            }
        return ret


class PseudoDataset(Dataset):
    # PseudoDataset: a Dataset class that provides extra functionalities for teacher-student training.
    def __init__(self, args, wsdataset, logger=None):
        super(PseudoDataset, self).__init__()
        self.args = args
        self.seed = args.seed
        self.method = wsdataset.method
        self.logger = logger
        self.num_labels = wsdataset.num_labels
        self.logger.info("copying data from {} dataset".format(wsdataset.method))
        self.original_data = deepcopy(wsdataset.data)
        self.data = deepcopy(self.original_data)
        self.no_accent_data = deepcopy(wsdataset.no_accent_data)
        self.student_data = {}
        self.teacher_data = {}
        self.logger.info("done")

    def keep(self, keep_indices, type='teacher'):
        self.logger.info("Creating Pseudo Dataset with {} items...".format(len(list(chain.from_iterable(keep_indices)))))
        new_dict = {}
        data = self.teacher_data if type=='teacher' else self.student_data
        for key, values in data.items():
            keep_sents = []
            for i, indices in enumerate(keep_indices):
                if len(indices)==0:
                    continue
                if key in ['id', 'align_index']:
                    keep_sent = [values[i][idx] for idx in indices]
                else:
                    keep_sent = values[i][np.array(indices)]
                keep_sents.append(keep_sent)
            new_dict[key] = keep_sents
        if type=='teacher':
            self.teacher_data = new_dict
        else:
            self.student_data = new_dict

    def downsample(self, sample_size):
        N = len(self.original_data['input'])
        if sample_size > N:
            self.logger.info("[WARNING] sample size = {} > {}".format(sample_size, N))
            sample_size = N
        self.logger.info("Downsampling {} data".format(sample_size))
        self.data = {}
        keep_indices = np.random.choice(N, sample_size, replace=False)
        for key, values in self.original_data.items():
            self.data[key] = [values[i] for i in keep_indices]
    
    def inference_downsample(self, start_idx, end_idx):
        # self.logger.info("Downsampling data from index {} to {}".format(start_idx, end_idx))
        for key, values in self.original_data.items():
            self.data[key] = values[start_idx:end_idx]

    def drop(self, col='teacher_labels', value=-1, type='teacher'):
        indices = []
        data = self.teacher_data if type=='teacher' else self.student_data
        for i, array in enumerate(data[col]):
            all_neg_one = np.all(array == -1, axis=1)
            keep = np.flatnonzero(~all_neg_one).tolist()
            indices.append(keep)
        
        self.keep(indices, type=type)

    def __len__(self):
        return len(self.data['id'])

    def __getitem__(self, item):
        ret = {
            'id': self.data['id'][item],
            'input': self.data['input'][item],
            'output': self.data['output'][item] if 'output' in self.data else None,
            'input_ids': self.data['input_ids'][item],
            'output_ids': self.data['output_ids'][item] if 'output' in self.data else None,
            'align_index': self.data['align_index'][item],
            'weak_labels': self.data['weak_labels'][item],
            'sent_len': self.data['sent_len'][item],
            }
        return ret