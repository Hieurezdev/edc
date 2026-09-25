import os
import numpy as np
from normalizer.rule_attention_network import RAN
from utils import sort_data, gen_dataIter

class Teacher:
    """
    Teacher:
        (1) considers multiple weak sources (1) multiple weak (heuristic) rules, (2) Student
        (2) aggregates weak sources with an aggregation model (e.g., RAN) to compute a single pseudo-label
    """

    def __init__(self, args, tokenizer, logger=None):
        self.name = args.teacher_name
        if self.name != "ran":
            raise (BaseException("Teacher not supported: {}".format(self.name)))
        self.args = args
        self.logger = logger
        self.seed = args.seed
        np.random.seed(self.seed)
        self.num_labels = len(tokenizer) + 1
        self.num_rules = args.num_rules
        self.agg_model = RAN(
            args=self.args, num_rules=self.num_rules, num_labels=self.num_labels, 
            logger=self.logger, name=self.name
        )
        self.name = 'ran'
        self.student = None

    def predict(self, dataset):
        dataset = sort_data(dataset)
        res = self.aggregate_sources(dataset)
        return res

    def predict_ran(self, dataset, inference_mode=False):
        self.logger.info("Getting RAN predictions")
        dataset = sort_data(dataset)
        len_ls = list(set(dataset['sent_len']))
        dataIter = gen_dataIter(dataset, self.agg_model.unsup_batch_size, len_ls)
        data_dict = self.student.predict(dataset=None, dataIter=dataIter)
        res = self.aggregate_sources(data_dict, inference_mode=inference_mode)
        return res

    def train_ran(self, train_dataset=None, dev_dataset=None, unlabeled_dataset=None):
        train_dataset = sort_data(train_dataset) if train_dataset is not None else None
        dev_dataset = sort_data(dev_dataset) if dev_dataset is not None else None
        unlabeled_dataset = sort_data(unlabeled_dataset) if unlabeled_dataset is not None else None

        self.logger.info("Getting rule predictions")
        train_len_ls = list(set(train_dataset['sent_len']))
        trainIter = gen_dataIter(train_dataset, self.agg_model.sup_batch_size, train_len_ls, shuffle=True, seed=self.seed)
        dev_len_ls = list(set(dev_dataset['sent_len']))
        devIter = gen_dataIter(dev_dataset, self.agg_model.sup_batch_size, dev_len_ls)
        unsup_len_ls = list(set(unlabeled_dataset['sent_len']))
        unsupIter = gen_dataIter(unlabeled_dataset, self.agg_model.unsup_batch_size, unsup_len_ls)

        self.logger.info("Getting student predictions on train (and dev) dataset")
        assert self.student is not None, "To train RAN we need access to the Student"
        train_data = self.student.predict(dataset=None, dataIter=trainIter) if train_dataset is not None else {'features': None, 'proba': None}
        dev_data = self.student.predict(dataset=None, dataIter=devIter) if dev_dataset is not None else {'features': None, 'proba': None}
        unsup_data = self.student.predict(dataset=None, dataIter=unsupIter) if unlabeled_dataset is not None else {'features': None, 'proba': None}

        self.logger.info("Training Rule Attention Network")
        self.agg_model.train(
            train_data=train_data,
            dev_data=dev_data,
            unsup_data=unsup_data,
        )
        del train_data, dev_data, unsup_data
        return {}

    def aggregate_sources(self, data_dict, inference_mode=False):
        if self.name != "ran":
            raise(BaseException("Teacher method not implemented: {}".format(self.name)))
        res = self.agg_model.predict_ran(data_dict, inference_mode=inference_mode)
        return res

    def save(self, savename=None):
        if savename is None:
            savefolder = os.path.join(self.args.logdir, 'teacher')
        else:
            savefolder = os.path.join(self.args.logdir, savename)

        self.logger.info("Saving teacher at {}".format(savefolder))
        os.makedirs(savefolder, exist_ok=True)
        model_file = os.path.join(savefolder, 'rule_attention_network.pt')
        self.agg_model.save(model_file)
        return

    def load(self, name):
        savefolder = os.path.join(self.args.logdir, name)
        self.logger.info("Loading teacher from {}".format(savefolder))
        model_file = os.path.join(savefolder, 'rule_attention_network.pt')
        self.agg_model.load(model_file)
        return