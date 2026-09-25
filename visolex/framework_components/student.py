import os
from normalizer.trainer import Trainer
from utils import sort_data

class Student:
    def __init__(self, args, tokenizer, logger=None):
        self.args = args
        self.logger = logger
        self.name = args.student_name
        self.tokenizer = tokenizer
        self.training_mode = args.training_mode
        self.remove_accents = args.remove_accents
        self.trainer = Trainer(args=self.args, tokenizer=self.tokenizer, logger=self.logger)

    def train(self, train_dataset, dev_dataset, mode='train'):
        assert mode in ['train', 'finetune', 'train_pseudo']
        if mode in ['train', 'finetune']:
            train_dataset = sort_data(train_dataset, remove_accents=self.remove_accents)
        dev_dataset = sort_data(dev_dataset, remove_accents=self.remove_accents)
        if mode == 'train':
            res = self.trainer.train(
                train_data=train_dataset,
                dev_data=dev_dataset,
            )
            return res
        if mode == 'finetune':
            res = self.trainer.finetune(
                train_data=train_dataset,
                dev_data=dev_dataset,
            )
            return res
        if mode == 'train_pseudo':
            res = self.trainer.train_pseudo(
                train_data=train_dataset.teacher_data if self.training_mode=='weakly_supervised' else train_dataset.student_data,
                dev_data=dev_dataset,
            )
            return res

    def predict(self, dataset, dataIter=None, inference_mode=False):
        res = self.trainer.predict(data=dataset, dataIter=dataIter, inference_mode=inference_mode)
        return res

    def inference(self, user_input):
        res = self.trainer.inference(user_input)
        return res

    def save(self, name='student'):
        savefolder = os.path.join(self.args.logdir, name)
        self.logger.info('Saving {} to {}'.format(name, savefolder))
        os.makedirs(savefolder, exist_ok=True)
        self.trainer.save(savefolder)

    def load(self, name):
        savefolder = os.path.join(self.args.logdir, name)
        if not os.path.exists(savefolder):
            raise(BaseException('Pre-trained student folder does not exist: {}'.format(savefolder)))
        self.trainer.load(savefolder)