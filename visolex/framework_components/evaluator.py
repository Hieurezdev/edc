import numpy as np
from statistics import mean
implemented_metrics = ['accuracy', 'f1_score', 'precision', 'recall']

class Evaluator:
    # A class that implements all evaluation metrics and prints relevant statistics
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.metric = args.metric
        assert self.metric in implemented_metrics, "Evaluation metric not implemented: {}".format(self.metric)

    def evaluate(self, preds, targets, sources, comment="", verbose=True):
        assert len(preds) == len(targets), "pred should have same length as true: pred={} gt={}".format(
            len(preds),
            len(labels)
        )
        normed_recall = []
        precision = []
        recall = []
        f1_score = []
        accuracy = []

        for i in range(len(preds)):
            pred = np.array(preds[i])
            source = np.array(sources[i])
            target = np.array(targets[i])
            total_num = pred.size

            normed = source==target
            need_norm = source!=target
            pred_need_norm = np.sum(source!=pred).item()

            # TP: number of need norm word are predicted correcly
            # FP: number of need norm word are predicted wrongly
            # TN: number of normed word are correcly predicted (not change)
            # FN: number of normed word are converted to non-standared
            tp = np.sum(target[need_norm]==pred[need_norm]).item()
            fp = np.sum(target[need_norm]!=pred[need_norm]).item()
            tn = np.sum(target[normed]==pred[normed]).item()
            fn = np.sum(target[normed]!=pred[normed]).item()

            r = tp/(tp+fp) if (tp+fp) > 0 else 0
            p = tp/pred_need_norm if pred_need_norm > 0 else 0
            norm_r = tn/(tn+fn) if (tn+fn) > 0 else 0
            f1 = (2*p*r)/(p+r) if (p+r) > 0 else 0
            acc = np.sum(pred==target).item()/total_num

            precision.append(p)
            recall.append(r)
            f1_score.append(f1)
            accuracy.append(acc)
            normed_recall.append(norm_r)

        res = {
            'precision': 100 * mean(precision),
            'recall': 100 * mean(recall),
            'normed_recall': 100 * mean(normed_recall),
            'f1_score': 100 * mean(f1_score),
            'accuracy': 100 * mean(accuracy),
        }
        res["perf"] = res[self.metric]

        self.logger.info("{} performance: {} = {:.2f}%".format(comment, self.metric, res["perf"]))
        return res