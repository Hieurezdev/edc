import os
import numpy as np
import torch
import shutil
from framework_components.log import get_logger, close
from framework_components.data_handler import DataHandler
from framework_components.student import Student
from framework_components.teacher import Teacher
from framework_components.evaluator import Evaluator
from utils import *
from arguments import parse_arguments
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

def visolex(args, logger):
    """
        Self-training with weak supervivsion
        Leverages labeled, unlabeled data and weak rules for training a neural network
    """

    teacher_dev_res_list = []
    teacher_test_res_list = []
    teacher_train_res_list = []
    dev_res_list = []
    test_res_list = []
    train_res_list = []
    results = {}

    student_pred_list = []

    ev = Evaluator(args, logger=logger)
    tokenizer = get_tokenizer(args.student_name)

    logger.info("Building student: {}".format(args.student_name))
    student = Student(args, tokenizer=tokenizer, logger=logger)

    logger.info("Loading data")
    dh = DataHandler(args, tokenizer=tokenizer, logger=logger)
    train_dataset = dh.load_dataset(method='train') 
    dev_dataset = dh.load_dataset(method='dev')
    test_dataset = dh.load_dataset(method='test')
    unlabeled_dataset = dh.load_dataset(method='unlabeled')

    logger.info("Creating pseudo-dataset")
    pseudodataset = dh.create_pseudodataset(unlabeled_dataset)
    pseudodataset.downsample(args.sample_size)

    # Train Student
    logger.info("\n\n\t*** Training Student on labeled data ***")
    newtraindataset = dh.create_pseudodataset(train_dataset)
    results['student_train'] = student.train(train_dataset=newtraindataset, dev_dataset=dev_dataset, mode='train')
    train_res_list.append(results['student_train'])
    if args.training_mode == 'only_student':
        student.save('student_best')

    logger.info("\n\n\t*** Evaluating student on dev data ***")
    results['supervised_student_dev'] = evaluate(student, dev_dataset, ev, comment="student dev", remove_accents=args.remove_accents)
    dev_res_list.append(results['supervised_student_dev'])

    logger.info("\n\n\t*** Evaluating student on test data ***")
    results['supervised_student_test'], s_test_dict = evaluate(student, test_dataset, ev, "test", comment="student test", remove_accents=args.remove_accents)
    test_res_list.append(results['supervised_student_test'])
    student_pred_list.append(s_test_dict)
    # write_predictions(args, logger, tokenizer, s_test_dict, file_name="student_test_predictions")

    if args.training_mode == 'only_student':
        save_and_report_results(args, results, logger)
        return results
                
    else:
        if args.training_mode == 'weakly_supervised':
            logger.info("Building teacher")
            teacher = Teacher(args, tokenizer=tokenizer, logger=logger)
            teacher.student = student
            if args.student_name in ['visobert', 'phobert']:
                teacher.agg_model.xdim = student.trainer.model.config.hidden_size
            else:
                teacher.agg_model.xdim = student.trainer.model.config.d_model

            logger.info("Initializing teacher")
            results['teacher_train'] = {'precision': 0, 'recall': 0, 'f1_score': 0, 'accuracy': 0, 'perf': 0}
            results['teacher_dev'] = {'precision': 0, 'recall': 0, 'f1_score': 0, 'accuracy': 0, 'perf': 0}
            results['teacher_test'] = {'precision': 0, 'recall': 0, 'f1_score': 0, 'accuracy': 0, 'perf': 0}
            teacher_train_res_list.append(results['teacher_train'])
            teacher_dev_res_list.append(results['teacher_dev'])
            teacher_test_res_list.append(results['teacher_test'])

        # Self-Training with Weak Supervision
        for iter in range(args.num_iter):
            logger.info("\n\n\t *** Starting loop {}/{} ***".format(iter+1, args.num_iter))
            # Create pseudo-labeled dataset
            pseudodataset.downsample(args.sample_size)

            if args.training_mode == 'weakly_supervised':
                _ = teacher.train_ran(train_dataset=train_dataset, dev_dataset=dev_dataset, unlabeled_dataset=pseudodataset)

                # Apply Teacher on unlabeled data
                teacher_pred_dict_unlabeled = teacher.predict_ran(dataset=pseudodataset)

                logger.info("\n\n\t*** Evaluating teacher on dev data ***")
                teacher_dev_res, t_dev_dict = evaluate(teacher, dev_dataset, ev, "ran", comment="teacher dev iter{}".format(iter+1))
                teacher_dev_res_list.append(teacher_dev_res)
                logger.info("\n\n\t*** Evaluating teacher on test data ***")
                teacher_test_res, t_test_dict = evaluate(teacher, test_dataset, ev, "ran", comment="teacher test iter{}".format(iter+1))
                teacher_test_res_list.append(teacher_test_res)

                # analyze_rule_attention_scores(t_test_dict, logger, args.logdir, name='test_iter{}'.format(iter))

                logger.info("Update unlabeled data with Teacher's predictions")
                pseudodataset.teacher_data['id'] = teacher_pred_dict_unlabeled['id']
                pseudodataset.teacher_data['input_ids'] = teacher_pred_dict_unlabeled['input_ids']
                pseudodataset.teacher_data['is_nsw'] = teacher_pred_dict_unlabeled['is_nsw']
                pseudodataset.teacher_data['align_index'] = teacher_pred_dict_unlabeled['align_index']
                pseudodataset.teacher_data['labels'] = teacher_pred_dict_unlabeled['preds']
                pseudodataset.teacher_data['proba'] = teacher_pred_dict_unlabeled['proba']
                pseudodataset.teacher_data['weights'] = [np.max(array, axis=-1) for array in teacher_pred_dict_unlabeled['proba']]
                pseudodataset.drop(col='labels', value=-1, type='teacher')
                del teacher_pred_dict_unlabeled

            else:
                sorted_pseudodataset = sort_data(pseudodataset)
                student_pred_dict_unlabeled = student.predict(dataset=sorted_pseudodataset)

                logger.info("Update unlabeled data with Student's predictions")
                pseudodataset.student_data['id'] = student_pred_dict_unlabeled['id']
                pseudodataset.student_data['input_ids'] = student_pred_dict_unlabeled['input_ids']
                pseudodataset.student_data['is_nsw'] = student_pred_dict_unlabeled['is_nsw']
                pseudodataset.student_data['align_index'] = student_pred_dict_unlabeled['align_index']
                pseudodataset.student_data['labels'] = student_pred_dict_unlabeled['preds']
                pseudodataset.student_data['proba'] = student_pred_dict_unlabeled['proba']
                pseudodataset.student_data['weights'] = [np.max(array, axis=-1) for array in student_pred_dict_unlabeled['proba']]
                pseudodataset.drop(col='labels', value=-1, type='student')
                del student_pred_dict_unlabeled

            logger.info('Re-train student on pseudo-labeled instances provided by the teacher')
            train_res = student.train(train_dataset=pseudodataset, dev_dataset=dev_dataset, mode='train_pseudo')

            logger.info('Fine-tuning the student on clean labeled data')
            train_res = student.train(train_dataset=newtraindataset, dev_dataset=dev_dataset, mode='finetune')
            train_res_list.append(train_res)

            logger.info("\n\n\t*** Evaluating student on dev data and update records ***")
            dev_res = evaluate(student, dev_dataset, ev, comment="student dev iter{}".format(iter+1))
            logger.info("Student Dev performance on iter {}: {}".format(iter, dev_res['perf']))
            logger.info("\n\n\t*** Evaluating student on dev data and update records ***")
            test_res, s_test_dict = evaluate(student, test_dataset, ev, "test", comment="student test iter{}".format(iter+1))
            logger.info("Student Test performance on iter {}: {}".format(iter, test_res['perf']))

            prev_max = max([x['perf'] for x in dev_res_list])
            if dev_res['perf'] > prev_max:
                logger.info("Improved dev performance from {:.2f} to {:.2f}".format(prev_max, dev_res['perf']))
                student.save("student_best")
                if args.training_mode == 'weakly_supervised':
                    teacher.save("teacher_best")
            dev_res_list.append(dev_res)
            test_res_list.append(test_res)
            student_pred_list.append(s_test_dict)

        # Store Final Results
        logger.info("Final Results")
        if args.training_mode == 'weakly_supervised':
            teacher_all_dev = [x['perf'] for x in teacher_dev_res_list]
            teacher_all_test = [x['perf'] for x in teacher_test_res_list]
            teacher_perf_str = ["{}:\t{:.2f}\t{:.2f}".format(i, teacher_all_dev[i], teacher_all_test[i]) for i in np.arange(len(teacher_all_dev))]
            logger.info("TEACHER PERFORMANCES:\n{}".format("\n".join(teacher_perf_str)))

        all_dev = [x['perf'] for x in dev_res_list]
        all_test = [x['perf'] for x in test_res_list]
        perf_str = ["{}:\t{:.2f}\t{:.2f}".format(i, all_dev[i], all_test[i]) for i in np.arange(len(all_dev))]
        logger.info("STUDENT PERFORMANCES:\n{}".format("\n".join(perf_str)))

        # Get results in the best epoch (if multiple best epochs keep last one)
        best_dev_epoch = len(all_dev) - np.argmax(all_dev[::-1]) - 1
        best_test_epoch = len(all_test) - np.argmax(all_test[::-1]) - 1
        logger.info("BEST DEV {} = {:.3f} for epoch {}".format(args.metric, all_dev[best_dev_epoch], best_dev_epoch))
        logger.info("FINAL TEST {} = {:.3f} for epoch {} (max={:.2f} for epoch {})".format(args.metric, all_test[best_dev_epoch], best_dev_epoch, all_test[best_test_epoch], best_test_epoch))

        if args.training_mode == 'weakly_supervised':
            results['teacher_train_iter'] = teacher_train_res_list
            results['teacher_dev_iter'] = teacher_dev_res_list
            results['teacher_test_iter'] = teacher_test_res_list

        results['student_train_iter'] = train_res_list
        results['student_dev_iter'] = dev_res_list
        results['student_test_iter'] = test_res_list

        results['student_dev'] = dev_res_list[best_dev_epoch]
        results['student_test'] = test_res_list[best_dev_epoch]
        if args.training_mode == 'weakly_supervised':
            results['teacher_dev'] = teacher_dev_res_list[best_dev_epoch]
            results['teacher_test'] = teacher_test_res_list[best_dev_epoch]

        write_predictions(
            args, logger, tokenizer, student_pred_list[best_dev_epoch], file_name="student_best_predictions"
        )
        
        # Save models and results
        student.save("student_last")
        if args.training_mode == 'weakly_supervised':
            teacher.save("teacher_last")
        save_and_report_results(args, results, logger)
        return results

def main():
    args = parse_arguments()
    np.random.seed(args.seed)

    if args.rm_accent_ratio != 0.0:
        args.remove_accents = True
    # Start Experiment
    args.logdir = os.path.join(args.experiment_folder, args.student_name, args.training_mode + '_accent_{}'.format(str(args.rm_accent_ratio)))

    # Check if the directory exists and remove it if it does (recursively deletes a directory and all its contents)
    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)

    os.makedirs(args.logdir, exist_ok=True)
    logger = get_logger(logfile=os.path.join(args.logdir, 'log.log'))

    # Setup CUDA, GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logger.info("\n\n\t\t *** NEW EXPERIMENT ***\nargs={}".format(args))
    visolex(args, logger=logger)
    logger.info("\n\n\t\t *** END EXPERIMENT ***")
    close(logger)

if __name__ == "__main__":
    main()