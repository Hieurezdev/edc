import argparse
from project_variables import DATASET_DIR, EXPERIMENT_DIR, NUM_RULES, UNLABELED_SAMPLE_SIZE

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Main Arguments
    parser.add_argument("--datapath", help="Path to base dataset folder", type=str, default=DATASET_DIR)
    parser.add_argument("--student_name", help="Student short name", type=str, default='visobert')
    parser.add_argument("--teacher_name", help="Teacher short name", type=str, default='ran')
    parser.add_argument("--training_mode", help="Training mode ['only_student', 'self-training', 'weakly_supervised']", type=str, default='weakly_supervised')
    parser.add_argument("--inference_model", help="Which model used for inference: student of teacher", type=str, default='student')

    # Extra Arguments
    parser.add_argument("--experiment_folder", help="Dataset name", type=str, default=EXPERIMENT_DIR)
    parser.add_argument("--logdir", help="Experiment log directory", type=str, default="./experiments/visobert")
    parser.add_argument("--metric", help="Evaluation metric", type=str, default='f1_score')
    parser.add_argument("--num_iter", help="Number of self/co-training iterations", type=int, default=10)
    parser.add_argument('--num_rules', type=int, default=NUM_RULES, help="Number of of rules")
    parser.add_argument("--num_epochs", default=10, type=int, help="Total number of training epochs for student.")
    parser.add_argument("--num_unsup_epochs", default=5, type=int, help="Total number of training epochs for training student on unlabeled data")
    parser.add_argument("--debug", action="store_true", help="Activate debug mode")
    parser.add_argument("--remove_accents", action="store_true", help="Remove accents in dataset")
    parser.add_argument("--rm_accent_ratio", default=0.0, type=float, help="The ratio of character in a sentence to remove accents")
    parser.add_argument('--append_n_mask', action="store_true", help="Append mask for training Student")
    parser.add_argument('--nsw_detect', action="store_true", help="Whether to detect NSWs or not")
    parser.add_argument("--soft_labels", action="store_true", help="Use soft labels for training Student")
    parser.add_argument("--loss_weights", action="store_true", help="Use instance weights in loss function according to Teacher's confidence")
    parser.add_argument("--convert_abstain_to_random", action="store_true", help="In Teacher, if rules abstain on dev/test then flip a coin")
    parser.add_argument("--hard_student_rule", action="store_true", help="When using Student as a rule in Teacher, use hard (instead of soft) student labels")
    parser.add_argument("--train_batch_size", help="Train batch size", type=int, default=16)
    parser.add_argument("--eval_batch_size", help="Dev batch size", type=int, default=128)
    parser.add_argument("--unsup_batch_size", help="Unsupervised batch size", type=int, default=128)
    parser.add_argument("--lower_case", action="store_true", help="Use uncased model")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--fine_tuning_strategy", help="Student finetuning strategy", type=str, default='flexible_lr')
    parser.add_argument("--sample_size", nargs="?", type=int, default=UNLABELED_SAMPLE_SIZE, help="number of unlabeled samples for evaluating uncetainty on in each self-training iteration")
    parser.add_argument("--topk", help="Return top K predictions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    
    args = parser.parse_args()
    return args