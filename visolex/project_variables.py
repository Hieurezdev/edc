import os

# OpenAI
OPENAI_KEY = "<your_api_key>"
OPENAI_MODEL = "gpt-4o"
MAX_TOKENS = 500
TEMPERATURE = 0.7

# Project directories
PROJECT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DATASET_DIR = os.path.join(PROJECT_PATH, "data")
EXPERIMENT_DIR = os.path.join(PROJECT_PATH, "experiments")
DICT_PATH = os.path.join(PROJECT_PATH, "dict", "dictionary.json")

# Tokenizer constants
MASK_TOKEN = '<mask>'
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
PAD_TOKEN = '<pad>'
NULL_STR = '<space>'
UNK_TOKEN = '<unk>'
NUM_LABELS_N_MASKS = 5
NULL_STR_TO_SHOW = '_'
SPECIAL_TOKEN_LS = [NULL_STR, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PRETRAINED_TOKENIZER_MAP = {
    'phobert': 'vinai/phobert-base',
    'visobert': 'uitnlp/visobert',
    'bartpho': 'vinai/bartpho-syllable',
}

# Student constants
SAMPLES_PER_TASK_TO_REPORT = {"normalize": ["all", "NEED_NORM", "NORMED", "PRED_NEED_NORM", "PRED_NORMED"],
                              "n_masks_pred": ["all", "n_masks_1", "n_masks_2", "n_masks_3", "n_masks_4", "n_masks_5"]}
AVAILABLE_FINE_TUNING_STRATEGY = ["standard", "flexible_lr"]
RM_ACCENTS_DICT = str.maketrans(
    "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴáàảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ",
    "A"*17 + "D" + "E"*11 + "I"*5 + "O"*17 + "U"*11 + "Y"*5 + "a"*17 + "d" + "e"*11 + "i"*5 + "o"*17 + "u"*11 + "y"*5
)

# RAN CONSTANTS
NUM_RULES = 2
UNLABELED_SAMPLE_SIZE = 8096