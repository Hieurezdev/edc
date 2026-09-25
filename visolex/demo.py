import os
import numpy as np
import torch
import copy
import json
from bs4 import BeautifulSoup
from framework_components.log import get_logger, close
from framework_components.student import Student
from utils import post_process, delete_special_tokens, bold, get_tokenizer
from project_variables import DICT_PATH
from arguments import parse_arguments
from chatgpt import run_chatgpt
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

def parse_html_to_string(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    cleaned_text = "\n".join([line for line in text.splitlines() if line.strip()])
    return cleaned_text

def load_nsw_dict():
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    return dictionary

def add_vocab(nsw, response, dictionary):
    dictionary[nsw] = {}
    dictionary[nsw]['response'] = response

    with open(DICT_PATH, 'w') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)

def dict_lookup(nsw):
    nsw_dict = load_nsw_dict()
    if nsw in nsw_dict:
        normalized_data = nsw_dict[nsw]
        result = normalized_data
        try:
            normalized_data = result['response']['normalized'][0]
            response = {
                'word': normalized_data['word'],
                'definition': normalized_data['definition'],
                'abbreviation': normalized_data.get('abbreviations', ''),
                'example': normalized_data.get('example', '')
            }
            response_str = ""
            for key, value in response.items():
                string = "- " + key.upper() + ": " + value + "\n"
                response_str += string
        except:
            response_str = json.dumps(result)
    else:
        response = run_chatgpt(nsw)
        response = response.replace("```html\n", "").replace("\n```", "")
        add_vocab(nsw, response, nsw_dict)
        response_str = parse_html_to_string(response)
    
    return response_str

def concatenate_nsw_spans(nsw_spans):
    result = []
    current_span = nsw_spans[0]

    for i in range(1, len(nsw_spans)):
        next_span = nsw_spans[i]
        if current_span['end_index'] == next_span['start_index']:
            current_span['nsw'] += next_span['nsw']
            current_span['end_index'] = next_span['end_index']
        else:
            result.append(current_span)
            current_span = next_span
    result.append(current_span)

    return result

def nsw_detection(source_tokens, is_nsw, tokenizer):
    source_tokens, keep_indices = delete_special_tokens(source_tokens)
    is_nsw = [is_nsw[i] for i in keep_indices]
    nsw_indices = [i for i, nsw in enumerate(is_nsw) if nsw == 1]
    nsw_tokens = [source_tokens[i] for i in nsw_indices]

    nsw_spans = []
    end_index = 0
    for i in range(len(source_tokens)):
        if source_tokens[i].startswith('▁'):
            end_index += 1
        current_text = tokenizer.convert_tokens_to_string([source_tokens[i]])
        full_text = tokenizer.convert_tokens_to_string(source_tokens[:(i+1)])
        if is_nsw[i] == 1:
            if current_text:
                nsw_spans.append({
                    'index': i,
                    'start_index': end_index,
                    'end_index': end_index + len(current_text),
                    'nsw': current_text
                })
        end_index = len(full_text) if current_text else len(full_text) + 1

    # nsw_spans = concatenate_nsw_spans(nsw_spans)
    return nsw_spans

def lexnorm(output, tokenizer):

    # NSW Detection
    nsw_spans = nsw_detection(output['source_tokens'], output['is_nsw'], tokenizer)
    nsw_indices = [span['index'] for span in nsw_spans]

    # Lexical Normalization
    # pred = [id for id in output['pred'] if id != -1]
    # proba = [output['proba'][i] for i, id in enumerate(output['pred']) if id != -1]
    pred = output['pred']
    proba = output['proba']
    decoded_pred = tokenizer.convert_ids_to_tokens(pred)
    for i, nsw_idx in enumerate(nsw_indices):
        nsw_spans[i]['prediction'] = tokenizer.convert_tokens_to_string([decoded_pred[nsw_idx+1]])
        nsw_spans[i]['confidence_score'] = round(proba[nsw_idx+1], 4)
    pred_tokens, keep_indices = delete_special_tokens(decoded_pred)
    proba = [proba[i] for i in keep_indices]
    pred_str = tokenizer.convert_tokens_to_string(pred_tokens)
    pred_str = post_process(pred_str)
    return nsw_spans, pred_str


def demo():
    args = parse_arguments()
    np.random.seed(args.seed)

    print("====================================================================")
    print(bold("Chose service:"))
    print("1. If you want to normalize non-standard words in a sentence, enter 'lexnorm'")
    print("2. If you want to look non-standard words up in NSW dictionary, enter 'dict_lookup'")
    service = input()
    while service not in ['lexnorm', 'dict_lookup']:
        print(bold("INVALID service name: Only support 'lexnorm' or 'dict_lookup'"))
        print(bold("Chose service:"))
        service = input()

    if service == 'lexnorm':
        print("====================================================================")
        print(bold("Chose model:"))
        print("1. BARTpho: enter 'bartpho'")
        print("2. ViSoBERT: enter 'visobert'")
        model = input()
        while model not in ['bartpho', 'visobert', 'phobert']:
            print(bold("NOT SUPPORTED model: Only support 'bartpho' and 'visobert'"))
            print(bold("Chose model:"))
            model = input()
        args.student_name = model

        print("====================================================================")
        print(bold("Enter a sentence for normalization:"))
        user_input = input()

        print("====================================================================")
        print(bold("Is your input sentence missing diacritics? (yes or no)"))
        remove_accents = input()
        remove_accents = remove_accents.lower()
        while remove_accents not in ['yes', 'no']:
            print(bold("INVALID value: Only answer 'yes' or 'no'"))
            print(bold("Is your input sentence missing diacritics? (yes or no)"))
            remove_accents = input()
            remove_accents = remove_accents.lower()
        args.rm_accent_ratio = 1.0 if remove_accents == "yes" else 0.0

        if args.rm_accent_ratio != 0.0:
            args.remove_accents = True

        # Start Experiment
        args.logdir = os.path.join(args.experiment_folder, args.student_name, args.training_mode + '_accent_{}'.format(str(args.rm_accent_ratio)))
        args.lower_case = True
        args.hard_student_rule = True
        args.soft_labels = True
        args.append_n_mask = True
        args.nsw_detect = True

        logger = get_logger(logfile=os.path.join(args.logdir, 'demo.log'))

        # Setup CUDA, GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        args.n_gpu = torch.cuda.device_count()
        args.device = device

        logger.info("\n\n\t\t *** START DEMO ***\nargs={}".format(args))

        tokenizer = get_tokenizer(args.student_name)
        student = Student(args, tokenizer=tokenizer, logger=logger)
        student.load("student_best")

        # Run inference
        output = student.inference(user_input=user_input)

        # Run service
        nsw_spans, pred_str = lexnorm(output, tokenizer=tokenizer)

        # NSW Detection
        copied_nsw_spans = copy.deepcopy(nsw_spans)
        concat_nsw_spans = concatenate_nsw_spans(copied_nsw_spans)

        print(bold("============================ NON-STANDARD WORDS DETECTION ============================"))
        for i in range(len(concat_nsw_spans)):
            print(f"{i + 1}. NSW '{concat_nsw_spans[i]['nsw']}' start from index {concat_nsw_spans[i]['start_index']} to index {concat_nsw_spans[i]['end_index']}")
        print("\n")

        # Lexical normalization
        print(bold("============================ LEXICAL NORMALIZATION ============================"))
        print(bold("NORMALIZED VERSION:"), pred_str)
        print("\n")
        for i in range(len(nsw_spans)):
            print(f"{i + 1}. NSW '{nsw_spans[i]['nsw']}' ==> STANDARD FORM: '{nsw_spans[i]['prediction']}' (CONFIDENCE : {nsw_spans[i]['confidence_score']})")

        logger.info("\n\n\t\t *** END DEMO ***")
        close(logger)

    elif service == "dict_lookup":
        print("====================================================================")
        print(bold("Enter a non-standard word:"))
        nsw = input()
        response = dict_lookup(nsw)
        print(bold("====================== NSW INTERPRETATION ======================"))
        print(response)

demo()