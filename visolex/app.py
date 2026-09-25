from flask import Flask, request, jsonify, render_template
import json
import logging
import os
import numpy as np
import torch
import copy
from framework_components.log import get_logger, close
from framework_components.student import Student
from utils import post_process, delete_special_tokens, get_tokenizer
from arguments import parse_arguments
from project_variables import DICT_PATH
from chatgpt import run_chatgpt
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

app = Flask(__name__, template_folder='interface', static_folder='interface/static')

# Load the dictionary initially
with open(DICT_PATH, 'r', encoding='utf-8') as f:
    dictionary = json.load(f)

# Global variables to hold the loaded model and tokenizer
loaded_model = None
loaded_tokenizer = None

# Setup CUDA, GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

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

    return nsw_spans

def lexnorm(output, tokenizer):
    # NSW Detection
    nsw_spans = nsw_detection(output['source_tokens'], output['is_nsw'], tokenizer)
    nsw_indices = [span['index'] for span in nsw_spans]

    # Lexical Normalization
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_model', methods=['POST'])
def load_model():
    global loaded_model, loaded_tokenizer, args
    model_name = request.form['model']
    percent = request.form['percent']  # Get the percent value from the request and cast to int

    args.student_name = model_name

    # Set the remove_accents argument if percent is not zero
    if percent != 0.0:
        args.remove_accents = True
        args.rm_accent_ratio = percent    

    try:
        # Start Experiment: set the log directory based on args
        args.logdir = os.path.join(args.experiment_folder, args.student_name, args.training_mode + '_accent_{}'.format(str(args.rm_accent_ratio)))

        # Initialize the logger for the experiment
        logger = get_logger(logfile=os.path.join(args.logdir, 'demo.log'))
        logger.info(f"Loading model: {model_name} with {percent}% unmarked text...")

        # Load the tokenizer and student model
        loaded_tokenizer = get_tokenizer(model_name)
        logger.info("Tokenizer loaded.")

        loaded_model = Student(args=args, tokenizer=loaded_tokenizer, logger=logger)  # Initialize student model with args
        logger.info("Initializing student model...")
        loaded_model.load("student_best")
        logger.info("Student model loaded successfully.")

        # Close the logger when done
        close(logger)

        # Model loaded successfully, send response to frontend
        response_message = f"Model {model_name} loaded successfully."
  
    
        return jsonify({
        'status': 'success',
        'message': response_message,
        'log': "\nModel loaded successfully."
      })

    except Exception as e:
        # Handle errors in loading the model
        print(f"Error loading model: {str(e)}")
        return jsonify({'status': 'error', 'message': f"Error loading model: {str(e)}"})

@app.route('/normalize_text', methods=['POST'])
def normalize_text():
    global loaded_model, loaded_tokenizer

    if loaded_model is None or loaded_tokenizer is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded. Please load the model first.'})

    input_text = request.form['input_text']

    try:
        # Perform text normalization using the loaded model and tokenizer
        output = loaded_model.inference(user_input=input_text)
        nsw_spans, pred_str = lexnorm(output, loaded_tokenizer)

        # Highlight the NSW tokens in pred_str
        highlighted_pred_str = input_text
        for i, span in enumerate(nsw_spans):
            nsw_word = span['nsw']
            highlighted_pred_str = highlighted_pred_str.replace(nsw_word, f"<mark>{nsw_word}</mark>")

        # Prepare the detection information
        detection_info = ""
        for i, span in enumerate(nsw_spans):
              detection_info += f"<tr><td>{span['nsw']}</td><td>{span['prediction']}</td><td>{span['confidence_score']}</td></tr>"

        # Return the highlighted normalized text and detection info as HTML
        return jsonify({
            'status': 'success',
            'normalized_text': pred_str,  # Raw normalized text
            'highlighted_text': highlighted_pred_str,  # Highlighted text
            'detection_info': detection_info  # Detection info details
        })

    except Exception as e:
        # Handle any errors during normalization
        print(f"Error during normalization: {str(e)}")
        return jsonify({'status': 'error', 'message': f"Error during normalization: {str(e)}"})


@app.route('/lookup_dict', methods=['POST'])
def load_dictionary():
    # Receive input word from the request
    nsw = request.form['nsw']

    # Check if the word exists in the dictionary
    if nsw in dictionary:
        normalized_data = dictionary[nsw]
        result = normalized_data
        normalized_data = result['response']['normalized'][0]
        return jsonify({'status': 'success', 
        'word': normalized_data['word'],
        'definition': normalized_data['definition'],
        'abbreviation': normalized_data.get('abbreviations', ''),
        'example': normalized_data.get('example', '')
        })             
    else:
        # If word is not in dictionary, return an error to the frontend
        return jsonify({
            'status': 'fail',
            'message': f"No result found in dictionary for '{nsw}'"
        })

@app.route('/lookup_API', methods=['POST'])
def lookup_API():
    nsw = request.form['nsw']
    try:
        # Call the run_chatgpt function to get the response
        response = run_chatgpt(nsw)
        response = response.replace("```html\n", "").replace("\n```", "")
        return jsonify({'status': 'success', 'message': response})
        
        if nsw in dictionary:
            return jsonify({'status': 'error', 'message': 'This word in Dictionary'})
        else:
            dictionary[nsw] = {}
            dictionary[nsw]['response'] = response

            with open(DICT_PATH, 'w') as f:
              json.dump(dictionary, f, ensure_ascii=False, indent=4)
        
            return jsonify({'status': 'success', 'message': response})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    # Parse args and store globally
    args = parse_arguments()
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.lower_case = True
    args.hard_student_rule = True
    args.soft_labels = True
    args.append_n_mask = True
    args.nsw_detect = True

    # Set up seed, logging, etc. here as needed
    np.random.seed(args.seed)

    # Start Flask app
    app.run(debug=True)