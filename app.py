import tiktoken
import logging
from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

logging.basicConfig(level=logging.ERROR)


@app.route('/')
@app.route('/test', methods=['POST'])
def test():
    data = request.json
    return jsonify(data, "post received")


@app.route('/bloomz', methods=['POST'])
def tokenize():
    model_name_or_path = 'bigscience/bloomz-560m'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    data = request.json
    try:
        if 'text' not in data:
            raise ValueError('No string was provided')
        if 'text' in data:
            text = data['text']
            print(tokenizer.encode(text))
            input_ids = tokenizer.encode(text)
            response = {'success': True, 'input_ids': input_ids}
            return jsonify(response), 200
    except Exception as error:
        app.logger.error(error)
        return jsonify({'error': str(error)}), 400


@app.route('/gpt3', methods=['POST'])
def gpt3_token():
    model_name_or_path = "gpt-4"
    tokenizer = tiktoken.encoding_for_model(model_name_or_path)
    data = request.json

    try:
        if 'text' not in data:
            raise ValueError('No string was provided')

        text = data['text']
        encoded_text = tokenizer.encode(text)
        decoded_text = tokenizer.decode(encoded_text)

        if not decoded_text:
            raise ValueError(
                'Tokenization and decoding resulted in an empty string')

        response = {'success': True, 'input_ids': encoded_text}
        return jsonify(response), 200

    except Exception as error:
        app.logger.error(error)
        return jsonify({'error': str(error)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0')
