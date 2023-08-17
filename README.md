# Tokenizer

This is a python flask server. It is the backend to an app that tokenizes strings. The tokenizers used are:

- tiktoken with the gpt4 model
- Hugging Face Autotokenizer with the bigscience/bloomz-560m model

Install dependencies:

```
pip install -r requirements.txt
```

Run the server:

```
FLASK_APP=app flask run
```
