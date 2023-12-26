import flask
import json
import os
from flask import send_from_directory, request

# Flask app should start in global layout

app = flask.Flask(__name__)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/favicon.png')

@app.route('/')
@app.route('/home')
def home():
    return "Hello World"

@app.route('/webhook-math', methods=['POST'])
def webhook_math():
    req = request.get_json(force=True)
    params = req['queryResult']['parameters']

    n1 = params['number'][0]
    n2 = params['number'][1]
    match params['operation']:
        case 'add':
            out = sum(n for n in params['number'])
        case 'subtract':
            out = n1-n2
        case 'multiply':
            out = n1*n2
        case 'divide':
            out = n1/n2
        case _:
            out = None
    if out:
        return {
            'fulfillmentText': f'The answer is: {out}'
        }
    else:
        return {
            'fulfillmentText': 'This type of math is unsupported or impossible.'
        }

if __name__ == "__main__":
    app.secret_key = 'ItIsASecret'
    app.debug = True
    app.run(port=5000)
