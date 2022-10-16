from integrate import init, translate
from flask import Flask, jsonify, request, abort, Response
from werkzeug.exceptions import BadRequest, InternalServerError

import socket
import os
import json
import sys

app = Flask(__name__)
# Get Unicode characters and not ASCII:
app.config['JSON_AS_ASCII'] = False


@app.route("/", methods=['GET'])
def root():
    return "This is a translation module. Use the /translation endpoint for translating text."


@app.route("/translation", methods=['POST'])
def translation():
    return handle_POST(translate)


def handle_POST(func):
    """
    Handles POST requests where the body of the request is JSON where one of the keys is "q". E.g. {"q": "hello world"}
    :param func. A function that takes a translation model, a string and a logger object and returns a python dictionary.
    """
    payload = request.json
    if not (payload or payload.get('q')):
        return BadRequest("No payload given")
    print("Flask input: ", payload["q"])
    data = func(payload["q"])
    print("Flask response: ", data.get("result"))
    return jsonify(data)


if __name__ == "__main__":

    port = sys.argv[-1]
    sys.argv = sys.argv[:-2]
    model = init()
    app.run(host='0.0.0.0', port=port)
