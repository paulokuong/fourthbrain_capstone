from flask import Flask, jsonify
from flask_swagger import swagger
from flask import redirect, session, request, json, render_template

app = Flask(__name__)


@app.route("/spec")
def spec():
    return jsonify(swagger(app))


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080, threaded=True)
