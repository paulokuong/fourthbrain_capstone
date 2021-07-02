"""
gunicorn --bind 0.0.0.0:5000 wsgi:app
"""
from flask import Flask, jsonify
from flask_swagger import swagger
from flask import redirect, session, request, json, render_template
from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt
import shap
import pickle
import numpy as np
from joblib import dump, load
from tensorflow import keras
from tensorflow_addons.metrics import F1Score

app = Flask(__name__)

"""
               score                feature
feature 3   0.067646  totalViewProducts
feature 8   0.062860  totalAddToCartQty
feature 4   0.055818    totalAddToCarts
feature 15  0.034540   productPriceMean
feature 9   0.033196          hourOfDay
feature 0   0.016931     uniqueSearches
feature 1   0.014983      totalSearches
feature 12  0.013691       has_campaign
"""
features = [
    {
        "id": "total_view_products", "name": "Total view products:",
        "min_value": 1, "max_value": 30, "default_value": 1, "step": 1
    },
    {
        "id": "total_add_to_cart_qty", "name": "Total add to cart qty",
        "min_value": 0, "max_value": 15, "default_value": 0, "step": 1
    },
    {
        "id": "total_add_to_carts", "name": "Total add to carts",
        "min_value": 0, "max_value": 10, "default_value": 0, "step": 1
    },
    {
        "id": "product_price_mean", "name": "Product price mean",
        "min_value": 1, "max_value": 50, "default_value": 1, "step": 0.1
    },
    {
        "id": "hour_of_day", "name": "Hour of day",
        "min_value": 0, "max_value": 23, "default_value": 0, "step": 1
    },
    {
        "id": "unique_searches", "name": "Unique searches",
        "min_value": 0, "max_value": 20, "default_value": 0, "step": 1
    },
    {
        "id": "total_searches", "name": "Total searches",
        "min_value": 0, "max_value": 20, "default_value": 0, "step": 1
    }
    # {
    #     "id": "has_campaign", "name": "Has Campaign",
    #     "min_value": 0, "max_value": 1, "default_value": 0, "step": 1
    # }
]


@app.route("/spec")
def spec():
    return jsonify(swagger(app))


@app.route('/user_conversion')
def user_conversion():
    """
    Best pipeline: GaussianNB(ExtraTreesClassifier(XGBClassifier(input_matrix, learning_rate=0.001, max_depth=10, min_child_weight=10, n_estimators=100, n_jobs=1, subsample=0.7500000000000001, verbosity=0), bootstrap=True, criterion=entropy, max_features=0.55, min_samples_leaf=10, min_samples_split=19, n_estimators=100))
    """
    img_path = 'static/img/shap_bar_graph.png'
    explanability = 'static/img/explanability.png'
    feature_importance = 'static/img/feature_importance.png'
    scores = 'static/img/scores.png'
    events_dist = 'static/img/events_dist.png'
    lstm_f1 = 'static/img/lstm_f1.png'
    lstm_model_loss = 'static/img/lstm_model_loss.png'

    return render_template(
        'user_conversion.html', shap_bar_graph=img_path,
        explanability=explanability, features=features,
        feature_importance=feature_importance, scores=scores,
        events_dist=events_dist, lstm_f1=lstm_f1, lstm_model_loss=lstm_model_loss)


@app.route('/user_conversion_lstm_predict')
def user_conversion_lstm_predict():
    seq = request.args.get('seq', '000')
    seq = str(seq)
    seq = seq.zfill(40)
    seq_arr = [int(i) for i in list(seq)]
    lstm_model_new = keras.models.load_model(
        'static/models/lstm/lstm_model.h5')
    predictions = lstm_model_new.predict_proba(
        np.array(seq_arr).reshape(1, 1, -1))
    return app.response_class(
        response=json.dumps({
            "input": seq_arr,
            "predictions": predictions.tolist()[0]
        }),
        status=200,
        mimetype='application/json'
    )


@app.route('/user_conversion_predict')
def user_conversion_predict():
    inputs = []
    for f in features:
        if 'price' in f['id'] or 'revenue' in f['id']:
            inputs.append(float(request.args.get(f['id'], 0)))
        else:
            inputs.append(int(request.args.get(f['id'], 0)))
    inputs.append(0)
    model = load('static/models/ensemble/stacking.joblib')
    predictions = np.round(model.predict_proba(
        np.array(inputs).reshape(1, -1)), 6)
    nonconvert, convert = predictions.tolist()[0]
    return app.response_class(
        response=json.dumps({
            "convert": convert,
            "nonconvert": nonconvert,
            "input": inputs
        }),
        status=200,
        mimetype='application/json'
    )


@app.route('/personalization')
def personalization():
    return render_template('personalization.html')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
