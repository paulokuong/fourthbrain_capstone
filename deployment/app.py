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

app = Flask(__name__)


@app.route("/spec")
def spec():
    return jsonify(swagger(app))


@app.route('/user_conversion')
def user_conversion():
    # model = XGBClassifier()
    # model.load_model('static/models/xgboost/xgboost_model')
    # preprocessed_data = pickle.load(
    #     open('static/data/preprocessed_data', 'rb'))
    #
    # selected_features = [
    #     'timeOnSiteSeconds',
    #     'totalViewProducts',
    #     'uniqueAddToCarts',
    #     'productPriceMean',
    #     'totalSearches',
    #     'totalAddToCartQty',
    #     'hourOfDay'
    # ]
    # new_features = preprocessed_data['features'][selected_features]
    # shap_values = shap.TreeExplainer(model).shap_values(new_features)
    # shap.summary_plot(
    #     shap_values, pd.DataFrame(new_features, columns=new_features.columns),
    #     plot_type="bar", show=False)
    img_path = 'static/img/shap_bar_graph.png'
    explanability = 'static/img/explanability.png'
    # plt.savefig(img_path)
    return render_template(
        'user_conversion.html', shap_bar_graph=img_path,
        explanability=explanability)


@app.route('/user_conversion_predict')
def user_conversion_predict():
    model = XGBClassifier()
    model.load_model('xgboost_model')
    model.predict_proba(np.array([100, 300, 30, 23, 30, 4, 0]).reshape(1, -1))


@app.route('/personalization')
def personalization():
    return render_template('personalization.html')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
