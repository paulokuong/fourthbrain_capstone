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
    time_on_site = request.args.get('time_on_site', 0)
    total_view_products = request.args.get('total_view_products', 0)
    unique_add_to_cart = request.args.get('unique_add_to_cart', 0)
    mean_product_price = request.args.get('mean_product_price', 0)
    total_searches = request.args.get('total_searches', 0)
    total_add_to_cart = request.args.get('total_add_to_cart', 0)
    hour_of_day = request.args.get('hour_of_day', 0)

    model = XGBClassifier()
    model.load_model('static/models/xgboost/xgboost_model')
    predictions = model.predict_proba(
        np.array([
            int(time_on_site),
            int(total_view_products),
            int(unique_add_to_cart),
            float(mean_product_price),
            int(total_searches),
            int(total_add_to_cart),
            int(hour_of_day)]).reshape(1, -1))

    nonconvert, convert = predictions.tolist()[0]
    return app.response_class(
        response=json.dumps({
            "convert": convert,
            "nonconvert": nonconvert,
            "input": {
                "time_on_site": int(time_on_site),
                "total_view_products": int(total_view_products),
                "unique_add_to_cart": int(unique_add_to_cart),
                "mean_product_price": float(mean_product_price),
                "total_searches": int(total_searches),
                "total_add_to_cart": int(total_add_to_cart),
                "hour_of_day": int(hour_of_day)
            }
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
