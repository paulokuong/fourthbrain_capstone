import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import os
import time

from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from tensorflow.keras.layers import GRU, Embedding, SimpleRNN, Activation
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Helper functions for this:
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import tensorflow_addons as tfa
import shap

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from tpot.export_utils import set_param_recursive
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB


import warnings
warnings.filterwarnings('ignore')

class Evaluation(object):

    @staticmethod
    def plot_history(history):
        # This function will plot the model fit process
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['f1_score'])
        # plt.plot(history.history['var_f1_score'])
        plt.title('model f1_score')
        plt.ylabel('f1')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    @staticmethod
    def evaluate_nn_model(X_test, y_test, training_model):
        # This function will evaluate the fit model on test data
        g_preds = training_model.predict_classes(X_test)
        gaccuracy = accuracy_score(y_test[:, 1], g_preds)
        print('Accuracy: %f' % gaccuracy)
        # precision tp / (tp + fp)
        gprecision = precision_score(y_test[:, 1], g_preds)
        print('Precision: %f' % gprecision)
        # recall: tp / (tp + fn)
        grecall = recall_score(y_test[:, 1], g_preds)
        print('Recall: %f' % grecall)
        # f1: 2 tp / (2 tp + fp + fn)
        gf1 = f1_score(y_test[:, 1], g_preds)
        print('F1 score: %f' % gf1)
        
    @staticmethod
    def evaluate_model(x_test, y_test, model):
        predictions = model.predict(x_test)
        return pd.DataFrame(classification_report(y_test, predictions, output_dict=True))


class FeatureSelection(object):
    @staticmethod
    def by_coorelation(x, threshold=0.8, debug=False):
        """Feature selection by eliminating highly correlated features.

        Args:
            x (pandas dataframe): features.
            threshold (float[optional]): score above which feature is highly correlated.
            debug (boolean[optional]): True to show debug messages.

        Return:
            pandas dataframe: dataframe with selected features.
        """
        cor = x.corr()
        keep_columns = np.full((cor.shape[0],), True, dtype=bool)
        for i in range(cor.shape[0]):
            for j in range(i + 1, cor.shape[0]):
                if np.abs(cor.iloc[i, j]) >= threshold:
                    if keep_columns[j]:
                        keep_columns[j] = False
                        if debug:
                            print((
                                f'Feature "{x.columns[j]}" is highly '
                                f'related to "{x.columns[i]}". '
                                f'Remove "{x.columns[j]}"'))
        if debug:
            print(len(np.full((cor.shape[0],), True, dtype=bool)))
            plt.figure(figsize=(12,10))
            cor = x.corr()
            sns.heatmap(cor)
            plt.show()
        selected_columns = x.columns[keep_columns]
        return x[selected_columns]

    @staticmethod
    def by_permutation_importance(
            x, y, threshold=0.01, n_repeats=10, random_state=42, n_jobs=2, debug=False):
        """Feature selection by permutation importance.

        Args:
            x (pandas dataframe): features.
            threshold (float[optional]): score above which the feature is
                considered as important.

        """
        feature_names = [f'feature {i}' for i in range(x.shape[1])]
        forest = RandomForestClassifier(random_state=random_state)
        forest.fit(x, y.values.ravel())
        start_time = time.time()
        result = permutation_importance(
            forest, x, y, n_repeats=n_repeats, random_state=random_state,
            n_jobs=n_jobs)
        elapsed_time = time.time() - start_time
        forest_importances = pd.Series(
            result.importances_mean, index=feature_names)
        importances = pd.DataFrame(forest_importances, columns=['score'])
        importances = importances.sort_values(by='score', ascending=False)
        importances.loc[:, 'feature'] = [
            x.columns[int(i.replace('feature ', ''))]
            for i in importances.index]

        if debug:
            print(importances[importances['score'] > threshold])
        return x[list(
            importances[importances['score'] > threshold]['feature'].values)]


class GroupBy(object):

    def __init__(self, raw_data_path):
        if not os.path.exists(raw_data_path):
            raise Exception(f'Path {raw_data_path} does not exist.')

        self.raw_data = pd.read_json(raw_data_path, lines=True)

    def preprocessing_for_bin_class(self):
        """Preprcess GroupBy data for binary classification training.

        Args:
            raw_data_path (str): local path to raw json data.
        Returns:
            dict: dictionary of training
        """

        df = self.raw_data
        transformed_df = df[
            ['customerId', 'customerVisitorId', 'customerSessionId',
             'sessionStartTime', 'sessionEndTime', 'customerSessionNumber']]
        transformed_df.loc[:, 'deviceCategory'] = df['trafficSource'].transform(
            lambda x: x.get('deviceCategory', ''))
        transformed_df.loc[:, 'browser'] = df['trafficSource'].transform(
            lambda x: x.get('browser', ''))
        transformed_df.loc[:, 'os'] = df['trafficSource'].transform(
            lambda x: x.get('os', ''))
        transformed_df.loc[:, 'userAgent'] = df['trafficSource'].transform(
            lambda x: x.get('userAgent', ''))
        transformed_df.loc[:, 'language'] = df['trafficSource'].transform(
            lambda x: x.get('language'))
        transformed_df.loc[:, 'source'] = df['trafficSource'].transform(
            lambda x: x.get('source'))
        transformed_df.loc[:, 'has_campaign'] = df['trafficSource'].transform(
            lambda x: 1 if x.get('campaign') is not None else 0)
        transformed_df.loc[:, 'sessionStartTime'] = df['sessionStartTime'].transform(
            lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f %Z'))
        transformed_df.loc[:, 'sessionEndTime'] = df['sessionEndTime'].transform(
            lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f %Z'))
        transformed_df.loc[:, 'sessionDuration'] = df[['sessionStartTime', 'sessionEndTime']].apply(
            lambda x: (datetime.strptime(x['sessionEndTime'], '%Y-%m-%d %H:%M:%S.%f %Z') -
                       datetime.strptime(x['sessionStartTime'], '%Y-%m-%d %H:%M:%S.%f %Z')).seconds, axis=1)
        transformed_df.loc[:, 'hourOfDay'] = df['sessionStartTime'].transform(
            lambda x: int(datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f %Z').strftime("%H")))
        total_df = []
        for i in range(len(df['totals'])):
            new_dict = {k: float(v) if 'total' in k or 'unique' in k else v
                        for k, v in df.iloc[i]['totals'].items()}
            total_df.append(new_dict)
        cleaned_df = pd.concat(
            [transformed_df, pd.DataFrame(total_df)], axis=1)
        cleaned_df = cleaned_df.fillna(0)

        all_browsers = sorted(pd.unique(cleaned_df['browser']))
        all_os = sorted(pd.unique(cleaned_df['os']))
        all_deviceCategory = sorted(pd.unique(cleaned_df['deviceCategory']))
        all_language = sorted(pd.unique(cleaned_df['language'].astype('str')))
        all_source = sorted(pd.unique(cleaned_df['source'].astype('str')))
        cleaned_df.loc[:, 'browser'] = cleaned_df['browser'].transform(
            lambda x: all_browsers.index(x))
        cleaned_df.loc[:, 'os'] = cleaned_df['os'].transform(
            lambda x: all_os.index(x))
        cleaned_df.loc[:, 'language'] = cleaned_df['language'].transform(
            lambda x: all_language.index(str(x)))
        cleaned_df.loc[:, 'source'] = cleaned_df['source'].transform(
            lambda x: all_source.index(str(x)))
        cleaned_df.loc[:, 'deviceCategory'] = cleaned_df['deviceCategory'].transform(
            lambda x: all_deviceCategory.index(x))
        cleaned_df.loc[:, 'bounce'] = cleaned_df['bounce'].transform(
            lambda x: int(x))
        cleaned_df.loc[:, 'events'] = cleaned_df['events'].transform(
            lambda x: int(x))
        cleaned_df.loc[:, 'timeOnSiteSeconds'] = cleaned_df['timeOnSite'].transform(
            lambda x: datetime.strptime(x, '%H:%M:%S').second + 60 * datetime.strptime(
                x, '%H:%M:%S').minute + 3600 * datetime.strptime(x, '%H:%M:%S').hour)
        cleaned_df.loc[:, 'newSession'] = cleaned_df['newSession'].transform(
            lambda x: 1 if x is True else 0)
        cleaned_df.loc[:, 'has_purchase'] = cleaned_df['totalOrders'].transform(
            lambda x: 1 if int(x) > 0 else 0)
        cleaned_df.loc[:, 'productPriceMean'] = df['hits'].apply(
            lambda x: np.nan_to_num(np.mean([np.mean([j.get('price') or 0
                                                      for j in i['product']]) for i in x])))
        cleaned_df = cleaned_df.drop(
            columns=[
                'sessionStartTime', 'sessionEndTime', 'userAgent', 'customerId',
                'customerVisitorId', 'totalOrders', 'timeOnSite',
                'queriesSearched', 'customerSessionId', 'totalOrderQty',
                'uniqueOrders', 'totalOrderRevenue'])
        # sorted(cleaned_df.columns)
        x = cleaned_df.loc[:, list(
            set(cleaned_df.columns) - set('has_purchase'))]
        del x['has_purchase']
        y = cleaned_df.loc[:, ['has_purchase']]
        return {"features": x, "label": y,"all_os":all_os, "all_source":all_source}

    def preprocessing_for_sequence_model(self, num_of_events=30, debug=False):
        df = self.raw_data
        oo = df[['hits']].apply(
            lambda x: [
                list(set([j.get('eventType').get('category')
                          for j in hit])) for hit in x])['hits']
        # Get event type map
        event_type_map = {y: index + 1 for index, y in enumerate(
            [i for i in pd.unique(oo.explode()) if type(i) == str])}
        # Get sequences and sort the events by hitSequence which shows the order
        # of each event. Apply event type map after sorting.
        sequence_df = df.copy(deep=True)
        sequence_df.loc[:, 'sequence'] = sequence_df[['hits']].apply(
            lambda x: [
                [event_type_map[j[0]]
                 for j in sorted(
                    [(j.get('eventType').get('category'),
                      j.get('hitSequence')) for j in hit])]
                for hit in x])['hits']
        if debug:
            length = sequence_df['sequence'].map(len).to_list()
            sns.distplot(length)
        # Find the target from the raw dataset.
        total_df = []
        for i in range(len(df['totals'])):
            new_dict = {k: float(v) if 'total' in k or 'unique' in k else v
                        for k, v in df.iloc[i]['totals'].items()}
            total_df.append(new_dict)
        sequence_df = pd.concat([sequence_df, pd.DataFrame(total_df)], axis=1)
        sequence_df = sequence_df.fillna(0)
        sequence_df.loc[:, 'has_purchase'] = sequence_df['totalOrders'].transform(
            lambda x: 1 if int(x) > 0 else 0)

        final_sequence_df = sequence_df[
            ['customerSessionId', 'sequence', 'has_purchase']
        ][sequence_df['sequence'].map(len) <= num_of_events]
        event_sequence = final_sequence_df['sequence'].to_list()
        # Pad 0 to make all sequences to have the same size.
        x = pad_sequences(event_sequence)
        # replace 4 to 0 since 4 is our target
        x = np.where(x==4, 0, x) 
        y = np.array(pd.get_dummies(
            final_sequence_df['has_purchase'], prefix='Purchase'))
        return {"features": x, "label": y,"event_type_map":event_type_map}

    @staticmethod
    def train_xgb_bin_class(
            features, label, test_size=0.33, random_state=42, debug=False):
        """Train binary classification using ensemble stacking classifier

        Args:
            preprocessed_data (pandas dataframe): preprocessed data.
            test_size (float): test data size in percentage.
            random_state (int): random state.
            debug (boolean): True for print out debug messages.
        """
        # Split dataset
        x_train, x_test, y_train, y_test = train_test_split(
            features.values, label, test_size=test_size,
            random_state=random_state)
        # Train model
        exported_pipeline = make_pipeline(
            StackingEstimator(estimator=RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.9500000000000001, min_samples_leaf=4, min_samples_split=4, n_estimators=100)),
            BernoulliNB(alpha=0.01, fit_prior=True)
        )
        # Fix random state for all the steps in exported pipeline
        set_param_recursive(exported_pipeline.steps, 'random_state', 42)

        exported_pipeline.fit(x_train, y_train)

        return exported_pipeline, x_test, y_test, x_train, y_train


    @staticmethod
    def train_lstm(
            features, label, op=30, neurons=40, epochs=150, batch_size=1000,
            validation_split=0.2, test_size=0.3, threshold=0.5):
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(features), label, test_size=test_size)
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
        tf.keras.backend.clear_session()
        model = Sequential()
        model.add(Bidirectional(
            LSTM(neurons, return_sequences=True), input_shape=(1, op)))
        model.add(Bidirectional(LSTM(2 * neurons)))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.0003),
            loss='binary_crossentropy',
            metrics=[tfa.metrics.F1Score(num_classes=2, threshold=threshold)])
        model.fit(
            x_train, y_train, epochs=epochs, batch_size=batch_size,
            validation_split=validation_split)
        return model, x_test, y_test, x_train, y_train
