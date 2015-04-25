__author__ = 'davidnola'

# notes:
# position is useless - not in test data
# prop_id is too hard to vectorize - too many levels, plus not much point because its an id
# Over 300,000 data points

import pandas as pd, sklearn.linear_model, numpy as np,sklearn.preprocessing, sklearn.feature_extraction, sklearn.ensemble, sklearn.decomposition, sklearn.cluster, sklearn.svm, sklearn.tree, sklearn.neighbors, sklearn.feature_selection, sklearn.pipeline
pd.options.mode.chained_assignment = None
from sklearn import metrics
import sklearn, sklearn.base, sklearn.naive_bayes, sklearn.neural_network
import sklearn as sk
import math
import itertools, random, copy
from joblib import Parallel, delayed
import multiprocessing

class Parameters:
    imputed_features = ['comp%s_rate_percent_diff'%str(i) for i in range(1,9)] + ['visitor_hist_starrating','visitor_hist_adr_usd', 'orig_destination_distance', 'srch_query_affinity_score', 'prop_log_historical_price', 'prop_review_score', 'prop_starrating', 'price_usd', 'srch_room_count']
    less_important = ['visitor_hist_starrating','visitor_hist_adr_usd', 'promotion_flag', 'prop_brand_bool']
    one_hot_encoded_features = ['site_id','visitor_location_country_id']
    best_features_numerical = ['prop_starrating', 'prop_review_score', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd' , 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_saturday_night_bool', 'orig_destination_distance', 'random_bool']
    extra_features_numerical = ['comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate','comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff']
    ratio_features = ['prop_review_score', 'prop_starrating', 'prop_location_score1', 'prop_location_score2']
    other_ratios = ['comp%s_rate_percent_diff'%str(i) for i in range(1,9)] + ['orig_destination_distance', 'srch_booking_window' ]
    testing_selection = []
    everything_orig = set(['site_id','visitor_location_country_id','prop_starrating', 'prop_review_score', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_saturday_night_bool', 'orig_destination_distance', 'random_bool', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'promotion_flag', 'prop_brand_bool', 'comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff', 'comp4_rate_percent_diff', 'comp5_rate_percent_diff', 'comp6_rate_percent_diff', 'comp7_rate_percent_diff', 'comp8_rate_percent_diff', 'ratio_other_comp1_rate_percent_diff_vs_price_usd', 'ratio_other_comp2_rate_percent_diff_vs_price_usd', 'ratio_other_comp3_rate_percent_diff_vs_price_usd', 'ratio_other_comp4_rate_percent_diff_vs_price_usd', 'ratio_other_comp5_rate_percent_diff_vs_price_usd', 'ratio_other_comp6_rate_percent_diff_vs_price_usd', 'ratio_other_comp7_rate_percent_diff_vs_price_usd', 'ratio_other_comp8_rate_percent_diff_vs_price_usd', 'ratio_usd_prop_review_score_vs_price_usd', 'ratio_usd_prop_starrating_vs_price_usd', 'ratio_usd_prop_location_score1_vs_price_usd', 'ratio_usd_prop_location_score2_vs_price_usd', 'ratio_other_comp1_rate_percent_diff_vs_price_usd', 'ratio_other_comp2_rate_percent_diff_vs_price_usd', 'ratio_other_comp3_rate_percent_diff_vs_price_usd', 'ratio_other_comp4_rate_percent_diff_vs_price_usd', 'ratio_other_comp5_rate_percent_diff_vs_price_usd', 'ratio_other_comp6_rate_percent_diff_vs_price_usd', 'ratio_other_comp7_rate_percent_diff_vs_price_usd', 'ratio_other_comp8_rate_percent_diff_vs_price_usd', 'ratio_other_orig_destination_distance_vs_price_usd', 'ratio_other_srch_booking_window_vs_price_usd', 'comp_diff'])

def one_hot_encode_data(train, test, features):

    vectorizer = sklearn.feature_extraction.DictVectorizer()


    one_hot_train = train[Parameters.one_hot_encoded_features]
    one_hot_test = test[Parameters.one_hot_encoded_features]
    combined = pd.concat([one_hot_train,one_hot_test])

    for s in features:
        print("one hot encoding:", s)

        combined_dict_series = combined[[s]].astype(str).T.to_dict()
        combined_ind = list(combined_dict_series)
        combined_tmp = list(combined_dict_series.values())

        train_dict_series = one_hot_train[[s]].astype(str).T.to_dict()
        train_ind = list(train_dict_series)
        train_tmp = list(train_dict_series.values())

        test_dict_series = one_hot_test[[s]].astype(str).T.to_dict()
        test_ind = list(test_dict_series)
        test_tmp = list(test_dict_series.values())

        print("fitting vectorizer")
        vectorizer.fit(combined_tmp)


        train_data = vectorizer.transform(train_tmp).toarray()
        test_data = vectorizer.transform(test_tmp).toarray()

        print(s, "number of categories:",len(train_data[0]),len(test_data[0]))


        transformed_train = pd.DataFrame(index = train_ind, data=train_data, columns = ['one_hot_%s_%s'%(s,x) for x in range(len(train_data[0]))])
        transformed_test = pd.DataFrame(index = test_ind, data=test_data, columns = ['one_hot_%s_%s'%(s,x) for x in range(len(test_data[0]))])

        one_hot_train = one_hot_train.join(transformed_train)
        one_hot_test = one_hot_test.join(transformed_test)


        print(s,"complete")

    return one_hot_train, one_hot_test

def encode_and_pickle_data():
    train_dataset = pd.io.parsers.read_csv('train.csv')
    train_dataset['srch-prop_id'] = train_dataset['srch_id'].astype(str) + "-" + train_dataset['prop_id'].astype(str)
    train_dataset = train_dataset.set_index('srch-prop_id')

    test_dataset = pd.io.parsers.read_csv('test.csv')
    test_dataset['srch-prop_id'] = test_dataset['srch_id'].astype(str) + "-" + test_dataset['prop_id'].astype(str)
    test_dataset = test_dataset.set_index('srch-prop_id')


    (one_hot_train, one_hot_test) = one_hot_encode_data(train_dataset,test_dataset, Parameters.one_hot_encoded_features)


    train_dataset = train_dataset.join(one_hot_train, lsuffix ="oh_")
    test_dataset = test_dataset.join(one_hot_test, lsuffix ="oh_")

    train_dataset.to_pickle('train_dataset.pkl')
    test_dataset.to_pickle('test_dataset.pkl')
    #one_hot_train.to_pickle('one_hot_train_features.pkl')
    #one_hot_test.to_pickle('one_hot_test_features.pkl')

def load_data():
    train_dataset = pd.io.pickle.read_pickle('train_dataset.pkl')
    test_dataset = pd.io.pickle.read_pickle('test_dataset.pkl')
    train_dataset = train_dataset.reindex(np.random.permutation(train_dataset.index))
    test_dataset = test_dataset.reindex(np.random.permutation(test_dataset.index))

    return (train_dataset, test_dataset)

def impute_features(features, NA_Val = 0):
    (train, test) = load_data()
    combined = pd.concat([train, test])

    estimators = list(set(Parameters.best_features_numerical) - set(Parameters.imputed_features))

    for feat in features:
        print("Imputing:", feat)

        # model = sklearn.ensemble.GradientBoostingRegressor()
        # model.fit(combined[combined[feat] != NA_Val][estimators],combined[combined[feat] != NA_Val][feat])

        new_mean = (combined[combined[feat] != NA_Val][feat].mean())

        train[feat+'_is_null'] = 0
        test[feat+'_is_null'] = 0

        train[train[feat] == NA_Val][feat+'_is_null'] = 1
        test[test[feat] == NA_Val][feat+'_is_null'] = 1

        train[train[feat] == NA_Val][feat] = new_mean
        test[test[feat] == NA_Val][feat] = new_mean

        # train[train[feat] == NA_Val][feat] = model.predict(train[train[feat] == NA_Val][estimators])
        # test[test[feat] == NA_Val][feat] = model.predict(test[test[feat] == NA_Val][estimators])

    print("writing imputed features to pickles")
    train.to_pickle('train_dataset.pkl')
    test.to_pickle('test_dataset.pkl')

def preprocess_data():
    encode_and_pickle_data()
    impute_features(Parameters.imputed_features)
    derive_features()
    print("Done Preprocessing Data. Data is pickled.")


def slice_data(dataset):
    dataset = dataset.reindex(np.random.permutation(dataset.index))

    cv = dataset[-50000:]
    train = dataset[:-50000]

    zs = train[train['booking_bool'] == 0]
    nzs = train[train['booking_bool'] == 1]

    l = min(len(zs), len(nzs))

    combined = [ zs[:int(l)], nzs[:l] ]
    train = pd.concat(combined)

    train = train.reindex(np.random.permutation(train.index))

    return (train, cv)

def generate_features(dataset, feats):
    features = pd.DataFrame(index=dataset.index)
    selected = feats
    features = features.join(dataset[selected])

    return features

def generate_features_and_classes(dataset, feats):
    return (generate_features(dataset, feats), dataset['booking_bool'])

def generate_submission(test, model, features):
    features += filter(lambda x: 'extra' in x, test.columns)
    print(features)

    test_feats = generate_features(test, features)
    preds = model.predict(test_feats)
    preds2 = list(map(lambda x: x[1], model.predict_proba(test_feats)))
    print(preds[0:10],preds2[0:10])

    to_write = pd.DataFrame(index=test_feats.index)
    to_write['booking_bool'] = preds2

    to_write.to_csv('expedia_submit3.csv')

def derive_features():
    (train, test) = load_data()
    for dataset in [train,test]:
        dataset['comp_diff'] =  dataset[['comp%s_rate_percent_diff'%str(i) for i in range(1,9)]].sum(axis=1)

        done = []
        for f1 in Parameters.ratio_features:
            name = "ratio_usd_"+f1+"_vs_price_usd"
            dataset[name] = (dataset[f1] * 1.0) / (dataset['price_usd'] * 1.0)
            dataset[name] = dataset[name].replace([np.inf, -np.inf], np.nan)

            new_mean = np.nanmean(dataset[name])
            dataset[name] = dataset[name].replace(np.nan, new_mean)
    
        for f1 in Parameters.other_ratios:
            name = "ratio_other_"+f1+"_vs_price_usd"
            dataset[name] = (dataset[f1] * 1.0) / (dataset['price_usd'] * 1.0)
            dataset[name] = dataset[name].replace([np.inf, -np.inf], np.nan)

            new_mean = np.nanmean(dataset[name])
            dataset[name] = dataset[name].replace(np.nan, new_mean)

    train.to_pickle('train_dataset.pkl')
    test.to_pickle('test_dataset.pkl')

###################################################################################################
# preprocess_data()
##################
(train_l, test_l) = load_data()

class Printer(sklearn.base.TransformerMixin):
    def transform(self, X, **transform_params):
        #print(len(X[0]))
        return X
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self,deep=False):
        return {}

def trial():
    global train_l, test_l

    train = copy.deepcopy(train_l)
    test = copy.deepcopy(test_l)

    (train, cv) = slice_data(train)

    train_stack = train[-5:]
    train = train[:-5]

    one_hots = list(filter(lambda x: 'visitor_location_country_id' in x, train.columns))
    rate_diffs= list(filter(lambda x: 'rate_percent_diff' in x, train.columns))
    ratio_usd_feats= list(filter(lambda x: 'ratio_usd_' in x, train.columns))
    ratio_other_feats= list(filter(lambda x: 'ratio_other_' in x, train.columns))
    null_feats= list(filter(lambda x: 'is_null' in x, train.columns))

    selected = list(set(Parameters.testing_selection)) # + one_hots

    (tr_feats, tr_class) = generate_features_and_classes(train, selected)
    (cv_feats, cv_class) = generate_features_and_classes(cv, selected)

    num = len(selected)
    print(num)

    p_model = sklearn.pipeline.Pipeline([
    # ('union',  sklearn.pipeline.FeatureUnion([
    #                                         ('printer', Printer() ),
    #
    #                                         ('inner', sklearn.pipeline.Pipeline([
    #                                                             ('normalizer', sklearn.preprocessing.MinMaxScaler()),
    #                                                             ('rbm', sklearn.neural_network.BernoulliRBM(n_components=3)) ]))
    #
    #                                         ]) ),
    ('feature_selection', sklearn.feature_selection.RFE(sklearn.linear_model.LogisticRegression(), step=1, n_features_to_select=31)),

    ('classification', sklearn.ensemble.GradientBoostingClassifier(n_estimators=200, max_depth=3, min_samples_split = 3, min_samples_leaf=2, max_features=16))

    ])

    # model = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=8, max_features=10)

    # dicts = []
    # for select, cls in itertools.product(list(range(25,35))+[40], list(range(5,20))):
    #     if select>cls:
    #         dicts.append({'feature_selection__n_features_to_select': [select], 'classification__max_features': [cls]})
    # model = sklearn.grid_search.GridSearchCV(p_model, dicts, scoring = 'roc_auc', n_jobs=8, cv=5, verbose=3)

    model = p_model
    # To perform grid search instead, reenable following line:
    #model = sklearn.grid_search.GridSearchCV(p_model, {'classification__n_estimators':[175,200,225],'classification__max_features':[15,16,17,18,19,20]}, scoring = 'roc_auc', n_jobs=8, cv=5, verbose=3)

    model.fit(np.array(tr_feats), list(tr_class))

    try:
        print(model.best_params_)
    except:
        pass

    pred = model.predict(np.array(cv_feats))

    auc = metrics.roc_auc_score(cv_class,pred)

    print("AUC:", auc)

    generate_submission(test, model, selected)
    return auc



Parameters.testing_selection = list(Parameters.everything_orig)
trial()
