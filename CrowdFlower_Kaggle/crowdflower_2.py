if __name__ == "__main__":
    import multiprocessing as mp; mp.set_start_method('forkserver')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search
import xgb_wrapper
from stemming.porter2 import stem
import scipy.sparse
# from xgb_wrapper import XGBoostClassifier
import pickle
import sklearn.ensemble

import os




# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


def porter_stem_data(data):
    print(data[:10])
    data = [" ".join([stem(kw) for kw in t.split(" ")]) for t in data]
    print(data[:10])
    return data

def filter_noun_data(data):
    pass

def td_idf_data(traindata, testdata, cvdata):
    tfv = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 5), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

    # Fit TFIDF
    tfv.fit(traindata)
    X =  tfv.transform(traindata)
    X_test = tfv.transform(testdata)
    X_cv = tfv.transform(cvdata)

    return X,X_test,X_cv


def filter_nouns(dat):
    import nltk
    final = []

    print(len(dat))
    for i, desc in enumerate(dat):
        tok = nltk.pos_tag(nltk.word_tokenize(desc))
        sent = []
        for j, t in enumerate(tok):
            if t[1][0]=='N' or j<5:
                sent.append(t[0])

        toadd = " ".join(sent).lower()
        if(i%50==0):
            print(i,toadd)
        final.append(toadd)
    return final


def preprocess_data():


    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    idx = test.id.values.astype(int)
    cv = train[train['id']%10==0]
    train = train[train['id']%10!=0]


    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)
    cv = cv.drop('id', axis=1)

    y = train.median_relevance.values
    y_cv = cv.median_relevance.values

    train = train.drop(['median_relevance', 'relevance_variance'], axis=1)
    cv = cv.drop(['median_relevance', 'relevance_variance'], axis=1)

    # do some lambda magic on text columns
    traindata = list(train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    testdata = list(test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    cvdata = list(cv.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))

    print("Filtering Nouns")
    traindata_full = filter_nouns(list(train.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'],x['product_description']),axis=1)))
    testdata_full = filter_nouns(list(test.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'],x['product_description']),axis=1)))
    cvdata_full = filter_nouns(list(cv.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'],x['product_description']),axis=1)))
    print("Done")

    traindata_stem = porter_stem_data(traindata_full)
    testdata_stem = porter_stem_data(testdata_full)
    cvdata_stem = porter_stem_data(cvdata_full)


    (X,X_test,X_cv) = td_idf_data(traindata, testdata,cvdata)
    # (X_noun,X_test_X_noun) = td_idf_data(traindata, testdata)
    (X_stem,X_test_stem,X_cv_stem) = td_idf_data(traindata_stem, testdata_stem,cvdata_stem)

    pickle.dump((X,X_test,y,idx,cv,X_cv, y_cv),open('dat.pkl','wb'))

    pickle.dump((X_stem,X_test_stem,X_cv_stem),open('dat_stem.pkl','wb'))

    # exit()


def get_predictions(X,y,X_test,X_cv):
    # Initialize SVD
    svd = TruncatedSVD()

    scl = StandardScaler()

    clf_model = SVC()

    clf = pipeline.Pipeline([('svd', svd),
    						 ('scl', scl),
                    	     ('clf', clf_model)])

    # Create a parameter grid to search for best parameters for everything in the pipeline
    param_grid = {'svd__n_components' : [400],
                  'clf__C': [12]}





    # Kappa Scorer
    kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)

    # Initialize Grid Search Model
    model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=kappa_scorer,
                                     verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

    # Fit Grid Search Model
    model.fit(X, y)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
    	print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # Get best model
    best_model = model.best_estimator_

    # Fit model with best parameters optimized for quadratic_weighted_kappa
    best_model.fit(X,y)
    preds = best_model.predict(X_test)
    preds_cv = best_model.predict(X_cv)
    return (preds, preds_cv)

if __name__ == '__main__':

    # preprocess_data()
    (X,X_test,y,idx,cv,X_cv, y_cv) = pickle.load(open('dat.pkl','rb'))

    (X_stem,X_test_stem,X_cv_stem) = pickle.load(open('dat_stem.pkl','rb'))

    print(X.shape)
    # X = scipy.sparse.hstack((X, X_stem))
    # X_test = scipy.sparse.hstack((X_test, X_test_stem))

    # X_test = np.array([np.concatenate(X_test[i]X_test_stem[i] for i in range(X.shape[0]) ])
    # (X,X_test) = (X_stem,X_test_stem)


    # (preds_stem, preds_cv_stem) = get_predictions(X_stem,y[:],X_test_stem,X_cv_stem)
    #
    # print("Final Stem:", quadratic_weighted_kappa(preds_cv_stem,y_cv))


    (preds, preds_cv) = get_predictions(X,y[:],X_test,X_cv)

    print("Final No Stem:", quadratic_weighted_kappa(preds_cv,y_cv))





    (bow,bow_cv,cv_qt,test_qt) = pickle.load(open('BoW2.pkl','rb'))


    ensemble_feats = [[preds_cv[i],bow_cv[i], cv_qt[i]] for i in range(len(preds_cv))]
    ensemble_feats_test = [[preds[i],bow[i],test_qt[i]] for i in range(len(preds))]


    # clf_model = sklearn.ensemble.RandomForestClassifier()
    # param_grid = {'clf__n_estimators': [50,100]}

    clf_model=sklearn.linear_model.LogisticRegression(class_weight='auto')
    param_grid = {'clf__C': [1]}

    clf = pipeline.Pipeline([('clf', clf_model)])

    # Create a parameter grid to search for best parameters for everything in the pipeline



    # Kappa Scorer
    kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)

    # Initialize Grid Search Model
    model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=kappa_scorer,
                                     verbose=10, n_jobs=-1, iid=True, refit=True, cv=5)

    # Fit Grid Search Model
    model.fit(ensemble_feats, y_cv)

    preds = model.best_estimator_.predict(ensemble_feats_test)

    print("Final Best score: %0.3f" % model.best_score_)


    # Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv("beating_the_benchmark_yet_again.csv", index=False)
