
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search
import scipy.sparse
from crowdflower import extract_features
from sklearn.feature_extraction.text import CountVectorizer

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
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
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


if __name__ == '__main__':


    train = pd.read_csv('train.csv').fillna("")
    test = pd.read_csv('test.csv').fillna("")

    train_full = train.copy(deep=True)
    test_full = test.copy(deep=True)

    extract_features(train_full)
    extract_features(test_full)


    idx = test.id.values.astype(int)
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)


    y = train.median_relevance.values
    train = train.drop(['median_relevance', 'relevance_variance'], axis=1)


    traindata = list(train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    testdata = list(test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))


    tfv = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 5), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')


    tfv.fit(traindata)
    X =  tfv.transform(traindata)
    X_test = tfv.transform(testdata)


    svd = TruncatedSVD(n_components=400)


    scl = StandardScaler()

    print(train_full.head())



    cvec_query = CountVectorizer(max_features=20)
    cvec_title = CountVectorizer(max_features=20)
    cvec_desc = CountVectorizer(max_features=20)

    bow_q = cvec_query.fit_transform(train_full['query'])
    bow_t = cvec_title.fit_transform(train_full['product_title'])
    bow_d = cvec_desc.fit_transform(train_full['product_description'])

    bow_t_q = cvec_query.transform(test_full['query'])
    bow_t_t = cvec_title.transform(test_full['product_title'])
    bow_t_d = cvec_desc.transform(test_full['product_description'])


    X = svd.fit_transform(X)
    in_desc = train_full['query_tokens_in_description'].tolist()
    in_title = train_full['query_tokens_in_title'].tolist()


    X_more = np.array([list(X[i])+[in_desc[i],in_title[i]] for i in range(len(in_desc))])
    X_more = scipy.sparse.hstack((X_more,bow_q,bow_t,bow_d))
    print(X_more.shape)

    X = scl.fit_transform(X_more.todense())




    X_test = svd.transform(X_test)
    in_desc = test_full['query_tokens_in_description'].tolist()
    in_title = test_full['query_tokens_in_title'].tolist()
    X_test_more = np.array([list(X_test[i])+[in_desc[i],in_title[i]] for i in range(len(in_desc))])
    X_test_more = scipy.sparse.hstack((X_test_more,bow_t_q,bow_t_t,bow_t_d))
    print(X_test_more.shape)

    X_test = scl.transform(X_test_more.todense())




    svm_model = SVC()


    clf = pipeline.Pipeline([('svm', svm_model)])


    param_grid = {'svm__C': [10, 12, 15]}


    kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)


    model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=kappa_scorer,
                                     verbose=10, n_jobs=-1, iid=True, refit=True, cv=3)

    model.fit(X, y)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
    	print("\t%s: %r" % (param_name, best_parameters[param_name]))


    best_model = model.best_estimator_


    best_model.fit(X,y)
    preds = best_model.predict(X_test)


    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv("final.csv", index=False)
