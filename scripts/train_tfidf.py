"""CPU baseline. Uses only training and validation targets; test kept for Final."""
import json
import time
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from threadpoolctl import threadpool_limits

from common import ROOT, load_data, targets, environment, write_json
from evaluate import save_evaluation


def main():
    config = json.loads((ROOT/'configs/midterm.json').read_text())
    frames, _ = load_data()
    params = config['tfidf'].copy()
    params['ngram_range'] = tuple(params['ngram_range'])
    vectorizer = TfidfVectorizer(**params)
    started = time.perf_counter()
    x = vectorizer.fit_transform(frames['train'].normalized)
    model = OneVsRestClassifier(LogisticRegression(**config['logistic_regression']), n_jobs=1)
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, targets(frames['train']))
    seconds = time.perf_counter()-started
    val = model.predict_proba(vectorizer.transform(frames['val'].normalized))
    parameters = sum(est.coef_.size + est.intercept_.size for est in model.estimators_)
    metadata = {'parameter_count':parameters, 'train_seconds':seconds}
    save_evaluation('tfidf_lr',frames,{'val':val},metadata)
    write_json(ROOT/'reports/tfidf_lr_run.json',dict(config=config,
        environment=environment(),features=len(vectorizer.vocabulary_),
        parameter_definition='27 x (TF-IDF feature count + intercept); IDF values excluded',
        time_definition='Training TF-IDF fit + OvR fit; excludes data loading and evaluation',
        **metadata))


if __name__ == '__main__':
    main()
