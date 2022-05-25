import numpy as np

from sklearn import svm, linear_model
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier

import random
import pickle
import torch

def set_seed(seed):
    # set seed
    np.random.seed(seed=seed)
    random.seed(seed)

set_seed(42)

def accuracy(pred, ref):
    """Compute simple accuracy."""
    assert len(pred) == len(ref)
    correct = sum((pred == ref).astype('int'))
    return correct/len(ref)


def make_train_test_splits(data, feature_set):
    """Randomly split data into train (90%) and test (10%)."""
    n = len(data)
    test_ids = np.random.choice(range(n), n//10, replace=False)
    test = data.loc[test_ids]

    train_ids = np.array(list(set(range(n)).difference(set(test_ids))))
    train = data.loc[train_ids]
    
    train_data = train[feature_set]
    train_label = train['label']
    test_data = test[feature_set]
    test_label = test['label']
        
    return train_data, train_label, test_data, test_label


def train_rbf_split(data, feats=['x', 'y', 'c1', 'c2'], num_seeds=1000):
    """
    Splits data into train and test, and then trains an SVM model with an
    RBF kernel on the train, and with the trained model, computes accuracy 
    on the test set. Repeats the above `num_seeds` times, and returns the 
    mean / std of accuracies on the test set (to reduce variance).
    """
    acc = []
    for _ in range(num_seeds):
        train_data, train_label, test_data, test_label = make_train_test_splits(data, feats)
        
        rbf = svm.SVC(gamma='auto', 
                      tol=1e-5, 
                      # C=1.0,
                      # gamma=0.10,
                      random_state=np.random.randint(0, 100))
        rbf.fit(train_data, train_label)
        
        rbf_predicted = rbf.predict(test_data)
        acc.append(accuracy(rbf_predicted, test_label.values))
    return f"RBF    acc: {np.mean(acc):.4f} (+/-{np.std(acc):.2f})"


def train_linear(data, feats=['x', 'y', 'c1', 'c2'], num_seeds=100, save_name=None):
    """
    Splits data into train and test, and then trains a linear model on the 
    train, and with the trained model, computes accuracy 
    on the test set. Repeats the above `num_seeds` times, and returns the 
    mean / std of accuracies on the test set (to reduce variance).
    """
    acc = []
    for seed in range(num_seeds):
        set_seed(seed)
        train_data, train_label, test_data, test_label = make_train_test_splits(data, feats)
        set_seed(seed)
        lin = linear_model.SGDClassifier(max_iter=1000, tol=1e-5)
        lin.fit(train_data, train_label)
        if save_name:
            pickle.dump(lin, open(f"{save_name}/{seed}.pkl", "wb"))
        
        lin_predicted = lin.predict(test_data)
        acc.append(accuracy(lin_predicted, test_label.values))
    return f"LINEAR acc: {np.mean(acc):.4f} (+/-{np.std(acc):.2f})"

def train_nn(data, feats=['x', 'y', 'c1'], 
             num_seeds=10, save_name=None, epochs=100,
             hidden_layer_sizes=[16], patience=10, tol=0.001):
    """
    Splits data into train and test, and then trains a linear model on the 
    train, and with the trained model, computes accuracy 
    on the test set. Repeats the above `num_seeds` times, and returns the 
    mean / std of accuracies on the test set (to reduce variance).
    """
    criterion = torch.nn.CrossEntropyLoss()
    train_acc, test_acc = [], []
    for seed in range(num_seeds):
        set_seed(seed)
        train_data, train_label, test_data, test_label = make_train_test_splits(data, feats)
        model = MLPClassifier(solver='adam', max_iter=epochs, verbose=True,
                              n_iter_no_change=patience, tol=tol, alpha=0.00,
                              hidden_layer_sizes=hidden_layer_sizes, random_state=seed)
        model.fit(train_data, train_label)
        if save_name:
            pickle.dump(model, open(f"{save_name}/{seed}.pkl", "wb"))
        
        predicted_train, predicted_test = model.predict(train_data), model.predict(test_data)
        train_acc.append(accuracy(predicted_train, train_label.values))
        test_acc.append(accuracy(predicted_test, test_label.values))
    return f"NN train acc: {np.mean(train_acc):.4f} (+/-{np.std(train_acc):.2f})\nNN in-distribution test acc: {np.mean(test_acc):.4f} (+/-{np.std(test_acc):.2f})"

def evaluate_model(save_prefix, data, feats=['x', 'y', 'c1'], num_seeds=10, train_eval=False):
    acc = []
    for seed in range(num_seeds):
        model = pickle.load(open(f'{save_prefix}/{seed}.pkl', 'rb'))
        set_seed(seed)
        train_data, train_label, test_data, test_label = make_train_test_splits(data, feats)
        if train_eval:
            model_predicted = model.predict(train_data)
            labels = train_label
        else:
            model_predicted = model.predict(test_data)
            labels = test_label
        acc.append(accuracy(model_predicted, labels.values))
    return f"acc: {np.mean(acc):.4f} (+/-{np.std(acc):.2f})"