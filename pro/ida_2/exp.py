import numpy as np
import scipy.io as sio
from scipy import interp
from sklearn.metrics import roc_curve, auc, accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import DataConversionWarning
import datetime
import sys
import warnings


class solam_classfier():
    def __init__(self, eta=0.1, times=5):
        self.wt = None
        self.eta = eta
        self.times = times

    def fit(self, X, y):
        np.random.seed(0)
        dim = X.shape[1]
        wt = np.random.randn((dim))
        wt_mean = wt
        T = len(y)
        n_pos = n_neg = 0
        at = bt = alphat = 0

        index = np.random.permutation(T)
        eta = self.eta
        for t in range(T * self.times):
            xt = X[index[t % T]]
            yt = y[index[t % T]]
            if yt == 1:
                n_pos = n_pos + 1
            p = n_pos / (t + 1)
            if yt == 1:
                wt = wt - eta * 2 * (1 - p) * (wt @ xt - at - 1 - alphat) * xt
                at = at - eta * (-2) * (1 - p) * (wt @ xt - at)
                bt = bt - eta * 0
                alphat = alphat + eta * (-2 * p * (1 - p) * alphat - 2 * (1 - p) * wt @ xt)
            else:
                wt = wt - eta * 2 * p * (wt @ xt - bt + 1 + alphat) * xt
                at = at - eta * 0
                bt = bt - eta * (-2) * p * (wt @ xt - bt)
                alphat = alphat + eta * 2 * p * (wt @ xt - (1 - p) * alphat)

            wt, at, bt, alphat = self.prox(wt, at, bt, alphat)
            wt_mean = wt_mean + (wt - wt_mean) / (1 + t)

        self.wt = wt_mean / np.linalg.norm(wt_mean)
        return self

    def prox(self, wt, at, bt, alphat):
        if np.linalg.norm(wt) == 0:
            pass
        else:
            norm_wt = np.linalg.norm(wt)
            wt = wt / norm_wt
            at = at / norm_wt
            bt = bt / norm_wt
            alphat = alphat / norm_wt
        return wt, at, bt, alphat

    def predict(self, X):
        return X @ self.wt


class Preprocess( ):
    def __init__(self):
        self.scaler = MinMaxScaler((-1, 1))

    def get_data(self, train, path):
        if train == True:
            # data = sio.loadmat("train.mat")
            data = sio.loadmat(path)
            X = data['x_tr']
            X.astype(float)
            y = data['y_tr']
            y = y.flatten()
            y.astype(float)
            self.scaler.fit(X)
            X = self.scaler.transform(X)
        else:
            # data = sio.loadmat("test.mat")
            data = sio.loadmat(path)
            X = data['x_tr']
            y = data['y_tr']
            y = y.flatten()
            X = self.scaler.transform(X)
        return X, y


if __name__ == '__main__':
    # begin = datetime.datetime.now()
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)
    train_path = sys.argv[1]
    clc_preprocessor = Preprocess()
    X_train, y_train = clc_preprocessor.get_data(True, train_path)
    classifier = solam_classfier(0.01844, 10)
    classifier = classifier.fit(X_train, y_train)
    if len(sys.argv) > 2:
        val_path = sys.argv[2]
        X_val, y_val = clc_preprocessor.get_data(False, val_path)
        probas_ = classifier.predict(X_val)
        fpr, tpr, thresholds = roc_curve(y_val, probas_)
        roc_auc = auc(fpr, tpr)
        print("AUC Score: %0.2f" % roc_auc)
    else:
        probas_ = classifier.predict(X_train)
        fpr, tpr, thresholds = roc_curve(y_train, probas_)
        roc_auc = auc(fpr, tpr)
        print("AUC Score: %0.2f" % roc_auc)

    # end = datetime.datetime.now()
    # print("time elapsed " + str(end - begin))
