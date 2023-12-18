from .utils import train, evaluate, two_one_norm
import numpy as np
from copy import deepcopy
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_random_state

class MondrianForestRegressor(RegressorMixin):

    def __init__(self, n_estimators = 10, lifetime = 1, delta = 0, random_state = 42) -> None:
        self.n_estimators = n_estimators
        self.lifetime = lifetime
        self.delta = delta
        self.history = []
        self.w_trees = []
        self.X = None
        self.y = None
        self.rng = check_random_state(random_state)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.history, self.w_trees = train(X, y, self.n_estimators, self.lifetime, self.delta, self.rng)
        return self
    
    def predict(self, X):
        return evaluate(self.y, X, self.n_estimators, self.history, self.w_trees)
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators, 
                "lifetime": self.lifetime, 
                "delta": self.delta
                }


class MondrianForestTransformer(RegressorMixin):

    def __init__(self, mf: MondrianForestRegressor = None,
                  n_estimators = 10, lifetime = 1, delta = 0, iteration = 1, step_size = 0.1) -> None:
        if mf is None:
            self.mf = MondrianForestRegressor(n_estimators, lifetime, delta)
        else:
            self.mf = mf
        self.step_size = step_size
        self.iteration = iteration
        self.X = None
        self.y = None
        self.H = None

    def estimate_H_finite_diff(self):
        importance = []

        for dim in range(self.X.shape[1]):
            x_eval_pos = deepcopy(self.X)
            x_eval_neg = deepcopy(self.X)
            x_eval_pos[:,dim] = x_eval_pos[:,dim] + self.step_size / 2.0
            x_eval_neg[:,dim] = x_eval_neg[:,dim] - self.step_size / 2.0
            y_eval_pos = self.mf.predict(x_eval_pos)
            y_eval_neg = self.mf.predict(x_eval_neg)
            y_diff = y_eval_pos - y_eval_neg
            importance_temp = y_diff/self.step_size
            importance.append(importance_temp)

        importance = np.vstack(importance)
        H = np.matmul(importance, np.transpose(importance))/self.X.shape[0]
        return H
    
    def fit(self, X, y):
        self.X = np.array(X)
        self.y = y
        self.mf.fit(X, y)
        if self.iteration > 0:
            self.H = self.estimate_H_finite_diff()
            for _ in range(self.iteration - 1):
                self.reiterate()
            self.mf.fit(self.transform(deepcopy(self.X)), self.y)
        return self

    def reiterate(self):
        X = self.transform(deepcopy(self.X))
        self.mf.fit(X, self.y)
        self.H = np.matmul(self.H, self.estimate_H_finite_diff())
    
    def transform(self, X):
        return np.matmul(X, self.H / two_one_norm(self.H))
    
    def predict(self, X):
        if self.iteration > 0:
            return self.mf.predict(self.transform(X))
        else:
            return self.mf.predict(X)
    
    def set_params(self, **params):
        for key, value in params.items():
            if key in ["n_estimators", "lifetime", "delta"]:
                setattr(self.mf, key, value)
            else:
                setattr(self, key, value)
        return self

    def get_params(self, deep=True):
        return {"n_estimators": self.mf.n_estimators, 
                "lifetime": self.mf.lifetime, 
                "delta": self.mf.delta,
                "step_size": self.step_size
                }