from Mondrian_forest import *
from sklearn.metrics import mean_squared_error
import pickle

dim_in = 5
active = 3
n = 5000
n_test = 1000
M = 100  # number of Mondrian trees to use
delta = 0  # ridge regression delta

x_train, y_train, x_test, y_test, rotation = pickle.load(open("data.pickle", "rb"))

x_train = x_train[:n]
y_train = y_train[:n]
x_test = x_test[:n_test]
y_test = y_test[:n_test]


lifetime = 5
y_hat_test, history, X_bd_all, w_trees = one_run(
    x_train, y_train, x_test, M, lifetime, delta
)
mse = mean_squared_error(y_test, y_hat_test)
print(f"mse before:{mse}")
model = [y_hat_test, history, X_bd_all, w_trees]

H = estimate_H_ind(x_train, y_train, X_bd_all, M, x_test, history, w_trees, y_hat_test)

H_ = estimate_H_ind_(
    x_train, y_train, X_bd_all, M, x_test, history, w_trees, y_hat_test
)


eval_data = []
x_temp = deepcopy(x_test)
for step_size in [0.01, 0.005]:
    x_temp[:, 0] = x_temp[:, 0] + step_size
    y_hat_eval = evaluate(y_train, x_temp, M, history, w_trees)
    eval_data.append([x_temp, y_hat_eval])


pickle.dump([model, [H, H_], eval_data], open("result.pickle", "wb"))
