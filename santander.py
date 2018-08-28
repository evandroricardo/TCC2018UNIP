import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data_path = "C:/Users/Fernando/Desktop/data/"


def __read_csv(path):
    return pandas.read_csv(path)


def __get_test():
    return __read_csv(data_path + "minitest.csv")


def __get_train():
    return __read_csv(data_path + "train.csv")


def filter_columns(df, l):
    return df[list(filter(l, df.columns))]


def gen_mini_test():
    data = __get_test().head(5000)
    data.to_csv(data_path + "minitest.csv", index=False)


def get_test():
    test = __get_test()
    return filter_columns(test, lambda c: c not in ["ID"]), test["ID"].values


def get_train():
    train = __get_train()
    return filter_columns(train, lambda c: c not in ['target', "ID"]), train["target"].values, train["ID"].values


def hash_ids(ids):
    return list(hash(_id) for _id in ids)


def run():
    test_X, test_ids = get_test()
    train_X, train_y, train_ids = get_train()
    
    lr = LinearRegression()
    lr.fit(train_X, train_y)
    pred_y = lr.predict(test_X)

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.scatter(hash_ids(train_ids), train_y, s=1, c='g', marker="s", label='Real')

    ax2 = fig.add_subplot(222)
    ax2.scatter(hash_ids(test_ids), pred_y, s=1, c='b', marker="s", label='Preditado Todo')

    plt.legend()
    plt.show()

run()
