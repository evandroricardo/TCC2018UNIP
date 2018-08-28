import datetime
import pandas
import matplotlib.pyplot as plt
from numpy import nan, ndarray
from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

data_path = "C:/Users/Fernando/Desktop/data/"


def int_or_0(item):
    try:
        return int(item)
    except:
        pass
    return 0


def data_para_datetime(item):
    if item is nan:
        return 0
    return datetime.datetime.strptime(item, "%d/%m/%Y").timestamp()


def data_para_datetime_normalizada(data_inicio):
    t0 = data_para_datetime(data_inicio)
    def __gen__(item):
        t = data_para_datetime(item)
        return (t - t0).days
    return __gen__


def __read_csv(path):
    return pandas.read_csv(path, sep=";")


def concat_number(n1, n2, n3, n4, n5, n6):
    # n1 = n1 if not pandas.isnull(n1) else 0
    # n2 = n2 if not pandas.isnull(n2) else 0
    # n3 = n3 if not pandas.isnull(n3) else 0
    # n4 = n4 if not pandas.isnull(n4) else 0
    # n5 = n5 if not pandas.isnull(n5) else 0
    # n6 = n6 if not pandas.isnull(n6) else 0
    return int("%02d%02d%02d%02d%02d%02d" % (n1, n2, n3, n4, n5, n6))


def get_data():
    df = __read_csv(data_path + "/mega.csv")
    # number_cols = ["n1", "n2", "n3", "n4", "n5", "n6"]
    # format_number = "".join(("%02d" for n in number_cols))
    df["n1"] = df["n1"].map(int_or_0)
    df["n2"] = df["n2"].map(int_or_0)
    df["n3"] = df["n3"].map(int_or_0)
    df["n4"] = df["n4"].map(int_or_0)
    df["n5"] = df["n5"].map(int_or_0)
    df["n6"] = df["n6"].map(int_or_0)
    # df["numeros"] = pandas.Series(data=(
    #     concat_number(
    #         df["n1"][i], df["n2"][i], df["n3"][i], df["n4"][i], df["n5"][i], df["n6"][i]
    #     ) for i in range(0, len(df))
    # ))
    df["data"] = df["data"].map(data_para_datetime)
    return df


class Solver:
    def __init__(self):
        self.__s = LinearRegression()
        # self.__s = GradientBoostingRegressor(
        #     n_estimators=500, max_depth=4, min_samples_split=2,
        #     learning_rate=0.01, loss='ls'
        # )
        self.X = None
        self.y = None
        self.train_X, self.test_X, self.train_y, self.test_y = None, None, None, None
    
    def def_model(self, X, y):
        self.X = X
        self.y = y
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
    
    def fit(self):
        self.__s.fit(self.train_X, self.train_y)
    
    def predict(self, X=None):
        return self.__s.predict(X if X is not None else self.test_X)


class LotoSolver:
    def __init__(self):
        self.__columns = [
            (["concurso", "n2", "n3", "n4", "n5", "n6"], "n1"),
            (["concurso", "n1", "n3", "n4", "n5", "n6"], "n2"),
            (["concurso", "n1", "n2", "n4", "n5", "n6"], "n3"),
            (["concurso", "n1", "n2", "n3", "n5", "n6"], "n4"),
            (["concurso", "n1", "n2", "n3", "n4", "n6"], "n5"),
            (["concurso", "n1", "n2", "n3", "n4", "n5"], "n6"),
        ]
        self.__solvers = [
            Solver()
            for i in self.range
        ]
    
    def __len__(self):
        return len(self.__columns)
    
    @property
    def range(self):
        return range(0, len(self))
    
    def def_model(self, df):
        for i in self.range:
            X_cols, y_cols = self.__columns[i]
            X, y = df[X_cols].values, df[y_cols].values
            s = self.__solvers[i]
            s.def_model(X, y)
    
    def fit(self):
        for s in self.__solvers:
            s.fit()

    def predict(self, Xs):
        return [
            self.__solvers[i].predict(Xs[i])[0]
            for i in self.range
        ]
    
    def plot(self, Xs, preds):
        fig = plt.figure()
        for i in self.range:
            ax = fig.add_subplot(2, 3, i + 1)
            s = self.__solvers[i]
            ax.scatter(s.train_X[:, 0], s.train_y, s=1, c='r', marker="s", label='Real')
            ax.scatter(Xs[i][:, 0], preds[i], s=1, c='b', marker="s", label='Preditado Todo')
        plt.legend()
        plt.show()
            

def run():
    df = get_data()
    loto = LotoSolver()
    Xs = [
        pandas.Series(data=[2053, 10, 20, 30, 40, 50]).values.reshape(1, 6)
        for i in loto.range
    ]
    loto.def_model(df)
    loto.fit()
    preds = loto.predict(Xs)
    loto.plot(Xs, preds)
    for pred in preds:
        print(pred)


def run2():
    df = get_data()
    solver = Solver()
    X = df[["concurso", "n2", "n3", "n4", "n5", "n6"]]
    y = df["n1"]
    solver.def_model(X, y)
    solver.fit()
    pred_y = [
        solver.predict(
            pandas.Series(data=[2051,5,6,37,44,53]).values.reshape(1, 6)
        )[0],
        solver.predict(
            pandas.Series(data=[2051,1,6,37,44,53]).values.reshape(1, 6)
        )[0],
        solver.predict(
            pandas.Series(data=[2051,1,5,37,44,53]).values.reshape(1, 6)
        )[0],
        solver.predict(
            pandas.Series(data=[2051,1,5,6,44,53]).values.reshape(1, 6)
        )[0],
        solver.predict(
            pandas.Series(data=[2051,1,5,6,37,53]).values.reshape(1, 6)
        )[0],
        solver.predict(
            pandas.Series(data=[2051,1,5,6,37,44]).values.reshape(1, 6)
        )[0]
    ]
    
    print(pred_y)

run2()
