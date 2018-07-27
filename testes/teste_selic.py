from dados import CSV
from numpy import random

class TesteSelic:
    @property
    def name(self):
        return self.__class__.__name__

    def pega_csv(self):
        return CSV.selic

    def clusterizacao(self, csv):
        from clusterizacao import run
        x = csv[["data", "puCompra"]].values
        return run(csv, x, 2, random_state=0)

    def regressao_linear(self, csv):
        from regressaolinear import run
        x = csv["data"].values.reshape(-1, 1)
        y = csv["puCompra"].values.reshape(-1, 1)
        xAmostra = csv["data"][:200].values.reshape(-1, 1)
        yAmostra = csv["puCompra"][:200].values.reshape(-1, 1)
        return run(csv, x, y, xAmostra, yAmostra)

    def nn_regressao(self, csv):
        from nnregression import run
        x = csv["data"].values
        y = csv["puCompra"].values
        _tamnho_amostra = int(len(x) * 0.8)
        _tamnho_teste = int(len(x) * 0.2)
        xAmostra = random.choice(x, _tamnho_amostra).reshape(-1, 1)
        yAmostra = random.choice(y, _tamnho_amostra).reshape(-1, 1)
        xTeste = random.choice(x, _tamnho_teste).reshape(-1, 1)
        yTeste = random.choice(y, _tamnho_teste).reshape(-1, 1)
        return run(
            csv, x, y, xTeste, yTeste, xAmostra, yAmostra,
            hidden_layer_sizes=(1000, 1000),  activation='relu', solver='lbfgs', alpha=0.001, batch_size='auto',
            learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=False,
            random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=False,
            early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08
        )

