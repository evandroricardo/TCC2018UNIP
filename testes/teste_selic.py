from dados import CSV

class TesteSelic:
    @property
    def name(self):
        return self.__class__.__name__

    def pega_csv(self):
        return CSV.selic

    def clusterizacao(self, csv):
        from clusterizacao import run
        x = csv[["puCompra", "data"]].values
        return run(csv, x, 2, random_state=0)

    def regressao_linear(self, csv):
        from regressaolinear import run
        x = csv["puCompra"].values.reshape(-1, 1)
        y = csv["data"].values.reshape(-1, 1)
        xAmostra = csv["puCompra"][:200].values.reshape(-1, 1)
        yAmostra = csv["data"][:200].values.reshape(-1, 1)
        return run(csv, x, y, xAmostra, yAmostra)

    def nn_regressao(self, csv):
        from nnregression import run
        x = csv["puCompra"].values.reshape(-1, 1)
        y = csv["data"].values.reshape(-1, 1)
        xAmostra = csv["puCompra"][:200].values.reshape(-1, 1)
        yAmostra = csv["data"][:200].values.reshape(-1, 1)
        return run(
            csv, x, y, xAmostra, yAmostra,
            hidden_layer_sizes=(500,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
            learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=100000, shuffle=True,
            random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08
        )

