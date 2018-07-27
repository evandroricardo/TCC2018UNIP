from dados import CSV

class TesteIPCA:
    @property
    def name(self):
        return self.__class__.__name__

    def pega_csv(self):
        return CSV.ipca

    def clusterizacao(self, csv):
        from clusterizacao import run
        x = csv[["puCompra", "taxaCompra"]].values
        return run(csv, x, 2, random_state=0)

    def regressao_linear(self, csv):
        from regressaolinear import run
        x = csv["puCompra"].values.reshape(-1, 1)
        y = csv["taxaCompra"].values.reshape(-1, 1)
        xAmostra = csv["puCompra"][:200].values.reshape(-1, 1)
        yAmostra = csv["taxaCompra"][:200].values.reshape(-1, 1)
        return run(csv, x, y, xAmostra, yAmostra)

    def nn_regressao(self, csv):
        from nnregression import run
        x = csv["puCompra"].values.reshape(-1, 1)
        y = csv["taxaCompra"].values.reshape(-1, 1)
        xAmostra = csv["puCompra"][:200].values.reshape(-1, 1)
        yAmostra = csv["taxaCompra"][:200].values.reshape(-1, 1)
        return run(
            csv, x, y, xAmostra, yAmostra,
            hidden_layer_sizes=(100,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
            learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
            random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=False,
            early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08
        )

