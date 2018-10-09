from data_layer.data import CSV
from numpy import random
from sklearn.model_selection import train_test_split

# Classe para preditar o valor do titulo TestePrefixadoSemestral
class TestePrefixadoSemestral:
    @property
    def name(self):
        return self.__class__.__name__

    # Leitura da serie historica presente no arquivo .csv
    def pega_csv(self):
        return CSV.prefixado_jur_semestral

    # Funcao de clusterizacao recebendo como parametro a serie historica presente no arquivo .csv
    def clusterizacao(self, csv):
        from neural.clusterizacao import run
        # Passagem dos parametros de data e preco unitario de compra para clusterizar
        x = csv[["data", "puCompra"]].values
        return run(csv, x, 2, random_state=0)

   # Funcao de regressao linear recebendo como parametro a serie historica presente no arquivo .csv
    def regressao_linear(self, csv, X_test=None, dont_plot=False):
        from neural.regressaolinear import run

        # Funcao para testes da regressao do simulador para demonstrar resultados antes da plotagem 
        # de graficos
        def before_plot(lnr, pred):
            pred = lnr.predict([[2610, 0.13]])[0] + 1025.55
            print("Regressao Linear:\n\tPreco Prefixado Juros Semestrais 2029: ", pred)

        # Passagem dos parametros de data, taxa de compra e preco unitario de compra para traçar 
        # a regressao 
        X = csv[["data", "taxaCompra"]].values
        y = csv["puCompra"].values

        # Matrizes e parametros de configurações, considerando percentual de teste (test_size)
        # bem como o shuffle setado como false para não embaralhar os valores    
        X_train, _X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        if X_test is None:
            X_test = _X_test
        return run(
            csv, X_test, y_test, X_train, y_train, before_plot=before_plot, dont_plot=dont_plot
        )

    # Funcao de chamada da rede neural de regressao recebendo como parametro a serie historica
    # considerando a matriz de treinamento (X_test) e inibicao da plotagem do grafico usando dont_plot
    def nn_regressao(self, csv, X_test=None, dont_plot=False):
        from neural.nnregression import run

        # Matrizes com data, taxa de compra e preco unitario de compra
        X = csv[["data", "taxaCompra"]].values
        y = csv["puCompra"].values

        # Matrizes e parametros de configurações, considerando percentual de teste (test_size)
        # bem como o shuffle setado como false para não embaralhar os valores     
        X_train, _X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        if X_test is None:
            X_test = _X_test

        # Parametros de configuracao para a rede neural considerando calibragens     
        return run(
            X, y, X_test, y_test, X_train, y_train, 
            dont_plot=dont_plot, 
            before_plot=self.before_plot_nn_regressao, plot_title="Dias corridos x Preco de compra",
            hidden_layer_sizes=(308,),  activation='relu', solver='adam', learning_rate='constant', alpha=0.001, 
            batch_size='auto', learning_rate_init=0.01, power_t=0.5, max_iter=100000, shuffle=True,
            random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=False,
            early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.9999, epsilon=1e-08
        )

    # Funcao para testes de predicao do simulador para demonstrar resultados antes da plotagem de graficos
    def before_plot_nn_regressao(self, nn, yPred, yPredTeste, score_amostra, score_todo):
        preco_prefixadosemestral_2008 = nn.predict([[2610, 0.13]])[0] + 1025.55
        print("Previsao preco Prefixado Juros Semestrais 2029 no vencimento: {0}; Min: {1}; Max: {2}".format(
            preco_prefixadosemestral_2008,
            preco_prefixadosemestral_2008 * score_todo,
            preco_prefixadosemestral_2008 * (2 - score_todo)
        ))
        return yPred, yPredTeste

    # Funcao para predicao dos valores usando metologia determinada para o titulo
    def predict(self, data, taxa_compra, K=0):
        result = self.regressao_linear(self.pega_csv(), X_test=[[data, taxa_compra]], dont_plot=True)
        return result + K, "Reg. Linear"
