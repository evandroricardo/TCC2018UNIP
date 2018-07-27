from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

def run(csv):
    x = csv["puCompra"].values.reshape(-1, 1)
    y = csv["taxaCompra"].values.reshape(-1, 1)
    xAmostra = csv["puCompra"][:200].values.reshape(-1, 1)
    yAmostra = csv["taxaCompra"][:200].values.reshape(-1, 1)

    nn = MLPRegressor(
        hidden_layer_sizes=(100,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
        random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    n = nn.fit(x, y)
    test_x = x
    test_y = nn.predict(test_x)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y, s=1, c='g', marker="s", label='Real')
    ax1.scatter(test_x,test_y, s=10, c='b', marker="o", label='NN Prediction')

    plt.title("Preco de Compra X Taxa de Compra")

    plt.legend()
    plt.show()
