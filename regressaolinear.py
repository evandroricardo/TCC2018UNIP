import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def run(csv, x, y, xAmostra, yAmostra, **config):
      lnr = LinearRegression(**config)
      lnr.fit(xAmostra, yAmostra)

      pred = lnr.predict(x)

      # Os coeficientes
      print('Coeficiente: %.20f' % lnr.coef_[0][0])
      # O erro quadrático médio
      print("Erro médio quadrático: %.2f"
            % mean_squared_error(x, pred))
      # Pontuação de variância: 1 é uma previsão perfeita
      print('Variância: %.2f' % r2_score(x, pred))

      # Plotagem das saídas
      plt.scatter(x, y, s=1, color='g', marker="s", label='Real')
      plt.plot(x, pred, color='b', linewidth=3, label='Prediction')
      plt.title("Preco de Compra X Taxa de Compra")

      plt.xticks(())
      plt.yticks(())

      plt.legend()
      plt.show()

      #print(pred[:10])
      #print("ok")

      #shape
      #print(csv.shape)

      #describe data set 
      #print(csv.describe())
