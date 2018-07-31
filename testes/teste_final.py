from testes.teste_ipca import TesteIPCA
from testes.teste_selic import TesteSelic

def testar():
    testes = [
        TesteSelic(),
        TesteIPCA(),
    ]

    print("Executando testes...\n")
    for test in testes:
        print(test.name)
        csv = test.pega_csv()
        test.regressao_linear(csv)
        test.clusterizacao(csv)
        test.nn_regressao(csv)

    print("\n\nFIM!")