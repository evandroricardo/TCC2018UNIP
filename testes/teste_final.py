from testes.teste_ipca import TesteIPCA
from testes.teste_selic import TesteSelic
from testes.teste_prefixado import TestePrefixado
from testes.teste_ipca_semestral import TesteIPCASemestral
from testes.teste_prefixado_semestral import TestePrefixadoSemestral

def testar():
    testes = [
        #TesteIPCA(),
        #TesteIPCASemestral(), 
        TestePrefixadoSemestral(),
        #TesteSelic(),
        #TestePrefixado(),       
        
    ]

    print("Executando testes...\n")
    for test in testes:
        print(test.name)
        csv = test.pega_csv()
        test.regressao_linear(csv)
        test.clusterizacao(csv)
        test.nn_regressao(csv)

    print("\n\nFIM!")