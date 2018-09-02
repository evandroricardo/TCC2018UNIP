from testes.teste_ipca import TesteIPCA
from testes.teste_ipca_semestral import TesteIPCASemestral
from testes.teste_prefixado import TestePrefixado
from testes.teste_prefixado_semestral import TestePrefixadoSemestral
from testes.teste_selic import TesteSelic

testes = [
    {"test": TesteIPCA(), "params": (2207, 0.0494), "K": 2119.26},
    {"test": TesteIPCASemestral(), "params": (2937, 0.10), "K": 3286.48},
    {"test": TestePrefixado(), "params": (1614, 0.10, 0.15), "K": 0},
    {"test": TestePrefixadoSemestral(), "params": (2610, 0.13), "K": 1025.55},
    {"test": TesteSelic(), "params": (1674, 0.15), "K": 8990},
]


def run():
    for t in testes:
        test, params, K = t["test"], t["params"], t["K"]
        print(test.name)
        result = test.predict(*params)[0][0] + K
        print("Resultado: {0}\n\n".format(result))
    print("Pronto!")  

run()
