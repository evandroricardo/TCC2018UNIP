# Herança das classes
from testes.teste_ipca import TesteIPCA
from testes.teste_ipca_semestral import TesteIPCASemestral
from testes.teste_prefixado import TestePrefixado
from testes.teste_prefixado_semestral import TestePrefixadoSemestral
from testes.teste_selic import TesteSelic

# Array com parametros de teste de todas as funcoes
testes = [
    {"test": TesteIPCA(), "params": (2207, 0.0494), "K": 2119.26},
    {"test": TesteIPCASemestral(), "params": (2937, 0.10), "K": 3286.48},
    {"test": TestePrefixado(), "params": (1614, 0.10, 0.15), "K": 0},
    {"test": TestePrefixadoSemestral(), "params": (2610, 0.13), "K": 1025.55},
    {"test": TesteSelic(), "params": (1674, 0.15), "K": 8990},
]

# Chamada de cada funcao percorrendo o array e passagem dos parametros para seus respectivos modulos
def run():
    for t in testes:
        test, params, K = t["test"], t["params"], t["K"]
        
        #Exibe o nome do titulo testado
        print(test.name)
        
        #Resgata resultados obtidos
        result = test.predict(*params)[0][0] + K
        
        # Print do resultado na log 
        print("Resultado: {0}\n\n".format(result))

    # Mensagem indicando o termino do processamento    
    print("Pronto!")  

run()
