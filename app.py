#importacao das bibliotecas
import json
from flask import Flask, request

#criacao da aplicacao flask para servir o client
app = Flask(__name__, static_url_path="/static")
app.url_map.strict_slashes = False


#hook para retirar cors 
#inidica que e seguro a comunicacao com este servico nos metodos listados
@app.after_request
def after_request(response):
    header = response.headers
    header["Access-Control-Allow-Origin"] = "*"
    header["Access-Control-Allow-Methods"] = "POST, GET, DELETE, PUT, OPTIONS"
    return response


#rota de teste para api utilizando o metodo get
@app.route("/", methods=["GET"])
def get_status():
    return "ok", 200


#rota de consumo para api utilizando o metodo post
@app.route("/consulta_titulo", methods=["POST"])
def post_consulta_titulo():
    """
       chamada da funcao de predicao para o titulo solicitado
       retorno dos valor de compra preditado juntamente com a metodologia utilizada na funcao 
       param: titulo - tipo de titulo do tesouro direto (ipca, ipca_semestral, prefixado, prefixado_semestral, selic)
       param: data - data no formato "%d/%m/%Y" para calculo dos dias corridos ou uteis 
       param: taxaCompra - valor da taxa de compra do titulo na data informada
       param: selic - valor da taxa selic na data informada  
    """
    import datetime
    body = request.json
    titulo = body["titulo"]
    if not titulo or type(titulo) is not str:
        return "Bad Request", 400
    
    #normalizacao do nome do titulo
    titulo = titulo.lower()

    #calculo de dias corridos a partir da data de extracao dos datasets até a data informada
    data = datetime.datetime.strptime(body["data"], "%d/%m/%Y")
    dias = (datetime.datetime(2018, 7, 14) - data).days

    if titulo == "ipca": #caso ipca importa a classe de teste e armazena os parametros
        from testes.teste_ipca import TesteIPCA as Teste
        test = Teste()
        params = dias, body["taxaCompra"]
        K = 2119.26
    elif titulo == "ipca_semestral": #caso ipca_semestral importa a classe de teste e armazena os parametros
        from testes.teste_ipca_semestral import TesteIPCASemestral as Teste
        test = Teste()
        params = dias, body["taxaCompra"]
        K = 3286.48
    elif titulo == "prefixado": #caso prefixado importa a classe de teste e armazena os parametros
        from testes.teste_prefixado import TestePrefixado as Teste
        test = Teste()
        params = dias, body["taxaCompra"], body["selic"]
        K = 527.59
    elif titulo == "prefixado_semestral": #caso prefixado_semestral importa a classe de teste e armazena os parametros
        from testes.teste_prefixado_semestral import TestePrefixadoSemestral as Teste
        test = Teste()
        params = dias, body["taxaCompra"]
        K = 1025.55
    elif titulo == "selic": #caso selic importa a classe de teste e armazena os parametros
        from testes.teste_selic import TesteSelic as Teste
        test = Teste()
        params = dias, body["selic"]
        K = 8990
    else:
        return "Bad Request", 400
    
    # predita o valor de compra para o titulo e parametros informados
    print("Efetua predicao")
    y_pred, solved_by = test.predict(*params, K=K)

    # estrutura o retorno do metodo
    return json.dumps({
        "preditado": list(y_pred)[0],
        "resolucao": solved_by
    }), 200
