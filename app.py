#importacao das bibliotecas
import json
from flask import Flask, request

#criacao da aplicacao flask para servir o client
app = Flask(__name__, static_url_path="/static")
app.url_map.strict_slashes = False


# Interpola n pontos entre 0 e y
def interpolacao(y: float, n_pontos: int=5):
    def lerp(v0, v1, t):
        # Extraido de:
        #     https://en.wikipedia.org/wiki/Linear_interpolation
        
        return v0 + t * (v1 - v0)
    return [
        round(lerp(0, y, 1/n_pontos * i), 2)
        for i in range(1, n_pontos)
    ]


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
    dias = (data - datetime.datetime(2018, 7, 14)).days
    if dias < 0:
        dias = -dias

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
        params = dias, body["taxaCompra"], body["taxaSelic"]
        K = 527.59
    elif titulo == "prefixado_semestral": #caso prefixado_semestral importa a classe de teste e armazena os parametros
        from testes.teste_prefixado_semestral import TestePrefixadoSemestral as Teste
        test = Teste()
        params = dias, body["taxaCompra"]
        K = 1025.55
    elif titulo == "selic": #caso selic importa a classe de teste e armazena os parametros
        from testes.teste_selic import TesteSelic as Teste
        test = Teste()
        params = dias, body["taxaSelic"]
        K = 8990
    else:
        return "Bad Request", 400
    
    # predita o valor de compra para o titulo e parametros informados
    print("Efetua predicao")
    y_pred, solved_by = test.predict(*params, K=K)
    result = list(y_pred)[0]
    n_pontos = 5

    # estrutura o retorno do metodo
    return json.dumps({
        "preditado": round(result, 2),
        "resolucao": solved_by,
        "serie": interpolacao(result, n_pontos),
        "datas": [
            "{0} dias".format(int(1/n_pontos * i * dias))
            for i in range(1, n_pontos)
        ]
    }), 200

#ativa o servidor para exposição dos serviços do webservice - porta:8080
if __name__ == '__main__':
    from gevent.pywsgi import WSGIServer
    http_server = WSGIServer(('0.0.0.0', 8080), app)
    print('Gevent serve forever on', 8080)
    http_server.serve_forever()
