import json
from flask import Flask, request

app = Flask(__name__, static_url_path="/static")
app.url_map.strict_slashes = False


@app.after_request
def after_request(response):
    header = response.headers
    header["Access-Control-Allow-Origin"] = "*"
    header["Access-Control-Allow-Methods"] = "POST, GET, DELETE, PUT, OPTIONS"
    return response


@app.route("/", methods=["GET"])
def get_status():
    return "ok", 200


@app.route("/consulta_titulo", methods=["POST"])
def post_consulta_titulo():
    body = request.json
    titulo = body["titulo"]
    if not titulo or type(titulo) is not str:
        return "Bad Request", 400
    titulo = titulo.lower()

    if titulo == "ipca":
        from testes.teste_ipca import TesteIPCA as Teste
        test = Teste().predict
        params = [body["data"], body["taxaCompra"]]
    elif titulo == "ipca_semestral":
        from testes.teste_ipca_semestral import TesteIPCASemestral as Teste
        test = Teste().predict
        params = [body["data"], body["taxaCompra"]]
    elif titulo == "prefixado":
        from testes.teste_prefixado import TestePrefixado as Teste
        test = Teste().predict
        params = [body["data"], body["taxaCompra"], body["selic"]]
    elif titulo == "prefixado_semestral":
        from testes.teste_prefixado_semestral import TestePrefixadoSemestral as Teste
        test = Teste().predict
        params = [body["data"], body["taxaCompra"]]
    elif titulo == "selic":
        from testes.teste_selic import TesteSelic as Teste
        test = Teste().predict
        params = [body["data"], body["selic"]]
    else:
        return "Bad Request", 400
    
    y_pred = test(*params)

    return json.dumps({
        "preditado": y_pred,
        "resolucao": "RNG - Rede Neural Gnomex"
    }), 200
