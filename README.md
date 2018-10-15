# API - Predcon

Api desenvolvida em Python para a chamada de redes neurais e previsão de preço de compra de títulos do Tesouro Direto (Ofertas de Dívidas Públicas).

[O que é tesouro direto?](http://www.tesouro.fazenda.gov.br/tesouro-direto)

## Como usar

Clone o projeto: `git clone https://github.com/evandroricardo/TCC2018UNIP.git`

Este projeto depende das seguintes bibliotecas:

* Flask
* Numpy
* Matplotlib
* Pandas
* Scikit_learn
* Gevent

Para mais informações observe o arquivo [requeriments.txt](https://github.com/evandroricardo/TCC2018UNIP/blob/master/requirements.txt)

Execute `python app.py` para iniciar o serviço utilizando a porta 8080.

### Endpoints

#### get_status

Rota para testes da api

##### Request

````
GET /
````

##### Resposta

````
HTTP/1.1 200 OK
"ok"
````

#### consulta_titulo

Chamada da funcao de predicao para o titulo solicitado retorno dos valor de compra preditado juntamente com a metodologia utilizada na funcao.

* param: titulo - tipo de titulo do tesouro direto (ipca, ipca_semestral, prefixado, prefixado_semestral, selic);
* param: data - data no formato "%d/%m/%Y" para calculo dos dias corridos ou uteis; 
* param: taxaCompra - valor da taxa de compra do titulo na data informada;
* param: selic - valor da taxa selic na data informada.  

##### Request

````
POST /consulta_titulo
Content-Type: application/json

{
    "titulo": "IPCA",
    "data": "21/12/2019",
    "taxaCompra": 12.12
}
````

##### Resposta

````
HTTP/1.1 200 OK
Content-Type: application/json

{
    "preditado": 7890.00,
    "resolucao": "Rede Neural",
    "serie": [
        2340.00
        5680.00
        ...
        7890.00
    ],
    "datas": [
        "120 dias",
        "240 dias",
        ...
        "360 dias"
    ]
}
````

### Datasets

Para a realização da previsão dos valores de preços do Tesouro Direto, o projeto deve conter as séries temporais com indicadores de cada título.

Os arquivos seguem o padrão de estrutura de acordo com as listas abaixo, sua extensão será ".csv" e cada registro será delimitado através do símbolo ";".

#### IPCA

[tesouroIPCA2005.csv](https://github.com/evandroricardo/TCC2018UNIP/blob/master/data_layer/source/tesouroIPCA2005.csv)

````
    ano;mes;indice;taxaMes;taxa3Meses;taxa6Meses;taxaAno;taxa12Meses
    1994;JAN;141,31;41,31;162,13;533,33;41,31;2.693,84
````

#### IPCA_SEMESTRAL

[tituloIPCAJurosSemestrais2015.csv](https://github.com/evandroricardo/TCC2018UNIP/blob/master/data_layer/source/tituloIPCAJurosSemestrais2015.csv)

````
    data;taxaCompra;taxaVenda;puCompra;puVenda;puBase
    21/10/2003;9,93%;9,99%;1.049,50;1.045,11;1.044,53
````

#### PREFIXADO

[tituloPrefixado2008.csv](https://github.com/evandroricardo/TCC2018UNIP/blob/master/data_layer/source/tituloPrefixado2008.csv)

````
    data;taxaCompra;taxaVenda;puCompra;puVenda;puBase;selic
    11/08/2006;14,52%;14,57%;803,77;803,21;802,77;14,75%
````

#### PREFIXADO_SEMESTRAL

[tituloPrefixadoJurosSemestrais2008.csv](https://github.com/evandroricardo/TCC2018UNIP/blob/master/data_layer/source/tituloPrefixadoJurosSemestrais2008.csv)

````
    data;taxaCompra;taxaVenda;puCompra;puVenda;puBase
    06/01/2004;17,14%;17,24%;813,14;810,86;810,35
````

#### SELIC 

[tituloSelic2008.csv](https://github.com/evandroricardo/TCC2018UNIP/blob/master/data_layer/source/tituloSelic2008.csv)

````
    data;taxaCompra;taxaVenda;puCompra;puVenda;puBase;selic
    13/11/2003;0,006100;0,006900;1772,37;1765,96;1764,69;19,00%
````
