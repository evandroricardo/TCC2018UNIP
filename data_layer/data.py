import pandas
import data_layer.etl as etl

# Classe para leitura do arquivo .csv com os valores presentes na serie historica
def _read_csv_from_file(__nome_arquivo: str):

    # Caminho e nome do arquivo para leitura e delimitador
    return pandas.read_csv("./data_layer/source/" + __nome_arquivo, sep=";")

# Classe para carga dos valores do arquivo no projeto
def __load_csv(__nome_arquivo: str, data_inicio: str):

    # Formato e endereco do arquivo
    csv = _read_csv_from_file(__nome_arquivo)

    # Resgate dos valores utilizados no projeto e tratamento das informacoes
    csv["data"] = csv["data"].map(etl.data_para_datetime_normalizada(data_inicio))
    csv["taxaCompra"] = csv["taxaCompra"].map(etl.porcentagem_para_float)
    csv["taxaVenda"] = csv["taxaVenda"].map(etl.porcentagem_para_float)
    csv["puCompra"] = csv["puCompra"].map(etl.real_para_float) 
    csv["puVenda"] = csv["puVenda"].map(etl.real_para_float) 

    # Condiciona a existencia dos valores selic no arquivo
    if "selic" in csv.columns:
        csv["selic"] = csv["selic"].map(etl.porcentagem_para_float)

    # Indexa a variavel data para utilizacao
    csv["index"] = csv["data"]
    csv.set_index("index", inplace=True)
    return csv[[c for c in csv.columns if c != "index" and c != "puBase"]]

# Classe para leitura do arquivo .csv com os valores IPCA presentes na serie historica
# Classe para carga dos valores do arquivo IPCA no projeto
def csv_ipca():
    return __load_csv("tesouroIPCA2005.csv", "11/08/2005")

# Classe para leitura do arquivo .csv com os valores Juros Semestrais presentes na serie historica
# Classe para carga dos valores do arquivo Juros Semestrais no projeto
def csv_ipca_jur_semestral():
    return __load_csv("tituloIPCAJurosSemestrais2015.csv", "21/10/2003")

# Classe para leitura do arquivo .csv com os valores Prefixados presentes na serie historica
# Classe para carga dos valores do arquivo Prefixados no projeto
def csv_prefixado():
    return __load_csv("tituloPrefixado2008.csv", "11/08/2006")

# Classe para leitura do arquivo .csv com os valores Prefixados Juros Semestrais presentes na serie 
# historica
# Classe para carga dos valores do arquivo Prefixados Juros Semestrais no projeto
def csv_prefixado_jur_semestral():
    return __load_csv("tituloPrefixadoJurosSemestrais2008.csv", "06/01/2004")

# Classe para leitura do arquivo .csv com os valores Selic presentes na serie historica
# Classe para carga dos valores do arquivo Selic no projeto
def csv_selic():
    return __load_csv("tituloSelic2008.csv", "13/11/2003")

# Classe de carga
class __CSVLoader:
    @property
    # Classe IPCA
    def ipca(self):
        return csv_ipca()
    
    # Classe IPCA Juros Semestrais
    @property
    def ipca_jur_semestral(self):
        return csv_ipca_jur_semestral()
    
    # Classe Prefixado
    @property
    def prefixado(self):
        return csv_prefixado()
    
    # Classe Prefixado Juros Semestrais
    @property
    def prefixado_jur_semestral(self):
        return csv_prefixado_jur_semestral()

    # Classe Selic
    @property
    def selic(self):
        return csv_selic()
    
    # Classe para carga de todos titulos
    def pegar_todos(self):
        def __gen__():
            yield self.ipca
            yield self.ipca_jur_semestral
            yield self.prefixado
            yield self.prefixado_jur_semestral
            yield self.selic
        return __gen__()

# Classe tratamento de carga
class __IndicadorLoader:
    @property
    # Classe tratamento meses
    def ipca(self):
        meses = {
            "JAN": 1, "FEV": 2, "MAR": 3, "ABR": 4, 
            "MAI": 5, "JUN": 6, "JUL": 7, "AGO": 8,
            "SET": 9, "OUT": 10, "NOV": 11, "DEZ": 12, 
        }

        # Classe leitura do arquivo
        data = _read_csv_from_file("ipca.csv")

        # Classe tratamento 
        t0 = etl.datetime.datetime(data["ano"][0], meses[data["mes"][0]], 1)
        data["data"] = pandas.Series((
            (etl.datetime.datetime(data["ano"][i], meses[data["mes"][i]], 1) - t0).days
            for i in range(0, len(data))
        ), index=data.index)

        # Chamada da classe etl para tratar informacoes
        data["indice"] = data["indice"].map(etl.real_para_float)
        data["taxaMes"] = data["taxaMes"].map(etl.real_para_float)
        data["taxa3Meses"] = data["taxa3Meses"].map(etl.porcentagem_para_float)
        data["taxa6Meses"] = data["taxa6Meses"].map(etl.porcentagem_para_float)
        data["taxaAno"] = data["taxaAno"].map(etl.porcentagem_para_float)
        data["taxa12Meses"] = data["taxa12Meses"].map(etl.porcentagem_para_float)

        data["index"] = data["data"]
        data.set_index("index", inplace=True)
        return data[["data", "indice", "taxaMes", "taxa3Meses", "taxa6Meses", "taxaAno", "taxa12Meses"]]

# Classe para carga dos arquivos csv
CSV = __CSVLoader()
Indicador = __IndicadorLoader()
