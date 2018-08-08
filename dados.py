import pandas
import etl


def _read_csv_from_file(__nome_arquivo: str):
    return pandas.read_csv("./data/" + __nome_arquivo, sep=";")


def __load_csv(__nome_arquivo: str, data_inicio: str):
    csv = _read_csv_from_file(__nome_arquivo)
    csv["data"] = csv["data"].map(etl.data_para_datetime_normalizada(data_inicio))
    csv["taxaCompra"] = csv["taxaCompra"].map(etl.porcentagem_para_float)
    csv["taxaVenda"] = csv["taxaVenda"].map(etl.porcentagem_para_float)
    csv["puCompra"] = csv["puCompra"].map(etl.real_para_float) 
    csv["puVenda"] = csv["puVenda"].map(etl.real_para_float) 
    if "selic" in csv.columns:
        csv["selic"] = csv["selic"].map(etl.porcentagem_para_float)

    csv["index"] = csv["data"]
    csv.set_index("index", inplace=True)
    return csv[[c for c in csv.columns if c != "index" and c != "puBase"]]


def csv_ipca():
    return __load_csv("tesouroIPCA2005.csv", "11/08/2005")


def csv_ipca_jur_semestral():
    return __load_csv("tituloIPCAJurosSemestrais2015.csv", "21/10/2003")


def csv_prefixado():
    return __load_csv("tituloPrefixado2008.csv", "11/08/2006")


def csv_prefixado_jur_semestral():
    return __load_csv("tituloPrefixadoJurosSemestrais2008.csv", "06/01/2004")


def csv_selic():
    return __load_csv("tituloSelic2008.csv", "13/11/2003")


class __CSVLoader:
    @property
    def ipca(self):
        return csv_ipca()
    
    @property
    def ipca_jur_semestral(self):
        return csv_ipca_jur_semestral()
    
    @property
    def prefixado(self):
        return csv_prefixado()
    
    @property
    def prefixado_jur_semestral(self):
        return csv_prefixado_jur_semestral()

    @property
    def selic(self):
        return csv_selic()
    
    def pegar_todos(self):
        def __gen__():
            yield self.ipca
            yield self.ipca_jur_semestral
            yield self.prefixado
            yield self.prefixado_jur_semestral
            yield self.selic
        return __gen__()


class __IndicadorLoader:
    @property
    def ipca(self):
        meses = {
            "JAN": 1, "FEV": 2, "MAR": 3, "ABR": 4, 
            "MAI": 5, "JUN": 6, "JUL": 7, "AGO": 8,
            "SET": 9, "OUT": 10, "NOV": 11, "DEZ": 12, 
        }

        data = _read_csv_from_file("ipca.csv")
        t0 = etl.datetime.datetime(data["ano"][0], meses[data["mes"][0]], 1)
        data["data"] = pandas.Series((
            (etl.datetime.datetime(data["ano"][i], meses[data["mes"][i]], 1) - t0).days
            for i in range(0, len(data))
        ), index=data.index)
        data["indice"] = data["indice"].map(etl.real_para_float)
        data["taxaMes"] = data["taxaMes"].map(etl.real_para_float)
        data["taxa3Meses"] = data["taxa3Meses"].map(etl.porcentagem_para_float)
        data["taxa6Meses"] = data["taxa6Meses"].map(etl.porcentagem_para_float)
        data["taxaAno"] = data["taxaAno"].map(etl.porcentagem_para_float)
        data["taxa12Meses"] = data["taxa12Meses"].map(etl.porcentagem_para_float)

        data["index"] = data["data"]
        data.set_index("index", inplace=True)
        return data[["data", "indice", "taxaMes", "taxa3Meses", "taxa6Meses", "taxaAno", "taxa12Meses"]]


CSV = __CSVLoader()
Indicador = __IndicadorLoader()
