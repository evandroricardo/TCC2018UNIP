import pandas
import etl


def __load_csv(__nome_arquivo: str):
    csv = pandas.read_csv("./data/" + __nome_arquivo, sep=";")
    csv["data"] = csv["data"].map(etl.data_para_datetime)
    csv["taxaCompra"] = csv["taxaCompra"].map(etl.porcentagem_para_float)
    csv["taxaVenda"] = csv["taxaVenda"].map(etl.porcentagem_para_float)
    csv["puCompra"] = csv["puCompra"].map(etl.real_para_float)
    csv["puVenda"] = csv["puVenda"].map(etl.real_para_float)
    return csv[["data", "taxaCompra", "taxaVenda", "puCompra", "puVenda"]]


def csv_ipca():
    return __load_csv("tesouroIPCA2005.csv")


def csv_ipca_jur_semestral():
    return __load_csv("tituloIPCAJurosSemestrais2015.csv")


def csv_prefixado():
    return __load_csv("tituloPrefixado2008.csv")


def csv_prefixado_jur_semestral():
    return __load_csv("tituloPrefixadoJurosSemestrais2008.csv")


def csv_selic():
    return __load_csv("tituloSelic2008.csv")


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

CSV = __CSVLoader()
