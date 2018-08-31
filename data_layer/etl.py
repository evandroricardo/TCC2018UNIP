import datetime
from numpy import nan

def data_para_datetime(item):
    if item is nan:
        return 0
    return datetime.datetime.strptime(item, "%d/%m/%Y")


def data_para_datetime_normalizada(data_inicio):
    t0 = data_para_datetime(data_inicio)
    def __gen__(item):
        t = data_para_datetime(item)
        return (t - t0).days
    return __gen__


def real_para_float(item):
    if item is nan:
        return 0
    return float(str(item).replace(".", "").replace(",", "."))


def porcentagem_para_float(item):
    if item is nan:
        return 0
    return real_para_float(item.replace("%","")) / 100


def normalizar_por_variacao():
    prev = None
    def __gen__(item):
        nonlocal prev
        if prev is not None:
            variacao = item - prev
        else:
            variacao = 0
        prev = item
        return variacao
    return __gen__
