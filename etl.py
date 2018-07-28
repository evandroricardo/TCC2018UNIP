import datetime
from numpy import nan

def data_para_datetime(item):
    if item is nan:
        return 0
    return datetime.datetime.strptime(item, "%d/%m/%Y").timestamp()


def real_para_float(item):
    if item is nan:
        return 0
    return float(str(item).replace(".", "").replace(",", "."))

def porcentagem_para_float(item):
    if item is nan:
        return 0
    return real_para_float(item.replace("%","")) / 100
