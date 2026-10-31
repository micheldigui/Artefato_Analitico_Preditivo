"""Camada Silver: padroniza as fontes da Bronze e integra uma linha por data.

Nada e imputado: consumo ou plano ausente permanece vazio. Reservas discentes e
estimativa da nutricionista ficam em colunas separadas, porque sao sinais de
natureza diferente.

Execute a partir da raiz do projeto: python src/processar_silver.py
"""
import unicodedata
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
BRONZE = RAIZ / "dados" / "bronze"
SILVER = RAIZ / "dados" / "silver"
ANO_NUTRICIONISTA = 2026
PAUSA_MINIMA_DIAS_UTEIS = 5
HORAS_MINIMAS = 20
HORAS_MANHA = (6, 13)
HORAS_MINIMAS_MANHA = 7
DIAS_ACUMULO = 3
EVENTOS_GREVE = ("greve", "atividade_durante_greve")
DIAS =["seg", "ter", "qua", "qui", "sex", "sab", "dom"]


def ler_csv(caminho):
    try:
        return pd.read_csv(caminho, sep=";", encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(caminho, sep=";", encoding="latin1")


def ler_pasta(pasta):
    return pd.concat([ler_csv(f) for f in sorted((BRONZE / pasta).glob("*.csv"))], ignore_index=True)


def padronizar_texto(texto):
    if pd.isna(texto):
        return texto
    sem_acento = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return " ".join(sem_acento.upper().replace("-", " ").split())


def processar_calendario():
    cal = ler_pasta("calendario")
    cal["data"] = pd.to_datetime(cal["data"], dayfirst=True)
    assert not cal["data"].duplicated().any()
    return cal[["data", "status_legal", "evento_tipo"]].sort_values("data").reset_index(drop=True)


def processar_consumo():
    con = ler_pasta("consumo")
    con["data"] = pd.to_datetime(con["data"], dayfirst=True)
    con = con[con["consumo"].notna()]
    assert not con["data"].duplicated().any()
    assert (con["consumo"] >= 0).all()
    con = con.rename(columns={"status_dado": "status_consumo", "observacao": "obs_consumo"})
    return con[["data", "consumo", "status_consumo", "obs_consumo"]].sort_values("data").reset_index(drop=True)


def processar_planos():
    res = ler_pasta("reservas")
    res["data"] = pd.to_datetime(res["data"], dayfirst=True)
    res = res[res["reservas"].notna()]
    assert not res["data"].duplicated().any()
    discente = res["data"].dt.year < ANO_NUTRICIONISTA
    res["reserva_discente"] = res["reservas"].where(discente)
    res["plano_nutricionista"] = res["reservas"].where(~discente)
    return res[["data", "reserva_discente", "plano_nutricionista"]].sort_values("data").reset_index(drop=True)


def processar_cardapio():
    car = ler_pasta("cardapio")
    car.columns = [c.lower() for c in car.columns]
    car["data"] = pd.to_datetime(car["data"], dayfirst=True)
    assert not car["data"].duplicated().any()
    car = car.rename(columns={
        "prato_principal": "prato",
        "tipo_proteina_principal": "proteina",
        "tipo_preparo_principal": "preparo",
        "opcao_vegetariana": "vegetariana",
    })
    car["prato_padronizado"] = car["prato"].map(padronizar_texto)
    return car[["data", "prato", "prato_padronizado", "proteina", "preparo", "vegetariana", "guarnicao", "sobremesa"]].sort_values("data").reset_index(drop=True)


def processar_clima():
    cli = ler_csv(BRONZE / "clima" / "clima_historico.csv")
    cli["data"] = pd.to_datetime(cli["data_iso"])
    cols = {
        "precipitation_sum": "chuva_mm",
        "precipitation_hours": "chuva_horas",
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "wind_speed_10m_max": "vento_max",
    }
    return cli.rename(columns=cols)[["data", *cols.values()]].sort_values("data").reset_index(drop=True)


def sexta_de_emissao(datas, deslocamento):
    return datas - pd.to_timedelta(datas.dt.dayofweek + deslocamento, unit="D")


def processar_previsao_t1():
    """Previsao Previous Runs. Usa uma rodada anterior a emissao, ate o limite de 7 dias do servico."""
    hor = pd.read_csv(BRONZE / "clima_previsoes" / "previous_runs_horario.csv", sep=";", parse_dates=["time"])
    hor["data"] = hor["time"].dt.normalize()
    colunas = [f"precipitation_previous_day{d}" for d in range(1, 8)]
    chuva = hor.groupby("data")[colunas].sum(min_count=HORAS_MINIMAS)
    manha = hor[hor["time"].dt.hour.between(*HORAS_MANHA)].groupby("data")[colunas].sum(min_count=HORAS_MINIMAS_MANHA)
    acumulada = chuva.rolling(DIAS_ACUMULO, min_periods=DIAS_ACUMULO).sum()
    temp = hor.groupby("data")[[f"temperature_2m_previous_day{d}" for d in range(1, 8)]].max()
    tab = pd.DataFrame({"data": chuva.index})
    tab = tab[tab["data"].dt.dayofweek < 5].reset_index(drop=True)
    tab["origem"] = sexta_de_emissao(tab["data"], 3)
    dias = ((tab["data"] - tab["origem"]).dt.days + 1).clip(upper=7)
    tab["previsao_chuva_mm"] = [chuva.at[d, f"precipitation_previous_day{n}"] for d, n in zip(tab["data"], dias)]
    tab["previsao_chuva_manha_mm"] = [manha.at[d, f"precipitation_previous_day{n}"] for d, n in zip(tab["data"], dias)]
    tab["previsao_chuva_acum3d_mm"] = [acumulada.at[d, f"precipitation_previous_day{n}"] for d, n in zip(tab["data"], dias)]
    tab["previsao_temp_max"] = [temp.at[d, f"temperature_2m_previous_day{n}"] for d, n in zip(tab["data"], dias)]
    return tab.assign(horizonte="t1").dropna(subset=["previsao_chuva_mm"])


def processar_previsao_t2():
    """Media dos modelos GFS e ECMWF na rodada de sexta-feira."""
    hor = pd.read_csv(BRONZE / "clima_previsoes" / "single_runs_horario.csv", sep=";", parse_dates=["time", "rodada"])
    hor["data"] = hor["time"].dt.normalize()
    grupos = hor.groupby(["rodada", "modelo", "data"])
    dia = pd.DataFrame({"chuva": grupos["precipitation"].sum(min_count=HORAS_MINIMAS), "temp": grupos["temperature_2m"].max()})
    dia["manha"] = hor[hor["time"].dt.hour.between(*HORAS_MANHA)].groupby(["rodada", "modelo", "data"])["precipitation"].sum(min_count=HORAS_MINIMAS_MANHA)
    dia = dia.reset_index().sort_values(["rodada", "modelo", "data"])
    dia["acum"] = dia.groupby(["rodada", "modelo"])["chuva"].transform(lambda s: s.rolling(DIAS_ACUMULO, min_periods=DIAS_ACUMULO).sum())
    tab = dia.groupby(["rodada", "data"], as_index=False).agg(
        previsao_chuva_mm=("chuva", "mean"), previsao_chuva_manha_mm=("manha", "mean"),
        previsao_chuva_acum3d_mm=("acum", "mean"), previsao_temp_max=("temp", "mean"))
    tab = tab[(tab["data"].dt.dayofweek < 5) & (sexta_de_emissao(tab["data"], 10) == tab["rodada"])]
    return tab.rename(columns={"rodada": "origem"}).assign(horizonte="t2").dropna(subset=["previsao_chuva_mm"])


def marcar_blocos_letivos(fato):
    """Posicao de cada dia util letivo dentro do bloco continuo de aulas.

    Um bloco termina quando ha ao menos cinco dias uteis seguidos nao letivos.
    As colunas dependem so do calendario e, portanto, sao conhecidas com antecedencia.
    """
    util = fato[fato["dia_semana"] < 5].copy()
    util["letivo"] = util["status_legal"].eq("letivo")
    pausas = (~util["letivo"]).groupby(util["letivo"].cumsum()).transform("sum")
    pausa_longa = ~util["letivo"] & (pausas >= PAUSA_MINIMA_DIAS_UTEIS)
    bloco = pausa_longa.astype(int).diff().fillna(0).eq(-1).cumsum()
    letivos = util[util["letivo"]].assign(bloco=bloco)
    grupo = letivos.groupby("bloco")["data"]
    letivos["dias_desde_inicio_bloco"] = grupo.cumcount()
    letivos["dias_ate_fim_bloco"] = grupo.cumcount(ascending=False)
    return fato.merge(letivos[["data", "bloco", "dias_desde_inicio_bloco", "dias_ate_fim_bloco"]], on="data", how="left")


def integrar_fato_diaria(calendario, consumo, planos, cardapio, clima):
    fato = calendario
    for tabela in (consumo, planos, cardapio, clima):
        fato = fato.merge(tabela, on="data", how="left")
    assert not fato["data"].duplicated().any()
    fato["ano"] = fato["data"].dt.year
    fato["mes"] = fato["data"].dt.month
    fato["dia_semana"] = fato["data"].dt.dayofweek
    fato["dia_nome"] = fato["dia_semana"].map(dict(enumerate(DIAS)))
    fato["regime_plano"] = pd.Series("reserva_discente", index=fato.index).where(fato["ano"] < ANO_NUTRICIONISTA, "nutricionista")
    fato["no_escopo"] = fato["dia_semana"] < 5
    fato["tem_consumo"] = fato["consumo"].notna()
    fato["tem_cardapio"] = fato["prato"].notna()
    greve = fato.loc[fato["evento_tipo"].isin(EVENTOS_GREVE), "data"]
    fato["periodo_greve"] = fato["data"].between(greve.min(), greve.max())
    return marcar_blocos_letivos(fato)


def salvar(tabela, nome):
    tabela.to_parquet(SILVER / f"{nome}.parquet", index=False)
    tabela.to_csv(SILVER / f"{nome}.csv", sep=";", index=False, encoding="utf-8-sig")


def main():
    SILVER.mkdir(parents=True, exist_ok=True)
    calendario = processar_calendario()
    consumo = processar_consumo()
    planos = processar_planos()
    cardapio = processar_cardapio()
    clima = processar_clima()
    fato = integrar_fato_diaria(calendario, consumo, planos, cardapio, clima)
    previsoes = pd.concat([processar_previsao_t1(), processar_previsao_t2()], ignore_index=True)
    for tabela, nome in [(calendario, "silver_calendario"), (consumo, "silver_consumo"), (planos, "silver_planos"),
                         (cardapio, "silver_cardapio"), (clima, "silver_clima"), (fato, "silver_fato_diaria"),
                         (previsoes, "silver_clima_previsto")]:
        salvar(tabela, nome)
    print(f"silver_fato_diaria: {fato.shape}; datas com consumo: {fato['tem_consumo'].sum()}")


if __name__ == "__main__":
    main()
