"""Registro das previsoes enviadas em outubro de 2026 e diagnostico do erro de 26 a 30/10."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .configuracao import PASTA_REVISAO, RAIZ, SILVER

REFERENCIA = RAIZ / "dados" / "referencia_dissertacao"
ENTRADAS_OUTUBRO = REFERENCIA / "outubro"
PROTEINAS_APP = ["bovina", "frango", "suina", "peixe", "ovo"]
LAGS = ["consumo_lag_5", "consumo_lag_10"]
NUMERICAS = [*LAGS, "media_movel_5d", "precipitation_sum", "temperature_2m_max"]
COLUNAS_APP = NUMERICAS + [f"is_dia_sem_{d}" for d in range(5)] + [f"is_prot_{p}" for p in PROTEINAS_APP]
FIM_TREINO_APP = "2026-08-31"
EMISSAO_FINAL = pd.Timestamp("2026-10-24")


def registro_de_outubro():
    h = pd.read_csv(ENTRADAS_OUTUBRO / "registro_previsoes.csv", sep=";")
    h["data"] = pd.to_datetime(h.data_iso)
    h["emitida_em"] = pd.to_datetime(h.timestamp_execucao, errors="coerce")
    return h


def extrair_previsoes_enviadas():
    """Ultima previsao com margem emitida antes da segunda-feira de cada semana."""
    h = registro_de_outubro()
    h = h[h.margem_aplicada.eq(12) & h.estimativa_nutri.notna()].copy()
    h["segunda"] = h.data - pd.to_timedelta(h.data.dt.dayofweek, unit="D")
    h = h[h.emitida_em < h.segunda].sort_values("emitida_em").groupby("data").tail(1).sort_values("data")
    assert len(h) == 10
    assert (h.recomendacao_final == h.pred_base_combinada + 12).all()
    s = h.assign(decisao=h.horizonte.str.extract(r"\((.*)\)")[0], modelo_puro=h.ia_pura, hibrido=h.pred_base_combinada, hibrido_com_margem=h.recomendacao_final)
    s = s[["data", "emitida_em", "decisao", "modelo_puro", "hibrido", "hibrido_com_margem"]]
    s.to_csv(REFERENCIA / "previsoes_enviadas_outubro.csv", sep=";", index=False)
    return s


def preparar_treino(t):
    """Repete o aplicativo em uso: coluna ausente vira a mediana do consumo."""
    t = t.copy()
    mediana = t["consumo"].median()
    for c in NUMERICAS:
        t[c] = mediana if c not in t.columns else t[c].fillna(mediana)
    for d in range(5):
        t[f"is_dia_sem_{d}"] = (t.data.dt.dayofweek == d).astype(int)
    for p in PROTEINAS_APP:
        t[f"is_prot_{p}"] = t["proteina_tipo"].astype(str).str.lower().str.contains(p, na=False).astype(int)
    return t


def treinar_aplicativo(g):
    t = preparar_treino(g[g.data <= FIM_TREINO_APP])
    return Ridge(alpha=10.0, random_state=42).fit(t[COLUNAS_APP].values, t["consumo"].values)


def prever_semana(modelo, consumo, semana):
    """Preve 26 a 30/10 com o consumo conhecido na emissao de 24/10."""
    s = consumo[consumo.data <= EMISSAO_FINAL]
    media = s.consumo.tail(5).mean()
    previsoes = []
    for _, linha in semana.iterrows():
        mesmo_dia = s[s.data.dt.dayofweek == linha.data.dayofweek].consumo
        v = {"consumo_lag_5": mesmo_dia.iloc[-1], "consumo_lag_10": mesmo_dia.iloc[-2], "media_movel_5d": media, "precipitation_sum": linha.chuva_mm, "temperature_2m_max": linha.temp_max_c}
        v |= {f"is_dia_sem_{d}": int(linha.data.dayofweek == d) for d in range(5)}
        v |= {f"is_prot_{p}": int(p in str(linha.proteina_tipo).lower()) for p in PROTEINAS_APP}
        previsoes.append(float(modelo.predict(np.array([[v[c] for c in COLUNAS_APP]]))[0]))
    return np.array(previsoes)


def diagnosticar_outubro():
    """Reproduz o modelo puro de 26 a 30/10 e mede o efeito das duas falhas encontradas."""
    g = pd.read_csv(ENTRADAS_OUTUBRO / "base_de_treino_em_uso.csv", sep=";")
    g["data"] = pd.to_datetime(g.data_iso)
    app = pd.read_csv(ENTRADAS_OUTUBRO / "consumo_no_aplicativo.csv", sep=";")
    app["data"] = pd.to_datetime(app.data_iso)
    app = app.dropna(subset=["consumo"]).sort_values("data")
    registrado = pd.read_parquet(SILVER / "silver_consumo.parquet").set_index("data").consumo

    h = registro_de_outubro()
    semana = h[(h.emitida_em.dt.normalize() == EMISSAO_FINAL) & h.data.between("2026-10-26", "2026-10-30")]
    semana = semana.sort_values("emitida_em").groupby("data").tail(1).sort_values("data")
    real = registrado.loc[semana.data].to_numpy(dtype=float)

    corrigido = app.copy()
    for dia in pd.to_datetime(["2026-10-19", "2026-10-20"]):
        corrigido.loc[corrigido.data == dia, "consumo"] = registrado.loc[dia]
    colunas_certas = g.assign(consumo_lag_5=g.consumo_mesmo_dia_sem_passada, consumo_lag_10=g.consumo_mesmo_dia_sem_retrasada)

    original = treinar_aplicativo(g)
    certo = treinar_aplicativo(colunas_certas)
    casos = {"Registro de 24/10": semana.ia_pura.to_numpy(),
             "Reproduzido com o aplicativo em uso": prever_semana(original, app, semana),
             "Com os consumos de 19 e 20/10 corrigidos": prever_semana(original, corrigido, semana),
             "Com as colunas de consumo passado corrigidas": prever_semana(certo, app, semana),
             "Com as duas correções": prever_semana(certo, corrigido, semana)}
    linhas = [{"Caso": nome, **{f"{d:%d/%m}": round(p) for d, p in zip(semana.data, previsao)}, "Erro médio": round(float(np.abs(previsao - real).mean()), 1), "Viés": round(float((previsao - real).mean()), 1)} for nome, previsao in casos.items()]
    linhas.append({"Caso": "Consumo real", **{f"{d:%d/%m}": round(p) for d, p in zip(semana.data, real)}})
    tabela = pd.DataFrame(linhas)
    PASTA_REVISAO.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(PASTA_REVISAO / "diagnostico_outubro.csv", sep=";", index=False, encoding="utf-8-sig", decimal=",")
    return tabela
