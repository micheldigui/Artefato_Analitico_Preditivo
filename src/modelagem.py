"""Cenarios, modelos e validacao semana a semana.

Em cada sexta-feira do planejamento, o modelo treina somente com dias anteriores a
ela e preve as datas daquela emissao. Os parametros sao fixos: nada e ajustado com
os dados de avaliacao. Os modelos preveem o consumo comparado ao tipico, o que
mantem a escala quando o patamar de refeicoes muda entre os anos.
"""
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HISTORICO = ["tendencia_nivel", "coef_variacao_10d", "idade_informacao_dias", "dias_observados_no_bloco", "rel_ultimo_consumo", "rel_media_5d",
             "rel_nivel_bloco_atual", "rel_mesmo_dia_ha_1_ocorrencias", "rel_mesmo_dia_ha_2_ocorrencias"]
CALENDARIO = ["dia_seg", "dia_ter", "dia_qua", "dia_qui", "dia_sex", "dias_ate_fim_bloco", "dias_desde_inicio_bloco", "semana_do_bloco", "fim_de_bloco",
              "inicio_de_bloco", "vespera_de_pausa", "pos_pausa", "dias_sem_aula_antes", "dias_letivos_na_semana", "evento_especial", "efeito_evento"]
CARDAPIO = ["efeito_proteina", "efeito_preparo", "efeito_prato", "sem_cardapio"]
CLIMA_PREVISTO = ["previsao_chuva_mm", "previsao_chuva_manha_mm", "previsao_chuva_acum3d_mm", "previsao_temp_max"]
CLIMA_OCORRIDO = ["oraculo_chuva_mm", "oraculo_chuva_horas", "oraculo_temp_max", "oraculo_vento_max"]

SAZONALIDADE = ["efeito_posicao_bloco", "mesma_epoca_ano_passado"]

CENARIOS = {
    "C1s": HISTORICO + CALENDARIO + SAZONALIDADE,
    "C2s": HISTORICO + CALENDARIO + CARDAPIO + SAZONALIDADE,
    "H": HISTORICO,
    "C1": HISTORICO + CALENDARIO,
    "C2": HISTORICO + CALENDARIO + CARDAPIO,
    "C3": HISTORICO + CALENDARIO + CARDAPIO + CLIMA_PREVISTO,
    "S_clima_perfeito": HISTORICO + CALENDARIO + CARDAPIO + CLIMA_OCORRIDO,
}
CENARIOS_POR_JANELA = {"t1": ["C1", "C2", "C3", "S_clima_perfeito"], "t2": ["C1", "C2", "S_clima_perfeito"]}
TRATAMENTOS_REGULARES = ("normal", "previsivel_calendario", "alto_mantido")
INICIO_AVALIACAO = "2025-01-01"
INICIO_PROVA = "2026-01-01"
LIMITES_ALVO = (0.2, 2.0)
ANO_ANTERIOR_E_ATUAL = "ano_anterior_e_atual"
# Comite fixado com 2024 e 2025: media de CatBoost, vizinhos parecidos e consumo tipico.
CAMPEAO = "campeao"
CENARIO_CAMPEAO = "C2"
HISTORICO_CAMPEAO = "1_ano"


def construir_modelo(nome):
    if nome == "regressao":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=30.0))
    if nome == "regressao_robusta":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), HuberRegressor(alpha=1.0, epsilon=1.35, max_iter=1000))
    if nome == "arvores":
        return HistGradientBoostingRegressor(loss="absolute_error", max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=20, l2_regularization=1.0, random_state=42)
    if nome == "catboost":
        return CatBoostRegressor(loss_function="MAE", iterations=200, learning_rate=0.05, depth=4, random_seed=42, verbose=0, allow_writing_files=False, thread_count=1)
    if nome == "vizinhos":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), KNeighborsRegressor(n_neighbors=15, weights="distance"))
    raise ValueError(nome)


MODELOS = ["regressao", "regressao_robusta", "arvores", "catboost", "vizinhos"]


def arredondar_refeicoes(valores):
    """Refeicao e unidade inteira: de ,5 para cima arredonda para cima."""
    return np.floor(valores + 0.5)


def matriz(tabela, colunas):
    return tabela[colunas].astype(float)


def dados_de_treino(gold, origem, colunas, meses=None):
    treino = gold[(gold["data"] < origem) & gold["usar_no_treino"] & gold["alvo_relativo"].notna()]
    if meses == ANO_ANTERIOR_E_ATUAL:
        treino = treino[treino["data"] >= pd.Timestamp(year=pd.Timestamp(origem).year - 1, month=1, day=1)]
    elif meses is not None:
        treino = treino[treino["data"] >= origem - pd.DateOffset(months=meses)]
    clima = [c for c in colunas if c.startswith("previsao_")]
    return treino.dropna(subset=clima) if clima else treino


def prever_emissao(gold, origem, colunas, nome_modelo, meses=None):
    """Treina com o passado da emissao e devolve a previsao, em refeicoes."""
    alvo = gold[gold["origem"] == origem]
    if nome_modelo == CAMPEAO:
        return (prever_emissao(gold, origem, colunas, "catboost", meses) + prever_emissao(gold, origem, colunas, "vizinhos", meses) + alvo["nivel_base"]) / 3
    treino = dados_de_treino(gold, origem, colunas, meses)
    assert treino["data"].max() < origem
    modelo = construir_modelo(nome_modelo)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo.fit(matriz(treino, colunas), treino["alvo_relativo"].clip(*LIMITES_ALVO))
    return pd.Series(np.maximum(modelo.predict(matriz(alvo, colunas)), 0) * alvo["nivel_base"].to_numpy(), index=alvo.index)


def prever_serie(gold, origens, colunas, nome, meses):
    return pd.concat([prever_emissao(gold, origem, colunas, nome, meses) for origem in origens])


def validar_janela(gold, janela, meses=None, cenarios=None):
    """Previsoes semana a semana de todos os cenarios e modelos, desde 2025."""
    origens = sorted(gold.loc[gold["data"] >= INICIO_AVALIACAO, "origem"].unique())
    linhas = []
    base = gold.loc[gold["origem"].isin(origens), ["origem", "data", "consumo", "tratamento", "nivel_base"]].copy()
    linhas.append(base.assign(janela=janela, cenario="C0", modelo="consumo_tipico", previsao=base["nivel_base"]))
    combinacoes = [(cenario, nome) for cenario in cenarios or CENARIOS_POR_JANELA[janela] for nome in MODELOS]
    series = Parallel(n_jobs=-2)(delayed(prever_serie)(gold, origens, CENARIOS[c], nome, meses) for c, nome in combinacoes)
    por_modelo = dict(zip(combinacoes, series))
    for (cenario, nome), serie in por_modelo.items():
        linhas.append(base.assign(janela=janela, cenario=cenario, modelo=nome, previsao=serie.reindex(base.index)))
    for cenario in dict(combinacoes):
        media = (por_modelo[cenario, "regressao_robusta"] + por_modelo[cenario, "arvores"]) / 2
        linhas.append(base.assign(janela=janela, cenario=cenario, modelo="media_robusta_arvores", previsao=media.reindex(base.index)))
        campeao = (por_modelo[cenario, "catboost"].reindex(base.index) + por_modelo[cenario, "vizinhos"].reindex(base.index) + base["nivel_base"]) / 3
        linhas.append(base.assign(janela=janela, cenario=cenario, modelo=CAMPEAO, previsao=campeao))
    previsoes = pd.concat(linhas, ignore_index=True)
    return previsoes.assign(previsao=arredondar_refeicoes(previsoes["previsao"]))


def medir(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    erro = p - y
    return {
        "dias": len(y),
        "mae": np.abs(erro).mean(),
        "rmse": np.sqrt((erro ** 2).mean()),
        "wape": np.abs(erro).sum() / y.sum() * 100,
        "r2": 1 - (erro ** 2).sum() / ((y - y.mean()) ** 2).sum(),
        "vies": erro.mean(),
        "excesso": np.maximum(erro, 0).sum(),
        "deficit": np.maximum(-erro, 0).sum(),
        "dias_deficit": (erro < 0).mean() * 100,
    }


def intervalo_diferenca(tabela, coluna_a, coluna_b, reamostras=5000, semente=42):
    """Diferenca de erro medio (A menos B), reamostrando semanas inteiras."""
    t = tabela.assign(ea=(tabela[coluna_a] - tabela["consumo"]).abs(), eb=(tabela[coluna_b] - tabela["consumo"]).abs())
    semanas = t.groupby("origem")[["ea", "eb"]].sum().assign(n=t.groupby("origem").size()).to_numpy()
    gerador = np.random.default_rng(semente)
    sorteio = gerador.integers(0, len(semanas), (reamostras, len(semanas)))
    amostra = semanas[sorteio]
    dif = (amostra[:, :, 0].sum(1) - amostra[:, :, 1].sum(1)) / amostra[:, :, 2].sum(1)
    return t["ea"].mean() - t["eb"].mean(), *np.percentile(dif, [2.5, 97.5])


def peso_adaptativo(passado, coluna_modelo, coluna_plano, janela_dias=40, minimo_dias=10):
    """Peso do modelo que teria dado o menor erro nos dias recentes ja conhecidos."""
    recente = passado.tail(janela_dias)
    if len(recente) < minimo_dias:
        return 0.5
    pesos = np.arange(0, 1.01, 0.1)
    erros = [(p * recente[coluna_modelo] + (1 - p) * recente[coluna_plano] - recente["consumo"]).abs().mean() for p in pesos]
    return float(pesos[int(np.argmin(erros))])
