"""Executa a validacao, escolhe os modelos em 2025 e avalia em 2026.

Execute depois de testar_gold.py: python src/avaliar_cenarios.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from modelagem import (CAMPEAO, CENARIO_CAMPEAO, INICIO_PROVA, TRATAMENTOS_REGULARES, arredondar_refeicoes, intervalo_diferenca, medir, peso_adaptativo, validar_janela)

RAIZ = Path(__file__).resolve().parents[1]
GOLD = RAIZ / "dados" / "gold"
SAIDA = RAIZ / "resultados" / "modelos"


def rotular_periodo(previsoes):
    previsoes = previsoes[previsoes["consumo"].notna() | (previsoes["data"] >= INICIO_PROVA)].copy()
    previsoes["periodo"] = np.where(previsoes["data"] < INICIO_PROVA, "escolha_2025", "prova_2026")
    previsoes["regular"] = previsoes["tratamento"].isin(TRATAMENTOS_REGULARES)
    return previsoes


def resumir(previsoes):
    linhas = []
    observados = previsoes[previsoes["consumo"].notna()]
    for chaves, g in observados.groupby(["janela", "periodo", "cenario", "modelo"]):
        for universo, d in (("sem_atipicos", g[g["regular"]]), ("todos_os_dias", g)):
            linhas.append(dict(zip(["janela", "periodo", "cenario", "modelo"], chaves), universo=universo, **medir(d["consumo"], d["previsao"])))
    return pd.DataFrame(linhas)


def escolher_modelos(metricas):
    """A receita campea em cada cenario, medida em 2025 sem atipicos."""
    base = metricas[(metricas["periodo"] == "escolha_2025") & (metricas["universo"] == "sem_atipicos") & (metricas["modelo"] == CAMPEAO)]
    return base[["janela", "cenario", "modelo", "mae"]].reset_index(drop=True)


def montar_combinacao(previsoes, escolhas, gold_t1):
    """Cenario 4: combina a previsao campea com o numero humano."""
    p = previsoes[(previsoes["janela"] == "t1") & (previsoes["cenario"] == CENARIO_CAMPEAO) & (previsoes["modelo"] == CAMPEAO)]
    return combinar_com_plano(p, gold_t1), CENARIO_CAMPEAO, CAMPEAO


def combinar_com_plano(p, gold_t1):
    t = p.merge(gold_t1[["data", "plano_nutricionista", "reserva_discente"]], on="data")
    t["plano"] = t["plano_nutricionista"].fillna(t["reserva_discente"])
    t["regime"] = np.where(t["data"] < INICIO_PROVA, "reservas", "nutricionista")
    t = t.dropna(subset=["plano"]).sort_values("data")
    t["previsao_50_50"] = arredondar_refeicoes((t["previsao"] + t["plano"]) / 2)
    pesos = []
    for origem, grupo in t.groupby("origem"):
        passado = t[(t["regime"] == grupo["regime"].iloc[0]) & (t["data"] < origem) & t["consumo"].notna() & t["regular"]]
        pesos.append((origem, peso_adaptativo(passado, "previsao", "plano")))
    t["peso_modelo"] = t["origem"].map(dict(pesos))
    t["previsao_adaptativa"] = arredondar_refeicoes(t["peso_modelo"] * t["previsao"] + (1 - t["peso_modelo"]) * t["plano"])
    return t


def comparar_com_humano(comb):
    linhas = []
    observados = comb[comb["consumo"].notna()]
    for (periodo, regime), g in observados.groupby(["periodo", "regime"]):
        for universo, d in (("sem_atipicos", g[g["regular"]]), ("todos_os_dias", g)):
            for nome, coluna in (("humano", "plano"), ("algoritmo", "previsao"), ("combinacao_50_50", "previsao_50_50"), ("combinacao_adaptativa", "previsao_adaptativa")):
                linha = dict(periodo=periodo, regime=regime, universo=universo, alternativa=nome, **medir(d["consumo"], d[coluna]))
                if nome != "humano":
                    linha["dif_para_humano"], linha["ic_inf"], linha["ic_sup"] = intervalo_diferenca(d, coluna, "plano")
                linhas.append(linha)
    return pd.DataFrame(linhas)


def comparar_com_tipico(previsoes, escolhas):
    linhas = []
    observados = previsoes[previsoes["consumo"].notna() & (previsoes["periodo"] == "prova_2026")]
    for _, e in escolhas.iterrows():
        escolhido = observados[(observados["janela"] == e["janela"]) & (observados["cenario"] == e["cenario"]) & (observados["modelo"] == e["modelo"])]
        tipico = observados[(observados["janela"] == e["janela"]) & (observados["cenario"] == "C0")]
        j = escolhido.merge(tipico[["data", "previsao"]], on="data", suffixes=("", "_tipico"))
        for universo, d in (("sem_atipicos", j[j["regular"]]), ("todos_os_dias", j)):
            dif, inf, sup = intervalo_diferenca(d, "previsao", "previsao_tipico")
            linhas.append(dict(janela=e["janela"], cenario=e["cenario"], modelo=e["modelo"], universo=universo, dif_para_tipico=dif, ic_inf=inf, ic_sup=sup))
    return pd.DataFrame(linhas)


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    t1 = pd.read_parquet(GOLD / "gold_t1.parquet")
    t2 = pd.read_parquet(GOLD / "gold_t2.parquet")
    previsoes = rotular_periodo(pd.concat([validar_janela(t1, "t1"), validar_janela(t2, "t2")], ignore_index=True))
    metricas = resumir(previsoes)
    escolhas = escolher_modelos(metricas)
    comb, cenario_base, modelo_base = montar_combinacao(previsoes, escolhas, t1)
    humano = comparar_com_humano(comb)
    tipico = comparar_com_tipico(previsoes, escolhas)
    previsoes.to_parquet(SAIDA / "previsoes_validacao.parquet", index=False)
    comb.to_parquet(SAIDA / "previsoes_combinacao.parquet", index=False)
    for tabela, nome in ((metricas, "metricas_cenarios"), (escolhas, "modelos_escolhidos"), (humano, "comparacao_humano"), (tipico, "comparacao_consumo_tipico")):
        tabela.to_csv(SAIDA / f"{nome}.csv", sep=";", index=False, encoding="utf-8-sig")
    print("Base do cenario 4:", cenario_base, modelo_base)
    print(escolhas.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
