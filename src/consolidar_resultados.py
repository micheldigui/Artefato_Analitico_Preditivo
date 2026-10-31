"""Fixa o comite campeao e mede o resultado em 2026.

O comitê foi escolhido com 2024 e 2025, sem atipicos. Combinacoes que foram melhores
em 2026 nao substituem a escolhida, porque isso usaria o resultado da prova.

Execute depois de avaliar_janelas_treino.py: python src/consolidar_resultados.py
"""
from pathlib import Path

import pandas as pd

from avaliar_cenarios import combinar_com_plano
from modelagem import CAMPEAO, CENARIO_CAMPEAO, HISTORICO_CAMPEAO, intervalo_diferenca, medir

RAIZ = Path(__file__).resolve().parents[1]
GOLD = RAIZ / "dados" / "gold"
SAIDA = RAIZ / "resultados" / "modelos"
JANELA_C4 = "t1"


def escolher(metricas):
    base = metricas[(metricas["periodo"] == "escolha_2025") & (metricas["universo"] == "sem_atipicos") & (metricas["modelo"] == CAMPEAO) & (metricas["janela_treino"] == HISTORICO_CAMPEAO)]
    return base[["janela", "cenario", "modelo", "janela_treino", "mae"]].reset_index(drop=True)


def montar_tabela_diaria(previsoes, referencia, escolhas, janela):
    ref = referencia[(referencia["janela"] == janela) & (referencia["cenario"] == "C0")]
    tabela = ref[["origem", "data", "consumo", "tratamento", "periodo", "regular"]].assign(consumo_tipico=ref["previsao"])
    for _, e in escolhas[escolhas["janela"] == janela].iterrows():
        filtro = (previsoes["janela"] == janela) & (previsoes["cenario"] == e["cenario"]) & (previsoes["modelo"] == e["modelo"]) & (previsoes["janela_treino"] == e["janela_treino"])
        tabela = tabela.merge(previsoes.loc[filtro, ["data", "previsao"]].rename(columns={"previsao": e["cenario"]}), on="data")
    return tabela


def adicionar_cenario_4(tabela, escolhas, gold_t1):
    melhor = CENARIO_CAMPEAO
    entrada = tabela.assign(previsao=tabela[melhor])
    combinada = combinar_com_plano(entrada, gold_t1)
    regulares = combinada[(combinada["periodo"] == "escolha_2025") & combinada["regular"] & combinada["consumo"].notna()]
    erro = {c: (regulares[c] - regulares["consumo"]).abs().mean() for c in ("previsao_50_50", "previsao_adaptativa")}
    metodo = min(erro, key=erro.get)
    colunas = ["data", "plano", "regime", "peso_modelo", metodo]
    return tabela.merge(combinada[colunas].rename(columns={metodo: "C4"}), on="data", how="left"), melhor, metodo, erro


def medir_alternativas(tabela, janela, colunas, referencia):
    linhas = []
    observados = tabela[tabela["consumo"].notna()]
    for periodo, g in observados.groupby("periodo"):
        for universo, d in (("sem_atipicos", g[g["regular"]]), ("todos_os_dias", g)):
            for coluna in colunas:
                pareado = d.dropna(subset=[coluna, *( [referencia] if referencia in d else [])])
                linha = dict(janela=janela, periodo=periodo, universo=universo, alternativa=coluna, **medir(pareado["consumo"], pareado[coluna]))
                if coluna != referencia:
                    linha["dif_referencia"], linha["ic_inf"], linha["ic_sup"] = intervalo_diferenca(pareado, coluna, referencia)
                linhas.append(linha)
    return pd.DataFrame(linhas)


def comparar_janela_treino(previsoes, escolhas):
    linhas = []
    obs = previsoes[(previsoes["periodo"] == "prova_2026") & previsoes["consumo"].notna() & previsoes["regular"]]
    for _, e in escolhas.iterrows():
        filtro = (obs["janela"] == e["janela"]) & (obs["cenario"] == e["cenario"]) & (obs["modelo"] == e["modelo"])
        largo = obs[filtro].pivot(index="data", columns="janela_treino", values="previsao").reset_index()
        largo = largo.merge(obs[filtro].drop_duplicates("data")[["data", "origem", "consumo"]], on="data")
        for outra in sorted(set(obs["janela_treino"]) - {e["janela_treino"]}):
            dif, inf, sup = intervalo_diferenca(largo, e["janela_treino"], outra)
            linhas.append(dict(janela=e["janela"], cenario=e["cenario"], escolhida=e["janela_treino"], comparada=outra, dif=dif, ic_inf=inf, ic_sup=sup))
    return pd.DataFrame(linhas)


def main():
    metricas = pd.read_csv(SAIDA / "metricas_janelas_treino.csv", sep=";", encoding="utf-8-sig")
    previsoes = pd.read_parquet(SAIDA / "previsoes_janelas_treino.parquet")
    referencia = pd.read_parquet(SAIDA / "previsoes_validacao.parquet")
    gold_t1 = pd.read_parquet(GOLD / "gold_t1.parquet")
    escolhas = escolher(metricas)
    preparo = montar_tabela_diaria(previsoes, referencia, escolhas, "t1")
    preparo, melhor, metodo, erro_metodo = adicionar_cenario_4(preparo, escolhas, gold_t1)
    compras = montar_tabela_diaria(previsoes, referencia, escolhas, "t2")
    resultado = pd.concat([
        medir_alternativas(preparo, "t1", ["plano", "consumo_tipico", "H", "C1", "C2", "C3", "C4"], "plano"),
        medir_alternativas(preparo, "t1_vs_tipico", ["consumo_tipico", "H", "C1", "C2", "C3"], "consumo_tipico"),
        medir_alternativas(compras, "t2", ["consumo_tipico", "H", "C1", "C2"], "consumo_tipico"),
    ], ignore_index=True)
    janelas = comparar_janela_treino(previsoes, escolhas)
    preparo.to_parquet(SAIDA / "preparo_diario_final.parquet", index=False)
    compras.to_parquet(SAIDA / "compras_diario_final.parquet", index=False)
    escolhas.to_csv(SAIDA / "configuracao_escolhida.csv", sep=";", index=False, encoding="utf-8-sig")
    resultado.to_csv(SAIDA / "resultado_final.csv", sep=";", index=False, encoding="utf-8-sig")
    janelas.to_csv(SAIDA / "comparacao_janelas_treino.csv", sep=";", index=False, encoding="utf-8-sig")
    print(escolhas.round(2).to_string(index=False))
    print("base do cenario 4:", melhor, "| metodo:", metodo, {k: round(v, 2) for k, v in erro_metodo.items()})
    cols = ["janela", "universo", "alternativa", "dias", "mae", "rmse", "r2", "vies", "excesso", "deficit", "dias_deficit", "dif_referencia", "ic_inf", "ic_sup"]
    print(resultado[resultado["periodo"] == "prova_2026"][cols].round(2).to_string(index=False))
    print(resultado[(resultado["periodo"] == "escolha_2025") & (resultado["universo"] == "sem_atipicos")][cols].round(2).to_string(index=False))
    print(janelas.round(2).to_string(index=False))
    p = preparo[(preparo["periodo"] == "prova_2026") & preparo["consumo"].notna()].copy()
    p["mes"] = p["data"].dt.strftime("%Y-%m")
    print(p.groupby("mes").apply(lambda d: pd.Series({
        "dias": len(d), **{c: (d[c] - d["consumo"]).abs().mean() for c in ("plano", melhor, "C4")}, "peso_modelo": d["peso_modelo"].mean()})).round(2).to_string())


if __name__ == "__main__":
    main()
