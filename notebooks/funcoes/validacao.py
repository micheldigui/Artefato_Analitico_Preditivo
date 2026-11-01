"""Walk-forward: treina a cada sexta-feira, mede cenarios e historicos e compara com os planos humanos."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .configuracao import (ANO_ANTERIOR_E_ATUAL, CAMPEAO, CENARIO_CAMPEAO, CENARIOS, CENARIOS_POR_JANELA, GOLD, FIM_PROVA, HISTORICO_CAMPEAO, INICIO_AVALIACAO, INICIO_PROVA,
                          MODELOS_BASE, PASTA_COMPARACAO, PASTA_MODELOS, TRATAMENTOS_REGULARES)
from .modelos import arredondar_refeicoes, peso_adaptativo, prever_emissao, prever_serie


def validar_janela(gold, janela, meses=None, cenarios=None):
    """Previsoes semana a semana de todos os cenarios e modelos, desde 2025."""
    origens = sorted(gold.loc[gold["data"] >= INICIO_AVALIACAO, "origem"].unique())
    linhas = []
    base = gold.loc[gold["origem"].isin(origens), ["origem", "data", "consumo", "tratamento", "nivel_base"]].copy()
    linhas.append(base.assign(janela=janela, cenario="C0", modelo="consumo_tipico", previsao=base["nivel_base"]))
    combinacoes = [(cenario, nome) for cenario in cenarios or CENARIOS_POR_JANELA[janela] for nome in MODELOS_BASE]
    series = Parallel(n_jobs=-2, max_nbytes=None)(delayed(prever_serie)(gold, origens, CENARIOS[c], nome, meses) for c, nome in combinacoes)
    por_modelo = dict(zip(combinacoes, series))
    for (cenario, nome), serie in por_modelo.items():
        linhas.append(base.assign(janela=janela, cenario=cenario, modelo=nome, previsao=serie.reindex(base.index)))
    for cenario in dict(combinacoes):
        media = (por_modelo[cenario, "regressao_robusta"] + por_modelo[cenario, "arvores"]) / 2
        linhas.append(base.assign(janela=janela, cenario=cenario, modelo="media_robusta_arvores", previsao=media.reindex(base.index)))
        comite = (por_modelo[cenario, "catboost"].reindex(base.index) + por_modelo[cenario, "vizinhos"].reindex(base.index) + base["nivel_base"]) / 3
        linhas.append(base.assign(janela=janela, cenario=cenario, modelo="comite_catboost_knn", previsao=comite))
        linhas.append(base.assign(janela=janela, cenario=cenario, modelo=CAMPEAO, previsao=por_modelo[cenario, "extratrees"].reindex(base.index)))
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


def rotular_periodo(previsoes):
    previsoes = previsoes[previsoes["consumo"].notna() | (previsoes["data"] >= INICIO_PROVA)].copy()
    previsoes["periodo"] = np.select([previsoes["data"] < INICIO_PROVA, previsoes["data"] <= FIM_PROVA], ["escolha_2025", "prova_2026"], "prospectiva_set_out")
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
    """O modelo congelado em cada cenario, medido em 2025 sem atipicos."""
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


def avaliar_cenarios():
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    t1 = pd.read_parquet(GOLD / "gold_t1.parquet")
    t2 = pd.read_parquet(GOLD / "gold_t2.parquet")
    previsoes = rotular_periodo(pd.concat([validar_janela(t1, "t1"), validar_janela(t2, "t2")], ignore_index=True))
    metricas = resumir(previsoes)
    escolhas = escolher_modelos(metricas)
    comb, cenario_base, modelo_base = montar_combinacao(previsoes, escolhas, t1)
    humano = comparar_com_humano(comb)
    tipico = comparar_com_tipico(previsoes, escolhas)
    previsoes.to_parquet(PASTA_MODELOS / "previsoes_validacao.parquet", index=False)
    comb.to_parquet(PASTA_MODELOS / "previsoes_combinacao.parquet", index=False)
    for tabela, nome in ((metricas, "metricas_cenarios"), (escolhas, "modelos_escolhidos"), (humano, "comparacao_humano"), (tipico, "comparacao_consumo_tipico")):
        tabela.to_csv(PASTA_MODELOS / f"{nome}.csv", sep=";", index=False, encoding="utf-8-sig")
    print("Base do cenario 4:", cenario_base, modelo_base)
    print(escolhas.round(2).to_string(index=False))


JANELAS_TREINO = {"tudo": None, "2_anos": 24, "1_ano": 12, "6_meses": 6, ANO_ANTERIOR_E_ATUAL: ANO_ANTERIOR_E_ATUAL}


# H mede quanto o calendario acrescenta e nao e opcao de uso. C3 usa previsao do tempo,
# que so existe em compras desde abril de 2026, por isso fica fora dessa janela.
CENARIOS_TESTADOS = {"t1": ["H", "C1", "C2", "C3"], "t2": ["H", "C1", "C2"]}


def avaliar_historicos():
    tabelas = []
    previsoes_todas = []
    for janela in ("t1", "t2"):
        gold = pd.read_parquet(GOLD / f"gold_{janela}.parquet")
        for nome, meses in JANELAS_TREINO.items():
            previsoes = rotular_periodo(validar_janela(gold, janela, meses, CENARIOS_TESTADOS[janela]))
            previsoes = previsoes[previsoes["cenario"] != "C0"].assign(janela_treino=nome)
            previsoes_todas.append(previsoes)
            tabelas.append(resumir(previsoes.drop(columns="janela_treino")).assign(janela_treino=nome))
    metricas = pd.concat(tabelas, ignore_index=True)
    metricas.to_csv(PASTA_MODELOS / "metricas_janelas_treino.csv", sep=";", index=False, encoding="utf-8-sig")
    pd.concat(previsoes_todas, ignore_index=True).to_parquet(PASTA_MODELOS / "previsoes_janelas_treino.parquet", index=False)
    base = metricas[metricas["universo"] == "sem_atipicos"]
    print(base.pivot_table(index=["janela", "cenario", "modelo"], columns=["periodo", "janela_treino"], values="mae").round(2).to_string())


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
    metodo = "previsao_adaptativa"
    colunas = ["data", "plano", "regime", "peso_modelo", metodo]
    return tabela.merge(combinada[colunas].rename(columns={metodo: "C4"}), on="data", how="left"), melhor, metodo, erro


def adicionar_mesmo_dia(tabela, gold):
    """Linha de base simples: mediana das ultimas ocorrencias do mesmo dia da semana."""
    t = tabela.merge(gold[["data", "nivel_mesmo_dia"]], on="data", how="left")
    return t.assign(mesmo_dia=arredondar_refeicoes(t["nivel_mesmo_dia"])).drop(columns="nivel_mesmo_dia")


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


def consolidar_resultados():
    metricas = pd.read_csv(PASTA_MODELOS / "metricas_janelas_treino.csv", sep=";", encoding="utf-8-sig")
    previsoes = pd.read_parquet(PASTA_MODELOS / "previsoes_janelas_treino.parquet")
    referencia = pd.read_parquet(PASTA_MODELOS / "previsoes_validacao.parquet")
    gold_t1 = pd.read_parquet(GOLD / "gold_t1.parquet")
    escolhas = escolher(metricas)
    preparo = montar_tabela_diaria(previsoes, referencia, escolhas, "t1")
    preparo, melhor, metodo, erro_metodo = adicionar_cenario_4(preparo, escolhas, gold_t1)
    preparo = adicionar_mesmo_dia(preparo, gold_t1)
    compras = adicionar_mesmo_dia(montar_tabela_diaria(previsoes, referencia, escolhas, "t2"), pd.read_parquet(GOLD / "gold_t2.parquet"))
    resultado = pd.concat([
        medir_alternativas(preparo, "t1", ["plano", "consumo_tipico", "mesmo_dia", "H", "C1", "C2", "C3", "C4"], "plano"),
        medir_alternativas(preparo, "t1_vs_tipico", ["consumo_tipico", "H", "C1", "C2", "C3"], "consumo_tipico"),
        medir_alternativas(preparo, "t1_vs_mesmo_dia", ["mesmo_dia", "C2", "C4"], "mesmo_dia"),
        medir_alternativas(compras, "t2", ["consumo_tipico", "mesmo_dia", "H", "C1", "C2"], "consumo_tipico"),
        medir_alternativas(compras, "t2_vs_mesmo_dia", ["mesmo_dia", "C2"], "mesmo_dia"),
    ], ignore_index=True)
    janelas = comparar_janela_treino(previsoes, escolhas)
    preparo.to_parquet(PASTA_MODELOS / "preparo_diario_final.parquet", index=False)
    compras.to_parquet(PASTA_MODELOS / "compras_diario_final.parquet", index=False)
    escolhas.to_csv(PASTA_MODELOS / "configuracao_escolhida.csv", sep=";", index=False, encoding="utf-8-sig")
    resultado.to_csv(PASTA_MODELOS / "resultado_final.csv", sep=";", index=False, encoding="utf-8-sig")
    janelas.to_csv(PASTA_MODELOS / "comparacao_janelas_treino.csv", sep=";", index=False, encoding="utf-8-sig")
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


INICIO = "2024-01-01"


MESES = {"tudo": None, "2_anos": 24, "1_ano": 12, "6_meses": 6, "ano_anterior_e_atual": "ano_anterior_e_atual"}


def prever_configuracao(gold, cenario, modelo, janela_treino):
    origens = sorted(gold.loc[gold["data"] >= INICIO, "origem"].unique())
    colunas, meses = CENARIOS[cenario], MESES[janela_treino]
    nomes = ["regressao_robusta", "arvores"] if modelo == "media_robusta_arvores" else [modelo]
    previsoes = [pd.concat([prever_emissao(gold, o, colunas, n, meses) for o in origens]) for n in nomes]
    return arredondar_refeicoes(sum(previsoes) / len(previsoes))


def montar_base(gold, cenario, modelo, janela_treino):
    base = gold[["origem", "data", "ano", "consumo", "tratamento", "nivel_base", "sem_cardapio"]].copy()
    base["nivel_base"] = arredondar_refeicoes(base["nivel_base"])
    base["previsao"] = prever_configuracao(gold, cenario, modelo, janela_treino).reindex(base.index)
    base["regular"] = base["tratamento"].isin(TRATAMENTOS_REGULARES)
    return base


def adicionar_planos(preparo, gold_t1):
    combinada = combinar_com_plano(preparo.dropna(subset=["previsao"]), gold_t1)
    planos = gold_t1[["data", "plano_nutricionista", "reserva_discente"]]
    t = preparo.merge(planos, on="data", how="left")
    t["plano"] = t["plano_nutricionista"].fillna(t["reserva_discente"])
    return t.merge(combinada[["data", "previsao_adaptativa", "peso_modelo"]], on="data", how="left").rename(columns={"previsao_adaptativa": "combinacao"})


def medir_por_ano(t):
    alternativas = {"plano": "plano", "consumo_tipico": "nivel_base", "artefato": "previsao", "combinacao": "combinacao"}
    linhas = []
    obs = t[t["consumo"].notna() & t["plano"].notna() & (t["data"] <= FIM_PROVA)]
    for ano, g in obs.groupby("ano"):
        for universo, d in (("sem_atipicos", g[g["regular"]]), ("todos_os_dias", g)):
            for nome, coluna in alternativas.items():
                pareado = d.dropna(subset=[coluna])
                if len(pareado) != len(d):
                    continue
                linha = dict(ano=ano, regime="Reservas dos alunos" if ano < 2026 else "Quantitativo da nutricionista", universo=universo, alternativa=nome, **medir(d["consumo"], d[coluna]))
                if nome != "plano":
                    linha["dif_plano"], linha["ic_inf"], linha["ic_sup"] = intervalo_diferenca(d, coluna, "plano")
                linhas.append(linha)
    return pd.DataFrame(linhas)


def comparar_planejamentos():
    PASTA_COMPARACAO.mkdir(parents=True, exist_ok=True)
    escolhas = pd.read_csv(PASTA_MODELOS / "configuracao_escolhida.csv", sep=";", encoding="utf-8-sig")
    g1 = pd.read_parquet(GOLD / "gold_t1.parquet")
    g2 = pd.read_parquet(GOLD / "gold_t2.parquet")
    opcoes = escolhas[escolhas["cenario"] == CENARIO_CAMPEAO]
    linha_t1 = opcoes[opcoes["janela"] == "t1"].iloc[0]
    linha_t2 = opcoes[opcoes["janela"] == "t2"].iloc[0]
    preparo = adicionar_planos(montar_base(g1, linha_t1["cenario"], linha_t1["modelo"], linha_t1["janela_treino"]), g1)
    compras = montar_base(g2, linha_t2["cenario"], linha_t2["modelo"], linha_t2["janela_treino"])
    anual = medir_por_ano(preparo)
    preparo.to_parquet(PASTA_COMPARACAO / "serie_preparo.parquet", index=False)
    compras.to_parquet(PASTA_COMPARACAO / "serie_compras.parquet", index=False)
    anual.to_csv(PASTA_COMPARACAO / "comparacao_anual.csv", sep=";", index=False, encoding="utf-8-sig")
    print(anual[anual["universo"] == "sem_atipicos"][["ano", "alternativa", "dias", "mae", "vies", "excesso", "deficit", "dias_deficit", "dif_plano", "ic_inf", "ic_sup"]].round(2).to_string(index=False))
