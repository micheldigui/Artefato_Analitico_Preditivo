"""Torneio de algoritmos e ideias de melhoria. Tudo so com o que se sabia em cada sexta-feira."""
import logging
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from joblib import Parallel, delayed
from prophet import Prophet
from scipy.optimize import nnls
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .configuracao import ANO_ANTERIOR_E_ATUAL, CAMPEAO, CENARIO_CAMPEAO, CENARIOS, FIM_ESCOLHA, FIM_PROSPECTIVA, FIM_PROVA, GOLD, INICIO_PROSPECTIVA, INICIO_PROVA, LIMITES_ALVO, PASTA_MODELOS, TRATAMENTOS_REGULARES
from .margens import aplicar_folgas, aplicar_margem_contexto
from .modelos import ALGORITMOS, arredondar_refeicoes, construir_modelo, dados_de_treino, matriz, peso_adaptativo, prever_serie
from .validacao import combinar_com_plano, medir


INICIO = "2024-01-01"


HISTORICOS = {"1_ano": 12, "tudo": None}


def prever_torneio(gold, janela, cenario, algoritmo, historico):
    colunas = CENARIOS[cenario]
    meses = HISTORICOS[historico]
    origens = sorted(gold.loc[gold["data"] >= INICIO, "origem"].unique())
    saidas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for origem in origens:
            alvo = gold[gold["origem"] == origem]
            treino = dados_de_treino(gold, origem, colunas, meses)
            assert treino["data"].max() < origem
            modelo = ALGORITMOS[algoritmo]()
            modelo.fit(matriz(treino, colunas), treino["alvo_relativo"].clip(*LIMITES_ALVO))
            previsao = np.maximum(modelo.predict(matriz(alvo, colunas)), 0) * alvo["nivel_base"].to_numpy()
            saidas.append(pd.DataFrame({"origem": origem, "data": alvo["data"].to_numpy(), "previsao": previsao}))
    return pd.concat(saidas).assign(janela=janela, cenario=cenario, algoritmo=algoritmo, historico=historico)


def rodar_torneio():
    golds = {j: pd.read_parquet(GOLD / f"gold_{j}.parquet") for j in ("t1", "t2")}
    tarefas = [(j, c, a, h) for j in golds for c in ("C1", "C2") for a in ALGORITMOS for h in HISTORICOS]
    print(len(tarefas), "combinacoes")
    resultados = Parallel(n_jobs=-2, verbose=1)(delayed(prever_torneio)(golds[j], j, c, a, h) for j, c, a, h in tarefas)
    previsoes = pd.concat(resultados, ignore_index=True)
    previsoes.to_parquet(PASTA_MODELOS / "torneio_previsoes.parquet", index=False)
    base = {j: g[["data", "consumo", "tratamento", "ano"]] for j, g in golds.items()}
    linhas = []
    for (j, c, a, h), g in previsoes.groupby(["janela", "cenario", "algoritmo", "historico"]):
        d = g.merge(base[j], on="data")
        d = d[d["consumo"].notna() & d["tratamento"].isin(TRATAMENTOS_REGULARES) & (d["data"] <= FIM_ESCOLHA)]
        e = (d["previsao"] - d["consumo"]).abs()
        linhas.append({"janela": j, "cenario": c, "algoritmo": a, "historico": h, "erro_2024": e[d["ano"] == 2024].mean(), "erro_2025": e[d["ano"] == 2025].mean()})
    t = pd.DataFrame(linhas)
    t["erro_medio_2024_2025"] = (t["erro_2024"] + t["erro_2025"]) / 2
    t.to_csv(PASTA_MODELOS / "torneio_ate_2025.csv", sep=";", index=False, encoding="utf-8-sig")
    print(t.sort_values(["janela", "erro_medio_2024_2025"]).groupby("janela").head(14).round(2).to_string(index=False))


MEIA_VIDA_DIAS = 180


JANELA_VIES_DIAS = 40


EXPERIMENTAIS = {**CENARIOS, "C2p": CENARIOS["C2"] + ["rel_plano"]}


def prever_variante(gold, janela, cenario, algoritmo, meses, meia_vida):
    colunas = EXPERIMENTAIS[cenario]
    saidas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for origem in sorted(gold.loc[gold["data"] >= INICIO, "origem"].unique()):
            alvo = gold[gold["origem"] == origem]
            treino = dados_de_treino(gold, origem, colunas, meses)
            modelo = ALGORITMOS[algoritmo]()
            X, y = matriz(treino, colunas), treino["alvo_relativo"].clip(*LIMITES_ALVO)
            # Sem ano anterior no treino, a medida fica neutra: igual ao tipico.
            vazias = X.columns[X.isna().all()]
            X[vazias] = 1.0
            alvo = alvo.assign(**{c: 1.0 for c in vazias})
            if meia_vida is None:
                modelo.fit(X, y)
            else:
                pesos = 0.5 ** ((origem - treino["data"]).dt.days / meia_vida)
                etapa = list(modelo.named_steps)[-1] if hasattr(modelo, "named_steps") else None
                modelo.fit(X, y, **{f"{etapa}__sample_weight": pesos} if etapa else {"sample_weight": pesos})
            previsao = np.maximum(modelo.predict(matriz(alvo, colunas)), 0) * alvo["nivel_base"].to_numpy()
            saidas.append(pd.DataFrame({"origem": origem, "data": alvo["data"].to_numpy(), "previsao": previsao}))
    return pd.concat(saidas).assign(janela=janela, cenario=cenario, algoritmo=algoritmo, historico="recencia" if meia_vida else ("1_ano" if meses else "tudo"))


def erro_medio(prev, base):
    d = prev.merge(base, on="data")
    d = d[d["consumo"].notna() & d["tratamento"].isin(TRATAMENTOS_REGULARES) & (d["data"] <= FIM_ESCOLHA)]
    e = (d["previsao"] - d["consumo"]).abs()
    return e[d["ano"] == 2024].mean(), e[d["ano"] == 2025].mean()


def resumir_ideia(nome, janela, prev, base):
    a, b = erro_medio(prev, base[janela])
    return {"janela": janela, "ideia": nome, "erro_2024": a, "erro_2025": b, "media": (a + b) / 2}


def com_plano(serie, gold, nome_modelo):
    """Formas de juntar a previsao e o plano, usando em cada sexta so o que ja era conhecido."""
    t = gold[["origem", "data", "consumo", "tratamento", "nivel_base", "plano"]].merge(serie.rename("previsao").reset_index(), on="data").dropna(subset=["plano"]).sort_values("data")
    regulares = t[t["tratamento"].isin(TRATAMENTOS_REGULARES) & t["consumo"].notna()]
    linhas = []
    for origem, grupo in t.groupby("origem"):
        passado = regulares[regulares["data"] < origem]
        recente = passado.tail(JANELA_VIES_DIAS)
        fator = (recente["consumo"] / recente["plano"]).median() if len(recente) >= 10 else 1.0
        peso = peso_adaptativo(passado, "previsao", "plano")
        linhas.append(grupo.assign(plano_corrigido=grupo["plano"] * fator, peso=peso))
    t = pd.concat(linhas)
    opcoes = {
        "Só o plano": t["plano"],
        "Plano com correção de viés": t["plano_corrigido"],
        f"Só {nome_modelo}": t["previsao"],
        f"50/50 {nome_modelo} + plano": (t["previsao"] + t["plano"]) / 2,
        f"50/50 {nome_modelo} + plano corrigido": (t["previsao"] + t["plano_corrigido"]) / 2,
        f"Adaptativa {nome_modelo} + plano": t["peso"] * t["previsao"] + (1 - t["peso"]) * t["plano"],
        f"Média de três {nome_modelo} + plano + referência": (t["previsao"] + t["plano"] + t["nivel_base"]) / 3,
    }
    return {k: t[["data"]].assign(previsao=v) for k, v in opcoes.items()}


def testar_melhorias():
    golds = {j: pd.read_parquet(GOLD / f"gold_{j}.parquet") for j in ("t1", "t2")}
    preparo = golds["t1"]
    preparo["plano"] = preparo["plano_nutricionista"].fillna(preparo["reserva_discente"])
    preparo["rel_plano"] = preparo["plano"] / preparo["nivel_base"]
    base = {j: g[["data", "consumo", "tratamento", "ano"]] for j, g in golds.items()}
    torneio = pd.read_parquet(PASTA_MODELOS / "torneio_previsoes.parquet")
    atual = pd.read_csv(PASTA_MODELOS / "torneio_ate_2025.csv", sep=";", encoding="utf-8-sig")
    linhas = []

    # Regra simples
    for j, g in golds.items():
        linhas.append(resumir_ideia("Consumo de referência", j, g[g["data"] >= INICIO][["data"]].assign(previsao=g["nivel_base"]), base))

    # Melhores do torneio por janela
    top = {j: atual[atual["janela"] == j].sort_values("erro_medio_2024_2025").head(8) for j in golds}
    for j, t in top.items():
        for _, l in t.head(3).iterrows():
            linhas.append({"janela": j, "ideia": f"Torneio: {l['algoritmo']} {l['cenario']} {l['historico']}", "erro_2024": l["erro_2024"], "erro_2025": l["erro_2025"], "media": l["erro_medio_2024_2025"]})

    # Medias simples entre pares do mesmo cenario e historico
    largas = {}
    for j, t in top.items():
        chaves = [tuple(x) for x in t[["cenario", "algoritmo", "historico"]].to_numpy()]
        q = torneio[torneio["janela"] == j]
        series = {k: q[(q["cenario"] == k[0]) & (q["algoritmo"] == k[1]) & (q["historico"] == k[2])].set_index("data")["previsao"] for k in chaves}
        largas[j] = series
        for a, b in combinations(chaves, 2):
            media = ((series[a] + series[b]) / 2).rename("previsao").reset_index()
            linhas.append(resumir_ideia(f"Média: {a[1]} {a[0]} {a[2]} + {b[1]} {b[0]} {b[2]}", j, media, base))
        for n in (3, 5):
            media = (sum(series[k] for k in chaves[:n]) / n).rename("previsao").reset_index()
            linhas.append(resumir_ideia(f"Média dos {n} melhores", j, media, base))
        tipico = golds[j].set_index("data")["nivel_base"]
        melhor = series[chaves[0]]
        for peso in (0.25, 0.5):
            mix = ((1 - peso) * melhor + peso * tipico.reindex(melhor.index)).rename("previsao").reset_index()
            linhas.append(resumir_ideia(f"Melhor com {peso:.0%} do consumo de referência", j, mix, base))

    # Atributos de sazonalidade e recencia, nos algoritmos mais fortes
    fortes = ["Ridge", "Huber", "HistGB", "RandomForest", "ExtraTrees", "GradientBoosting"]
    tarefas = [(j, c, a, 12, None) for j in golds for c in ("C1s", "C2s") for a in fortes]
    tarefas += [(j, c, a, None, MEIA_VIDA_DIAS) for j in golds for c in ("C1", "C2", "C1s", "C2s") for a in fortes]
    tarefas += [("t1", "C2p", a, 12, None) for a in fortes]
    extras = Parallel(n_jobs=-2, max_nbytes=None)(delayed(prever_variante)(golds[j], j, c, a, m, v) for j, c, a, m, v in tarefas)
    previsoes_extras = pd.concat(extras, ignore_index=True)
    previsoes_extras.to_parquet(PASTA_MODELOS / "experimentos_previsoes.parquet", index=False)
    for (j, c, a, h), g in previsoes_extras.groupby(["janela", "cenario", "algoritmo", "historico"]):
        linhas.append(resumir_ideia(f"{a} {c} {h}", j, g[["data", "previsao"]], base))

    t = pd.DataFrame(linhas)
    t.to_csv(PASTA_MODELOS / "experimentos_ate_2025.csv", sep=";", index=False, encoding="utf-8-sig")
    print(t.sort_values(["janela", "media"]).groupby("janela").head(16).round(2).to_string(index=False))
    print(t[t["ideia"] == "Consumo de referência"].round(2).to_string(index=False))

    # Combinacoes com o plano, todas medidas nas mesmas datas
    q = torneio[torneio["janela"] == "t1"]
    candidatos = {"média dos 3 melhores": sum(largas["t1"][k] for k in list(largas["t1"])[:3]) / 3}
    for nome, cenario, algoritmo in (("CatBoost", "C1", "CatBoost"), ("Huber", "C2", "Huber"), ("KNN", "C2", "KNN")):
        candidatos[nome] = q[(q["cenario"] == cenario) & (q["algoritmo"] == algoritmo) & (q["historico"] == "1_ano")].set_index("data")["previsao"]
    planos = [resumir_ideia(ideia, "t1", prev, base) for nome, serie in candidatos.items() for ideia, prev in com_plano(serie, golds["t1"], nome).items()]
    p = pd.DataFrame(planos).drop_duplicates("ideia")
    p.to_csv(PASTA_MODELOS / "experimentos_plano_ate_2025.csv", sep=";", index=False, encoding="utf-8-sig")
    print(p.sort_values("media").round(2).to_string(index=False))


BLOCO = ["dias_observados_no_bloco", "idade_informacao_dias", "semana_do_bloco", "dias_desde_inicio_bloco", "dias_sem_aula_antes", "dias_ate_fim_bloco", "fim_de_bloco", "inicio_de_bloco"]


ENXUTAS = [c for c in CENARIOS["C2"] if c not in BLOCO]


def ajustar(gold, colunas, alpha, absoluto, limitar, meses):
    saidas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for origem in sorted(gold.loc[gold["data"] >= INICIO, "origem"].unique()):
            alvo = gold[gold["origem"] == origem]
            treino = dados_de_treino(gold, origem, colunas, meses)
            X, Xa = matriz(treino, colunas), matriz(alvo, colunas)
            if limitar:
                baixo, alto = X.quantile(0.01), X.quantile(0.99)
                X, Xa = X.clip(baixo, alto, axis=1), Xa.clip(baixo, alto, axis=1)
            modelo = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=alpha))
            if absoluto:
                X, Xa = X.assign(nivel_base=treino["nivel_base"].to_numpy()), Xa.assign(nivel_base=alvo["nivel_base"].to_numpy())
                modelo.fit(X, treino["consumo"])
                p = modelo.predict(Xa)
            else:
                modelo.fit(X, treino["alvo_relativo"].clip(*LIMITES_ALVO))
                p = modelo.predict(Xa)
                if limitar:
                    p = np.clip(p, 0.5, 1.4)
                p = p * alvo["nivel_base"].to_numpy()
            saidas.append(pd.Series(np.maximum(p, 0), index=alvo["data"].to_numpy()))
    return pd.concat(saidas)


VARIANTES = {
    "Atual: relativo, 29 informacoes, alpha 30": (CENARIOS["C2"], 30, False, False),
    "Relativo, alpha 300": (CENARIOS["C2"], 300, False, False),
    "Relativo, limitando extremos": (CENARIOS["C2"], 30, False, True),
    "Relativo, limitando extremos, alpha 300": (CENARIOS["C2"], 300, False, True),
    "Relativo, sem medidas de bloco": (ENXUTAS, 30, False, False),
    "Relativo, sem medidas de bloco, limitando": (ENXUTAS, 30, False, True),
    "Relativo, sem bloco, limitando, alpha 300": (ENXUTAS, 300, False, True),
    "Numero absoluto, 29 informacoes": (CENARIOS["C2"], 30, True, False),
    "Numero absoluto, limitando extremos": (CENARIOS["C2"], 30, True, True),
    "Numero absoluto, sem medidas de bloco": (ENXUTAS, 30, True, False),
}


def medir_variante(prev, gold):
    b = gold.set_index("data")
    d = pd.DataFrame({"p": prev}).join(b[["consumo", "tratamento", "ano"]])
    d = d[d.consumo.notna() & d.tratamento.isin(TRATAMENTOS_REGULARES)]
    e = (np.floor(d.p + 0.5) - d.consumo).abs()
    a, c, f = e[d.ano == 2024].mean(), e[d.ano == 2025].mean(), e[(d.ano == 2026) & (d.index <= FIM_PROVA)].mean()
    return {"2024": round(a, 2), "2025": round(c, 2), "media_24_25": round((a + c) / 2, 2), "2026": round(f, 2)}


def testar_ajustes_ridge():
    golds = {j: pd.read_parquet(GOLD / f"gold_{j}.parquet") for j in ("t1", "t2")}
    T = pd.read_parquet(PASTA_MODELOS / "torneio_previsoes.parquet")
    tarefas = [(j, n, m) for j in golds for n in VARIANTES for m in (12, None)]
    series = Parallel(n_jobs=-2, max_nbytes=None)(delayed(ajustar)(golds[j], *VARIANTES[n], m) for j, n, m in tarefas)
    res = dict(zip(tarefas, series))
    for j, g in golds.items():
        print("\n===", j)
        linhas = [{"variante": n, "historico": "12m" if m else "tudo", **medir_variante(res[j2, n, m], g)} for j2, n, m in tarefas if j2 == j]
        for nome, f in (("CatBoost C2", ("CatBoost",)), ("KNN C2", ("KNN",))):
            q = T[(T.janela == j) & (T.cenario == "C2") & (T.historico == "1_ano") & (T.algoritmo == f[0])].set_index("data")["previsao"]
            linhas.append({"variante": nome, "historico": "12m", **medir_variante(q, g)})
        cat = T[(T.janela == j) & (T.cenario == "C2") & (T.historico == "1_ano") & (T.algoritmo == "CatBoost")].set_index("data")["previsao"]
        knn = T[(T.janela == j) & (T.cenario == "C2") & (T.historico == "1_ano") & (T.algoritmo == "KNN")].set_index("data")["previsao"]
        tip = g.set_index("data")["nivel_base"].reindex(cat.index)
        linhas.append({"variante": "COMITE ATUAL", "historico": "12m", **medir_variante((cat + knn + tip) / 3, g)})
        linhas.append({"variante": "Consumo de referencia", "historico": "-", **medir_variante(tip, g)})
        print(pd.DataFrame(linhas).sort_values("media_24_25").to_string(index=False))


QTDS = (3, 5, 8, 12, 16, 20, 29)




def prever_com_quantidade(gold, algoritmo, k):
    colunas = CENARIOS["C2"]
    saidas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for origem in sorted(gold.loc[gold["data"] >= INICIO, "origem"].unique()):
            alvo = gold[gold["origem"] == origem]
            treino = dados_de_treino(gold, origem, colunas, 12)
            y = treino["alvo_relativo"].clip(*LIMITES_ALVO)
            X = matriz(treino, colunas)
            forca = X.apply(lambda c: abs(c.corr(y, method="spearman")) if c.nunique() > 1 else 0.0).fillna(0)
            escolhidas = list(forca.sort_values(ascending=False).index[:k])
            modelo = construir_modelo(algoritmo)
            modelo.fit(X[escolhidas], y)
            p = np.maximum(modelo.predict(matriz(alvo, escolhidas)), 0) * alvo["nivel_base"].to_numpy()
            saidas.append(pd.Series(p, index=alvo["data"].to_numpy()))
    return pd.concat(saidas)


def medir_quantidade(prev, gold):
    b = gold.set_index("data")
    d = pd.DataFrame({"p": prev}).join(b[["consumo", "tratamento", "ano"]])
    d = d[d.consumo.notna() & d.tratamento.isin(TRATAMENTOS_REGULARES)]
    e = (np.floor(d.p + 0.5) - d.consumo).abs()
    a, c, f = e[d.ano == 2024].mean(), e[d.ano == 2025].mean(), e[(d.ano == 2026) & (d.index <= FIM_PROVA)].mean()
    return {"2024": round(a, 1), "2025": round(c, 1), "media_24_25": round((a + c) / 2, 2), "2026": round(f, 1)}


def testar_quantidade_preditoras():
    golds = {j: pd.read_parquet(GOLD / f"gold_{j}.parquet") for j in ("t1", "t2")}
    tarefas = [(j, a, k) for j in golds for a in ("regressao", "vizinhos", "catboost") for k in QTDS]
    series = Parallel(n_jobs=-2, max_nbytes=None)(delayed(prever_com_quantidade)(golds[j], a, k) for j, a, k in tarefas)
    res = dict(zip(tarefas, series))
    for j, g in golds.items():
        print("\n===", j)
        print(pd.DataFrame([{"algoritmo": a, "k": k, **medir_quantidade(res[j2, a, k], g)} for j2, a, k in tarefas if j2 == j]).to_string(index=False))
        tip = g.set_index("data")["nivel_base"]
        print("comites (mesmo k):")
        l = []
        for k in QTDS:
            for nome, partes in (("Cat+KNN+ref", ("catboost", "vizinhos")), ("Cat+Ridge+ref", ("catboost", "regressao")), ("Cat+KNN+Ridge+ref", ("catboost", "vizinhos", "regressao"))):
                s = [res[j, a, k] for a in partes]
                p = (sum(s) + tip.reindex(s[0].index)) / (len(s) + 1)
                l.append({"comite": nome, "k": k, **medir_quantidade(p, g)})
        print(pd.DataFrame(l).sort_values("media_24_25").head(12).to_string(index=False))


PERIODOS_RIDGE = {"2024 e 2025": ("2024-01-01", "2025-12-31"), "Fevereiro a agosto de 2026": (INICIO_PROVA, FIM_PROVA), "Setembro e outubro de 2026": (INICIO_PROSPECTIVA, FIM_PROSPECTIVA), "Fevereiro a outubro de 2026": (INICIO_PROVA, FIM_PROSPECTIVA)}


def politicas_de_modelo(gold, janela, modelo):
    """Mesmo caminho do modelo congelado: 12 meses, regra do cardapio, hibrido e margens por contexto."""
    origens = origens_de(gold)
    t = gold[gold["origem"].isin(origens)][["origem", "data", "consumo", "tratamento", "sem_cardapio"]].copy()
    t["regular"] = t["tratamento"].isin(TRATAMENTOS_REGULARES)
    for cenario in ("C1", "C2"):
        t[cenario] = arredondar_refeicoes(prever_serie(gold, origens, CENARIOS[cenario], modelo, 12).reindex(t.index))
    t["regra"] = t["C1"].where(t["sem_cardapio"], t["C2"])
    if janela == "t1":
        hibrido = combinar_com_plano(t.assign(previsao=t["C2"]), gold)[["data", "previsao_adaptativa"]].rename(columns={"previsao_adaptativa": "hibrido"})
        t = t.merge(hibrido, on="data", how="left")
    margens = aplicar_margem_contexto(aplicar_folgas(t.assign(previsao=t["regra"]), "previsao"), gold)
    return t.merge(margens[["data", "plano_moderada", "plano_alta"]], on="data", how="left").assign(janela=janela, modelo=modelo)


def comparar_ridge_e_extratrees():
    """Compara Ridge e ExtraTrees nas mesmas datas e nas mesmas politicas."""
    golds = {j: pd.read_parquet(GOLD / f"gold_{j}.parquet") for j in ("t1", "t2")}
    series = Parallel(n_jobs=4)(delayed(politicas_de_modelo)(g, j, m) for j, g in golds.items() for m in ("regressao", "extratrees"))
    linhas = []
    for t in series:
        for periodo, (inicio, fim) in PERIODOS_RIDGE.items():
            q = t[t["data"].between(inicio, fim) & t["consumo"].notna()]
            for universo, d in (("sem_atipicos", q[q["regular"]]), ("todos_os_dias", q)):
                for politica, coluna in (("Previsão", "regra"), ("Híbrido", "hibrido"), ("Margem moderada", "plano_moderada"), ("Margem alta", "plano_alta")):
                    if coluna in d:
                        e = d.dropna(subset=[coluna])
                        linhas.append({"janela": t["janela"].iloc[0], "modelo": t["modelo"].iloc[0], "periodo": periodo, "universo": universo, "politica": politica, **medir(e["consumo"], e[coluna])})
    pd.DataFrame(linhas).to_csv(PASTA_MODELOS / "comparacao_ridge_extratrees.csv", sep=";", index=False, encoding="utf-8-sig")


INICIO_FINAL = "2024-01-01"
HISTORICOS_TESTE = {"Todo o histórico": None, "Últimos 2 anos": 24, "Últimos 12 meses": 12, "Ano anterior + ano atual": ANO_ANTERIOR_E_ATUAL, "Últimos 6 meses": 6}
EXOGENAS = ["fim_de_bloco", "inicio_de_bloco", "evento_especial", "vespera_de_pausa", "pos_pausa"]


def ler_golds_com_nota():
    golds = {j: pd.read_parquet(GOLD / f"gold_{j}.parquet") for j in ("t1", "t2")}
    for g in golds.values():
        g["nota_cardapio"] = g[["efeito_proteina", "efeito_preparo", "efeito_prato"]].mean(axis=1)
    return golds


def origens_de(gold):
    return sorted(gold.loc[gold["data"] >= INICIO_FINAL, "origem"].unique())


def medir_final(serie, gold):
    d = pd.DataFrame({"p": serie}).join(gold.set_index("data")[["consumo", "tratamento", "ano"]])
    d = d[d["consumo"].notna() & d["tratamento"].isin(TRATAMENTOS_REGULARES)]
    e = (arredondar_refeicoes(d["p"]) - d["consumo"]).abs()
    a, b = e[d["ano"] == 2024].mean(), e[d["ano"] == 2025].mean()
    prova, prospectiva = e[d.index.to_series().between(INICIO_PROVA, FIM_PROVA)].mean(), e[d.index > FIM_PROVA].mean()
    return {"erro_2024": round(a, 2), "erro_2025": round(b, 2), "media_2024_2025": round((a + b) / 2, 2), "erro_fev_ago_2026": round(prova, 2), "erro_set_out_2026": round(prospectiva, 2)}


def previsao_por_data(gold, colunas, nome, meses):
    s = prever_serie(gold, origens_de(gold), colunas, nome, meses)
    return pd.Series(s.to_numpy(), index=gold.loc[s.index, "data"].to_numpy())


def comparar_historicos():
    """Compara o comite com todo o historico, 2 anos, 12 meses, ano anterior + atual e 6 meses."""
    golds = ler_golds_com_nota()
    tarefas = [(j, nome, meses, cenario) for j in golds for nome, meses in HISTORICOS_TESTE.items() for cenario in ("C1", "C2")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        series = Parallel(n_jobs=-2, max_nbytes=None)(delayed(previsao_por_data)(golds[j], CENARIOS[c], CAMPEAO, m) for j, _, m, c in tarefas)
    t = pd.DataFrame([{"janela": j, "historico": n, "cenario": c, **medir_final(s, golds[j])} for (j, n, _, c), s in zip(tarefas, series)])
    t.to_csv(PASTA_MODELOS / "experimento_historico.csv", sep=";", index=False, encoding="utf-8-sig")
    return t.sort_values(["janela", "cenario", "media_2024_2025"])


def comparar_nota_cardapio():
    """A nota antiga era fixa por categoria. Aqui ela e refeita a cada sexta, so com o passado."""
    golds = ler_golds_com_nota()
    colunas = CENARIOS[CENARIO_CAMPEAO]
    sem_efeitos = [c for c in colunas if c not in ("efeito_proteina", "efeito_preparo", "efeito_prato")]
    opcoes = {"Três efeitos de cardápio (atual)": colunas, "Nota do cardápio": sem_efeitos + ["nota_cardapio"], "Três efeitos e nota": colunas + ["nota_cardapio"], "Sem cardápio": CENARIOS["C1"]}
    tarefas = [(j, n) for j in golds for n in opcoes]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        series = Parallel(n_jobs=-2, max_nbytes=None)(delayed(previsao_por_data)(golds[j], opcoes[n], CAMPEAO, 12) for j, n in tarefas)
    t = pd.DataFrame([{"janela": j, "opcao": n, **medir_final(s, golds[j])} for (j, n), s in zip(tarefas, series)])
    t.to_csv(PASTA_MODELOS / "experimento_nota_cardapio.csv", sep=";", index=False, encoding="utf-8-sig")
    return t


def base_empilhada(j, gold, nomes=("CatBoost", "KNN", "ExtraTrees", "Ridge", "HistGB")):
    torneio = pd.read_parquet(PASTA_MODELOS / "torneio_previsoes.parquet")
    t = torneio[(torneio["janela"] == j) & (torneio["cenario"] == "C2") & (torneio["historico"] == "1_ano") & torneio["algoritmo"].isin(nomes)]
    largura = t.pivot(index="data", columns="algoritmo", values="previsao")
    largura["Referencia"] = gold.drop_duplicates("data").set_index("data")["nivel_base"].reindex(largura.index)
    return largura


def stacking(gold, previsoes, dias=180, minimo=40):
    """Pesos nao negativos aprendidos com previsoes ja feitas em dias anteriores a cada sexta."""
    b = gold.drop_duplicates("data").set_index("data")
    x = previsoes.join(b[["consumo", "tratamento", "origem"]])
    regulares = x[x["consumo"].notna() & x["tratamento"].isin(TRATAMENTOS_REGULARES)].dropna()
    nomes = list(previsoes.columns)
    saida = []
    for origem, grupo in x.groupby("origem"):
        passado = regulares[regulares.index < origem].tail(dias)
        pesos = np.ones(len(nomes)) / len(nomes)
        if len(passado) >= minimo:
            ajuste, _ = nnls(passado[nomes].to_numpy(), passado["consumo"].to_numpy())
            if ajuste.sum() > 0:
                pesos = ajuste / ajuste.sum()
        saida.append(pd.Series(grupo[nomes].to_numpy() @ pesos, index=grupo.index))
    return pd.concat(saida)


def ridge_forte():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=300.0))


def catboost_raso(iteracoes=100, classe=False):
    args = dict(iterations=iteracoes, learning_rate=0.05, depth=3, random_seed=42, verbose=0, allow_writing_files=False, thread_count=1)
    return CatBoostClassifier(**args) if classe else CatBoostRegressor(loss_function="MAE", **args)


def residual_e_regimes(gold, limiar=0.85):
    """Residual: Ridge e CatBoost no que sobrou. Regimes: especialistas pesados pela chance de dia baixo."""
    colunas = CENARIOS[CENARIO_CAMPEAO]
    resultado = {"Residual": [], "Regimes": []}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for origem in origens_de(gold):
            alvo = gold[gold["origem"] == origem]
            treino = dados_de_treino(gold, origem, colunas, 12)
            X, y, Xa = matriz(treino, colunas), treino["alvo_relativo"].clip(*LIMITES_ALVO), matriz(alvo, colunas)
            baixo_lim, alto_lim = X.quantile(0.01), X.quantile(0.99)
            X, Xa = X.clip(baixo_lim, alto_lim, axis=1), Xa.clip(baixo_lim, alto_lim, axis=1)
            nivel = alvo["nivel_base"].to_numpy()
            base = ridge_forte().fit(X, y)
            ajuste = catboost_raso().fit(X, y - base.predict(X))
            resultado["Residual"].append(pd.Series((base.predict(Xa) + ajuste.predict(Xa)) * nivel, index=alvo["data"].to_numpy()))
            baixo = y < limiar
            if 10 <= baixo.sum() <= len(y) - 10:
                chance = catboost_raso(classe=True).fit(X, baixo.astype(int)).predict_proba(Xa)[:, 1]
                prev = chance * catboost_raso().fit(X[baixo], y[baixo]).predict(Xa) + (1 - chance) * catboost_raso().fit(X[~baixo], y[~baixo]).predict(Xa)
            else:
                prev = catboost_raso().fit(X, y).predict(Xa)
            resultado["Regimes"].append(pd.Series(prev * nivel, index=alvo["data"].to_numpy()))
    return {k: pd.concat(v) for k, v in resultado.items()}


def serie_de_treino(gold, origem, dias=500):
    t = gold[(gold["data"] < origem) & gold["usar_no_treino"] & gold["consumo"].notna()].drop_duplicates("data").sort_values("data")
    return t.tail(dias)


def prever_sarimax(gold):
    saidas = []
    for origem in origens_de(gold):
        alvo = gold[gold["origem"] == origem].sort_values("data")
        t = serie_de_treino(gold, origem)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ajuste = SARIMAX(t["consumo"].to_numpy(float), exog=t[EXOGENAS].astype(float).to_numpy(), order=(1, 0, 1), seasonal_order=(0, 1, 1, 5), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=50)
                prev = ajuste.forecast(len(alvo), exog=alvo[EXOGENAS].astype(float).to_numpy())
            except Exception:
                prev = alvo["nivel_base"].to_numpy()
        saidas.append(pd.Series(np.maximum(prev, 0), index=alvo["data"].to_numpy()))
    return pd.concat(saidas)


def prever_prophet(gold):
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    saidas = []
    for origem in origens_de(gold):
        alvo = gold[gold["origem"] == origem].sort_values("data")
        t = serie_de_treino(gold, origem)
        modelo = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05)
        for coluna in EXOGENAS:
            modelo.add_regressor(coluna)
        modelo.fit(t[["data", "consumo", *EXOGENAS]].rename(columns={"data": "ds", "consumo": "y"}).astype({c: float for c in EXOGENAS}))
        futuro = alvo[["data", *EXOGENAS]].rename(columns={"data": "ds"}).astype({c: float for c in EXOGENAS})
        saidas.append(pd.Series(np.maximum(modelo.predict(futuro)["yhat"].to_numpy(), 0), index=alvo["data"].to_numpy()))
    return pd.concat(saidas)


def testar_ideias_finais():
    """Stacking, residual, regimes, SARIMAX e Prophet, contra o comite atual."""
    golds = ler_golds_com_nota()
    linhas = []
    for j, g in golds.items():
        base = base_empilhada(j, g)
        comite = base[["CatBoost", "KNN", "Referencia"]].mean(axis=1)
        linhas.append({"janela": j, "ideia": "Comitê de CatBoost, KNN e referência", **medir_final(comite, g)})
        for dias in (90, 180, 365):
            linhas.append({"janela": j, "ideia": f"Stacking com {dias} dias", **medir_final(stacking(g, base, dias), g)})
    avancados = dict(zip(golds, Parallel(n_jobs=2, max_nbytes=None)(delayed(residual_e_regimes)(g) for g in golds.values())))
    for j, g in golds.items():
        for nome, serie in avancados[j].items():
            linhas.append({"janela": j, "ideia": nome, **medir_final(np.maximum(serie, 0), g)})
    chaves = [(j, n) for j in golds for n in ("SARIMAX", "Prophet")]
    temporais = dict(zip(chaves, Parallel(n_jobs=-2, max_nbytes=None)(delayed(prever_sarimax if n == "SARIMAX" else prever_prophet)(golds[j]) for j, n in chaves)))
    for (j, nome), serie in temporais.items():
        base = base_empilhada(j, golds[j])
        linhas.append({"janela": j, "ideia": nome, **medir_final(serie, golds[j])})
        linhas.append({"janela": j, "ideia": f"Comitê + {nome}", **medir_final((base[["CatBoost", "KNN", "Referencia"]].sum(axis=1) + serie.reindex(base.index)) / 4, golds[j])})
    t = pd.DataFrame(linhas).sort_values(["janela", "media_2024_2025"])
    t.to_csv(PASTA_MODELOS / "experimentos_finais.csv", sep=";", index=False, encoding="utf-8-sig")
    return t


JANELA_FATOR_PLANO = 40


def preparar_plano(gold):
    """Plano como preditor. A correcao usa so a relacao plano-consumo anterior a cada sexta."""
    g = gold.sort_values("data").copy()
    g["plano"] = g["plano_nutricionista"].fillna(g["reserva_discente"])
    g["plano_da_nutricionista"] = g["plano_nutricionista"].notna().astype(float)
    g["rel_plano"] = g["plano"] / g["nivel_base"]
    conhecidos = g[g["consumo"].notna() & g["plano"].notna() & g["tratamento"].isin(TRATAMENTOS_REGULARES)]
    fatores = {}
    for origem in g["origem"].unique():
        recente = conhecidos[conhecidos["data"] < origem].tail(JANELA_FATOR_PLANO)
        fatores[origem] = (recente["consumo"] / recente["plano"]).median() if len(recente) >= 10 else 1.0
    g["rel_plano_corrigido"] = g["rel_plano"] * g["origem"].map(fatores)
    return g


def cenarios_completos():
    c1, c2, c3 = CENARIOS["C1"], CENARIOS["C2"], CENARIOS["C3"]
    bruto, corrigido = ["rel_plano", "plano_da_nutricionista"], ["rel_plano_corrigido", "plano_da_nutricionista"]
    return {"Consumo e calendário": c1, "Inclui cardápio": c2, "Inclui clima previsto": c3, "Consumo, calendário e plano": c1 + bruto, "Inclui cardápio e plano": c2 + bruto, "Inclui clima e plano": c3 + bruto,
            "Consumo, calendário e plano corrigido": c1 + corrigido, "Inclui cardápio e plano corrigido": c2 + corrigido, "Inclui clima e plano corrigido": c3 + corrigido}


def prever_cenario(gold, algoritmo, colunas, meses=12):
    fabrica = ridge_forte if algoritmo == "Ridge forte" else ALGORITMOS[algoritmo]
    saidas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for origem in origens_de(gold):
            alvo = gold[gold["origem"] == origem]
            treino = dados_de_treino(gold, origem, colunas, meses)
            prev = alvo["nivel_base"].to_numpy(float)
            if len(treino) >= 30:
                X = matriz(treino, colunas)
                vazias = X.columns[X.isna().all()]
                X[vazias] = 0.0
                Xa = matriz(alvo.assign(**{c: 0.0 for c in vazias}), colunas)
                modelo = fabrica()
                modelo.fit(X, treino["alvo_relativo"].clip(*LIMITES_ALVO))
                prev = np.maximum(modelo.predict(Xa), 0) * alvo["nivel_base"].to_numpy()
            saidas.append(pd.Series(prev, index=alvo["data"].to_numpy()))
    return pd.concat(saidas)


def quatro_metricas_serie(serie, gold):
    d = pd.DataFrame({"p": serie}).join(gold.set_index("data")[["consumo", "tratamento"]])
    d = d[d["consumo"].notna() & d["tratamento"].isin(TRATAMENTOS_REGULARES) & d.index.to_series().between(INICIO_PROVA, FIM_PROVA)]
    e = arredondar_refeicoes(d["p"]) - d["consumo"]
    return {"mae": round(e.abs().mean(), 2), "rmse": round(np.sqrt((e ** 2).mean()), 2), "wape": round(e.abs().sum() / d["consumo"].sum() * 100, 2), "r2": round(1 - (e ** 2).sum() / ((d["consumo"] - d["consumo"].mean()) ** 2).sum(), 3)}


def previsoes_do_cenario(gold, janela, cenarios):
    algoritmos = [*ALGORITMOS, "Ridge forte"]
    tarefas = [(a, c) for c in cenarios for a in algoritmos]
    series = Parallel(n_jobs=-2, max_nbytes=None)(delayed(prever_cenario)(gold, a, cenarios[c]) for a, c in tarefas)
    previsoes = dict(zip(tarefas, series))
    tipico = gold.drop_duplicates("data").set_index("data")["nivel_base"]
    for c in cenarios:
        previsoes["Comitê", c] = (previsoes["CatBoost", c] + previsoes["KNN", c] + tipico.reindex(previsoes["KNN", c].index)) / 3
    return pd.concat([pd.DataFrame({"janela": janela, "modelo": a, "cenario": c, "data": s.index, "previsao": arredondar_refeicoes(s.to_numpy())}) for (a, c), s in previsoes.items()], ignore_index=True)


def gerar_matriz_de_previsoes():
    """Previsoes de todos os algoritmos em todos os cenarios, salvas para avaliar as politicas."""
    t1 = preparar_plano(pd.read_parquet(GOLD / "gold_t1.parquet"))
    t2 = pd.read_parquet(GOLD / "gold_t2.parquet")
    matriz_prev = pd.concat([previsoes_do_cenario(t1, "t1", cenarios_completos()), previsoes_do_cenario(t2, "t2", {"Consumo e calendário": CENARIOS["C1"], "Inclui cardápio": CENARIOS["C2"], "Inclui clima previsto": CENARIOS["C3"]})], ignore_index=True)
    matriz_prev.to_parquet(PASTA_MODELOS / "previsoes_matriz_completa.parquet", index=False)
    return matriz_prev


def metricas_de_periodo(d, coluna):
    e = d[coluna] - d["consumo"]
    return {"dias": len(d), "mae": round(e.abs().mean(), 2), "rmse": round(np.sqrt((e ** 2).mean()), 2), "wape": round(e.abs().sum() / d["consumo"].sum() * 100, 2), "r2": round(1 - (e ** 2).sum() / ((d["consumo"] - d["consumo"].mean()) ** 2).sum(), 3),
            "excesso": int(e.clip(lower=0).sum()), "deficit": int((-e).clip(lower=0).sum()), "dias_deficit_pct": round((e < 0).mean() * 100, 1)}


def politicas_do_modelo(grupo, gold):
    """Sem margem, margens por contexto e, no preparo, hibrido com o plano humano."""
    janela, cenario, modelo = grupo[["janela", "cenario", "modelo"]].iloc[0]
    base = gold[["origem", "data", "consumo", "tratamento", "ano", "sem_cardapio"]].merge(grupo[["data", "previsao"]], on="data")
    base["regular"] = base["tratamento"].isin(TRATAMENTOS_REGULARES)
    t = aplicar_margem_contexto(aplicar_folgas(base, "previsao"), gold)
    colunas = {"Sem margem": "plano_sem_folga", "Margem moderada": "plano_moderada", "Margem alta": "plano_alta"}
    if janela == "t1":
        hibrido = combinar_com_plano(base, gold)[["data", "previsao_50_50", "previsao_adaptativa"]]
        t = t.merge(hibrido, on="data", how="left")
        colunas |= {"Híbrido adaptativo": "previsao_adaptativa", "Híbrido 50/50": "previsao_50_50"}
    t["periodo"] = np.select([t["ano"] <= 2025, t["data"] <= FIM_PROVA], ["2024-2025", "fev-ago 2026"], "set-out 2026")
    linhas = []
    for politica, coluna in colunas.items():
        q = t[t["regular"] & t["consumo"].notna() & t[coluna].notna()]
        for periodo, d in q.groupby("periodo"):
            linhas.append({"janela": janela, "cenario": cenario, "modelo": modelo, "politica": politica, "periodo": periodo, **metricas_de_periodo(d, coluna)})
    return linhas


def testar_matriz_completa():
    """Cada algoritmo, em cada cenario, sem margem, com margens e no hibrido."""
    arquivo = PASTA_MODELOS / "previsoes_matriz_completa.parquet"
    matriz_prev = pd.read_parquet(arquivo) if arquivo.exists() else gerar_matriz_de_previsoes()
    golds = {"t1": pd.read_parquet(GOLD / "gold_t1.parquet"), "t2": pd.read_parquet(GOLD / "gold_t2.parquet")}
    grupos = [g for _, g in matriz_prev.groupby(["janela", "cenario", "modelo"])]
    linhas = Parallel(n_jobs=-2, max_nbytes=None)(delayed(politicas_do_modelo)(g, golds[g["janela"].iloc[0]]) for g in grupos)
    t = pd.DataFrame([l for bloco in linhas for l in bloco])
    t.to_csv(PASTA_MODELOS / "matriz_completa.csv", sep=";", index=False, encoding="utf-8-sig")
    return t
