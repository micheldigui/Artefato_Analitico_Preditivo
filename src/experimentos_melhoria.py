"""Testa melhorias sobre o torneio, olhando apenas 2024 e 2025.

- atributos de sazonalidade (posicao no periodo de aulas e mesma epoca do ano anterior);
- peso maior para dias recentes no aprendizado;
- media simples entre algoritmos;
- aproximacao da regra simples do consumo tipico;
- formas de combinar a previsao com o plano humano.

Regra fixada antes de ver os resultados: uma ideia so substitui a configuracao atual
se reduzir o erro medio de 2024-2025 em pelo menos 0,5 refeicao por dia.

Execute depois de torneio_ampliado.py: python src/experimentos_melhoria.py
"""
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from modelagem import CENARIOS, LIMITES_ALVO, TRATAMENTOS_REGULARES, dados_de_treino, matriz, peso_adaptativo
from torneio_ampliado import ALGORITMOS, INICIO

RAIZ = Path(__file__).resolve().parents[1]
GOLD = RAIZ / "dados" / "gold"
SAIDA = RAIZ / "resultados" / "modelos"
FIM_ESCOLHA = "2025-12-31"
GANHO_MINIMO = 0.5
MEIA_VIDA_DIAS = 180
JANELA_VIES_DIAS = 40
EXPERIMENTAIS = {**CENARIOS, "C2p": CENARIOS["C2"] + ["rel_plano"]}


def rodar(gold, janela, cenario, algoritmo, meses, meia_vida):
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


def resumo(nome, janela, prev, base):
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


def main():
    golds = {j: pd.read_parquet(GOLD / f"gold_{j}.parquet") for j in ("t1", "t2")}
    preparo = golds["t1"]
    preparo["plano"] = preparo["plano_nutricionista"].fillna(preparo["reserva_discente"])
    preparo["rel_plano"] = preparo["plano"] / preparo["nivel_base"]
    base = {j: g[["data", "consumo", "tratamento", "ano"]] for j, g in golds.items()}
    torneio = pd.read_parquet(SAIDA / "torneio_previsoes.parquet")
    atual = pd.read_csv(SAIDA / "torneio_ate_2025.csv", sep=";", encoding="utf-8-sig")
    linhas = []

    # Regra simples
    for j, g in golds.items():
        linhas.append(resumo("Consumo de referência", j, g[g["data"] >= INICIO][["data"]].assign(previsao=g["nivel_base"]), base))

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
            linhas.append(resumo(f"Média: {a[1]} {a[0]} {a[2]} + {b[1]} {b[0]} {b[2]}", j, media, base))
        for n in (3, 5):
            media = (sum(series[k] for k in chaves[:n]) / n).rename("previsao").reset_index()
            linhas.append(resumo(f"Média dos {n} melhores", j, media, base))
        tipico = golds[j].set_index("data")["nivel_base"]
        melhor = series[chaves[0]]
        for peso in (0.25, 0.5):
            mix = ((1 - peso) * melhor + peso * tipico.reindex(melhor.index)).rename("previsao").reset_index()
            linhas.append(resumo(f"Melhor com {peso:.0%} do consumo de referência", j, mix, base))

    # Atributos de sazonalidade e recencia, nos algoritmos mais fortes
    fortes = ["Ridge", "Huber", "HistGB", "RandomForest", "ExtraTrees", "GradientBoosting"]
    tarefas = [(j, c, a, 12, None) for j in golds for c in ("C1s", "C2s") for a in fortes]
    tarefas += [(j, c, a, None, MEIA_VIDA_DIAS) for j in golds for c in ("C1", "C2", "C1s", "C2s") for a in fortes]
    tarefas += [("t1", "C2p", a, 12, None) for a in fortes]
    extras = Parallel(n_jobs=-2)(delayed(rodar)(golds[j], j, c, a, m, v) for j, c, a, m, v in tarefas)
    previsoes_extras = pd.concat(extras, ignore_index=True)
    previsoes_extras.to_parquet(SAIDA / "experimentos_previsoes.parquet", index=False)
    for (j, c, a, h), g in previsoes_extras.groupby(["janela", "cenario", "algoritmo", "historico"]):
        linhas.append(resumo(f"{a} {c} {h}", j, g[["data", "previsao"]], base))

    t = pd.DataFrame(linhas)
    t.to_csv(SAIDA / "experimentos_ate_2025.csv", sep=";", index=False, encoding="utf-8-sig")
    print(t.sort_values(["janela", "media"]).groupby("janela").head(16).round(2).to_string(index=False))
    print(t[t["ideia"] == "Consumo de referência"].round(2).to_string(index=False))

    # Combinacoes com o plano, todas medidas nas mesmas datas
    q = torneio[torneio["janela"] == "t1"]
    candidatos = {"média dos 3 melhores": sum(largas["t1"][k] for k in list(largas["t1"])[:3]) / 3}
    for nome, cenario, algoritmo in (("CatBoost", "C1", "CatBoost"), ("Huber", "C2", "Huber"), ("KNN", "C2", "KNN")):
        candidatos[nome] = q[(q["cenario"] == cenario) & (q["algoritmo"] == algoritmo) & (q["historico"] == "1_ano")].set_index("data")["previsao"]
    planos = [resumo(ideia, "t1", prev, base) for nome, serie in candidatos.items() for ideia, prev in com_plano(serie, golds["t1"], nome).items()]
    p = pd.DataFrame(planos).drop_duplicates("ideia")
    p.to_csv(SAIDA / "experimentos_plano_ate_2025.csv", sep=";", index=False, encoding="utf-8-sig")
    print(p.sort_values("media").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
