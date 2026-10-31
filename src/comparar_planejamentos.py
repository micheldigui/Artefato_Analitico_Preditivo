"""Compara, ano a ano, as formas de planejar com o consumo registrado.

Usa a configuracao escolhida em 2025 e repete a validacao semanal desde 2024, para
comparar o artefato com as reservas dos alunos e com o quantitativo da nutricionista
nas mesmas datas. A ideia e comparar, nao competir: reservas e quantitativo tem
origens e momentos diferentes. Em 2023 nao ha historico anterior para treinar.

Execute depois de consolidar_resultados.py: python src/comparar_planejamentos.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from avaliar_cenarios import combinar_com_plano
from modelagem import CENARIO_CAMPEAO, CENARIOS, TRATAMENTOS_REGULARES, arredondar_refeicoes, intervalo_diferenca, medir, prever_emissao

RAIZ = Path(__file__).resolve().parents[1]
GOLD = RAIZ / "dados" / "gold"
MODELOS = RAIZ / "resultados" / "modelos"
SAIDA = RAIZ / "resultados" / "comparacao"
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
    obs = t[t["consumo"].notna() & t["plano"].notna()]
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


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    escolhas = pd.read_csv(MODELOS / "configuracao_escolhida.csv", sep=";", encoding="utf-8-sig")
    g1 = pd.read_parquet(GOLD / "gold_t1.parquet")
    g2 = pd.read_parquet(GOLD / "gold_t2.parquet")
    opcoes = escolhas[escolhas["cenario"] == CENARIO_CAMPEAO]
    linha_t1 = opcoes[opcoes["janela"] == "t1"].iloc[0]
    linha_t2 = opcoes[opcoes["janela"] == "t2"].iloc[0]
    preparo = adicionar_planos(montar_base(g1, linha_t1["cenario"], linha_t1["modelo"], linha_t1["janela_treino"]), g1)
    compras = montar_base(g2, linha_t2["cenario"], linha_t2["modelo"], linha_t2["janela_treino"])
    anual = medir_por_ano(preparo)
    preparo.to_parquet(SAIDA / "serie_preparo.parquet", index=False)
    compras.to_parquet(SAIDA / "serie_compras.parquet", index=False)
    anual.to_csv(SAIDA / "comparacao_anual.csv", sep=";", index=False, encoding="utf-8-sig")
    print(anual[anual["universo"] == "sem_atipicos"][["ano", "alternativa", "dias", "mae", "vies", "excesso", "deficit", "dias_deficit", "dif_plano", "ic_inf", "ic_sup"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
