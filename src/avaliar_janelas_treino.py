"""Compara quanto historico usar no treino.

Opcoes: tudo, ultimos 2 anos, ultimos 12 meses, ultimos 6 meses, ou o ano anterior
inteiro somado ao que ja passou do ano atual.

A janela de treino de cada combinacao e escolhida em 2025; 2026 so mede o resultado.
Execute depois de avaliar_cenarios.py: python src/avaliar_janelas_treino.py
"""
from pathlib import Path

import pandas as pd

from avaliar_cenarios import resumir, rotular_periodo
from modelagem import ANO_ANTERIOR_E_ATUAL, validar_janela

RAIZ = Path(__file__).resolve().parents[1]
GOLD = RAIZ / "dados" / "gold"
SAIDA = RAIZ / "resultados" / "modelos"
JANELAS_TREINO = {"tudo": None, "2_anos": 24, "1_ano": 12, "6_meses": 6, ANO_ANTERIOR_E_ATUAL: ANO_ANTERIOR_E_ATUAL}
# H mede quanto o calendario acrescenta e nao e opcao de uso. C3 usa previsao do tempo,
# que so existe em compras desde abril de 2026, por isso fica fora dessa janela.
CENARIOS_TESTADOS = {"t1": ["H", "C1", "C2", "C3"], "t2": ["H", "C1", "C2"]}


def main():
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
    metricas.to_csv(SAIDA / "metricas_janelas_treino.csv", sep=";", index=False, encoding="utf-8-sig")
    pd.concat(previsoes_todas, ignore_index=True).to_parquet(SAIDA / "previsoes_janelas_treino.parquet", index=False)
    base = metricas[metricas["universo"] == "sem_atipicos"]
    print(base.pivot_table(index=["janela", "cenario", "modelo"], columns=["periodo", "janela_treino"], values="mae").round(2).to_string())


if __name__ == "__main__":
    main()
