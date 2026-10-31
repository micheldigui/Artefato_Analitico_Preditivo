"""Prova que a previsao de uma sexta-feira nao muda quando o futuro muda.

Embaralha consumo e rotulos a partir da emissao e exige previsao identica.
Execute a partir da raiz do projeto: python src/testar_modelagem.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from modelagem import ANO_ANTERIOR_E_ATUAL, CAMPEAO, CENARIOS, CENARIOS_POR_JANELA, MODELOS, dados_de_treino, prever_emissao

GOLD = Path(__file__).resolve().parents[1] / "dados" / "gold"
SEMENTE = 42
ORIGENS = ["2025-05-09", "2026-03-06", "2026-09-04"]


def embaralhar_futuro(gold, origem):
    alterado = gold.copy()
    futuro = alterado["data"] >= origem
    gerador = np.random.default_rng(SEMENTE)
    alterado.loc[futuro, "consumo"] = gerador.integers(1, 400, futuro.sum()).astype(float)
    alterado.loc[futuro, "alvo_relativo"] = gerador.uniform(0.1, 3, futuro.sum())
    alterado.loc[futuro, "usar_no_treino"] = gerador.random(futuro.sum()) > 0.5
    return alterado


def main():
    for janela in ("t1", "t2"):
        gold = pd.read_parquet(GOLD / f"gold_{janela}.parquet")
        for texto in ORIGENS:
            origem = pd.Timestamp(texto)
            alterado = embaralhar_futuro(gold, origem)
            for cenario in CENARIOS_POR_JANELA[janela]:
                colunas = CENARIOS[cenario]
                assert dados_de_treino(gold, origem, colunas)["data"].max() < origem
                assert not any(c in colunas for c in ("plano_nutricionista", "reserva_discente", "consumo", "alvo_relativo"))
                for modelo in [*MODELOS, CAMPEAO]:
                    for meses in (None, 6, ANO_ANTERIOR_E_ATUAL):
                        a = prever_emissao(gold, origem, colunas, modelo, meses)
                        b = prever_emissao(alterado, origem, colunas, modelo, meses)
                        pd.testing.assert_series_equal(a, b)
        print(f"{janela}: previsoes identicas com o futuro embaralhado")
    print("Nenhum modelo usou informacao posterior a sexta-feira do planejamento.")


if __name__ == "__main__":
    main()
