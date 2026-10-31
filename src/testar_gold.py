"""Prova que a Gold nao usa informacao posterior a emissao.

O teste embaralha todo o consumo a partir de cada sexta-feira e reconstroi a Gold.
Os atributos daquela emissao precisam ficar identicos.

Execute depois de processar_gold.py: python src/testar_gold.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from processar_gold import construir_gold

RAIZ = Path(__file__).resolve().parents[1]
SILVER = RAIZ / "dados" / "silver"
GOLD = RAIZ / "dados" / "gold"
SEMENTE = 42
RESULTADOS_E_ROTULOS = {"consumo", "alvo_relativo", "tratamento", "usar_no_treino", "tipo_sugerido"}


def testar_estrutura(t1, t2):
    for nome, g, minimo, maximo in [("t1", t1, 3, 7), ("t2", t2, 10, 14)]:
        assert g["origem"].dt.dayofweek.eq(4).all(), nome
        assert g["antecedencia_dias"].between(minimo, maximo).all(), nome
        assert g["ultimo_consumo_usado"].lt(g["origem"]).all(), nome
        assert not g.duplicated(["origem", "data"]).any(), nome
        assert not g["data"].duplicated().any(), nome
    assert "plano_nutricionista" not in t2.columns and "reserva_discente" not in t2.columns
    print("Estrutura: origens em sextas, antecedencias corretas, sem plano humano em t+2.")


def testar_qualidade(t1, t2):
    for nome, g in (("t1", t1), ("t2", t2)):
        numericas = g.select_dtypes("number")
        assert not np.isinf(numericas.to_numpy(dtype=float, na_value=np.nan)).any(), nome
        assert g["nivel_base"].gt(0).all(), nome
        assert g["consumo"].dropna().ge(0).all(), nome
        assert g.loc[g["usar_no_treino"], "consumo"].notna().all(), nome
        constantes = [c for c in numericas if numericas[c].nunique(dropna=True) <= 1]
        assert not constantes, (nome, constantes)
        vazias = g.columns[g.isna().mean() > 0.5].tolist()
        print(f"{nome}: {g.shape[1]} colunas sem infinitos ou constantes; mais de 50% vazias: {vazias}")


def testar_vazamento(fato, t1, t2, origens):
    gerador = np.random.default_rng(SEMENTE)
    atributos = lambda g: [c for c in g.columns if c not in RESULTADOS_E_ROTULOS]
    for origem in origens:
        alterado = fato.copy()
        futuro = alterado["data"] >= origem
        alterado.loc[futuro & alterado["consumo"].notna(), "consumo"] = gerador.integers(1, 400, (futuro & alterado["consumo"].notna()).sum()).astype(float)
        novo = construir_gold(alterado)
        for nome, original in (("t1", t1), ("t2", t2)):
            a = original[original["origem"] == origem].reset_index(drop=True)
            b = novo[nome][novo[nome]["origem"] == origem].reset_index(drop=True)
            pd.testing.assert_frame_equal(a[atributos(a)], b[atributos(b)], check_dtype=False)
        print(f"  {origem.date()}: atributos identicos")


def main():
    fato = pd.read_parquet(SILVER / "silver_fato_diaria.parquet")
    t1 = pd.read_parquet(GOLD / "gold_t1.parquet")
    t2 = pd.read_parquet(GOLD / "gold_t2.parquet")
    testar_estrutura(t1, t2)
    testar_qualidade(t1, t2)
    comuns = sorted(set(t1["origem"]) & set(t2["origem"]))
    origens = [comuns[int(i)] for i in np.linspace(5, len(comuns) - 1, 6)]
    testar_vazamento(fato, t1, t2, origens)
    print("Nenhum atributo mudou com o futuro embaralhado.")


if __name__ == "__main__":
    main()
