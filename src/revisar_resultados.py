"""Revisao adversarial: recalcula numeros centrais por um caminho independente.

Parte dos CSVs da Bronze e das previsoes diarias salvas, sem usar as funcoes de
metricas do pipeline. Cada verificacao falha de forma explicita. O relatorio fica em
resultados/revisao/verificacoes.csv.

Execute depois de comparar_planejamentos.py: python src/revisar_resultados.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
BRONZE = RAIZ / "dados" / "bronze"
GOLD = RAIZ / "dados" / "gold"
MODELOS = RAIZ / "resultados" / "modelos"
SAIDA = RAIZ / "resultados" / "revisao"
TOLERANCIA = 0.005
verificacoes = []


def registrar(nome, ok, detalhe=""):
    verificacoes.append({"verificacao": nome, "resultado": "ok" if ok else "FALHOU", "detalhe": detalhe})
    print(("OK    " if ok else "FALHOU"), nome, detalhe)


def ler_bronze(pasta, coluna):
    tabelas = [pd.read_csv(f, sep=";", encoding="utf-8-sig") for f in sorted(glob.glob(str(BRONZE / pasta / "*.csv")))]
    t = pd.concat(tabelas, ignore_index=True)
    t["data"] = pd.to_datetime(t["data"], dayfirst=True)
    return t[t[coluna].notna()].set_index("data")[coluna]


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    consumo, planos = ler_bronze("consumo", "consumo"), ler_bronze("reservas", "reservas")

    # Dados brutos contra Gold
    g1 = pd.read_parquet(GOLD / "gold_t1.parquet").set_index("data")
    g2 = pd.read_parquet(GOLD / "gold_t2.parquet").set_index("data")
    for nome, g in (("preparo", g1), ("compras", g2)):
        d = g["consumo"].dropna()
        registrar(f"Consumo da Gold igual ao CSV ({nome})", (d == consumo.reindex(d.index)).all(), f"{len(d)} dias")
    p = g1["plano_nutricionista"].dropna()
    registrar("Quantitativo da Gold igual ao CSV", (p == planos.reindex(p.index)).all(), f"{len(p)} dias")
    r = g1["reserva_discente"].dropna()
    registrar("Reservas da Gold iguais ao CSV", (r == planos.reindex(r.index)).all(), f"{len(r)} dias")

    # Totais do planejamento, direto dos CSVs
    par = pd.DataFrame({"c": consumo, "p": planos}).dropna()
    par = par[par.index.dayofweek < 6]
    discente = par[par.index.year <= 2025]
    nutri = par[(par.index.year == 2026) & (par.index <= "2026-08-31")]
    registrar("Reservas 2023-2025: 458 datas, 2.916 a mais, 9.494 a menos", (len(discente), int((discente.p - discente.c).clip(lower=0).sum()), int((discente.c - discente.p).clip(lower=0).sum())) == (458, 2916, 9494))
    registrar("Quantitativo até 31/08/2026: 123 datas, 2.561 a mais, 663 a menos", (len(nutri), int((nutri.p - nutri.c).clip(lower=0).sum()), int((nutri.c - nutri.p).clip(lower=0).sum())) == (123, 2561, 663))

    # Resultado final recalculado a partir das previsoes diarias
    final = pd.read_csv(MODELOS / "resultado_final.csv", sep=";", encoding="utf-8-sig")
    preparo = pd.read_parquet(MODELOS / "preparo_diario_final.parquet")
    d = preparo[(preparo["data"] >= "2026-01-01") & preparo["consumo"].notna() & preparo["regular"]]
    for alternativa in ("plano", "C1", "C2", "C4"):
        calculado = (d[alternativa] - d["consumo"]).abs().sum() / len(d)
        tabela = final[(final["janela"] == "t1") & (final["periodo"] == "prova_2026") & (final["universo"] == "sem_atipicos") & (final["alternativa"] == alternativa)]["mae"].iloc[0]
        registrar(f"Erro médio de {alternativa} confere", abs(calculado - tabela) < TOLERANCIA, f"{calculado:.2f}")

    # Consumo das previsoes igual ao CSV
    registrar("Consumo das previsões igual ao CSV", (d.set_index("data")["consumo"] == consumo.reindex(d["data"])).all())

    # Ordem temporal
    registrar("Toda data avaliada vem depois da sua sexta-feira", (preparo["data"] > preparo["origem"]).all())
    registrar("Antecedência de preparo entre 3 e 7 dias", (preparo["data"] - preparo["origem"]).dt.days.between(3, 7).all())
    compras = pd.read_parquet(MODELOS / "compras_diario_final.parquet")
    registrar("Antecedência de compras entre 10 e 14 dias", (compras["data"] - compras["origem"]).dt.days.between(10, 14).all())

    # A configuracao precisa ser a receita fixada com 2024 e 2025
    config = pd.read_csv(MODELOS / "configuracao_escolhida.csv", sep=";", encoding="utf-8-sig")
    registrar("Configuração é o comitê campeão", (config["modelo"] == "campeao").all() and (config["janela_treino"] == "1_ano").all())

    # O plano humano nao pode explicar o modelo
    sem_plano = preparo[preparo["plano"].notna() & preparo["C2"].notna() & (preparo["data"] >= "2026-01-01")]
    registrar("Previsão não copia o quantitativo", (sem_plano["C2"] - sem_plano["plano"]).abs().median() > 5, f"diferença mediana {(sem_plano['C2'] - sem_plano['plano']).abs().median():.1f}")

    # O limite de fim de periodo precisa aparecer sem olhar 2026
    t = g1[(g1.index <= "2025-12-31") & g1["usar_no_treino"]]
    fim, resto = t[t["fim_de_bloco"]]["alvo_relativo"].median(), t[~t["fim_de_bloco"]]["alvo_relativo"].median()
    registrar("Queda de fim de período existe até 2025", fim < 0.8 * resto, f"{fim:.2f} contra {resto:.2f}")

    tabela = pd.DataFrame(verificacoes)
    tabela.to_csv(SAIDA / "verificacoes.csv", sep=";", index=False, encoding="utf-8-sig")
    print(f"\n{(tabela['resultado'] == 'ok').sum()} de {len(tabela)} verificações ok")


if __name__ == "__main__":
    main()
