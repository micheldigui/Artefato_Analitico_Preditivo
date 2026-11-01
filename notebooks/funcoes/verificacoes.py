"""Testes de vazamento e conferencia dos resultados por caminhos independentes."""
import glob
import numpy as np
import pandas as pd

from .atributos import construir_gold
from .configuracao import ANO_ANTERIOR_E_ATUAL, BRONZE, FIM_PROVA, INICIO_PROVA, CENARIOS, CENARIOS_POR_JANELA, GOLD, MODELOS_BASE, PASTA_MODELOS, PASTA_REVISAO, SILVER
from .modelos import dados_de_treino, prever_emissao


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


def testar_gold():
    fato = pd.read_parquet(SILVER / "silver_fato_diaria.parquet")
    t1 = pd.read_parquet(GOLD / "gold_t1.parquet")
    t2 = pd.read_parquet(GOLD / "gold_t2.parquet")
    testar_estrutura(t1, t2)
    testar_qualidade(t1, t2)
    comuns = sorted(set(t1["origem"]) & set(t2["origem"]))
    origens = [comuns[int(i)] for i in np.linspace(5, len(comuns) - 1, 6)]
    testar_vazamento(fato, t1, t2, origens)
    print("Nenhum atributo mudou com o futuro embaralhado.")


ORIGENS = ["2025-05-09", "2026-03-06", "2026-09-04"]


def embaralhar_futuro(gold, origem):
    alterado = gold.copy()
    futuro = alterado["data"] >= origem
    gerador = np.random.default_rng(SEMENTE)
    alterado.loc[futuro, "consumo"] = gerador.integers(1, 400, futuro.sum()).astype(float)
    alterado.loc[futuro, "alvo_relativo"] = gerador.uniform(0.1, 3, futuro.sum())
    alterado.loc[futuro, "usar_no_treino"] = gerador.random(futuro.sum()) > 0.5
    return alterado


def testar_vazamento_modelos():
    for janela in ("t1", "t2"):
        gold = pd.read_parquet(GOLD / f"gold_{janela}.parquet")
        for texto in ORIGENS:
            origem = pd.Timestamp(texto)
            alterado = embaralhar_futuro(gold, origem)
            for cenario in CENARIOS_POR_JANELA[janela]:
                colunas = CENARIOS[cenario]
                assert dados_de_treino(gold, origem, colunas)["data"].max() < origem
                assert not any(c in colunas for c in ("plano_nutricionista", "reserva_discente", "consumo", "alvo_relativo"))
                for modelo in MODELOS_BASE:
                    for meses in (None, 12, 6, ANO_ANTERIOR_E_ATUAL):
                        a = prever_emissao(gold, origem, colunas, modelo, meses)
                        b = prever_emissao(alterado, origem, colunas, modelo, meses)
                        pd.testing.assert_series_equal(a, b)
        print(f"{janela}: previsoes identicas com o futuro embaralhado")
    print("Nenhum modelo usou informacao posterior a sexta-feira do planejamento.")


TOLERANCIA = 0.005


registros = []


def registrar(nome, ok, detalhe=""):
    registros.append({"verificacao": nome, "resultado": "ok" if ok else "FALHOU", "detalhe": detalhe})
    print(("OK    " if ok else "FALHOU"), nome, detalhe)


def ler_bronze(pasta, coluna):
    tabelas = [pd.read_csv(f, sep=";", encoding="utf-8-sig") for f in sorted(glob.glob(str(BRONZE / pasta / "*.csv")))]
    t = pd.concat(tabelas, ignore_index=True)
    t["data"] = pd.to_datetime(t["data"], dayfirst=True)
    return t[t[coluna].notna()].set_index("data")[coluna]


def revisar_resultados():
    registros.clear()
    PASTA_REVISAO.mkdir(parents=True, exist_ok=True)
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
    final = pd.read_csv(PASTA_MODELOS / "resultado_final.csv", sep=";", encoding="utf-8-sig")
    preparo = pd.read_parquet(PASTA_MODELOS / "preparo_diario_final.parquet")
    d = preparo[preparo["data"].between(INICIO_PROVA, FIM_PROVA) & preparo["consumo"].notna() & preparo["regular"]]
    for alternativa in ("plano", "C1", "C2", "C4"):
        calculado = (d[alternativa] - d["consumo"]).abs().sum() / len(d)
        tabela = final[(final["janela"] == "t1") & (final["periodo"] == "prova_2026") & (final["universo"] == "sem_atipicos") & (final["alternativa"] == alternativa)]["mae"].iloc[0]
        registrar(f"Erro médio de {alternativa} confere", abs(calculado - tabela) < TOLERANCIA, f"{calculado:.2f}")

    # Consumo das previsoes igual ao CSV
    registrar("Consumo das previsões igual ao CSV", (d.set_index("data")["consumo"] == consumo.reindex(d["data"])).all())

    # Ordem temporal
    registrar("Toda data avaliada vem depois da sua sexta-feira", (preparo["data"] > preparo["origem"]).all())
    registrar("Antecedência de preparo entre 3 e 7 dias", (preparo["data"] - preparo["origem"]).dt.days.between(3, 7).all())
    compras = pd.read_parquet(PASTA_MODELOS / "compras_diario_final.parquet")
    registrar("Antecedência de compras entre 10 e 14 dias", (compras["data"] - compras["origem"]).dt.days.between(10, 14).all())

    # A configuracao precisa ser a receita fixada com 2024 e 2025
    config = pd.read_csv(PASTA_MODELOS / "configuracao_escolhida.csv", sep=";", encoding="utf-8-sig")
    registrar("Configuração é o modelo congelado", (config["modelo"] == "campeao").all() and (config["janela_treino"] == "1_ano").all())

    # O plano humano nao pode explicar o modelo
    sem_plano = preparo[preparo["plano"].notna() & preparo["C2"].notna() & preparo["data"].between(INICIO_PROVA, FIM_PROVA)]
    registrar("Previsão não copia o quantitativo", (sem_plano["C2"] - sem_plano["plano"]).abs().median() > 5, f"diferença mediana {(sem_plano['C2'] - sem_plano['plano']).abs().median():.1f}")

    # O limite de fim de periodo precisa aparecer sem olhar 2026
    t = g1[(g1.index <= "2025-12-31") & g1["usar_no_treino"]]
    fim, resto = t[t["fim_de_bloco"]]["alvo_relativo"].median(), t[~t["fim_de_bloco"]]["alvo_relativo"].median()
    registrar("Queda de fim de período existe até 2025", fim < 0.8 * resto, f"{fim:.2f} contra {resto:.2f}")

    tabela = pd.DataFrame(registros)
    tabela.to_csv(PASTA_REVISAO / "registros.csv", sep=";", index=False, encoding="utf-8-sig")
    print(f"\n{(tabela['resultado'] == 'ok').sum()} de {len(tabela)} verificações ok")
