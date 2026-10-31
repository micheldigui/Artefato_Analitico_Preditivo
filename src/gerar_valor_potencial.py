"""Gera as pecas da dissertacao com o valor potencial do excesso, em R$.

Os valores por refeicao ficam em config_local.json, e a saida fica em
resultados_locais/. Os dois estao fora do Git: nada aqui deve ir para o repositorio.
O valor e parametrico: nao mede gasto, descarte nem economia. O deficit segue em
refeicoes, sem valoracao.

Execute depois de gerar_figuras_tabelas.py: python src/gerar_valor_potencial.py
"""
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gerar_figuras_tabelas import ARTEFATO, COMBINACAO, CONSUMO, HUMANO, NOME_ARTEFATO, NOME_HIBRIDO, QUANTITATIVO, TEXTO2, barras_horizontais, quebrar

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "resultados_locais"
OPCOES = {QUANTITATIVO: ("plano", HUMANO), NOME_ARTEFATO: ("previsao", ARTEFATO), NOME_HIBRIDO: ("combinacao", COMBINACAO)}


def moeda(v):
    return f"R$ {v:,.2f}".replace(",", "#").replace(".", ",").replace("#", ".")


def resumo(t, coluna, valor):
    e = t[coluna] - t["consumo"]
    return {"Datas": len(t), "Erro médio": round(e.abs().mean(), 1), "Excesso potencial (refeições)": int(e.clip(lower=0).sum()), "Déficit potencial (refeições)": int((-e).clip(lower=0).sum()),
            "% de dias com déficit potencial": round((e < 0).mean() * 100, 1), "Valor potencial do excesso": moeda(e.clip(lower=0).sum() * valor)}


def main():
    SAIDA.mkdir(exist_ok=True)
    config = json.loads((RAIZ / "config_local.json").read_text(encoding="utf-8"))
    valor = config["valor_refeicao"]
    atual, anterior = valor["2026"], valor["ate_2025"]
    serie = pd.read_parquet(RAIZ / "resultados" / "comparacao" / "serie_preparo.parquet")
    margens = pd.read_csv(RAIZ / "resultados" / "planejamento" / "verificacao_folgas_2026.csv", sep=";", encoding="utf-8-sig")
    so = pd.read_csv(RAIZ / "dados" / "referencia_dissertacao" / "setembro_outubro.csv", sep=";", parse_dates=["data"]).merge(serie[["data", "previsao", "combinacao"]], on="data", how="left")
    so = so[so["consumo"].notna()].assign(plano=lambda t: t["nutricionista"])

    x = serie[(serie["data"] >= "2026-01-01") & serie["consumo"].notna() & serie["regular"] & serie["plano"].notna()]
    linhas = [{"Opção": n, **resumo(x, c, atual)} for n, (c, _) in OPCOES.items()]
    m = margens[(margens["decisao"] == "preparo") & (margens["universo"] == "sem_atipicos")].set_index("opcao")
    for chave, nome in (("moderada", "Artefato com margem moderada"), ("alta", "Artefato com margem alta")):
        l = m.loc[chave]
        linhas.append({"Opção": f"{nome} (+{l['folga_media']:.0f} refeições)", "Datas": int(l["dias"]), "Erro médio": round(l["erro_medio"], 1), "Excesso potencial (refeições)": int(l["a_mais"]), "Déficit potencial (refeições)": int(l["a_menos"]),
                       "% de dias com déficit potencial": round(l["dias_a_menos_pct"], 1), "Valor potencial do excesso": moeda(l["a_mais"] * atual)})
    opcoes = pd.DataFrame(linhas)
    opcoes.to_csv(SAIDA / "tabela_opcoes_com_valor_2026.csv", sep=";", index=False, encoding="utf-8-sig")

    fig, eixos = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    nomes = [quebrar(n, 30) for n in opcoes["Opção"]]
    cores = [HUMANO, ARTEFATO, COMBINACAO, ARTEFATO, ARTEFATO]
    barras_horizontais(eixos[0], nomes, opcoes["% de dias com déficit potencial"].to_numpy(), cores, "{:.0f}%")
    barras_horizontais(eixos[1], nomes, opcoes["Excesso potencial (refeições)"].to_numpy() * atual / 1000, cores, "{:.1f} mil")
    eixos[1].invert_yaxis()
    eixos[0].set_xlabel("Dias com déficit potencial (%)")
    eixos[1].set_xlabel("Valor potencial do excesso (R$ mil)")
    fig.tight_layout()
    fig.savefig(SAIDA / "figura_protecao_e_valor_potencial.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    anual = []
    base = serie[serie["consumo"].notna() & serie["regular"] & serie["plano"].notna()]
    for ano, g in base.groupby("ano"):
        preco = atual if ano == 2026 else anterior
        anual.append({"Ano": ano, "Opção": "Quantitativo da nutricionista" if ano == 2026 else "Reservas dos alunos", **resumo(g, "plano", preco)})
        if g["previsao"].notna().all():
            anual += [{"Ano": ano, "Opção": NOME_ARTEFATO, **resumo(g, "previsao", preco)}, {"Ano": ano, "Opção": NOME_HIBRIDO, **resumo(g.dropna(subset=["combinacao"]), "combinacao", preco)}]
    pd.DataFrame(anual).to_csv(SAIDA / "tabela_sintese_por_ano_com_valor.csv", sep=";", index=False, encoding="utf-8-sig")

    recortes = {"28_dias_regulares": so[so["status_qualidade"] == "REGULAR"], "5_dias_atipicos": so[so["status_qualidade"] == "EVENTO_EXTREMO_EXTERNO"],
                "33_dias_com_consumo": so, "semana_de_retorno": so[so["data"].between("2026-10-19", "2026-10-23")]}
    for nome, t in recortes.items():
        pd.DataFrame([{"Opção": n, **resumo(t, c, atual)} for n, (c, _) in OPCOES.items()]).to_csv(SAIDA / f"tabela_{nome}_com_valor.csv", sep=";", index=False, encoding="utf-8-sig")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    for n, (c, cor) in OPCOES.items():
        e = so[c] - so["consumo"]
        traco = "--" if n == NOME_HIBRIDO else "-"
        a.plot(so["data"], e.clip(lower=0).cumsum() * atual / 1000, color=cor, linestyle=traco, linewidth=2, label=n)
        b.plot(so["data"], (-e).clip(lower=0).cumsum(), color=cor, linestyle=traco, linewidth=2)
    a.set_ylabel("Valor potencial do excesso, acumulado (R$ mil)")
    b.set_ylabel("Déficit potencial, acumulado (refeições)")
    b.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(SAIDA / "figura_acumulado_bimestre.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(opcoes.to_string(index=False))


if __name__ == "__main__":
    main()
