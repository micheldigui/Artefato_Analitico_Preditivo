"""Monta e executa o notebook com os resultados dos cenarios.

Execute depois de consolidar_resultados.py: python src/montar_notebook_resultados.py
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "notebooks" / "04_resultados_cenarios.ipynb"


def md(texto):
    return nbf.v4.new_markdown_cell(texto.strip())


def code(texto):
    return nbf.v4.new_code_cell(texto.strip())


CELULAS = [
    md("""
# 04. Resultados dos cenários

**Como a avaliação funciona.** Em cada sexta-feira do planejamento, o algoritmo aprende só com dias anteriores e prevê a semana-alvo, como aconteceria na prática.

- **2024 e 2025** serviram para escolher o comitê do artefato: média de CatBoost, KNN e consumo de referência, com os últimos 12 meses. A escolha foi fixada antes de medir 2026.
- **2026 (fevereiro a outubro)** é a prova: nada foi ajustado depois de ver o resultado.
- Os resultados aparecem **com e sem os dias atípicos**.
- O intervalo de 95% mostra a incerteza. **Se ele inclui o zero, a diferença pode ser apenas acaso.**

Os dados de 2026 já haviam sido examinados em etapas anteriores do projeto, então esta prova é retrospectiva. Os termos estão no `GLOSSARIO.md`.
"""),
    code("""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append("../src")
from nomes import ALTERNATIVAS, CENARIOS, HISTORICOS, JANELAS, MODELOS, UNIVERSOS, rotulo

pd.set_option("display.max_rows", 90, "display.max_columns", 30, "display.width", 180, "display.max_colwidth", 60)
RESULTADOS = Path("..") / "resultados" / "modelos"
FIGURAS = Path("..") / "resultados" / "figuras"

SUPERFICIE, TEXTO2, GRADE = "#fcfcfb", "#52514e", "#e6e5e0"
AZUL, LARANJA, AQUA, VIOLETA, CINZA = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#9b9a93"
plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE, "axes.edgecolor": GRADE,
    "axes.labelcolor": TEXTO2, "xtick.color": TEXTO2, "ytick.color": TEXTO2, "axes.grid": True, "grid.color": GRADE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True, "font.size": 10,
})

def ler(nome):
    return pd.read_csv(RESULTADOS / f"{nome}.csv", sep=";", encoding="utf-8-sig")

resultado, escolhas, janelas, metricas = ler("resultado_final"), ler("configuracao_escolhida"), ler("comparacao_janelas_treino"), ler("metricas_janelas_treino")
preparo = pd.read_parquet(RESULTADOS / "preparo_diario_final.parquet")
compras = pd.read_parquet(RESULTADOS / "compras_diario_final.parquet")

def intervalo(linha):
    return "" if pd.isna(linha["ic_inf"]) else f"{linha['ic_inf']:.1f} a {linha['ic_sup']:.1f}"

def tabela(janela, periodo, universo, nome_ref):
    d = resultado[(resultado["janela"] == janela) & (resultado["periodo"] == periodo) & (resultado["universo"] == universo)].copy()
    d["Intervalo de 95%"] = d.apply(intervalo, axis=1)
    d["alternativa"] = d["alternativa"].map(ALTERNATIVAS)
    colunas = {"alternativa": "Alternativa", "mae": rotulo("mae"), "vies": rotulo("vies"), "excesso": rotulo("excesso"), "deficit": rotulo("deficit"),
               "dias_deficit": "% de dias com déficit potencial", "dif_referencia": f"Diferença para {nome_ref}", "Intervalo de 95%": "Intervalo de 95%"}
    return d[list(colunas)].rename(columns=colunas).round(1).reset_index(drop=True)
"""),
    md("## 1. Quanto histórico usar para aprender\n\nCada linha é o comitê campeão com um histórico diferente. Menor erro médio é melhor."),
    code("""
base = metricas[(metricas["universo"] == "sem_atipicos")].merge(escolhas[["janela", "cenario", "modelo"]], on=["janela", "cenario", "modelo"])
tab = base.pivot_table(index=["janela", "cenario"], columns=["periodo", "janela_treino"], values="mae").round(1)
tab = tab.rename(columns=HISTORICOS, level=1).rename(columns={"escolha_2025": "Escolha em 2025", "prova_2026": "Prova em 2026"}, level=0)
tab.rename(index=JANELAS, level=0).rename(index=CENARIOS, level=1).rename_axis(index=["Decisão", "Cenário"], columns=["Período", "Histórico"])
"""),
    code("""
ordem = ["tudo", "2_anos", "1_ano", "ano_anterior_e_atual", "6_meses"]
fig, eixos = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
for ax, (janela, cenario) in zip(eixos, [("t1", "C2"), ("t2", "C2")]):
    d = base[(base["janela"] == janela) & (base["cenario"] == cenario)].set_index("janela_treino")
    p25 = d[d["periodo"] == "escolha_2025"]["mae"].reindex(ordem)
    p26 = d[d["periodo"] == "prova_2026"]["mae"].reindex(ordem)
    y = range(len(ordem))
    ax.scatter(p25, y, s=60, color=AZUL, label="Escolha em 2025", zorder=3)
    ax.scatter(p26, y, s=60, color=LARANJA, marker="D", label="Prova em 2026", zorder=3)
    ax.set_yticks(list(y), [HISTORICOS[o] for o in ordem])
    ax.invert_yaxis() if ax is eixos[0] else None
    ax.set_title(JANELAS[janela], fontsize=10, color=TEXTO2)
    ax.set_xlabel("Erro médio (refeições por dia)")
fig.legend(*eixos[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(FIGURAS / "06_historico_de_treino.png", dpi=200)
plt.show()

c = janelas.assign(Decisão=janelas["janela"].map(JANELAS), Cenário=janelas["cenario"].map(CENARIOS), Comparado_com=janelas["comparada"].map(HISTORICOS))
c["Diferença (refeições por dia)"] = c["dif"].round(1)
c["Intervalo de 95%"] = c.apply(intervalo, axis=1)
c[["Decisão", "Cenário", "Comparado_com", "Diferença (refeições por dia)", "Intervalo de 95%"]].rename(columns={"Comparado_com": "Últimos 12 meses contra"})
"""),
    md("## 2. Comitê campeão em cada cenário"),
    code("""
e = escolhas.assign(Decisão=escolhas["janela"].map(JANELAS), Cenário=escolhas["cenario"].map(CENARIOS), Modelo=escolhas["modelo"].map(MODELOS),
                    Histórico=escolhas["janela_treino"].map(HISTORICOS), Erro_em_2025=escolhas["mae"].round(1))
e[["Decisão", "Cenário", "Modelo", "Histórico", "Erro_em_2025"]].rename(columns={"Erro_em_2025": "Erro médio em 2025"})
"""),
    md("## 3. Preparo: prova em 2026\n\nComparação com o quantitativo da nutricionista, nas mesmas datas. O quantitativo chega na sexta-feira à tarde, com a mesma antecedência da previsão."),
    code("""
for universo in ("sem_atipicos", "todos_os_dias"):
    print(UNIVERSOS[universo])
    display(tabela("t1", "prova_2026", universo, "a nutricionista"))
"""),
    code("""
d = resultado[(resultado["janela"] == "t1") & (resultado["periodo"] == "prova_2026") & (resultado["universo"] == "sem_atipicos")].set_index("alternativa")
ordem = ["consumo_tipico", "C1", "C2", "plano", "C4"]
fig, ax = plt.subplots(figsize=(8.5, 3.4))
cores = [CINZA, AZUL, AZUL, LARANJA, AQUA]
ax.barh([ALTERNATIVAS[o] for o in ordem], d.loc[ordem, "mae"], height=0.6, color=cores)
for y, o in enumerate(ordem):
    ax.text(d.loc[o, "mae"] + 0.3, y, f"{d.loc[o, 'mae']:.1f}", va="center", color=TEXTO2)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel("Erro médio (refeições por dia), 149 dias de 2026 sem atípicos")
fig.tight_layout()
fig.savefig(FIGURAS / "07_preparo_prova_2026.png", dpi=200)
plt.show()
"""),
    md("### Mês a mês\n\nO resultado não é uniforme: o artefato e a nutricionista erram em momentos diferentes, e é isso que dá valor ao híbrido."),
    code("""
p = preparo[(preparo["periodo"] == "prova_2026") & preparo["consumo"].notna()].copy()
p["Mês"] = p["data"].dt.strftime("%Y-%m")
mensal = p.groupby("Mês").apply(lambda g: pd.Series({
    "Dias": len(g), "Nutricionista": (g["plano"] - g["consumo"]).abs().mean(), "Artefato (cenário 2)": (g["C2"] - g["consumo"]).abs().mean(),
    "Híbrido (cenário 4)": (g["C4"] - g["consumo"]).abs().mean()}), include_groups=False).round(1)
mensal
"""),
    md("## 4. Compras: prova em 2026\n\nNão existe quantitativo com duas semanas de antecedência, então a referência é o consumo de referência."),
    code("""
for universo in ("sem_atipicos", "todos_os_dias"):
    print(UNIVERSOS[universo])
    display(tabela("t2", "prova_2026", universo, "o consumo de referência"))
"""),
    md("## 5. O algoritmo contra o consumo de referência, no preparo"),
    code("""
tabela("t1_vs_tipico", "prova_2026", "sem_atipicos", "o consumo de referência")
"""),
    md("""
## 6. Limites desta avaliação

- Os excessos e déficits são diferenças entre plano e consumo registrado. Não medem descarte nem pessoas sem refeição.
- A prova cobre um ano só e um único restaurante.
- 2026 já tinha sido observado nas etapas exploratórias do projeto; a confirmação independente depende de previsões guardadas antes do serviço.
- O cenário 3 ficou fora do comitê porque não reduziu o erro em 2025, e a previsão do tempo segue disponível como alerta.
- A configuração anterior foi melhor em 2026, mas não voltou: trocar agora usaria a prova para escolher. Os detalhes estão no `REVISAO.md`.
- O cenário 4 usa os pesos aprendidos em cada ano; em fevereiro de 2026 ainda não há histórico do quantitativo, e o peso começa em 50%.
"""),
]


def main():
    nb = nbf.v4.new_notebook(cells=CELULAS, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(DESTINO.parent)}}).execute()
    nbf.write(nb, DESTINO)
    print(f"Notebook executado: {DESTINO}")


if __name__ == "__main__":
    main()
