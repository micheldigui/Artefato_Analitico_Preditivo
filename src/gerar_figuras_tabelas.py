"""Gera as figuras e tabelas para a dissertacao, com legendas sugeridas.

As figuras nao tem titulo na imagem, porque a legenda fica no texto (ABNT). Cada
alternativa usa sempre a mesma cor. A analise exploratoria usa dias ate 2025 quando
apoia escolhas de modelo; as tabelas descritivas informam o proprio periodo.

Execute depois de revisar_resultados.py: python src/gerar_figuras_tabelas.py
"""
from pathlib import Path

import locale
import textwrap

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter
import nbformat as nbf
import numpy as np
import pandas as pd

from nomes import EVENTOS, HISTORICOS, JANELAS, MODELOS as ALGORITMOS, PREPAROS, PROTEINAS, TRATAMENTOS, rotulo

RAIZ = Path(__file__).resolve().parents[1]
SILVER = RAIZ / "dados" / "silver"
GOLD = RAIZ / "dados" / "gold"
MODELOS = RAIZ / "resultados" / "modelos"
COMPARACAO = RAIZ / "resultados" / "comparacao"
PLANEJAMENTO = RAIZ / "resultados" / "planejamento"
SAIDA = RAIZ / "resultados" / "dissertacao"
FIGURAS, TABELAS = SAIDA / "figuras", SAIDA / "tabelas"
CORTE = "2025-12-31"

SUPERFICIE, TEXTO, TEXTO2, GRADE = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e0"
CONSUMO, HUMANO, ARTEFATO, COMBINACAO, TIPICO = "#1f2933", "#eb6834", "#2a78d6", "#1baf7a", "#9b9a93"
NOME_PLANO = {"Reservas dos alunos": HUMANO, "Quantitativo da nutricionista": HUMANO}
NOMES_CENARIOS = {"consumo_tipico": "Consumo de referência", "H": "Só histórico de consumo", "C1": "Consumo e calendário", "C2": "Inclui o cardápio", "C3": "Inclui a previsão do tempo", "C4": "Híbrido (artefato com quantitativo)", "plano": "Quantitativo da nutricionista"}
QUANTITATIVO = "Quantitativo da nutricionista"
NOME_ARTEFATO = "Artefato (comitê de CatBoost, KNN e consumo de referência)"
NOME_HIBRIDO = "Híbrido (artefato com quantitativo)"
NOMES_ANUAL = {"plano": "Plano humano", "consumo_tipico": "Consumo de referência", "artefato": NOME_ARTEFATO, "combinacao": NOME_HIBRIDO}
CORES_ANUAL = {"plano": HUMANO, "consumo_tipico": TIPICO, "artefato": ARTEFATO, "combinacao": COMBINACAO}
ANOS = ["2023", "2024", "2025", "2026"]

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE, "axes.edgecolor": GRADE, "axes.labelcolor": TEXTO2,
    "xtick.color": TEXTO2, "ytick.color": TEXTO2, "text.color": TEXTO, "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True, "font.size": 11, "legend.frameon": False,
})
for nome_local in ("pt_BR.UTF-8", "Portuguese_Brazil.1252"):
    try:
        locale.setlocale(locale.LC_NUMERIC, nome_local)
        plt.rcParams["axes.formatter.use_locale"] = True
        break
    except locale.Error:
        continue
lista = []


def salvar(fig, nome, legenda):
    numero = sum(1 for tipo, *_ in lista if tipo == "Figura") + 1
    arquivo = f"figura_{numero:02d}_{nome}.png"
    fig.savefig(FIGURAS / arquivo, dpi=300, bbox_inches="tight")
    plt.close(fig)
    lista.append(("Figura", numero, arquivo, legenda))


def tabela(df, nome, legenda, indice=False):
    numero = sum(1 for tipo, *_ in lista if tipo == "Tabela") + 1
    arquivo = f"tabela_{numero:02d}_{nome}.csv"
    df.to_csv(TABELAS / arquivo, sep=";", index=indice, encoding="utf-8-sig", decimal=",")
    lista.append(("Tabela", numero, arquivo, legenda))


def barras_horizontais(ax, nomes, valores, cores, formato="{:.2f}"):
    ax.barh(nomes, valores, height=0.62, color=cores)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    ax.margins(x=0.1)
    for y, v in enumerate(valores):
        rotulo_valor = formato.format(v).replace(".", ",")
        ax.text(v, y, f" {rotulo_valor} " if v >= 0 else f"{rotulo_valor} ", va="center", ha="left" if v >= 0 else "right", color=TEXTO2, fontsize=9)


def eixo_correlacao(ax, minimo=-1):
    """Escala completa da correlacao, com marcas em palavras."""
    marcas = {-1: "-1\nquando um sobe,\no outro cai", 0: "0\nsem relação", 0.5: "0,5\nmoderada", 1: "1\nandam juntos"}
    if minimo < 0:
        marcas.pop(0.5)
    posicoes = [v for v in marcas if v >= minimo]
    ax.set_xlim(minimo, 1.15)
    ax.set_xticks(posicoes, [marcas[v] for v in posicoes], fontsize=9)
    ax.axvline(0, color=TEXTO2, linewidth=1)
    ax.set_xlabel("Correlação com o consumo")


def quebrar(nome, largura=20):
    return textwrap.fill(nome, largura)


def percentual(v):
    return f"{v:+.0f}%".replace("+0%", "0%").replace("-0%", "0%")


def barras_a_partir_do_tipico(ax, nomes, valores, cor=ARTEFATO):
    """Mostra quanto cada grupo ficou acima ou abaixo do consumo tipico, em porcentagem."""
    diferencas = (np.asarray(valores, float) - 1) * 100
    ax.barh(nomes, diferencas, height=0.62, color=cor)
    ax.axvline(0, color=TEXTO2, linewidth=1)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    ax.margins(x=0.15)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: percentual(v)))
    for y, v in enumerate(diferencas):
        ax.text(v, y, f" {percentual(v)} ", va="center", ha="left" if v >= 0 else "right", color=TEXTO2, fontsize=9)
    ax.set_xlabel("Diferença para o consumo de referência (0% é igual à referência)")


def carregar():
    d = {}
    d["fato"] = pd.read_parquet(SILVER / "silver_fato_diaria.parquet")
    d["dias"] = pd.read_parquet(SILVER / "silver_dias_classificados.parquet")
    d["t1"] = pd.read_parquet(GOLD / "gold_t1.parquet")
    d["t2"] = pd.read_parquet(GOLD / "gold_t2.parquet")
    d["base"] = d["fato"][d["fato"]["no_escopo"] & d["fato"]["tem_consumo"]].merge(d["dias"].drop(columns="consumo"), on="data").sort_values("data").reset_index(drop=True)
    d["relativo"] = d["t1"][d["t1"]["usar_no_treino"] & d["t1"]["alvo_relativo"].notna()]
    d["preparo"] = pd.read_parquet(COMPARACAO / "serie_preparo.parquet")
    d["compras"] = pd.read_parquet(COMPARACAO / "serie_compras.parquet")
    d["anual"] = pd.read_csv(COMPARACAO / "comparacao_anual.csv", sep=";", encoding="utf-8-sig")
    d["final"] = pd.read_csv(MODELOS / "resultado_final.csv", sep=";", encoding="utf-8-sig")
    d["janelas"] = pd.read_csv(MODELOS / "metricas_janelas_treino.csv", sep=";", encoding="utf-8-sig")
    d["config"] = pd.read_csv(MODELOS / "configuracao_escolhida.csv", sep=";", encoding="utf-8-sig")
    d["diario"] = pd.read_parquet(MODELOS / "preparo_diario_final.parquet")
    d["folgas"] = pd.read_csv(PLANEJAMENTO / "verificacao_folgas_2026.csv", sep=";", encoding="utf-8-sig")
    d["situacao"] = pd.read_csv(PLANEJAMENTO / "verificacao_por_situacao_2026.csv", sep=";", encoding="utf-8-sig")
    opcoes = d["config"][d["config"]["cenario"] == "C2"]
    d["vencedor_t1"] = opcoes[opcoes["janela"] == "t1"].iloc[0]
    d["vencedor_t2"] = opcoes[opcoes["janela"] == "t2"].iloc[0]
    return d


def figuras_dados(d):
    base = d["base"]
    grupos = [("Enchente registrada", base["tipo_sugerido"].eq("evento_externo_registrado"), "#eb6834", "o"),
              ("Queda sem explicação", base["tipo_sugerido"].isin(["isolado_sem_explicacao", "sequencia_de_dias"]), "#6250d6", "D"),
              ("Queda prevista pelo calendário", base["tratamento"].eq("previsivel_calendario"), "#1baf7a", "s")]
    fig, eixos = plt.subplots(4, 1, figsize=(11, 9.5), sharey=True)
    for ax, ano in zip(eixos, range(2023, 2027)):
        s = base[base["ano"] == ano]
        for i, (_, t) in enumerate(s.groupby(s["data"].diff().dt.days.gt(14).cumsum())):
            ax.plot(t["data"], t["consumo"], color=ARTEFATO, linewidth=1.3, label="Refeições consumidas" if i == 0 else None)
        for nome, filtro, cor, marca in grupos:
            q = s[filtro.loc[s.index]]
            ax.scatter(q["data"], q["consumo"], s=34, color=cor, marker=marca, edgecolor=SUPERFICIE, linewidth=1, zorder=3, label=nome)
        ax.set_ylabel(str(ano))
        ax.set_xlim(pd.Timestamp(f"{ano}-01-20"), pd.Timestamp(f"{ano}-12-20"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    eixos[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.3))
    fig.supylabel("Refeições por dia", color=TEXTO2)
    fig.tight_layout()
    salvar(fig, "consumo_diario", "Consumo diário de refeições e dias fora da rotina (fevereiro de 2023 a outubro de 2026)")

    mapa = base.pivot_table(index="ano", columns="mes", values="consumo", aggfunc="mean").reindex(columns=range(2, 13))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(mapa, cmap="Blues", aspect="auto", vmin=0, vmax=230)
    ax.set_xticks(range(mapa.shape[1]), ["Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"])
    ax.set_yticks(range(mapa.shape[0]), mapa.index)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(mapa.to_numpy()):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="white" if v > 130 else TEXTO, fontsize=9)
    fig.colorbar(im, ax=ax, label="Refeições por dia", fraction=0.025)
    fig.tight_layout()
    salvar(fig, "consumo_ano_mes", "Consumo médio diário por ano e mês")

    r = d["relativo"]
    nomes_dia = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]
    v = r.groupby("dia_semana")["alvo_relativo"].median()
    fig, ax = plt.subplots(figsize=(7, 3))
    barras_a_partir_do_tipico(ax, nomes_dia, v.to_numpy())
    fig.tight_layout()
    salvar(fig, "dia_da_semana", "Consumo comparado à referência por dia da semana")

    blocos = base[base["bloco"].notna()].copy()
    blocos["rel"] = blocos["consumo"] / blocos.groupby("bloco")["consumo"].transform("median")
    p = (blocos[(blocos["dias_ate_fim_bloco"] <= 12) & (blocos["data"] <= CORTE)].groupby("dias_ate_fim_bloco")["rel"].median() - 1) * 100
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(p.index, p.values, width=0.62, color=[HUMANO if i <= 3 else ARTEFATO for i in p.index])
    ax.axhline(0, color=TEXTO2, linewidth=1)
    ax.invert_xaxis()
    ax.set_xticks(p.index)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: percentual(v)))
    ax.set_xlabel("Dias de aula até o fim do período")
    ax.set_ylabel("Diferença para o consumo de referência")
    fig.tight_layout()
    salvar(fig, "fim_do_periodo", "Consumo nos dias finais dos períodos de aulas, com dados até 2025")

    ev = r.groupby("evento_tipo")["alvo_relativo"].agg(["size", "median"])
    ev = ev[ev["size"] >= 3].sort_values("median")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    barras_a_partir_do_tipico(ax, [f"{EVENTOS.get(i, i)} ({n} dias)" for i, n in zip(ev.index, ev["size"])], ev["median"].to_numpy(), [HUMANO if m < 0.95 else ARTEFATO for m in ev["median"]])
    fig.tight_layout()
    salvar(fig, "eventos_do_calendario", "Consumo comparado à referência por evento do calendário")

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, coluna, nomes in ((a, "proteina", PROTEINAS), (b, "preparo", PREPAROS)):
        v = r.groupby(coluna)["alvo_relativo"].agg(["size", "median"])
        v = v[v["size"] >= 5].sort_values("median")
        barras_a_partir_do_tipico(ax, [f"{nomes[i]} ({n} dias)" for i, n in zip(v.index, v["size"])], v["median"].to_numpy())
    fig.tight_layout()
    salvar(fig, "cardapio", "Consumo comparado à referência por proteína e forma de preparo")

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    faixas = [-1, 1, 5, 15, 500]
    nomes = ["Até 1 mm", "1 a 5 mm", "5 a 15 mm", "15 mm ou mais"]
    for ax, coluna, rotulo_x in ((a, "oraculo_chuva_mm", "Chuva que ocorreu no dia"), (b, "previsao_chuva_mm", "Chuva prevista na sexta-feira")):
        q = r.dropna(subset=[coluna])
        v = q.groupby(pd.cut(q[coluna], faixas, labels=nomes), observed=False)["alvo_relativo"].agg(["size", "median"])
        difs = (v["median"] - 1) * 100
        ax.bar(range(4), difs, width=0.62, color=ARTEFATO)
        for x, m in enumerate(difs):
            ax.text(x, m, percentual(m), ha="center", va="bottom" if m >= 0 else "top", color=TEXTO2, fontsize=9)
        ax.set_xticks(range(4), [f"{n}\n({c} dias)" for n, c in zip(nomes, v["size"])], fontsize=9)
        ax.axhline(0, color=TEXTO2, linewidth=1)
        ax.set_ylim(-15, 15)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: percentual(v)))
        ax.set_xlabel(rotulo_x)
    a.set_ylabel("Diferença para o consumo de referência")
    fig.tight_layout()
    salvar(fig, "chuva", "Consumo comparado à referência por faixa de chuva")

    return defasagens(d)


def defasagens(d):
    b = d["base"].set_index("data")
    s = b["consumo"]
    L = pd.DataFrame(index=s.index)
    for k in (1, 2, 3, 5, 10):
        L[f"lag_{k}_dias_servidos"] = s.shift(k)
    for k in (1, 2, 3, 4):
        L[f"mesmo_dia_{k}_semana(s)_antes"] = s.reindex(s.index - pd.Timedelta(days=7 * k)).values
    a = s.shift(1)
    for j in (5, 10, 20):
        L[f"media_movel_{j}d"] = a.rolling(j).mean()
        L[f"mediana_movel_{j}d"] = a.rolling(j).median()
    for j in (5, 10):
        L[f"desvio_padrao_{j}d"] = a.rolling(j).std()
    L["minimo_5d"], L["maximo_5d"] = a.rolling(5).min(), a.rolling(5).max()
    filtro = b["usar_no_treino"] & (b.index <= CORTE)
    linhas = []
    for c in L:
        x = pd.DataFrame({"y": s[filtro], "x": L.loc[filtro, c]}).dropna()
        linhas.append({"Variável": rotulo(c), "Dias analisados": len(x), "Relação linear": x["y"].corr(x["x"]), "Relação pela ordem": x["y"].corr(x["x"], method="spearman"), "Relação média por ano": x.groupby(x.index.year).apply(lambda g: g["y"].corr(g["x"])).mean()})
    tab = pd.DataFrame(linhas).sort_values("Relação linear", ascending=False).round(3)
    tabela(tab, "relacao_defasagens", "Relação das defasagens e medidas móveis com o consumo, conhecendo o consumo até a véspera (até 2025)")
    o = tab.sort_values("Relação linear", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    barras_horizontais(ax, o["Variável"], o["Relação linear"].to_numpy(), [ARTEFATO if v >= 0 else HUMANO for v in o["Relação linear"]])
    eixo_correlacao(ax, 0)
    fig.tight_layout()
    salvar(fig, "relacao_defasagens", "As 10 medidas de consumo passado mais relacionadas ao consumo do dia (até 2025). A lista completa está na tabela")

    linhas = []
    colunas = ["nivel_base", "nivel_exponencial", "nivel_10d", "nivel_20d", "nivel_mesmo_dia", "nivel_bloco_atual", "ultimo_consumo", "media_5d", "mediana_5d", "media_10d", "media_20d",
               "mesmo_dia_ha_1_ocorrencias", "mesmo_dia_ha_2_ocorrencias", "mesmo_dia_ha_3_ocorrencias", "mesmo_dia_ha_4_ocorrencias", "desvio_5d", "desvio_10d", "minimo_5d", "maximo_5d"]
    p = d["t1"][d["t1"]["usar_no_treino"] & (d["t1"]["data"] <= CORTE)]
    c = d["t2"][d["t2"]["usar_no_treino"] & (d["t2"]["data"] <= CORTE)]
    for col in colunas:
        linhas.append({"Variável conhecida na sexta-feira": rotulo(col), JANELAS["t1"]: p["consumo"].corr(p[col]), JANELAS["t2"]: c["consumo"].corr(c[col])})
    tabela(pd.DataFrame(linhas).round(3), "relacao_atributos_sexta", "Relação com o consumo das medidas realmente conhecidas na sexta-feira do planejamento (até 2025)")

    r = d["relativo"][d["relativo"]["data"] <= CORTE].copy()
    grupos = {
        "Consumo recente": ["tendencia_nivel", "coef_variacao_10d", "idade_informacao_dias", "dias_observados_no_bloco", "rel_ultimo_consumo", "rel_media_5d", "rel_nivel_bloco_atual", "rel_mesmo_dia_ha_1_ocorrencias", "rel_mesmo_dia_ha_2_ocorrencias"],
        "Calendário": ["dia_seg", "dia_ter", "dia_qua", "dia_qui", "dia_sex", "dias_ate_fim_bloco", "semana_do_bloco", "fim_de_bloco", "inicio_de_bloco", "vespera_de_pausa", "pos_pausa", "dias_sem_aula_antes", "dias_letivos_na_semana", "evento_especial", "efeito_evento"],
        "Cardápio": ["efeito_proteina", "efeito_preparo", "efeito_prato"],
        "Clima previsto": ["previsao_chuva_mm", "previsao_chuva_manha_mm", "previsao_chuva_acum3d_mm", "previsao_temp_max"],
        "Clima que ocorreu": ["oraculo_chuva_mm", "oraculo_chuva_horas", "oraculo_temp_max", "oraculo_vento_max"],
    }
    linhas = []
    for g, cols in grupos.items():
        for c_ in cols:
            v = r[c_].astype(float)
            linhas.append({"Grupo": g, "Variável": rotulo(c_), "Dias analisados": int(v.notna().sum()), "Relação linear": r["alvo_relativo"].corr(v), "Relação pela ordem": r["alvo_relativo"].corr(v, method="spearman")})
    rel = pd.DataFrame(linhas).sort_values("Relação pela ordem", key=np.abs, ascending=False).round(3)
    tabela(rel, "relacao_variaveis", "Relação das variáveis com o consumo comparado à referência (até 2025)")
    cores = {"Consumo recente": ARTEFATO, "Calendário": COMBINACAO, "Cardápio": HUMANO, "Clima previsto": "#6250d6", "Clima que ocorreu": "#e87ba4"}
    t = rel.head(10)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    barras_horizontais(ax, t["Variável"], t["Relação pela ordem"].to_numpy(), [cores[g] for g in t["Grupo"]])
    presentes = [g for g in cores if g in set(t["Grupo"])]
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=cores[g]) for g in presentes], presentes, ncol=len(presentes), loc="lower center", bbox_to_anchor=(0.4, 1))
    eixo_correlacao(ax)
    fig.tight_layout()
    salvar(fig, "relacao_variaveis", "As 10 variáveis mais relacionadas ao consumo comparado à referência (até 2025). A lista completa está na tabela")

    todas = p[colunas[:13]].corr()
    tabela(todas.rename(index=rotulo, columns=rotulo).round(2).reset_index(names="Medida"), "informacoes_repetidas", "Correlação entre as medidas de consumo recente")
    nivel = ["nivel_base", "media_5d", "media_20d", "ultimo_consumo", "mesmo_dia_ha_1_ocorrencias", "mesmo_dia_ha_4_ocorrencias"]
    m = p[nivel].corr()
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    nomes = [quebrar(rotulo(c), 22) for c in nivel]
    ax.set_xticks(range(len(nivel)), nomes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(nivel)), nomes, fontsize=9)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(m.to_numpy()):
        ax.text(j, i, f"{v:.2f}".replace(".", ","), ha="center", va="center", fontsize=10, color="white" if v > 0.75 else TEXTO)
    fig.tight_layout()
    salvar(fig, "informacoes_repetidas", "Correlação entre seis medidas de consumo recente. Perto de 1, duas medidas contam quase a mesma história")

    linhas = []
    for nome, g, decisao in (("t1", d["t1"], "Preparo"), ("t2", d["t2"], "Compras")):
        for ant, s in g.dropna(subset=["previsao_chuva_mm", "oraculo_chuva_mm"]).groupby("antecedencia_dias"):
            linhas.append({"Decisão": decisao, "Antecedência (dias)": ant, "Dias": len(s), "Relação entre previsão e chuva real": s["previsao_chuva_mm"].corr(s["oraculo_chuva_mm"])})
    q = pd.DataFrame(linhas).round(3)
    tabela(q, "qualidade_previsao_tempo", "Relação entre a chuva prevista e a chuva que ocorreu, por antecedência")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(q["Antecedência (dias)"], q["Relação entre previsão e chuva real"], width=0.7, color=[ARTEFATO if x == "Preparo" else HUMANO for x in q["Decisão"]])
    ax.set_xticks(q["Antecedência (dias)"])
    ax.set_xlabel("Antecedência da previsão (dias)")
    for x, v in zip(q["Antecedência (dias)"], q["Relação entre previsão e chuva real"]):
        ax.text(x, max(v, 0), f"{v:.2f}".replace(".", ","), ha="center", va="bottom", color=TEXTO2, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1], ["0\nnenhum", "0,5\nmédio", "1\ntotal"])
    ax.set_ylabel("Acerto da previsão de chuva")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=ARTEFATO), plt.Rectangle((0, 0), 1, 1, color=HUMANO)], ["Preparo", "Compras"])
    fig.tight_layout()
    salvar(fig, "qualidade_previsao_tempo", "Qualidade da previsão de chuva por antecedência")


def figuras_regimes(d):
    t = d["preparo"][d["preparo"]["consumo"].notna() & d["preparo"]["plano"].notna()].copy()
    base = d["t1"][d["t1"]["consumo"].notna()].copy()
    base["plano"] = base["plano_nutricionista"].fillna(base["reserva_discente"])
    base = base.dropna(subset=["plano"])
    fig, eixos = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    for ax, q, nome in ((eixos[0], base[base["ano"] < 2026], "Reservas dos alunos"), (eixos[1], base[base["ano"] == 2026], "Quantitativo da nutricionista")):
        ax.scatter(q["plano"], q["consumo"], s=16, color=HUMANO, alpha=0.55, linewidth=0)
        ax.plot([0, 300], [0, 300], color=TEXTO2, linestyle="--", linewidth=1)
        ax.set_xlabel(f"{nome} (refeições)")
    eixos[0].set_ylabel("Refeições consumidas")
    fig.tight_layout()
    salvar(fig, "plano_contra_consumo", "Plano humano e consumo registrado; a linha indica plano igual ao consumo")

    dif = [base[base["ano"] == a]["plano"] - base[base["ano"] == a]["consumo"] for a in (2023, 2024, 2025, 2026)]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.boxplot(dif, tick_labels=["2023\nReservas", "2024\nReservas", "2025\nReservas", "2026\nQuantitativo"], patch_artist=True, widths=0.5, showfliers=True,
               boxprops={"facecolor": HUMANO, "alpha": 0.5, "edgecolor": TEXTO2}, medianprops={"color": TEXTO}, flierprops={"markersize": 3, "markerfacecolor": TEXTO2, "markeredgecolor": TEXTO2})
    ax.axhline(0, color=TEXTO2, linestyle="--", linewidth=1)
    ax.set_ylabel("Plano menos consumo (refeições)")
    fig.tight_layout()
    salvar(fig, "diferenca_plano_consumo", "Diferença diária entre plano humano e consumo; valores positivos indicam plano acima do consumo")

    linhas = []
    for a, g in base.groupby("ano"):
        e = g["plano"] - g["consumo"]
        linhas.append({"Ano": a, "Fonte do plano": "Quantitativo da nutricionista" if a == 2026 else "Reservas dos alunos", "Dias": len(g), "Consumo médio": g["consumo"].mean(), "Plano médio": g["plano"].mean(),
                       "Excesso potencial (refeições)": e.clip(lower=0).sum(), "Déficit potencial (refeições)": (-e).clip(lower=0).sum(), "% de dias com déficit potencial": (e < 0).mean() * 100, "Erro médio": e.abs().mean()})
    tab = pd.DataFrame(linhas).round(1)
    tabela(tab, "planos_por_ano", "Plano humano e consumo registrado por ano. O ano de 2026 vai até outubro")
    x = np.arange(len(tab))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - 0.2, tab["Excesso potencial (refeições)"], 0.38, color=ARTEFATO, label="Excesso potencial")
    ax.bar(x + 0.2, tab["Déficit potencial (refeições)"], 0.38, color=HUMANO, label="Déficit potencial")
    ax.set_xticks(x, tab["Ano"].astype(str))
    ax.set_ylabel("Refeições no ano")
    ax.legend()
    fig.tight_layout()
    salvar(fig, "excesso_deficit_ano", "Excesso potencial e déficit potencial do plano humano, por ano")


def torneio_de_algoritmos(d):
    """Mostra como o comite campeao se compara aos 15 algoritmos, so com 2024 e 2025."""
    torneio = pd.read_csv(MODELOS / "torneio_ate_2025.csv", sep=";", encoding="utf-8-sig")
    previsoes = pd.read_parquet(MODELOS / "torneio_previsoes.parquet")
    ideias = pd.read_csv(MODELOS / "experimentos_ate_2025.csv", sep=";", encoding="utf-8-sig")
    fig, eixos = plt.subplots(1, 2, figsize=(11, 6.2))
    linhas = []
    for ax, janela in zip(eixos, ("t1", "t2")):
        gold = d[janela].set_index("data")
        q = previsoes[(previsoes["janela"] == janela) & (previsoes["cenario"] == "C2") & (previsoes["historico"] == "1_ano")]
        partes = [q[q["algoritmo"] == a].set_index("data")["previsao"] for a in ("CatBoost", "KNN")]
        juntos = gold.loc[partes[0].index, ["consumo", "tratamento", "ano", "nivel_base"]].assign(a=partes[0], b=partes[1])
        juntos = juntos[juntos["consumo"].notna() & juntos["tratamento"].isin(["normal", "previsivel_calendario", "alto_mantido"]) & (juntos.index <= CORTE)]
        erro_campeao = (juntos[["a", "b", "nivel_base"]].mean(axis=1) - juntos["consumo"]).abs().groupby(juntos["ano"]).mean().mean()
        base = torneio[(torneio["janela"] == janela) & (torneio["cenario"] == "C2") & (torneio["historico"] == "1_ano")]
        v = dict(zip(base["algoritmo"], base["erro_medio_2024_2025"]))
        v["Comitê campeão"] = erro_campeao
        v["Consumo de referência"] = ideias[(ideias["janela"] == janela) & (ideias["ideia"] == "Consumo de referência")]["media"].iloc[0]
        ordem = sorted(v, key=v.get)
        cores = [ARTEFATO if n == "Comitê campeão" else TIPICO if n == "Consumo de referência" else "#a6c8ee" for n in ordem]
        barras_horizontais(ax, ordem, [v[n] for n in ordem], cores, "{:.1f}")
        ax.set_title(JANELAS[janela], fontsize=10, color=TEXTO2)
        ax.set_xlabel("Erro médio em 2024 e 2025 (refeições por dia)")
        linhas += [{"Decisão": JANELAS[janela], "Alternativa": n, "Erro médio em 2024 e 2025": round(v[n], 2)} for n in ordem]
    fig.tight_layout()
    salvar(fig, "torneio_de_algoritmos", "Algoritmos testados com a mesma informação e os últimos 12 meses de aprendizado. Dados de 2024 e 2025, usados para escolher o comitê campeão")
    tabela(pd.DataFrame(linhas), "torneio_de_algoritmos", "Erro médio dos algoritmos testados, com os dados usados para a escolha")
    tabela(ideias.rename(columns={"janela": "Decisão", "ideia": "Ideia testada", "erro_2024": "Erro médio em 2024", "erro_2025": "Erro médio em 2025", "media": "Média dos dois anos"}).assign(**{"Decisão": lambda t: t["Decisão"].map(JANELAS)}).sort_values(["Decisão", "Média dos dois anos"]).round(2), "ideias_testadas", "Ideias de melhoria testadas com 2024 e 2025. Uma ideia só seria adotada com ganho de pelo menos 0,5 refeição por dia")


def figuras_resultados(d):
    f = d["final"]
    prova = f[(f["periodo"] == "prova_2026") & (f["universo"] == "sem_atipicos")]
    ordem_t1 = ["consumo_tipico", "H", "C1", "C2", "C3", "plano", "C4"]
    ordem_t2 = ["consumo_tipico", "H", "C1", "C2"]
    cores = {"consumo_tipico": TIPICO, "H": ARTEFATO, "C1": ARTEFATO, "C2": ARTEFATO, "C3": ARTEFATO, "plano": HUMANO, "C4": COMBINACAO}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    for ax, janela, ordem, titulo in ((a, "t1", ordem_t1, JANELAS["t1"]), (b, "t2", ordem_t2, JANELAS["t2"])):
        v = prova[prova["janela"] == janela].set_index("alternativa").reindex(ordem).dropna(subset=["mae"])
        barras_horizontais(ax, [NOMES_CENARIOS[i] for i in v.index], v["mae"].to_numpy(), [cores[i] for i in v.index], "{:.1f}")
        ax.set_title(titulo, fontsize=10, color=TEXTO2)
        ax.set_xlabel("Erro médio (refeições por dia)")
    fig.tight_layout()
    salvar(fig, "cenarios_2026", "Erro médio por cenário em 2026, sem dias atípicos")

    linhas = []
    nomes = {"prova_2026": "Prova em 2026", "escolha_2025": "Escolha em 2025"}
    for (periodo, universo, janela), g in f[f["janela"].isin(["t1", "t2"])].groupby(["periodo", "universo", "janela"]):
        for _, l in g.iterrows():
            linhas.append({"Período": nomes[periodo], "Dias": "Sem atípicos" if universo == "sem_atipicos" else "Todos", "Decisão": JANELAS[janela], "Alternativa": NOMES_CENARIOS.get(l["alternativa"]), "Datas": l["dias"], "Erro médio": l["mae"], "Erro com peso nos grandes desvios": l["rmse"],
                           "Erro em relação ao total (%)": l["wape"], "Variação explicada": l["r2"], "Tendência": l["vies"], "Excesso potencial": l["excesso"], "Déficit potencial": l["deficit"], "% de dias com déficit potencial": l["dias_deficit"],
                           "Diferença para a referência": l["dif_referencia"], "Intervalo de 95%, início": l["ic_inf"], "Intervalo de 95%, fim": l["ic_sup"]})
    tabela(pd.DataFrame(linhas).round(2), "resultado_cenarios", "Desempenho das alternativas por período. No preparo a referência é o quantitativo; nas compras, o consumo de referência")

    j = d["janelas"][d["janelas"]["universo"] == "sem_atipicos"].merge(d["config"][["janela", "cenario", "modelo"]], on=["janela", "cenario", "modelo"])
    ordem = ["tudo", "2_anos", "1_ano", "ano_anterior_e_atual", "6_meses"]
    fig, eixos = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    opcoes = d["config"][d["config"]["cenario"] == "C2"].set_index("janela")
    for ax, janela in zip(eixos, ("t1", "t2")):
        q = j[(j["janela"] == janela) & (j["cenario"] == opcoes.loc[janela, "cenario"]) & (j["modelo"] == opcoes.loc[janela, "modelo"])].set_index("janela_treino")
        ax.scatter(q[q["periodo"] == "escolha_2025"]["mae"].reindex(ordem), range(5), s=60, color=ARTEFATO, label="Medido em 2025", zorder=3)
        ax.scatter(q[q["periodo"] == "prova_2026"]["mae"].reindex(ordem), range(5), s=60, color=HUMANO, marker="D", label="Prova em 2026", zorder=3)
        ax.set_yticks(range(5), [HISTORICOS[o] for o in ordem])
        ax.set_title(JANELAS[janela], fontsize=10, color=TEXTO2)
        ax.set_xlabel("Erro médio (refeições por dia)")
    eixos[0].invert_yaxis()
    fig.legend(*eixos[0].get_legend_handles_labels(), loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    salvar(fig, "historico_de_treino", "Erro médio conforme o histórico usado para aprender")
    p = d["janelas"][d["janelas"]["universo"] == "sem_atipicos"].pivot_table(index=["janela", "cenario", "modelo"], columns=["periodo", "janela_treino"], values="mae").round(2).reset_index()
    p.columns = [" ".join(str(x) for x in c if x).replace("escolha_2025", "2025").replace("prova_2026", "2026") for c in p.columns]
    tabela(p, "historico_de_treino", "Erro médio por modelo e histórico usado para aprender, sem dias atípicos")

    cfg = d["config"].assign(Decisão=d["config"]["janela"].map(JANELAS), Cenário=d["config"]["cenario"].map(lambda c: NOMES_CENARIOS.get(c, c)), Histórico=d["config"]["janela_treino"].map(HISTORICOS))
    cfg["Algoritmo"] = cfg["modelo"].map(ALGORITMOS)
    tabela(cfg[["Decisão", "Cenário", "Algoritmo", "Histórico", "mae"]].rename(columns={"mae": "Erro médio em 2025"}).round(2), "configuracao_escolhida", "Comitê campeão em cada cenário. A escolha usou 2024 e 2025")
    torneio_de_algoritmos(d)

    a = d["anual"]
    s = a[a["universo"] == "sem_atipicos"]
    anos = sorted(s["ano"].unique())
    nomes_alt = list(NOMES_ANUAL)
    fig, ax = plt.subplots(figsize=(9, 4))
    largura = 0.2
    for i, alt in enumerate(nomes_alt):
        q = s[s["alternativa"] == alt].set_index("ano").reindex(anos)
        ax.bar(np.arange(len(anos)) + (i - 1.5) * largura, q["mae"], largura * 0.92, color=CORES_ANUAL[alt], label=NOMES_ANUAL[alt])
    ax.set_xticks(range(len(anos)), [f"{y}\n{'Quantitativo' if y == 2026 else 'Reservas'}" for y in anos])
    ax.set_ylabel("Erro médio (refeições por dia)")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "comparacao_anual", "Erro médio por ano nas mesmas datas, sem dias atípicos. Em 2023 não há histórico anterior para o artefato")
    t = a.assign(Alternativa=a["alternativa"].map(NOMES_ANUAL), Dias_avaliados=a["universo"].map({"sem_atipicos": "Sem atípicos", "todos_os_dias": "Todos"}))
    t = t[["ano", "regime", "Dias_avaliados", "Alternativa", "dias", "mae", "rmse", "r2", "vies", "excesso", "deficit", "dias_deficit", "dif_plano", "ic_inf", "ic_sup"]].round(2)
    t.columns = ["Ano", "Plano humano do ano", "Dias considerados", "Alternativa", "Datas", "Erro médio", "Erro com peso nos grandes desvios", "Variação explicada", "Tendência", "Excesso potencial", "Déficit potencial", "% de dias com déficit potencial", "Diferença para o plano humano", "Intervalo de 95%, início", "Intervalo de 95%, fim"]
    tabela(t, "comparacao_anual", "Comparação anual entre plano humano, consumo de referência, artefato e combinação, nas mesmas datas")

    x = d["preparo"][(d["preparo"]["data"] >= "2026-01-01") & d["preparo"]["consumo"].notna()]
    paineis = [("plano", QUANTITATIVO, HUMANO), ("previsao", NOME_ARTEFATO, ARTEFATO), ("combinacao", NOME_HIBRIDO, COMBINACAO)]
    fig, eixos = plt.subplots(3, 1, figsize=(11, 8), sharex=True, sharey=True)
    for ax, (coluna, nome, cor) in zip(eixos, paineis):
        for _, t in x.groupby(x["data"].diff().dt.days.gt(10).cumsum()):
            ax.plot(t["data"], t["consumo"], color=CONSUMO, linewidth=1.5, label="Consumo real" if t.index[0] == x.index[0] else None)
            ax.plot(t["data"], t[coluna], color=cor, linewidth=1.5, label=nome if t.index[0] == x.index[0] else None)
        ax.legend(ncol=2, loc="lower left")
        ax.set_ylabel("Refeições por dia")
    eixos[2].xaxis.set_major_locator(mdates.MonthLocator())
    eixos[2].xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    fig.tight_layout()
    salvar(fig, "serie_2026", "Consumo registrado e cada forma de planejamento em 2026")

    x = x.assign(mes=x["data"].dt.strftime("%m/%Y"))
    cols = {QUANTITATIVO: ("plano", HUMANO), NOME_ARTEFATO: ("previsao", ARTEFATO), NOME_HIBRIDO: ("combinacao", COMBINACAO)}

    medias = pd.DataFrame({"Consumo real": x.groupby("mes", sort=False)["consumo"].mean(), **{n: x.groupby("mes", sort=False)[c].mean() for n, (c, _) in cols.items()}})
    fig, ax = plt.subplots(figsize=(11, 4.2))
    posicoes = np.arange(len(medias))
    for i, (nome, cor) in enumerate(zip(medias.columns, (CONSUMO, HUMANO, ARTEFATO, COMBINACAO))):
        ax.bar(posicoes + (i - 1.5) * 0.2, medias[nome], 0.18, color=cor, label=nome)
    ax.set_xticks(posicoes, medias.index)
    ax.set_ylabel("Refeições por dia, em média")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "consumo_real_e_alternativas_por_mes", "Consumo real médio e planejamento médio de cada alternativa por mês, em 2026")
    tabela(medias.round(0).astype(int).reset_index().rename(columns={"mes": "Mês"}), "consumo_real_e_alternativas_por_mes", "Consumo real médio e planejamento médio de cada alternativa por mês, em 2026")

    erros_abs = pd.DataFrame({n: (x[c] - x["consumo"]).abs() for n, (c, _) in cols.items()})
    mais_perto = erros_abs.eq(erros_abs.min(axis=1), axis=0).sum().reindex(cols).astype(int)
    fig, ax = plt.subplots(figsize=(9.5, 2.8))
    ax.barh([quebrar(n, 34) for n in mais_perto.index], mais_perto.to_numpy(), height=0.62, color=[c for _, c in cols.values()])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.18)
    for y, v in enumerate(mais_perto):
        ax.text(v, y, f" {v} de {len(x)} dias ({v / len(x) * 100:.0f}%)", va="center", color=TEXTO2, fontsize=9)
    ax.set_xlabel("Dias em que foi a opção que mais se aproximou do consumo real")
    fig.tight_layout()
    salvar(fig, "mais_perto_do_consumo_real", f"Em cada um dos {len(x)} dias de 2026, qual opção ficou mais perto do consumo real. Empates contam para todas as opções empatadas, por isso a soma passa de {len(x)}")
    tabela(pd.DataFrame({"Alternativa": mais_perto.index, "Dias mais perto do consumo real": mais_perto.values, "% dos dias": (mais_perto.values / len(x) * 100).round(1)}), "mais_perto_do_consumo_real", "Dias em que cada alternativa ficou mais perto do consumo real em 2026. Empates contam para todas as opções empatadas")

    j_todos = d["janelas"][(d["janelas"]["universo"] == "sem_atipicos") & (d["janelas"]["periodo"] == "prova_2026")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, janela, vencedor, referencia in ((a, "t1", d["vencedor_t1"], ("Quantitativo da nutricionista", d["final"].query("janela=='t1' and periodo=='prova_2026' and universo=='sem_atipicos' and alternativa=='plano'")["mae"].iloc[0], HUMANO)), (b, "t2", d["vencedor_t2"], ("Consumo de referência", d["final"].query("janela=='t2' and periodo=='prova_2026' and universo=='sem_atipicos' and alternativa=='consumo_tipico'")["mae"].iloc[0], TIPICO))):
        q = j_todos[(j_todos["janela"] == janela) & (j_todos["cenario"] == vencedor["cenario"]) & (j_todos["janela_treino"] == vencedor["janela_treino"])].sort_values("mae")
        nomes_q = [quebrar(ALGORITMOS[m] + (" (campeão)" if m == vencedor["modelo"] else ""), 42) for m in q["modelo"]] + [referencia[0]]
        barras_horizontais(ax, nomes_q, [*q["mae"], referencia[1]], [ARTEFATO if m == vencedor["modelo"] else "#a6c8ee" for m in q["modelo"]] + [referencia[2]], "{:.1f}")
        ax.set_title(JANELAS[janela], fontsize=10, color=TEXTO2)
        ax.set_xlabel("Erro médio em 2026 (refeições por dia)")
    fig.tight_layout()
    salvar(fig, "algoritmos_comparados", "Erro médio de cada algoritmo em 2026, sem dias atípicos, com destaque para o comitê campeão, fixado antes desta medição")

    m = pd.DataFrame({n: x.groupby("mes", sort=False).apply(lambda g: (g[c] - g["consumo"]).abs().mean(), include_groups=False) for n, (c, _) in cols.items()})
    m.insert(0, "Datas", x.groupby("mes", sort=False).size())
    tabela(m.round(1).reset_index().rename(columns={"mes": "Mês"}), "erro_mensal_2026", "Erro médio por mês em 2026, com todos os dias")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    consumo_mes = x.groupby("mes", sort=False)["consumo"].mean()
    ax.bar(consumo_mes.index, consumo_mes.values, width=0.62, color="#dddcd6", label="Consumo real (média do mês)")
    for mes, v in consumo_mes.items():
        ax.text(mes, v, f"{v:.0f}", ha="center", va="bottom", color=TEXTO2, fontsize=9)
    for n, (_, cor) in cols.items():
        ax.plot(m.index, m[n], color=cor, linewidth=2, marker="o", markersize=6, linestyle="--" if n == NOME_HIBRIDO else "-", label=f"Erro médio: {n}")
    ax.set_ylabel("Refeições por dia")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "erro_mensal_2026", "Consumo real médio (barras) e erro médio de cada opção (linhas) por mês em 2026, na mesma escala de refeições, com todos os dias. Setembro: híbrido igual ao quantitativo")

    reg = x[x["regular"]]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.boxplot([reg[c] - reg["consumo"] for c, _ in cols.values()], tick_labels=[quebrar(n, 28) for n in cols], patch_artist=True, widths=0.5, showfliers=True,
               boxprops={"facecolor": TIPICO, "alpha": 0.35, "edgecolor": TEXTO2}, medianprops={"color": TEXTO}, flierprops={"markersize": 3, "markerfacecolor": TEXTO2, "markeredgecolor": TEXTO2})
    ax.axhline(0, color=TEXTO2, linestyle="--", linewidth=1)
    ax.set_ylabel("Plano menos consumo (refeições)")
    fig.tight_layout()
    salvar(fig, "distribuicao_erros_2026", "Distribuição dos desvios em 2026, sem dias atípicos")

    fig, eixos = plt.subplots(1, 3, figsize=(11, 3.9), sharex=True, sharey=True)
    for ax, (n, (c, cor)) in zip(eixos, cols.items()):
        ax.scatter(reg[c], reg["consumo"], s=16, color=cor, alpha=0.6, linewidth=0)
        ax.plot([60, 300], [60, 300], color=TEXTO2, linestyle="--", linewidth=1)
        ax.set_xlabel(quebrar(n, 34))
    eixos[0].set_ylabel("Refeições consumidas")
    fig.tight_layout()
    salvar(fig, "planejado_contra_consumo_2026", "Planejado e consumido em 2026, sem dias atípicos; a linha indica plano igual ao consumo")

    v = d["folgas"]
    fv = v[v["universo"] == "sem_atipicos"]
    nomes_f = {"sem_folga": "Sem margem", "moderada": "Margem moderada", "alta": "Margem alta", "nutricionista": "Quantitativo da nutricionista"}
    prep = fv[fv["decisao"] == "preparo"].set_index("opcao")
    hibrido = reg["combinacao"] - reg["consumo"]
    opcoes = [(QUANTITATIVO, prep.loc["nutricionista", "a_mais"], prep.loc["nutricionista", "a_menos"]),
              ("Artefato sem margem", prep.loc["sem_folga", "a_mais"], prep.loc["sem_folga", "a_menos"]),
              (f"Artefato com margem moderada (+{prep.loc['moderada', 'folga_media']:.0f} refeições)", prep.loc["moderada", "a_mais"], prep.loc["moderada", "a_menos"]),
              (f"Artefato com margem alta (+{prep.loc['alta', 'folga_media']:.0f} refeições)", prep.loc["alta", "a_mais"], prep.loc["alta", "a_menos"]),
              (NOME_HIBRIDO, hibrido.clip(lower=0).sum(), (-hibrido).clip(lower=0).sum())]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    posicoes = np.arange(len(opcoes))
    for deslocamento, indice, cor, nome in ((-0.2, 1, ARTEFATO, "Excesso potencial"), (0.2, 2, HUMANO, "Déficit potencial")):
        valores = [o[indice] for o in opcoes]
        ax.bar(posicoes + deslocamento, valores, 0.38, color=cor, label=nome)
        for x_p, valor in zip(posicoes + deslocamento, valores):
            ax.text(x_p, valor, f"{valor:,.0f}".replace(",", "."), ha="center", va="bottom", color=TEXTO2, fontsize=9)
    ax.set_xticks(posicoes, [quebrar(o[0], 22) for o in opcoes], fontsize=9)
    ax.set_ylabel(f"Refeições somadas em {len(reg)} dias de 2026")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "excesso_e_deficit_2026", "Soma do excesso potencial e do déficit potencial de cada opção de preparo em 2026, sem dias atípicos. A margem muda a cada semana; entre parênteses está o acréscimo médio por dia")

    q = fv[fv["opcao"].isin(["sem_folga", "moderada", "alta"])]
    fig, eixos = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, decisao in zip(eixos, ("preparo", "compras")):
        s = q[q["decisao"] == decisao].set_index("opcao").reindex(["sem_folga", "moderada", "alta"])
        posicoes = np.arange(3)
        for deslocamento, coluna, cor, nome in ((-0.2, "prometido_dias_a_menos_pct", TIPICO, "Risco informado antes da semana"), (0.2, "dias_a_menos_pct", ARTEFATO, "O que aconteceu")):
            ax.bar(posicoes + deslocamento, s[coluna], 0.38, color=cor, label=nome)
            for x_p, valor in zip(posicoes + deslocamento, s[coluna]):
                ax.text(x_p, valor, f"{valor:.0f}%", ha="center", va="bottom", color=TEXTO2, fontsize=9)
        ax.set_xticks(posicoes, ["Sem margem", "Margem moderada", "Margem alta"])
        ax.set_title(decisao.capitalize(), fontsize=10, color=TEXTO2)
    eixos[0].set_ylabel("% de dias com déficit potencial")
    fig.legend(*eixos[0].get_legend_handles_labels(), loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    salvar(fig, "risco_informado_e_observado", "Risco de déficit potencial informado antes de cada semana e o que aconteceu em 2026, sem dias atípicos")
    t = v.assign(Decisão=v["decisao"].str.capitalize(), Opção=v["opcao"].map(nomes_f), Dias_c=v["universo"].map({"sem_atipicos": "Sem atípicos", "todos_os_dias": "Todos"}))
    t = t[["Decisão", "Dias_c", "Opção", "dias", "folga_media", "erro_medio", "a_mais", "a_menos", "prometido_dias_a_menos_pct", "dias_a_menos_pct"]].round(1)
    t.columns = ["Decisão", "Dias considerados", "Opção", "Datas", "Margem média", "Erro médio", "Excesso potencial", "Déficit potencial", "Chance informada (%)", "O que aconteceu (%)"]
    tabela(t, "opcoes_de_margem", "Resultado das opções de margem de segurança em 2026")

    s = d["situacao"]
    preparo = s[s["decisao"] == "preparo"]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x_pos = np.arange(len(preparo))
    for i, (col, nome, tom) in enumerate((("dias_a_menos_sem_folga", "Sem margem", "#a6c8ee"), ("dias_a_menos_moderada", "Margem moderada", "#5c9ee4"), ("dias_a_menos_alta", "Margem alta", ARTEFATO))):
        ax.bar(x_pos + (i - 1) * 0.26, preparo[col], 0.24, color=tom, label=nome)
    ax.set_xticks(x_pos, [f"{a.replace(' ou ', chr(10) + 'ou ')}\n({n} dias)" for a, n in zip(preparo["situacao"], preparo["dias"])], fontsize=9)
    ax.set_ylabel("% de dias com déficit potencial")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "margem_por_situacao", "Dias com déficit potencial por situação do dia, no preparo em 2026")
    t = s.assign(Decisão=s["decisao"].str.capitalize())[["Decisão", "situacao", "dias", "erro_previsao", "dias_a_menos_sem_folga", "dias_a_menos_moderada", "dias_a_menos_alta"]].round(1)
    t.columns = ["Decisão", "Situação do dia", "Datas", "Tendência da previsão", "% abaixo, sem margem", "% abaixo, margem de segurança moderada", "% abaixo, margem de segurança alta"]
    tabela(t, "margem_por_situacao", "Tendência da previsão e proteção das margens de segurança por situação do dia em 2026")

    linhas = []
    for nome, g in (("Preparo", d["preparo"]), ("Compras", d["compras"])):
        q = g[(g["data"] >= "2026-01-01") & g["regular"] & g["consumo"].notna()]
        for ant, u in q.groupby((q["data"] - q["origem"]).dt.days):
            linhas.append({"Decisão": nome, "Antecedência (dias)": ant, "Datas": len(u), "Erro médio": (u["previsao"] - u["consumo"]).abs().mean()})
    a = pd.DataFrame(linhas).round(1)
    tabela(a, "erro_por_antecedencia", "Erro médio do artefato por antecedência em 2026, sem dias atípicos")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(a["Antecedência (dias)"], a["Erro médio"], width=0.7, color=[ARTEFATO if n == "Preparo" else HUMANO for n in a["Decisão"]])
    ax.set_xticks(a["Antecedência (dias)"])
    ax.set_xlabel("Antecedência (dias)")
    ax.set_ylabel("Erro médio (refeições por dia)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=ARTEFATO), plt.Rectangle((0, 0), 1, 1, color=HUMANO)], ["Preparo", "Compras"])
    fig.tight_layout()
    salvar(fig, "erro_por_antecedencia", "Erro médio do artefato por antecedência em 2026, sem dias atípicos")


def tabelas_dados(d):
    base = d["base"]
    res = base.groupby("ano").agg(Datas=("consumo", "size"), Consumo_medio=("consumo", "mean"), Menor=("consumo", "min"), Maior=("consumo", "max"), Dias_fora_da_rotina=("tratamento", lambda s: (s != "normal").sum())).round(1).reset_index()
    tabela(res.rename(columns={"ano": "Ano", "Consumo_medio": "Consumo médio", "Dias_fora_da_rotina": "Dias fora da rotina"}), "consumo_por_ano", "Consumo registrado de segunda a sexta-feira, por ano. O ano de 2026 vai até outubro")
    t = d["dias"]["tratamento"].value_counts().rename_axis("tratamento").reset_index(name="Dias")
    t["Situação"] = t["tratamento"].map(TRATAMENTOS)
    t["Usado para aprender"] = t["tratamento"].map({"normal": "Sim", "previsivel_calendario": "Sim", "alto_mantido": "Sim"}).fillna("Não")
    tabela(t[["Situação", "Dias", "Usado para aprender"]], "tratamento_dos_dias", "Classificação dos dias com consumo registrado e uso no aprendizado")
    r = d["relativo"]
    linhas = []
    for coluna, nome, nomes in (("proteina", "Proteína", PROTEINAS), ("preparo", "Preparo", PREPAROS)):
        for k, v in r.groupby(coluna)["alvo_relativo"]:
            linhas.append({"Grupo": nome, "Categoria": nomes.get(k, k), "Dias": len(v), "Diferença para o consumo de referência (%)": (v.median() - 1) * 100})
    for k, v in r.groupby("evento_tipo")["alvo_relativo"]:
        linhas.append({"Grupo": "Evento", "Categoria": EVENTOS.get(k, k), "Dias": len(v), "Diferença para o consumo de referência (%)": (v.median() - 1) * 100})
    tabela(pd.DataFrame(linhas).round(3), "efeitos_categorias", "Consumo comparado à referência por categoria de cardápio e evento")


def mae(y, p):
    return (y - p).abs().mean()


def figuras_detalhes(d):
    ref = RAIZ / "dados" / "referencia_dissertacao"
    so = pd.read_csv(ref / "setembro_outubro.csv", sep=";", parse_dates=["data"])
    so = so.merge(d["preparo"][["data", "previsao", "combinacao", "plano"]], on="data", how="left")
    atipicos = so[so["status_qualidade"] == "EVENTO_EXTREMO_EXTERNO"]["data"]
    continuo = so.set_index("data").reindex(pd.bdate_range(so["data"].min(), so["data"].max())).rename_axis("data").reset_index()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for dia in atipicos:
        ax.axvspan(dia - pd.Timedelta(hours=12), dia + pd.Timedelta(hours=12), color="#fde8df", zorder=0)
    ax.plot(continuo["data"], continuo["consumo"], color=CONSUMO, linewidth=2, marker="o", markersize=3, label="Consumo real")
    for coluna, nome, cor, traco, espessura in (("previsao", NOME_ARTEFATO, ARTEFATO, "-", 1.5), ("nutricionista", QUANTITATIVO, HUMANO, "-", 2.4), ("combinacao", NOME_HIBRIDO, COMBINACAO, "--", 1.6)):
        ax.plot(continuo["data"], continuo[coluna], color=cor, linewidth=espessura, linestyle=traco, label=nome)
    ax.set_ylabel("Refeições por dia")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.legend([*ax.get_legend_handles_labels()[0], Patch(color="#fde8df")], [*ax.get_legend_handles_labels()[1], "Dias atípicos"], ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "setembro_outubro", "Consumo real e planejamento em setembro e outubro de 2026. Na maior parte do período, o híbrido coincide com o quantitativo, porque o peso do artefato foi zero. As linhas são interrompidas nas datas sem registro")

    obs = so[so["consumo"].notna()]
    recortes = {"28 dias regulares": obs[obs["status_qualidade"] == "REGULAR"], "5 dias atípicos": obs[obs["status_qualidade"] == "EVENTO_EXTREMO_EXTERNO"], "33 dias com consumo": obs}
    alternativas = {QUANTITATIVO: "nutricionista", NOME_ARTEFATO: "previsao", NOME_HIBRIDO: "combinacao"}
    linhas = []
    for recorte, q in recortes.items():
        for nome, coluna in alternativas.items():
            e = q[coluna] - q["consumo"]
            linhas.append({"Recorte": recorte, "Alternativa": nome, "Erro médio": e.abs().mean(), "Excesso potencial": e.clip(lower=0).sum(), "Déficit potencial": (-e).clip(lower=0).sum(), "% de dias com déficit potencial": (e < 0).mean() * 100})
    t = pd.DataFrame(linhas).round(1)
    tabela(t, "setembro_outubro", "Resultado em setembro e outubro de 2026, por recorte de dias")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, q, titulo in ((a, recortes["28 dias regulares"], "28 dias regulares"), (b, recortes["5 dias atípicos"], "5 dias atípicos")):
        ax.boxplot([q["nutricionista"] - q["consumo"], q["previsao"] - q["consumo"], q["combinacao"] - q["consumo"]], tick_labels=[quebrar(n, 20) for n in alternativas], patch_artist=True, widths=0.5,
                   boxprops={"facecolor": TIPICO, "alpha": 0.35, "edgecolor": TEXTO2}, medianprops={"color": TEXTO}, flierprops={"markersize": 3, "markerfacecolor": TEXTO2, "markeredgecolor": TEXTO2})
        ax.axhline(0, color=TEXTO2, linestyle="--", linewidth=1)
        ax.set_title(titulo, fontsize=10, color=TEXTO2)
    a.set_ylabel("Plano menos consumo real (refeições)")
    fig.tight_layout()
    salvar(fig, "regulares_e_atipicos", "Desvios em setembro e outubro de 2026, em dias regulares e atípicos; os painéis têm escalas próprias")

    sem = so[so["data"].between("2026-10-19", "2026-10-23")]
    x = np.arange(len(sem))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, (coluna, nome, cor) in enumerate((("consumo", "Consumo real", CONSUMO), ("nutricionista", QUANTITATIVO, HUMANO), ("previsao", NOME_ARTEFATO, ARTEFATO), ("combinacao", NOME_HIBRIDO, COMBINACAO))):
        ax.bar(x + (i - 1.5) * 0.2, sem[coluna], 0.18, color=cor, label=nome)
    ax.set_xticks(x, sem["data"].dt.strftime("%d/%m"))
    ax.set_ylabel("Refeições por dia")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "semana_de_retorno", "Consumo real e planejamento na semana de retorno, de 19 a 23 de outubro de 2026")
    tabela(pd.DataFrame([{"Alternativa": n, "Total na semana": sem[c].sum(), "Erro médio": mae(sem["consumo"], sem[c])} for n, c in {"Consumo real": "consumo", **{k: v for k, v in alternativas.items()}}.items()]).round(1), "semana_de_retorno", "Planejamento na semana de retorno, de 19 a 23 de outubro de 2026")

    x26 = d["preparo"][(d["preparo"]["data"] >= "2026-01-01") & d["preparo"]["peso_modelo"].notna()].drop_duplicates("origem")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.step(x26["origem"], x26["peso_modelo"] * 100, where="post", color=COMBINACAO, linewidth=2, label="Peso do artefato no híbrido")
    ax.axhline(50, color=TEXTO2, linestyle="--", linewidth=1, label="Pesos iguais: 50% artefato e 50% quantitativo")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Peso do artefato (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "peso_da_combinacao", "Peso do artefato no híbrido, definido a cada semana pelos erros recentes. O restante é o peso do quantitativo")

    q = d["preparo"][(d["preparo"]["data"] >= "2026-01-01") & d["preparo"]["consumo"].notna() & d["preparo"]["regular"]].copy()
    q["dia"] = q["data"].dt.dayofweek
    nomes = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]
    t = pd.DataFrame([{"Dia": nomes[k], "Datas": len(g), QUANTITATIVO: mae(g["consumo"], g["plano"]), NOME_ARTEFATO: mae(g["consumo"], g["previsao"]), NOME_HIBRIDO: mae(g["consumo"], g["combinacao"])} for k, g in q.groupby("dia")]).round(1)
    tabela(t, "erro_por_dia_da_semana", "Erro médio por dia da semana em 2026, sem dias atípicos")
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for i, (nome, cor) in enumerate(((QUANTITATIVO, HUMANO), (NOME_ARTEFATO, ARTEFATO), (NOME_HIBRIDO, COMBINACAO))):
        ax.bar(x + (i - 1) * 0.26, t[nome], 0.24, color=cor, label=nome)
    ax.set_xticks(x, t["Dia"])
    ax.set_ylabel("Erro médio (refeições por dia)")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "erro_por_dia_da_semana", "Erro médio por dia da semana em 2026, sem dias atípicos")


def caixa(ax, x, y, largura, altura, titulo, texto, cor):
    ax.add_patch(FancyBboxPatch((x, y), largura, altura, boxstyle="round,pad=0,rounding_size=0.12", facecolor=cor, edgecolor=TEXTO2, linewidth=0.8))
    ax.text(x + largura / 2, y + altura - 0.17, titulo, ha="center", va="top", fontsize=10, fontweight="bold", color=TEXTO)
    ax.text(x + largura / 2, y + altura - 0.5, texto, ha="center", va="top", fontsize=8.5, color=TEXTO, linespacing=1.3)


def seta(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "-|>", "color": TEXTO2, "linewidth": 1.2})


def diagramas():
    AZ, VD, LJ, CZ = "#dbe9fa", "#d4f1e6", "#fde3d8", "#efeeea"
    fig, ax = plt.subplots(figsize=(12, 2.4))
    etapas = [("Bronze", "Consumo, planos,\ncardápio, calendário,\nclima e previsões\nguardadas", CZ), ("Silver", "Uma linha por dia,\nsem preencher\nausências, com\ndias classificados", AZ), ("Gold", "Uma linha por\nsexta-feira e dia\nprevisto, só com o\nque se sabia", AZ),
              ("Modelos", "CatBoost, KNN e\nconsumo de\nreferência, com os\núltimos 12 meses", VD), ("Decisão", "Margem de\nsegurança, contexto\ndo dia e pontos\nde atenção", LJ), ("Aplicação", "Nutricionista e\ngestora comparam\ne decidem", LJ)]
    for i, (t, x, c) in enumerate(etapas):
        caixa(ax, i * 2.1, 0, 1.75, 1.7, t, x, c)
        if i:
            seta(ax, i * 2.1 - 0.35, 0.85, i * 2.1, 0.85)
    ax.set_xlim(-0.1, 12.4); ax.set_ylim(-0.1, 1.8); ax.axis("off")
    salvar(fig, "diagrama_fluxo_do_artefato", "Fluxo do artefato, dos dados brutos à decisão da nutricionista e da gestora")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    niveis = [("Cenário 1: consumo e calendário", "Sempre disponível. Funciona sem cardápio e sem quantitativo.", AZ), ("Cenário 2: inclui o cardápio", "Usado quando o cardápio da semana já foi informado. Base do artefato.", AZ),
              ("Cenário 3: inclui a previsão do tempo", "Testado no preparo. Fica como alerta de chuva.", CZ), ("Cenário 4: híbrido com o quantitativo", "Junta o artefato ao quantitativo ou às reservas, quando existem. Só no preparo.", VD)]
    for i, (t, x, c) in enumerate(niveis):
        topo = 4.5 - i * 1.5
        caixa(ax, 0, topo, 8, 0.95, t, x, c)
        if i:
            seta(ax, 4, topo + 1.5, 4, topo + 0.95)
    ax.set_xlim(-0.1, 8.1); ax.set_ylim(-0.1, 5.5); ax.axis("off")
    salvar(fig, "diagrama_cenarios", "Cenários do artefato: cada um acrescenta uma fonte de dados, e o artefato usa o mais completo disponível na semana")

    fig, ax = plt.subplots(figsize=(10.5, 2.1))
    caixa(ax, 0, 0.5, 3.4, 1, "Aprendizado", "Últimos 12 meses antes da sexta-feira", AZ)
    caixa(ax, 4, 0.5, 1.7, 1, "Sexta-feira", "Previsão emitida", LJ)
    caixa(ax, 6.3, 0.5, 2.1, 1, "Preparo", "3 a 7 dias depois", VD)
    caixa(ax, 8.6, 0.5, 2.1, 1, "Compras", "10 a 14 dias depois", VD)
    seta(ax, 3.4, 1, 4, 1); seta(ax, 5.7, 1, 6.3, 1)
    ax.text(5.35, 0.18, "A cada semana, a sexta-feira avança e o processo se repete, sem usar dados posteriores a ela.", ha="center", fontsize=9, color=TEXTO2)
    ax.set_xlim(-0.1, 10.8); ax.set_ylim(0, 1.6); ax.axis("off")
    salvar(fig, "diagrama_validacao_semanal", "Validação semana a semana: em cada sexta-feira, o modelo aprende só com o passado")


def escrever_lista():
    linhas = ["# Figuras e tabelas para a dissertação", "", "Figuras a 300 DPI, sem título na imagem; as legendas abaixo são sugestões. Tabelas em CSV com separador `;` e decimal com vírgula.", ""]
    for tipo, titulo in (("Figura", "Figuras"), ("Tabela", "Tabelas")):
        linhas += [f"## {titulo}", "", "| Número | Arquivo | Legenda sugerida |", "|---:|---|---|"]
        linhas += [f"| {n} | `{a}` | {l} |" for t, n, a, l in lista if t == tipo]
        linhas.append("")
    (SAIDA / "LISTA_FIGURAS_TABELAS.md").write_text("\n".join(linhas), encoding="utf-8")
    celulas = ["# 06. Figuras e tabelas para a dissertação\n\nAs legendas são sugestões. Arquivos em `resultados/dissertacao/`."]
    for _, numero, arquivo, legenda in [item for item in lista if item[0] == "Figura"]:
        celulas.append(f"**Figura {numero}.** {legenda}\n\n![Figura {numero}](../resultados/dissertacao/figuras/{arquivo})")
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(c) for c in celulas], metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    nbf.write(nb, RAIZ / "notebooks" / "06_figuras_e_tabelas_dissertacao.ipynb")


def main():
    FIGURAS.mkdir(parents=True, exist_ok=True)
    TABELAS.mkdir(parents=True, exist_ok=True)
    for antigo in list(FIGURAS.glob("*.png")) + list(TABELAS.glob("*.csv")):
        antigo.unlink()
    d = carregar()
    tabelas_dados(d)
    figuras_dados(d)
    figuras_regimes(d)
    figuras_resultados(d)
    figuras_detalhes(d)
    diagramas()
    escrever_lista()
    print(sum(t == "Figura" for t, *_ in lista), "figuras e", sum(t == "Tabela" for t, *_ in lista), "tabelas")


if __name__ == "__main__":
    main()
