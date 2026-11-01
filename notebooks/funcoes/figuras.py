"""Figuras e tabelas da dissertacao e, em pasta local, os valores em reais."""
import io
import json
import locale
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import textwrap
from IPython.display import Image, Markdown, display
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter

from .configuracao import (CALENDARIO, FIM_PROVA, INICIO_PROVA, CAMPEAO, CARDAPIO, CENARIO_CAMPEAO, CENARIOS, CLIMA_PREVISTO, FIGURAS, GOLD, HISTORICO, HISTORICO_CAMPEAO, LIMITES_ALVO, PASTA_COMPARACAO, PASTA_DISSERTACAO,
                          PASTA_LOCAL, PASTA_MODELOS, PASTA_PLANEJAMENTO, RAIZ, SILVER, TABELAS)
from .modelos import arredondar_refeicoes, construir_modelo, dados_de_treino, matriz, prever_serie
from .nomes import EVENTOS, HISTORICOS, JANELAS, MODELOS as ALGORITMOS, PREPAROS, PROTEINAS, TRATAMENTOS, rotulo


CORTE = "2025-12-31"


SUPERFICIE, TEXTO, TEXTO2, GRADE = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e0"


CONSUMO, HUMANO, ARTEFATO, COMBINACAO, TIPICO = "#1f2933", "#eb6834", "#2a78d6", "#1baf7a", "#9b9a93"


MESMO_DIA = "#c9c8c1"


NOME_PLANO = {"Reservas dos alunos": HUMANO, "Quantitativo da nutricionista": HUMANO}


NOMES_CENARIOS = {"consumo_tipico": "Consumo de referência", "mesmo_dia": "Mediana do mesmo dia da semana", "H": "Só histórico de consumo", "C1": "Consumo e calendário", "C2": "Inclui o cardápio", "C3": "Inclui a previsão do tempo", "C4": "Híbrido (artefato com quantitativo)", "plano": "Quantitativo da nutricionista"}


QUANTITATIVO = "Quantitativo da nutricionista"


NOME_ARTEFATO = "Artefato (ExtraTrees)"


NOME_HIBRIDO = "Híbrido (artefato com quantitativo)"


NOME_COMITE = "Comitê de CatBoost, KNN e referência"


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


PERIODOS_EDA = {"2023_2025": ("2023-01-01", CORTE, "2023 a 2025"), "2026": (INICIO_PROVA, FIM_PROVA, "fevereiro a agosto de 2026")}


CORES_PERIODOS = {"2023_2025": "#a6c8ee", "2026": ARTEFATO}


DIAS_SEMANA = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]


FAIXAS_CHUVA, NOMES_CHUVA = [-1, 0, 5, 15, 500], ["Sem chuva", "Até 5 mm", "5 a 15 mm", "Acima de 15 mm"]


FAIXAS_TEMPERATURA, NOMES_TEMPERATURA = [0, 24, 27, 30, 33, 50], ["Até 24 °C", "24 a 27 °C", "27 a 30 °C", "30 a 33 °C", "33 °C ou mais"]


DIFERENCA = "Diferença para o consumo de referência (%)"


lista = {}


def numero_na_historia(tipo, nome):
    nomes = [n for secao in HISTORIA for n in secao[1 if tipo == "Figura" else 2]]
    assert nome in nomes, f"{tipo} fora da historia: {nome}"
    return nomes.index(nome) + 1


def salvar(fig, nome, legenda):
    numero = numero_na_historia("Figura", nome)
    arquivo = f"figura_{numero:02d}_{nome}.png"
    fig.savefig(FIGURAS / arquivo, dpi=300, bbox_inches="tight")
    imagem = io.BytesIO()
    fig.savefig(imagem, dpi=90, bbox_inches="tight")
    plt.close(fig)
    lista["Figura", numero] = (arquivo, legenda)
    display(Markdown(f"**Figura {numero}.** {legenda}"), Image(data=imagem.getvalue()))


def tabela(df, nome, legenda, indice=False):
    numero = numero_na_historia("Tabela", nome)
    arquivo = f"tabela_{numero:02d}_{nome}.csv"
    df.to_csv(TABELAS / arquivo, sep=";", index=indice, encoding="utf-8-sig", decimal=",")
    lista["Tabela", numero] = (arquivo, legenda)
    print(f"Tabela {numero}: {nome}")


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
    d["base"] = d["base"][d["base"]["data"] <= FIM_PROVA].reset_index(drop=True)
    d["relativo"] = d["t1"][d["t1"]["usar_no_treino"] & d["t1"]["alvo_relativo"].notna() & (d["t1"]["data"] <= FIM_PROVA)]
    d["preparo"] = pd.read_parquet(PASTA_COMPARACAO / "serie_preparo.parquet")
    d["compras"] = pd.read_parquet(PASTA_COMPARACAO / "serie_compras.parquet")
    d["anual"] = pd.read_csv(PASTA_COMPARACAO / "comparacao_anual.csv", sep=";", encoding="utf-8-sig")
    d["final"] = pd.read_csv(PASTA_MODELOS / "resultado_final.csv", sep=";", encoding="utf-8-sig")
    d["janelas"] = pd.read_csv(PASTA_MODELOS / "metricas_janelas_treino.csv", sep=";", encoding="utf-8-sig")
    d["config"] = pd.read_csv(PASTA_MODELOS / "configuracao_escolhida.csv", sep=";", encoding="utf-8-sig")
    d["diario"] = pd.read_parquet(PASTA_MODELOS / "preparo_diario_final.parquet")
    d["folgas"] = pd.read_csv(PASTA_PLANEJAMENTO / "verificacao_folgas_2026.csv", sep=";", encoding="utf-8-sig")
    d["folgas_prospectiva"] = pd.read_csv(PASTA_PLANEJAMENTO / "verificacao_folgas_prospectiva.csv", sep=";", encoding="utf-8-sig")
    d["situacao"] = pd.read_csv(PASTA_PLANEJAMENTO / "verificacao_por_situacao_2026.csv", sep=";", encoding="utf-8-sig")
    opcoes = d["config"][d["config"]["cenario"] == "C2"]
    d["vencedor_t1"] = opcoes[opcoes["janela"] == "t1"].iloc[0]
    d["vencedor_t2"] = opcoes[opcoes["janela"] == "t2"].iloc[0]
    return d


def consumo_ao_longo_do_tempo(d):
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
    salvar(fig, "consumo_diario", "Consumo diário de refeições e dias fora da rotina (fevereiro de 2023 a agosto de 2026)")

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

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    anos = sorted(base["ano"].unique())
    ax.boxplot([base[base["ano"] == a]["consumo"] for a in anos], tick_labels=[f"{a}\n({(base['ano'] == a).sum()} dias)" for a in anos], patch_artist=True, widths=0.5,
               boxprops={"facecolor": ARTEFATO, "alpha": 0.5, "edgecolor": TEXTO2}, medianprops={"color": TEXTO}, flierprops={"markersize": 3, "markerfacecolor": TEXTO2, "markeredgecolor": TEXTO2})
    ax.set_ylabel("Refeições por dia")
    fig.tight_layout()
    salvar(fig, "consumo_medio_por_ano", "Distribuição do consumo diário por ano, com todos os dias. A linha é a mediana, e a caixa reúne a metade central dos dias. O ano de 2026 vai até agosto")


def entre(t, periodo):
    inicio, fim = PERIODOS_EDA[periodo][:2]
    return t[t["data"].between(inicio, fim)]


def efeitos_por_grupo(r):
    """Compara cada grupo de dias com o consumo de referencia, sempre com a quantidade de dias."""
    grupos = {"Dia da semana": pd.Series(pd.Categorical(r["dia_semana"].map(dict(enumerate(DIAS_SEMANA))), DIAS_SEMANA), index=r.index),
              "Proteína": r["proteina"].map(lambda k: PROTEINAS.get(k, k)), "Forma de preparo": r["preparo"].map(lambda k: PREPAROS.get(k, k)), "Evento": r["evento_tipo"].map(lambda k: EVENTOS.get(k, k)),
              "Chuva": pd.cut(r["oraculo_chuva_mm"], FAIXAS_CHUVA, labels=NOMES_CHUVA), "Temperatura máxima": pd.cut(r["oraculo_temp_max"], FAIXAS_TEMPERATURA, labels=NOMES_TEMPERATURA)}
    linhas = []
    for grupo, chave in grupos.items():
        for categoria, v in r["alvo_relativo"].groupby(chave, observed=True):
            linhas.append({"Grupo": grupo, "Categoria": str(categoria), "Dias": len(v), "Consumo médio": round(r.loc[v.index, "consumo"].mean(), 1), DIFERENCA: round((v.median() - 1) * 100, 1)})
    return pd.DataFrame(linhas)


def barras_do_grupo(ax, e, grupo, minimo=1, ordenar=True, destacar_quedas=False):
    v = e[(e["Grupo"] == grupo) & (e["Dias"] >= minimo)]
    if ordenar:
        v = v.sort_values(DIFERENCA)
    cor = [HUMANO if m < -5 else ARTEFATO for m in v[DIFERENCA]] if destacar_quedas else ARTEFATO
    barras_a_partir_do_tipico(ax, [f"{c} ({n} dias)" for c, n in zip(v["Categoria"], v["Dias"])], 1 + v[DIFERENCA].to_numpy() / 100, cor)


def memoria_do_consumo(serie):
    """Autocorrelacao total e propria das ultimas 20 defasagens."""
    defasagens = np.arange(1, 21)
    acf = np.array([np.corrcoef(serie[:-k], serie[k:])[0, 1] for k in defasagens])
    completa = np.r_[1.0, acf]
    pacf = np.array([np.linalg.solve(np.array([[completa[abs(i - j)] for j in range(k)] for i in range(k)]), completa[1:k + 1])[-1] for k in defasagens])
    return defasagens, acf, pacf


def memoria_por_ano(base):
    """Media das autocorrelacoes de cada ano, para diferencas de nivel entre anos nao inflarem a relacao."""
    medidas = [memoria_do_consumo(g["consumo"].to_numpy(float)) for _, g in base.groupby("ano")]
    return medidas[0][0], np.mean([m[1] for m in medidas], axis=0), np.mean([m[2] for m in medidas], axis=0)


def eda_consumo(d, periodo):
    texto = PERIODOS_EDA[periodo][2]
    serie = entre(d["base"], periodo).query("usar_no_treino")
    defasagens_acf, acf, pacf = memoria_por_ano(serie)
    fig, eixos = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, valores, titulo in ((eixos[0], acf, "Relação total com o consumo passado"), (eixos[1], pacf, "Relação própria, descontados os dias intermediários")):
        ax.bar(defasagens_acf, valores, width=0.7, color=[HUMANO if k % 5 == 0 else ARTEFATO for k in defasagens_acf])
        ax.axhline(0, color=TEXTO2, linewidth=1)
        ax.set_ylim(-0.25, 1)
        ax.set_yticks([0, 0.5, 1], ["0\nsem relação", "0,5\nmoderada", "1\nandam juntos"])
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_title(titulo, fontsize=10, color=TEXTO2)
    eixos[0].set_ylabel("Correlação")
    fig.supxlabel("Dias de serviço antes (em laranja, o mesmo dia da semana)", fontsize=10, color=TEXTO2)
    fig.tight_layout()
    salvar(fig, f"autocorrelacao_{periodo}", f"Quanto o consumo de um dia se relaciona com o consumo de dias anteriores ({texto}, {len(serie)} dias). Média das correlações de cada ano")
    tabela(pd.DataFrame({"Dias de serviço antes": defasagens_acf, "Relação total": acf, "Relação própria": pacf}).round(3), f"autocorrelacao_{periodo}", f"Autocorrelação do consumo ({texto})")

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
    filtro = b["usar_no_treino"] & b.index.isin(entre(d["base"], periodo)["data"])
    linhas = []
    for c in L:
        x = pd.DataFrame({"y": s[filtro], "x": L.loc[filtro, c]}).dropna()
        linhas.append({"Variável": rotulo(c), "Dias analisados": len(x), "Relação linear": x["y"].corr(x["x"]), "Relação pela ordem": x["y"].corr(x["x"], method="spearman"), "Relação média por ano": x.groupby(x.index.year).apply(lambda g: g["y"].corr(g["x"])).mean()})
    tab = pd.DataFrame(linhas).sort_values("Relação média por ano", ascending=False).round(3)
    tabela(tab, f"relacao_defasagens_{periodo}", f"Relação das defasagens e medidas móveis com o consumo, conhecendo o consumo até a véspera ({texto})")
    o = tab.head(10)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    barras_horizontais(ax, o["Variável"], o["Relação média por ano"].to_numpy(), [ARTEFATO if v >= 0 else HUMANO for v in o["Relação média por ano"]])
    eixo_correlacao(ax, 0)
    fig.tight_layout()
    salvar(fig, f"relacao_defasagens_{periodo}", f"As 10 medidas de consumo passado mais relacionadas ao consumo do dia ({texto}), pela média da relação em cada ano. A lista completa está na tabela")


def medidas_de_sexta(d):
    """Mostra o que se sabe na sexta-feira e o que se repete entre as medidas, com o historico da escolha."""
    linhas = []
    colunas = ["nivel_base", "nivel_exponencial", "nivel_10d", "nivel_20d", "nivel_mesmo_dia", "nivel_bloco_atual", "ultimo_consumo", "media_5d", "mediana_5d", "media_10d", "media_20d",
               "mesmo_dia_ha_1_ocorrencias", "mesmo_dia_ha_2_ocorrencias", "mesmo_dia_ha_3_ocorrencias", "mesmo_dia_ha_4_ocorrencias", "desvio_5d", "desvio_10d", "minimo_5d", "maximo_5d"]
    p = d["t1"][d["t1"]["usar_no_treino"] & (d["t1"]["data"] <= CORTE)]
    c = d["t2"][d["t2"]["usar_no_treino"] & (d["t2"]["data"] <= CORTE)]
    for col in colunas:
        linhas.append({"Variável conhecida na sexta-feira": rotulo(col), JANELAS["t1"]: p["consumo"].corr(p[col]), JANELAS["t2"]: c["consumo"].corr(c[col])})
    tabela(pd.DataFrame(linhas).round(3), "relacao_atributos_sexta", "Relação com o consumo das medidas realmente conhecidas na sexta-feira do planejamento (2023 a 2025)")

    todas = p[colunas[:13]].corr()
    tabela(todas.rename(index=rotulo, columns=rotulo).round(2).reset_index(names="Medida"), "informacoes_repetidas", "Correlação entre as medidas de consumo recente (2023 a 2025)")
    nivel = ["nivel_base", "media_5d", "media_20d", "ultimo_consumo", "mesmo_dia_ha_1_ocorrencias", "mesmo_dia_ha_4_ocorrencias"]
    m = p[nivel].corr()
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    nomes = [quebrar(rotulo(c_), 22) for c_ in nivel]
    ax.set_xticks(range(len(nivel)), nomes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(nivel)), nomes, fontsize=9)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(m.to_numpy()):
        ax.text(j, i, f"{v:.2f}".replace(".", ","), ha="center", va="center", fontsize=10, color="white" if v > 0.75 else TEXTO)
    fig.tight_layout()
    salvar(fig, "informacoes_repetidas", "Correlação entre seis medidas de consumo recente (2023 a 2025). Perto de 1, duas medidas contam quase a mesma história")


def eda_calendario(d, periodo):
    texto = PERIODOS_EDA[periodo][2]
    e = efeitos_por_grupo(entre(d["relativo"], periodo))
    v = e[e["Grupo"] == "Dia da semana"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.4))
    a.bar(v["Categoria"], v["Consumo médio"], width=0.62, color=ARTEFATO)
    for x, (m, n) in enumerate(zip(v["Consumo médio"], v["Dias"])):
        a.text(x, m, f"{m:.0f}\n({n} dias)", ha="center", va="bottom", color=TEXTO2, fontsize=9)
    a.set_ylim(0, v["Consumo médio"].max() * 1.25)
    a.set_ylabel("Refeições por dia, em média")
    barras_do_grupo(b, e, "Dia da semana", ordenar=False)
    fig.tight_layout()
    salvar(fig, f"dia_da_semana_{periodo}", f"Consumo médio por dia da semana e diferença para o consumo de referência ({texto}, sem dias atípicos). A referência já considera o mesmo dia da semana, por isso a diferença fica menor que a distância entre as médias")

    blocos = entre(d["base"], periodo)
    blocos = blocos[blocos["bloco"].notna()].copy()
    blocos["rel"] = blocos["consumo"] / blocos.groupby("bloco")["consumo"].transform("median")
    blocos = blocos.drop(columns="dias_ate_fim_bloco").merge(d["t1"][["data", "dias_ate_fim_bloco"]], on="data")
    finais = blocos[blocos["dias_ate_fim_bloco"] <= 12]
    v = (finais.groupby("dias_ate_fim_bloco")["rel"].median() - 1) * 100
    quantos = finais["bloco"].nunique()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(v.index, v.values, width=0.62, color=[HUMANO if i <= 3 else ARTEFATO for i in v.index])
    ax.axhline(0, color=TEXTO2, linewidth=1)
    ax.invert_xaxis()
    ax.set_xticks(v.index)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: percentual(x)))
    ax.set_xlabel("Dias de aula até o fim do período")
    ax.set_ylabel("Diferença para o consumo típico do período")
    fig.tight_layout()
    salvar(fig, f"fim_do_periodo_{periodo}", f"Consumo nos dias finais dos períodos de aulas ({texto}, {quantos} {'fim de período' if quantos == 1 else 'fins de período'})")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    barras_do_grupo(ax, e, "Evento", minimo=3, destacar_quedas=True)
    fig.tight_layout()
    salvar(fig, f"eventos_do_calendario_{periodo}", f"Consumo comparado à referência por evento do calendário ({texto}, sem dias atípicos, eventos com pelo menos 3 dias)")


def eda_cardapio(d, periodo):
    e = efeitos_por_grupo(entre(d["relativo"], periodo))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    barras_do_grupo(a, e, "Proteína", minimo=5)
    barras_do_grupo(b, e, "Forma de preparo", minimo=5)
    fig.tight_layout()
    salvar(fig, f"cardapio_{periodo}", f"Consumo comparado à referência por proteína e forma de preparo ({PERIODOS_EDA[periodo][2]}, sem dias atípicos, categorias com pelo menos 5 dias)")


def eda_chuva(d, periodo):
    texto = PERIODOS_EDA[periodo][2]
    r = entre(d["relativo"], periodo)
    tabelas = []
    for coluna in ("oraculo_chuva_mm", "previsao_chuva_mm"):
        q = r.dropna(subset=[coluna])
        v = q.groupby(pd.cut(q[coluna], FAIXAS_CHUVA, labels=NOMES_CHUVA), observed=False)["alvo_relativo"].agg(["size", "median"])
        tabelas.append((len(q), v["size"], (v["median"] - 1) * 100))
    limite = max(15, np.ceil(np.nanmax([np.abs(x[2]).max() for x in tabelas]) / 5) * 5 + 5)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, (dias, tamanhos, difs), rotulo_x in ((a, tabelas[0], "Chuva que ocorreu no dia"), (b, tabelas[1], "Chuva prevista na sexta-feira")):
        ax.bar(range(4), difs, width=0.62, color=ARTEFATO)
        for x, m in enumerate(difs):
            if not np.isnan(m):
                ax.text(x, m, percentual(m), ha="center", va="bottom" if m >= 0 else "top", color=TEXTO2, fontsize=9)
        ax.set_xticks(range(4), [f"{n}\n({c} dias)" for n, c in zip(NOMES_CHUVA, tamanhos)], fontsize=9)
        ax.axhline(0, color=TEXTO2, linewidth=1)
        ax.set_ylim(-limite, limite)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: percentual(x)))
        ax.set_xlabel(rotulo_x)
    a.set_ylabel("Diferença para o consumo de referência")
    fig.tight_layout()
    aviso = " A previsão guardada começa em 2024" if periodo == "2023_2025" else ""
    salvar(fig, f"chuva_{periodo}", f"Consumo comparado à referência por faixa de chuva ({texto}, sem dias atípicos).{aviso}")



def eda_temperatura(d, periodo):
    texto = PERIODOS_EDA[periodo][2]
    e = efeitos_por_grupo(entre(d["relativo"], periodo))
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    barras_do_grupo(ax, e, "Temperatura máxima", ordenar=False)
    fig.tight_layout()
    salvar(fig, f"temperatura_{periodo}", f"Consumo comparado à referência por faixa de temperatura máxima ({texto}, sem dias atípicos)")


def chuva_bruta_e_ajustada(d, periodo):
    """Separa a diferenca bruta da chuva forte do que coincide com ano, mes e fim de periodo."""
    texto = PERIODOS_EDA[periodo][2]
    b = entre(d["base"], periodo).merge(d["t1"][["data", "fim_de_bloco"]], on="data", how="left")
    b = b[b["chuva_mm"].notna()].copy()
    b["faixa"] = pd.cut(b["chuva_mm"], FAIXAS_CHUVA, labels=NOMES_CHUVA)
    b["mes_numero"] = b["data"].dt.month
    b["dia"] = b["data"].dt.dayofweek
    v = b.groupby("faixa", observed=False)["consumo"].agg(["size", "mean"])
    sem, forte = b[b["faixa"] == NOMES_CHUVA[0]], b[b["faixa"] == NOMES_CHUVA[3]]

    def comparar(chaves):
        medias = sem.groupby(chaves)["consumo"].agg(media_sem_chuva="mean", dias="size").reset_index()
        q = forte.merge(medias[medias["dias"] >= 3], on=chaves, how="inner")
        return (q["consumo"] - q["media_sem_chuva"]).mean(), len(q)

    sem_fim, forte_fim = sem[sem["fim_de_bloco"].ne(True)], forte[forte["fim_de_bloco"].ne(True)]
    varios_anos = b["ano"].nunique() > 1
    prefixo, nome = (["ano"], "Mesmo ano e ") if varios_anos else ([], "Mesmo ")
    comparacoes = [("Média bruta", forte["consumo"].mean() - sem["consumo"].mean(), len(forte))]
    if varios_anos:
        comparacoes.append(("Mesmo ano", *comparar(["ano"])))
    comparacoes += [(f"{nome}dia da semana", *comparar(prefixo + ["dia"])), (f"{nome}mês", *comparar(prefixo + ["mes_numero"])),
                    ("Fora dos últimos dias de período", forte_fim["consumo"].mean() - sem_fim["consumo"].mean(), len(forte_fim))]
    fig, (a, c) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.1, 1]})
    a.bar(range(4), v["mean"], width=0.62, color=[TIPICO, "#a6c8ee", "#5c9ee4", ARTEFATO])
    for x, (n, m) in enumerate(zip(v["size"], v["mean"])):
        a.text(x, m, f"{m:.0f}\n({n} dias)", ha="center", va="bottom", color=TEXTO2, fontsize=9)
    a.set_xticks(range(4), NOMES_CHUVA, fontsize=9)
    a.set_ylim(0, v["mean"].max() * 1.22)
    a.set_ylabel("Refeições por dia, em média")
    nomes = [f"{n} ({q} dias)" for n, _, q in comparacoes]
    valores = [x for _, x, _ in comparacoes]
    barras_horizontais(c, [quebrar(n, 30) for n in nomes], valores, [HUMANO if i == 0 else ARTEFATO for i in range(len(valores))], "{:.1f}")
    c.axvline(0, color=TEXTO2, linewidth=1)
    c.set_xlabel("Chuva forte menos dias sem chuva (refeições por dia)")
    fig.tight_layout()
    salvar(fig, f"chuva_bruta_e_ajustada_{periodo}", f"Consumo por faixa de chuva e diferença da chuva forte para dias sem chuva, na média bruta e comparando dias parecidos ({texto}, com todos os dias). Dias parecidos só contam com pelo menos 3 dias sem chuva para comparar")
    tabela(pd.DataFrame([{"Comparação": n, "Dias de chuva forte": q, "Diferença (refeições por dia)": round(x, 1)} for n, x, q in comparacoes]), f"chuva_bruta_e_ajustada_{periodo}", f"Diferença de consumo entre chuva forte e dias sem chuva, na média bruta e comparando dias parecidos ({texto}). Dias parecidos só contam com pelo menos 3 dias sem chuva para comparar")


def eda_relacao_variaveis(d, periodo):
    texto = PERIODOS_EDA[periodo][2]
    r = entre(d["relativo"], periodo)
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
    tabela(rel, f"relacao_variaveis_{periodo}", f"Relação das variáveis com o consumo comparado à referência ({texto})")
    cores = {"Consumo recente": ARTEFATO, "Calendário": COMBINACAO, "Cardápio": HUMANO, "Clima previsto": "#6250d6", "Clima que ocorreu": "#e87ba4"}
    t = rel.head(10)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    barras_horizontais(ax, t["Variável"], t["Relação pela ordem"].to_numpy(), [cores[g] for g in t["Grupo"]])
    presentes = [g for g in cores if g in set(t["Grupo"])]
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=cores[g]) for g in presentes], presentes, ncol=len(presentes), loc="lower center", bbox_to_anchor=(0.4, 1))
    eixo_correlacao(ax)
    fig.tight_layout()
    salvar(fig, f"relacao_variaveis_{periodo}", f"As 10 variáveis mais relacionadas ao consumo comparado à referência ({texto}). A lista completa está na tabela")


def qualidade_previsao_tempo(d):
    linhas = []
    for g, decisao in ((d["t1"], "Preparo"), (d["t2"], "Compras")):
        for ant, s in g[g["data"] <= FIM_PROVA].dropna(subset=["previsao_chuva_mm", "oraculo_chuva_mm"]).groupby("antecedencia_dias"):
            linhas.append({"Decisão": decisao, "Antecedência (dias)": ant, "Dias": len(s), "Relação entre previsão e chuva real": s["previsao_chuva_mm"].corr(s["oraculo_chuva_mm"])})
    q = pd.DataFrame(linhas).round(3)
    tabela(q, "qualidade_previsao_tempo", "Relação entre a chuva prevista e a chuva que ocorreu, por antecedência. Preparo desde 2024 e compras desde abril de 2026, ambos até agosto de 2026")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(q["Antecedência (dias)"], q["Relação entre previsão e chuva real"], width=0.7, color=[ARTEFATO if x == "Preparo" else HUMANO for x in q["Decisão"]])
    ax.set_xticks(q["Antecedência (dias)"])
    ax.set_xlabel("Antecedência da previsão (dias)")
    for x, v in zip(q["Antecedência (dias)"], q["Relação entre previsão e chuva real"]):
        ax.text(x, v, f"{v:.2f}".replace(".", ","), ha="center", va="bottom" if v >= 0 else "top", color=TEXTO2, fontsize=9)
    ax.axhline(0, color=TEXTO2, linewidth=1)
    ax.set_ylim(-0.35, 1)
    ax.set_yticks([-0.25, 0, 0.5, 1], ["-0,25", "0\nsem relação", "0,5", "1"])
    ax.set_ylabel("Correlação entre previsão e chuva real")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=ARTEFATO), plt.Rectangle((0, 0), 1, 1, color=HUMANO)], ["Preparo", "Compras"])
    fig.tight_layout()
    salvar(fig, "qualidade_previsao_tempo", "Qualidade da previsão de chuva por antecedência. Preparo desde 2024 e compras desde abril de 2026, ambos até agosto de 2026")


def base_de_planos(d):
    base = d["base"].copy()
    base["plano"] = base["plano_nutricionista"].fillna(base["reserva_discente"])
    return base.dropna(subset=["plano"])


def resumo_do_plano(g):
    e = g["plano"] - g["consumo"]
    return {"Dias": len(g), "Consumo médio": g["consumo"].mean(), "Plano médio": g["plano"].mean(), "Saldo (plano menos consumo)": e.sum(), "Excesso potencial (refeições)": e.clip(lower=0).sum(),
            "Déficit potencial (refeições)": (-e).clip(lower=0).sum(), "% de dias com déficit potencial": (e < 0).mean() * 100, "Erro médio": e.abs().mean()}


def plano_e_consumo(q, nome, arquivo, legenda):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    limite = max(q["plano"].max(), q["consumo"].max()) * 1.05
    a.scatter(q["plano"], q["consumo"], s=16, color=HUMANO, alpha=0.55, linewidth=0)
    a.plot([0, limite], [0, limite], color=TEXTO2, linestyle="--", linewidth=1)
    a.set_xlim(0, limite)
    a.set_ylim(0, limite)
    a.set_xlabel(f"{nome} (refeições)")
    a.set_ylabel("Refeições consumidas")
    b.hist(q["plano"] - q["consumo"], bins=25, color=HUMANO, alpha=0.75)
    b.axvline(0, color=TEXTO2, linestyle="--", linewidth=1)
    b.set_xlabel("Plano menos consumo (refeições por dia)")
    b.set_ylabel("Dias")
    b.text(0.02, 0.95, "Plano abaixo\ndo consumo", transform=b.transAxes, ha="left", va="top", color=TEXTO2, fontsize=9)
    b.text(0.98, 0.95, "Plano acima\ndo consumo", transform=b.transAxes, ha="right", va="top", color=TEXTO2, fontsize=9)
    fig.tight_layout()
    salvar(fig, arquivo, legenda)


def barras_de_excesso_e_deficit(t, chave, arquivo, legenda, rotulo_y):
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for deslocamento, coluna, cor, nome in ((-0.2, "Excesso potencial (refeições)", ARTEFATO, "Excesso potencial"), (0.2, "Déficit potencial (refeições)", HUMANO, "Déficit potencial")):
        ax.bar(x + deslocamento, t[coluna], 0.38, color=cor, label=nome)
        for x_p, valor in zip(x + deslocamento, t[coluna]):
            ax.text(x_p, valor, f"{valor:,.0f}".replace(",", "."), ha="center", va="bottom", color=TEXTO2, fontsize=9)
    ax.set_xticks(x, t[chave].astype(str))
    ax.set_ylabel(rotulo_y)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, arquivo, legenda)


def eda_reservas(d):
    q = base_de_planos(d)
    q = q[q["ano"] < 2026]
    plano_e_consumo(q, "Reservas dos alunos", "reservas_e_consumo", "Reservas dos alunos e consumo registrado de 2023 a 2025. À esquerda, a linha indica reserva igual ao consumo; à direita, a distribuição da diferença diária")
    t = pd.DataFrame([{"Ano": a, **resumo_do_plano(g)} for a, g in q.groupby("ano")])
    barras_de_excesso_e_deficit(t, "Ano", "excesso_deficit_reservas", "Excesso potencial e déficit potencial das reservas dos alunos, por ano", "Refeições no ano")
    t = pd.concat([t, pd.DataFrame([{"Ano": "2023 a 2025", **resumo_do_plano(q)}])], ignore_index=True)
    tabela(t.round(1), "reservas_por_ano", "Reservas dos alunos e consumo registrado por ano")


def eda_quantitativo(d):
    q = base_de_planos(d)
    q = q[q["ano"] == 2026]
    plano_e_consumo(q, QUANTITATIVO, "quantitativo_e_consumo", "Quantitativo da nutricionista e consumo registrado de fevereiro a agosto de 2026. À esquerda, a linha indica quantitativo igual ao consumo; à direita, a distribuição da diferença diária")
    t = pd.DataFrame([{"Mês": m, **resumo_do_plano(g)} for m, g in q.groupby(q["data"].dt.strftime("%m/%Y"), sort=False)])
    barras_de_excesso_e_deficit(t, "Mês", "excesso_deficit_quantitativo", "Excesso potencial e déficit potencial do quantitativo da nutricionista, por mês de 2026", "Refeições no mês")
    t = pd.concat([t, pd.DataFrame([{"Mês": "Fevereiro a agosto", **resumo_do_plano(q)}])], ignore_index=True)
    tabela(t.round(1), "quantitativo_por_mes", "Quantitativo da nutricionista e consumo registrado por mês de 2026")


def repetiu_e_mudou(d):
    """Coloca os dois periodos lado a lado. Mudou significa 5 pontos ou mais de distancia entre eles."""
    a, b = (efeitos_por_grupo(entre(d["relativo"], periodo)) for periodo in PERIODOS_EDA)
    ordem_grupos = {g: i for i, g in enumerate(a["Grupo"].unique())}
    ordem = pd.concat([a, b])[["Grupo", "Categoria"]].drop_duplicates().sort_values("Grupo", key=lambda s: s.map(ordem_grupos), kind="stable")
    colunas = ["Grupo", "Categoria", "Dias", DIFERENCA]
    t = ordem.merge(a[colunas], on=["Grupo", "Categoria"], how="left").merge(b[colunas], on=["Grupo", "Categoria"], how="left", suffixes=(" 2023 a 2025", " fev. a ago. 2026"))
    deslocamento = {p: (entre(d["relativo"], p)["alvo_relativo"].median() - 1) * 100 for p in PERIODOS_EDA}
    t[f"{DIFERENCA} 2023 a 2025"] = (t[f"{DIFERENCA} 2023 a 2025"] - deslocamento["2023_2025"]).round(1)
    t[f"{DIFERENCA} fev. a ago. 2026"] = (t[f"{DIFERENCA} fev. a ago. 2026"] - deslocamento["2026"]).round(1)
    d_a, d_b, c_a, c_b = "Dias 2023 a 2025", "Dias fev. a ago. 2026", f"{DIFERENCA} 2023 a 2025", f"{DIFERENCA} fev. a ago. 2026"
    t[[d_a, d_b]] = t[[d_a, d_b]].fillna(0).astype(int)
    t["Distância entre os períodos (pontos)"] = (t[c_b] - t[c_a]).round(1)
    t["Leitura"] = np.select([t[[d_a, d_b]].min(axis=1) < 10, t["Distância entre os períodos (pontos)"].abs() >= 5], ["Poucos dias", "Mudou"], "Repetiu")
    tabela(t, "repetiu_e_mudou", "Consumo comparado à referência nos dois períodos, descontado o deslocamento geral de cada período. Mudou indica 5 pontos percentuais ou mais de distância; Poucos dias indica menos de 10 dias em algum período")

    grupos = ["Dia da semana", "Proteína", "Forma de preparo", "Chuva"]
    fig, eixos = plt.subplots(2, 2, figsize=(12, 8))
    for ax, grupo in zip(eixos.ravel(), grupos):
        s = t[(t["Grupo"] == grupo) & (t["Leitura"] != "Poucos dias")]
        y = np.arange(len(s))
        for deslocamento, coluna, periodo in ((-0.19, c_a, "2023_2025"), (0.19, c_b, "2026")):
            ax.barh(y + deslocamento, s[coluna], 0.36, color=CORES_PERIODOS[periodo], label=PERIODOS_EDA[periodo][2].capitalize())
            for y_p, valor in zip(y + deslocamento, s[coluna]):
                ax.text(valor, y_p, f" {percentual(valor)} ", va="center", ha="left" if valor >= 0 else "right", color=TEXTO2, fontsize=8)
        ax.set_yticks(y, [quebrar(c_, 22) for c_ in s["Categoria"]], fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=TEXTO2, linewidth=1)
        ax.margins(x=0.18)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: percentual(v)))
        ax.set_title(grupo, fontsize=10, color=TEXTO2)
    fig.supxlabel("Diferença para o consumo de referência (0% é igual à referência)", fontsize=10, color=TEXTO2)
    fig.legend(*eixos[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    salvar(fig, "repetiu_e_mudou", "O que repetiu e o que mudou entre os dois períodos, descontado o deslocamento geral de cada período, com categorias de pelo menos 10 dias em cada período e sem dias atípicos")


def resumo_dos_periodos(d):
    numero = lambda v, casas=1: f"{v:.{casas}f}".replace(".", ",")
    linhas = {}
    for periodo, (_, _, texto) in PERIODOS_EDA.items():
        r, planos = entre(d["relativo"], periodo), entre(base_de_planos(d), periodo)
        e = efeitos_por_grupo(r)
        semana = e[e["Grupo"] == "Dia da semana"].set_index("Categoria")["Consumo médio"]
        chuva_forte = e[e["Categoria"] == NOMES_CHUVA[3]].iloc[0]
        _, acf, _ = memoria_por_ano(entre(d["base"], periodo).query("usar_no_treino"))
        plano = resumo_do_plano(planos)
        linhas[texto.capitalize()] = {
            "Dias sem atípicos": str(len(r)), "Consumo médio por dia": numero(r["consumo"].mean()), "Dia de maior consumo": f"{semana.idxmax()} ({numero(semana.max(), 0)})", "Dia de menor consumo": f"{semana.idxmin()} ({numero(semana.min(), 0)})",
            "Relação com o dia de serviço anterior (média por ano)": numero(acf[0], 2), "Relação com o mesmo dia da semana anterior (média por ano)": numero(acf[4], 2), "Chuva forte comparada à referência": f"{percentual(chuva_forte[DIFERENCA])} ({chuva_forte['Dias']} dias)",
            "Fonte do plano": "Reservas dos alunos" if periodo == "2023_2025" else QUANTITATIVO, "Plano médio por dia": numero(plano["Plano médio"]), "Erro médio do plano": numero(plano["Erro médio"]), "Dias com déficit potencial": f"{numero(plano['% de dias com déficit potencial'], 0)}%"}
    tabela(pd.DataFrame(linhas).rename_axis("Pergunta").reset_index(), "resumo_dos_periodos", "Resumo dos dois períodos: as mesmas perguntas respondidas em 2023 a 2025 e em fevereiro a agosto de 2026")


def efeitos_do_periodo(d, periodo):
    texto = PERIODOS_EDA[periodo][2]
    tabela(efeitos_por_grupo(entre(d["relativo"], periodo)), f"efeitos_{periodo}", f"Consumo comparado à referência por dia da semana, cardápio, evento e clima, com a quantidade de dias ({texto}, sem dias atípicos)")


def comparar_planos(d):
    base = base_de_planos(d)
    dif = [base[base["ano"] == a]["plano"] - base[base["ano"] == a]["consumo"] for a in (2023, 2024, 2025, 2026)]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.boxplot(dif, tick_labels=["2023\nReservas", "2024\nReservas", "2025\nReservas", "2026\nQuantitativo"], patch_artist=True, widths=0.5, showfliers=True,
               boxprops={"facecolor": HUMANO, "alpha": 0.5, "edgecolor": TEXTO2}, medianprops={"color": TEXTO}, flierprops={"markersize": 3, "markerfacecolor": TEXTO2, "markeredgecolor": TEXTO2})
    ax.axhline(0, color=TEXTO2, linestyle="--", linewidth=1)
    ax.set_ylabel("Plano menos consumo (refeições)")
    fig.tight_layout()
    salvar(fig, "diferenca_plano_consumo", "Diferença diária entre plano humano e consumo; valores positivos indicam plano acima do consumo. O ano de 2026 vai até agosto")

    tab = pd.DataFrame([{"Ano": a, "Fonte do plano": QUANTITATIVO if a == 2026 else "Reservas dos alunos", **resumo_do_plano(g)} for a, g in base.groupby("ano")]).round(1)
    tabela(tab, "planos_por_ano", "Plano humano e consumo registrado por ano. O ano de 2026 vai até agosto")
    barras_de_excesso_e_deficit(tab, "Ano", "excesso_deficit_ano", "Excesso potencial e déficit potencial do plano humano, por ano. O ano de 2026 vai até agosto", "Refeições no ano")

    x = np.arange(len(tab))
    soma = tab["Excesso potencial (refeições)"] + tab["Déficit potencial (refeições)"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for deslocamento, valores, cor, nome in ((-0.2, tab["Saldo (plano menos consumo)"], TIPICO, "Saldo final (excesso e déficit se compensam)"), (0.2, soma, HUMANO, "Soma das diferenças de cada dia")):
        ax.bar(x + deslocamento, valores, 0.38, color=cor, label=nome)
        for x_p, valor in zip(x + deslocamento, valores):
            ax.text(x_p, valor, f"{valor:,.0f}".replace(",", "."), ha="center", va="bottom" if valor >= 0 else "top", color=TEXTO2, fontsize=9)
    ax.axhline(0, color=TEXTO2, linewidth=1)
    ax.margins(y=0.14)
    ax.set_xticks(x, tab["Ano"].astype(str))
    ax.set_ylabel("Refeições no ano")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "saldo_e_diferencas", "Saldo final do plano humano e soma das diferenças de cada dia. Um saldo pequeno pode esconder desvios grandes. O ano de 2026 vai até agosto")


def torneio_de_algoritmos(d):
    """Compara os 15 algoritmos, um comite e a referencia com 2024 e 2025."""
    torneio = pd.read_csv(PASTA_MODELOS / "torneio_ate_2025.csv", sep=";", encoding="utf-8-sig")
    previsoes = pd.read_parquet(PASTA_MODELOS / "torneio_previsoes.parquet")
    ideias = pd.read_csv(PASTA_MODELOS / "experimentos_ate_2025.csv", sep=";", encoding="utf-8-sig")
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
        v[NOME_COMITE] = erro_campeao
        v["Consumo de referência"] = ideias[(ideias["janela"] == janela) & (ideias["ideia"] == "Consumo de referência")]["media"].iloc[0]
        ordem = sorted(v, key=v.get)
        cores = [ARTEFATO if n == "ExtraTrees" else TIPICO if n == "Consumo de referência" else "#a6c8ee" for n in ordem]
        barras_horizontais(ax, [quebrar(n, 62) for n in ordem], [v[n] for n in ordem], cores, "{:.1f}")
        ax.set_title(JANELAS[janela], fontsize=10, color=TEXTO2)
        ax.set_xlabel("Erro médio em 2024 e 2025 (refeições por dia)")
        linhas += [{"Decisão": JANELAS[janela], "Alternativa": n, "Erro médio em 2024 e 2025": round(v[n], 2)} for n in ordem]
    fig.tight_layout()
    salvar(fig, "torneio_de_algoritmos", "Algoritmos testados com a mesma informação e os últimos 12 meses de aprendizado, em 2024 e 2025, primeira etapa da escolha. Em destaque, o ExtraTrees, eleito depois de somar a etapa de fevereiro a agosto de 2026")
    tabela(pd.DataFrame(linhas), "torneio_de_algoritmos", "Erro médio dos algoritmos testados, com os dados usados para a escolha")
    tabela(ideias.rename(columns={"janela": "Decisão", "ideia": "Ideia testada", "erro_2024": "Erro médio em 2024", "erro_2025": "Erro médio em 2025", "media": "Média dos dois anos"}).assign(**{"Decisão": lambda t: t["Decisão"].map(JANELAS)}).sort_values(["Decisão", "Média dos dois anos"]).round(2), "ideias_testadas", "Ideias de melhoria testadas com 2024 e 2025. Uma ideia só seria adotada com ganho de pelo menos 0,5 refeição por dia")


def dias_ate_agosto(d):
    x = d["preparo"][d["preparo"]["data"].between(INICIO_PROVA, FIM_PROVA) & d["preparo"]["consumo"].notna()]
    return x.assign(mes=x["data"].dt.strftime("%m/%Y"))


def cenarios_2026(d):
    f = d["final"]
    prova = f[(f["periodo"] == "prova_2026") & (f["universo"] == "sem_atipicos")]
    ordem_t1 = ["consumo_tipico", "mesmo_dia", "H", "C1", "C2", "C3", "plano", "C4"]
    ordem_t2 = ["consumo_tipico", "mesmo_dia", "H", "C1", "C2"]
    cores = {"consumo_tipico": TIPICO, "mesmo_dia": MESMO_DIA, "H": ARTEFATO, "C1": ARTEFATO, "C2": ARTEFATO, "C3": ARTEFATO, "plano": HUMANO, "C4": COMBINACAO}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    for ax, janela, ordem, titulo in ((a, "t1", ordem_t1, JANELAS["t1"]), (b, "t2", ordem_t2, JANELAS["t2"])):
        v = prova[prova["janela"] == janela].set_index("alternativa").reindex(ordem).dropna(subset=["mae"])
        barras_horizontais(ax, [NOMES_CENARIOS[i] for i in v.index], v["mae"].to_numpy(), [cores[i] for i in v.index], "{:.1f}")
        ax.set_title(titulo, fontsize=10, color=TEXTO2)
        ax.set_xlabel("Erro médio (refeições por dia)")
    fig.tight_layout()
    salvar(fig, "cenarios_2026", "Erro médio por cenário de fevereiro a agosto de 2026, sem dias atípicos")


def resultado_cenarios(d):
    f = d["final"]
    linhas = []
    nomes = {"prova_2026": "Desenvolvimento, fevereiro a agosto de 2026", "escolha_2025": "Escolha em 2025", "prospectiva_set_out": "Avaliação prospectiva, setembro e outubro de 2026"}
    for (periodo, universo, janela), g in f[f["janela"].isin(["t1", "t2"])].groupby(["periodo", "universo", "janela"]):
        for _, l in g.iterrows():
            linhas.append({"Período": nomes[periodo], "Dias": "Sem atípicos" if universo == "sem_atipicos" else "Todos", "Decisão": JANELAS[janela], "Alternativa": NOMES_CENARIOS.get(l["alternativa"]), "Datas": l["dias"], "MAE": l["mae"], "RMSE": l["rmse"],
                           "WAPE (%)": l["wape"], "R²": l["r2"], "Tendência": l["vies"], "Excesso potencial": l["excesso"], "Déficit potencial": l["deficit"], "% de dias com déficit potencial": l["dias_deficit"],
                           "Diferença para a referência": l["dif_referencia"], "Intervalo de 95%, início": l["ic_inf"], "Intervalo de 95%, fim": l["ic_sup"]})
    tabela(pd.DataFrame(linhas).round(2), "resultado_cenarios", "Desempenho das alternativas por período. No preparo a referência é o quantitativo; nas compras, o consumo de referência")


def historico_de_treino(d):
    j = d["janelas"][d["janelas"]["universo"] == "sem_atipicos"].merge(d["config"][["janela", "cenario", "modelo"]], on=["janela", "cenario", "modelo"])
    ordem = ["tudo", "2_anos", "1_ano", "ano_anterior_e_atual", "6_meses"]
    fig, eixos = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    opcoes = d["config"][d["config"]["cenario"] == "C2"].set_index("janela")
    for ax, janela in zip(eixos, ("t1", "t2")):
        q = j[(j["janela"] == janela) & (j["cenario"] == opcoes.loc[janela, "cenario"]) & (j["modelo"] == opcoes.loc[janela, "modelo"])].set_index("janela_treino")
        ax.scatter(q[q["periodo"] == "escolha_2025"]["mae"].reindex(ordem), range(5), s=60, color=ARTEFATO, label="Medido em 2025", zorder=3)
        ax.scatter(q[q["periodo"] == "prova_2026"]["mae"].reindex(ordem), range(5), s=60, color=HUMANO, marker="D", label="Desenvolvimento, fevereiro a agosto de 2026", zorder=3)
        ax.set_yticks(range(5), [HISTORICOS[o] for o in ordem])
        ax.set_title(JANELAS[janela], fontsize=10, color=TEXTO2)
        ax.set_xlabel("Erro médio (refeições por dia)")
    eixos[0].invert_yaxis()
    fig.legend(*eixos[0].get_legend_handles_labels(), loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    salvar(fig, "historico_de_treino", "Erro médio conforme o histórico usado para aprender")
    p = d["janelas"][d["janelas"]["universo"] == "sem_atipicos"].pivot_table(index=["janela", "cenario", "modelo"], columns=["periodo", "janela_treino"], values="mae").round(2).reset_index()
    p.columns = [" ".join(str(x) for x in c if x).replace("escolha_2025", "2025").replace("prova_2026", "desenvolvimento, fev. a ago. 2026").replace("prospectiva_set_out", "set. e out. 2026") for c in p.columns]
    tabela(p, "historico_de_treino", "Erro médio por modelo e histórico usado para aprender, sem dias atípicos")


def configuracao_escolhida(d):
    cfg = d["config"].assign(Decisão=d["config"]["janela"].map(JANELAS), Cenário=d["config"]["cenario"].map(lambda c: NOMES_CENARIOS.get(c, c)), Histórico=d["config"]["janela_treino"].map(HISTORICOS))
    cfg["Algoritmo"] = cfg["modelo"].map(ALGORITMOS)
    tabela(cfg[["Decisão", "Cenário", "Algoritmo", "Histórico", "mae"]].rename(columns={"mae": "Erro médio em 2025"}).round(2), "configuracao_escolhida", "ExtraTrees em cada cenário. A escolha usou 2024, 2025 e fevereiro a agosto de 2026")


def comparacao_anual(d):
    a = d["anual"]
    s = a[a["universo"] == "sem_atipicos"]
    anos = sorted(s["ano"].unique())
    nomes_alt = list(NOMES_ANUAL)
    fig, ax = plt.subplots(figsize=(9, 4))
    largura = 0.2
    for i, alt in enumerate(nomes_alt):
        q = s[s["alternativa"] == alt].set_index("ano").reindex(anos)
        ax.bar(np.arange(len(anos)) + (i - 1.5) * largura, q["mae"], largura * 0.92, color=CORES_ANUAL[alt], label=NOMES_ANUAL[alt])
        for x_p, v in zip(np.arange(len(anos)) + (i - 1.5) * largura, q["mae"]):
            if pd.notna(v):
                ax.text(x_p, v, f"{v:.0f}", ha="center", va="bottom", color=TEXTO2, fontsize=8)
    ax.set_xticks(range(len(anos)), [f"{y}\n{'Quantitativo' if y == 2026 else 'Reservas'}" for y in anos])
    ax.set_ylabel("Erro médio (refeições por dia)")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "comparacao_anual", "Erro médio por ano nas mesmas datas, sem dias atípicos. Em 2023 não há histórico anterior para o artefato")
    t = a.assign(Alternativa=a["alternativa"].map(NOMES_ANUAL), Dias_avaliados=a["universo"].map({"sem_atipicos": "Sem atípicos", "todos_os_dias": "Todos"}))
    t = t[["ano", "regime", "Dias_avaliados", "Alternativa", "dias", "mae", "rmse", "wape", "r2", "vies", "excesso", "deficit", "dias_deficit", "dif_plano", "ic_inf", "ic_sup"]].round(2)
    t.columns = ["Ano", "Plano humano do ano", "Dias considerados", "Alternativa", "Datas", "MAE", "RMSE", "WAPE (%)", "R²", "Tendência", "Excesso potencial", "Déficit potencial", "% de dias com déficit potencial", "Diferença para o plano humano", "Intervalo de 95%, início", "Intervalo de 95%, fim"]
    tabela(t, "comparacao_anual", "Comparação anual entre plano humano, consumo de referência, artefato e combinação, nas mesmas datas")


def serie_2026(d):
    x = dias_ate_agosto(d)
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
    salvar(fig, "serie_2026", "Consumo registrado e cada forma de planejamento de fevereiro a agosto de 2026")


def consumo_real_e_alternativas_por_mes(d):
    x, cols = dias_ate_agosto(d), OPCOES_VALOR
    medias = pd.DataFrame({"Consumo real": x.groupby("mes", sort=False)["consumo"].mean(), **{n: x.groupby("mes", sort=False)[c].mean() for n, (c, _) in cols.items()}})
    fig, ax = plt.subplots(figsize=(11, 4.2))
    posicoes = np.arange(len(medias))
    for i, (nome, cor) in enumerate(zip(medias.columns, (CONSUMO, HUMANO, ARTEFATO, COMBINACAO))):
        ax.bar(posicoes + (i - 1.5) * 0.2, medias[nome], 0.18, color=cor, label=nome)
    ax.set_xticks(posicoes, medias.index)
    ax.set_ylabel("Refeições por dia, em média")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "consumo_real_e_alternativas_por_mes", "Consumo real médio e planejamento médio de cada alternativa por mês, de fevereiro a agosto de 2026")
    tabela(medias.round(0).astype(int).reset_index().rename(columns={"mes": "Mês"}), "consumo_real_e_alternativas_por_mes", "Consumo real médio e planejamento médio de cada alternativa por mês, de fevereiro a agosto de 2026")


def mais_perto_do_consumo_real(d):
    x, cols = dias_ate_agosto(d), OPCOES_VALOR
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
    salvar(fig, "mais_perto_do_consumo_real", f"Em cada um dos {len(x)} dias, de fevereiro a agosto de 2026, qual opção ficou mais perto do consumo real. Empates contam para todas as opções empatadas, por isso a soma passa de {len(x)}")
    tabela(pd.DataFrame({"Alternativa": mais_perto.index, "Dias mais perto do consumo real": mais_perto.values, "% dos dias": (mais_perto.values / len(x) * 100).round(1)}), "mais_perto_do_consumo_real", "Dias em que cada alternativa ficou mais perto do consumo real de fevereiro a agosto de 2026. Empates contam para todas as opções empatadas")


def algoritmos_comparados(d):
    j_todos = d["janelas"][(d["janelas"]["universo"] == "sem_atipicos") & (d["janelas"]["periodo"] == "prova_2026") & (d["janelas"]["modelo"] != "extratrees")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, janela, vencedor, referencia in ((a, "t1", d["vencedor_t1"], ("Quantitativo da nutricionista", d["final"].query("janela=='t1' and periodo=='prova_2026' and universo=='sem_atipicos' and alternativa=='plano'")["mae"].iloc[0], HUMANO)), (b, "t2", d["vencedor_t2"], ("Consumo de referência", d["final"].query("janela=='t2' and periodo=='prova_2026' and universo=='sem_atipicos' and alternativa=='consumo_tipico'")["mae"].iloc[0], TIPICO))):
        q = j_todos[(j_todos["janela"] == janela) & (j_todos["cenario"] == vencedor["cenario"]) & (j_todos["janela_treino"] == vencedor["janela_treino"])].sort_values("mae")
        nomes_q = [quebrar(ALGORITMOS[m] + (" (congelado)" if m == vencedor["modelo"] else ""), 42) for m in q["modelo"]] + [referencia[0]]
        barras_horizontais(ax, nomes_q, [*q["mae"], referencia[1]], [ARTEFATO if m == vencedor["modelo"] else "#a6c8ee" for m in q["modelo"]] + [referencia[2]], "{:.1f}")
        ax.set_title(JANELAS[janela], fontsize=10, color=TEXTO2)
        ax.set_xlabel("Erro médio de fevereiro a agosto de 2026 (refeições por dia)")
    fig.tight_layout()
    salvar(fig, "algoritmos_comparados", "Erro médio de cada algoritmo de fevereiro a agosto de 2026, sem dias atípicos, com destaque para o ExtraTrees")


def erro_mensal_2026(d):
    x, cols = dias_ate_agosto(d), OPCOES_VALOR
    m = pd.DataFrame({n: x.groupby("mes", sort=False).apply(lambda g: (g[c] - g["consumo"]).abs().mean(), include_groups=False) for n, (c, _) in cols.items()})
    m.insert(0, "Datas", x.groupby("mes", sort=False).size())
    tabela(m.round(1).reset_index().rename(columns={"mes": "Mês"}), "erro_mensal_2026", "Erro médio por mês de fevereiro a agosto de 2026, com todos os dias")
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
    salvar(fig, "erro_mensal_2026", "Consumo real médio (barras) e erro médio de cada opção (linhas) por mês de fevereiro a agosto de 2026, na mesma escala de refeições, com todos os dias.")


def distribuicao_erros_2026(d):
    x, cols = dias_ate_agosto(d), OPCOES_VALOR
    reg = x[x["regular"]]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.boxplot([reg[c] - reg["consumo"] for c, _ in cols.values()], tick_labels=[quebrar(n, 28) for n in cols], patch_artist=True, widths=0.5, showfliers=True,
               boxprops={"facecolor": TIPICO, "alpha": 0.35, "edgecolor": TEXTO2}, medianprops={"color": TEXTO}, flierprops={"markersize": 3, "markerfacecolor": TEXTO2, "markeredgecolor": TEXTO2})
    ax.axhline(0, color=TEXTO2, linestyle="--", linewidth=1)
    ax.set_ylabel("Plano menos consumo (refeições)")
    fig.tight_layout()
    salvar(fig, "distribuicao_erros_2026", "Distribuição dos desvios de fevereiro a agosto de 2026, sem dias atípicos")


def planejado_contra_consumo_2026(d):
    x, cols = dias_ate_agosto(d), OPCOES_VALOR
    reg = x[x["regular"]]
    fig, eixos = plt.subplots(1, 3, figsize=(11, 3.9), sharex=True, sharey=True)
    for ax, (n, (c, cor)) in zip(eixos, cols.items()):
        ax.scatter(reg[c], reg["consumo"], s=16, color=cor, alpha=0.6, linewidth=0)
        ax.plot([60, 300], [60, 300], color=TEXTO2, linestyle="--", linewidth=1)
        ax.set_xlabel(quebrar(n, 34))
    eixos[0].set_ylabel("Refeições consumidas")
    fig.tight_layout()
    salvar(fig, "planejado_contra_consumo_2026", "Planejado e consumido de fevereiro a agosto de 2026, sem dias atípicos; a linha indica plano igual ao consumo")


def excesso_e_deficit_2026(d):
    reg = dias_ate_agosto(d).query("regular")
    v = d["folgas"]
    fv = v[v["universo"] == "sem_atipicos"]
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
    ax.set_ylabel(f"Refeições somadas em {len(reg)} dias")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "excesso_e_deficit_2026", "Soma do excesso potencial e do déficit potencial de cada opção de preparo de fevereiro a agosto de 2026, sem dias atípicos. A margem muda a cada semana; entre parênteses está o acréscimo médio por dia. O artefato sem margem usa o cenário sem cardápio nos 13 dias sem cardápio registrado")


def risco_das_margens_ate_agosto(d):
    v = d["folgas"]
    fv = v[v["universo"] == "sem_atipicos"]
    risco_das_margens(fv, "risco_informado_e_observado", "Risco de déficit potencial informado antes de cada semana e o que aconteceu de fevereiro a agosto de 2026, sem dias atípicos")


def opcoes_de_margem(d):
    v = d["folgas"]
    nomes_f = {"sem_folga": "Sem margem", "moderada": "Margem moderada", "alta": "Margem alta", "nutricionista": QUANTITATIVO}
    t = v.assign(Decisão=v["decisao"].str.capitalize(), Opção=v["opcao"].map(nomes_f), Dias_c=v["universo"].map({"sem_atipicos": "Sem atípicos", "todos_os_dias": "Todos"}))
    t = t[["Decisão", "Dias_c", "Opção", "dias", "folga_media", "erro_medio", "a_mais", "a_menos", "prometido_dias_a_menos_pct", "dias_a_menos_pct"]].round(1)
    t.columns = ["Decisão", "Dias considerados", "Opção", "Datas", "Margem média", "Erro médio", "Excesso potencial", "Déficit potencial", "Chance informada (%)", "O que aconteceu (%)"]
    tabela(t, "opcoes_de_margem", "Resultado das opções de margem de segurança de fevereiro a agosto de 2026")


def margem_por_situacao(d):
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
    salvar(fig, "margem_por_situacao", "Dias com déficit potencial por situação do dia, no preparo de fevereiro a agosto de 2026")
    t = s.assign(Decisão=s["decisao"].str.capitalize())[["Decisão", "situacao", "dias", "erro_previsao", "dias_a_menos_sem_folga", "dias_a_menos_moderada", "dias_a_menos_alta"]].round(1)
    t.columns = ["Decisão", "Situação do dia", "Datas", "Tendência da previsão", "% abaixo, sem margem", "% abaixo, margem de segurança moderada", "% abaixo, margem de segurança alta"]
    tabela(t, "margem_por_situacao", "Tendência da previsão e proteção das margens de segurança por situação do dia de fevereiro a agosto de 2026")


def erro_por_antecedencia(d):
    linhas = []
    for nome, g in (("Preparo", d["preparo"]), ("Compras", d["compras"])):
        q = g[g["data"].between(INICIO_PROVA, FIM_PROVA) & g["regular"] & g["consumo"].notna()]
        for ant, u in q.groupby((q["data"] - q["origem"]).dt.days):
            linhas.append({"Decisão": nome, "Antecedência (dias)": ant, "Datas": len(u), "Erro médio": (u["previsao"] - u["consumo"]).abs().mean()})
    a = pd.DataFrame(linhas).round(1)
    tabela(a, "erro_por_antecedencia", "Erro médio do artefato por antecedência de fevereiro a agosto de 2026, sem dias atípicos")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(a["Antecedência (dias)"], a["Erro médio"], width=0.7, color=[ARTEFATO if n == "Preparo" else HUMANO for n in a["Decisão"]])
    ax.set_xticks(a["Antecedência (dias)"])
    ax.set_xlabel("Antecedência (dias)")
    ax.set_ylabel("Erro médio (refeições por dia)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=ARTEFATO), plt.Rectangle((0, 0), 1, 1, color=HUMANO)], ["Preparo", "Compras"])
    fig.tight_layout()
    salvar(fig, "erro_por_antecedencia", "Erro médio do artefato por antecedência de fevereiro a agosto de 2026, sem dias atípicos")


def risco_das_margens(fv, arquivo, legenda):
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
    salvar(fig, arquivo, legenda)


def consumo_por_ano(d):
    base = d["base"]
    res = base.groupby("ano").agg(Datas=("consumo", "size"), Consumo_medio=("consumo", "mean"), Menor=("consumo", "min"), Maior=("consumo", "max"), Dias_fora_da_rotina=("tratamento", lambda s: (s != "normal").sum())).round(1).reset_index()
    tabela(res.rename(columns={"ano": "Ano", "Consumo_medio": "Consumo médio", "Dias_fora_da_rotina": "Dias fora da rotina"}), "consumo_por_ano", "Consumo registrado de segunda a sexta-feira, por ano. O ano de 2026 vai até agosto")


def tratamento_dos_dias(d):
    t = d["dias"][d["dias"]["data"] <= FIM_PROVA]["tratamento"].value_counts().rename_axis("tratamento").reset_index(name="Dias")
    t["Situação"] = t["tratamento"].map(TRATAMENTOS)
    t["Usado para aprender"] = t["tratamento"].map({"normal": "Sim", "previsivel_calendario": "Sim", "alto_mantido": "Sim"}).fillna("Não")
    tabela(t[["Situação", "Dias", "Usado para aprender"]], "tratamento_dos_dias", "Classificação dos dias com consumo registrado e uso no aprendizado")


def mae(y, p):
    return (y - p).abs().mean()


ALTERNATIVAS_PROSPECTIVA = {QUANTITATIVO: "nutricionista", NOME_ARTEFATO: "previsao", NOME_HIBRIDO: "combinacao"}


def setembro_outubro(d):
    so = pd.read_csv(RAIZ / "dados" / "referencia_dissertacao" / "setembro_outubro.csv", sep=";", parse_dates=["data"])
    return so.merge(d["preparo"][["data", "previsao", "combinacao", "plano"]], on="data", how="left")


def consumo_e_planos_setembro_outubro(d):
    so = setembro_outubro(d)
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
    salvar(fig, "setembro_outubro", "Avaliação prospectiva: consumo real e planejamento em setembro e outubro de 2026. O peso do artefato no híbrido variou de 20% a 100%, conforme os erros recentes. As linhas são interrompidas nas datas sem registro")


def regulares_e_atipicos(d):
    so = setembro_outubro(d)
    obs = so[so["consumo"].notna()]
    recortes = {"28 dias regulares": obs[obs["status_qualidade"] == "REGULAR"], "5 dias atípicos": obs[obs["status_qualidade"] == "EVENTO_EXTREMO_EXTERNO"], "33 dias com consumo": obs}
    linhas = []
    for recorte, q in recortes.items():
        for nome, coluna in ALTERNATIVAS_PROSPECTIVA.items():
            e = q[coluna] - q["consumo"]
            linhas.append({"Recorte": recorte, "Alternativa": nome, "Erro médio": e.abs().mean(), "Excesso potencial": e.clip(lower=0).sum(), "Déficit potencial": (-e).clip(lower=0).sum(), "% de dias com déficit potencial": (e < 0).mean() * 100})
    t = pd.DataFrame(linhas).round(1)
    tabela(t, "setembro_outubro", "Avaliação prospectiva: resultado em setembro e outubro de 2026, por recorte de dias")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, q, titulo in ((a, recortes["28 dias regulares"], "28 dias regulares"), (b, recortes["5 dias atípicos"], "5 dias atípicos")):
        ax.boxplot([q["nutricionista"] - q["consumo"], q["previsao"] - q["consumo"], q["combinacao"] - q["consumo"]], tick_labels=[quebrar(n, 20) for n in ALTERNATIVAS_PROSPECTIVA], patch_artist=True, widths=0.5,
                   boxprops={"facecolor": TIPICO, "alpha": 0.35, "edgecolor": TEXTO2}, medianprops={"color": TEXTO}, flierprops={"markersize": 3, "markerfacecolor": TEXTO2, "markeredgecolor": TEXTO2})
        ax.axhline(0, color=TEXTO2, linestyle="--", linewidth=1)
        ax.set_title(titulo, fontsize=10, color=TEXTO2)
    a.set_ylabel("Plano menos consumo real (refeições)")
    fig.tight_layout()
    salvar(fig, "regulares_e_atipicos", "Avaliação prospectiva: desvios em setembro e outubro de 2026, em dias regulares e atípicos; os painéis têm escalas próprias")


def semana_de_retorno(d):
    so = setembro_outubro(d)
    sem = so[so["data"].between("2026-10-19", "2026-10-23")]
    x = np.arange(len(sem))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, (coluna, nome, cor) in enumerate((("consumo", "Consumo real", CONSUMO), ("nutricionista", QUANTITATIVO, HUMANO), ("previsao", NOME_ARTEFATO, ARTEFATO), ("combinacao", NOME_HIBRIDO, COMBINACAO))):
        ax.bar(x + (i - 1.5) * 0.2, sem[coluna], 0.18, color=cor, label=nome)
    ax.set_xticks(x, sem["data"].dt.strftime("%d/%m"))
    ax.set_ylabel("Refeições por dia")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "semana_de_retorno", "Avaliação prospectiva: consumo real e planejamento na semana de retorno, de 19 a 23 de outubro de 2026")
    tabela(pd.DataFrame([{"Alternativa": n, "Total na semana": sem[c].sum(), "Erro médio": mae(sem["consumo"], sem[c])} for n, c in {"Consumo real": "consumo", **{k: v for k, v in ALTERNATIVAS_PROSPECTIVA.items()}}.items()]).round(1), "semana_de_retorno", "Avaliação prospectiva: planejamento na semana de retorno, de 19 a 23 de outubro de 2026")


def peso_da_combinacao(d):
    x26 = d["preparo"][d["preparo"]["data"].between(INICIO_PROVA, FIM_PROVA) & d["preparo"]["peso_modelo"].notna()].drop_duplicates("origem")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.step(x26["origem"], x26["peso_modelo"] * 100, where="post", color=COMBINACAO, linewidth=2, label="Peso do artefato no híbrido")
    ax.axhline(50, color=TEXTO2, linestyle="--", linewidth=1, label="Pesos iguais: 50% artefato e 50% quantitativo")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Peso do artefato (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "peso_da_combinacao", "Peso do artefato no híbrido, definido a cada semana pelos erros recentes. O restante é o peso do quantitativo")


def erro_por_dia_da_semana(d):
    q = d["preparo"][d["preparo"]["data"].between(INICIO_PROVA, FIM_PROVA) & d["preparo"]["consumo"].notna() & d["preparo"]["regular"]].copy()
    q["dia"] = q["data"].dt.dayofweek
    nomes = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]
    t = pd.DataFrame([{"Dia": nomes[k], "Datas": len(g), QUANTITATIVO: mae(g["consumo"], g["plano"]), NOME_ARTEFATO: mae(g["consumo"], g["previsao"]), NOME_HIBRIDO: mae(g["consumo"], g["combinacao"])} for k, g in q.groupby("dia")]).round(1)
    tabela(t, "erro_por_dia_da_semana", "Erro médio por dia da semana de fevereiro a agosto de 2026, sem dias atípicos")
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for i, (nome, cor) in enumerate(((QUANTITATIVO, HUMANO), (NOME_ARTEFATO, ARTEFATO), (NOME_HIBRIDO, COMBINACAO))):
        ax.bar(x + (i - 1) * 0.26, t[nome], 0.24, color=cor, label=nome)
    ax.set_xticks(x, t["Dia"])
    ax.set_ylabel("Erro médio (refeições por dia)")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "erro_por_dia_da_semana", "Erro médio por dia da semana de fevereiro a agosto de 2026, sem dias atípicos")


def caixa(ax, x, y, largura, altura, titulo, texto, cor):
    ax.add_patch(FancyBboxPatch((x, y), largura, altura, boxstyle="round,pad=0,rounding_size=0.12", facecolor=cor, edgecolor=TEXTO2, linewidth=0.8))
    ax.text(x + largura / 2, y + altura - 0.17, titulo, ha="center", va="top", fontsize=10, fontweight="bold", color=TEXTO)
    ax.text(x + largura / 2, y + altura - 0.5, texto, ha="center", va="top", fontsize=8.5, color=TEXTO, linespacing=1.3)


def seta(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "-|>", "color": TEXTO2, "linewidth": 1.2})


AZ, VD, LJ, CZ = "#dbe9fa", "#d4f1e6", "#fde3d8", "#efeeea"


def diagrama_arquitetura():
    fig, ax = plt.subplots(figsize=(12, 2.1))
    pecas = [("Fontes", "CSVs do restaurante\ne API de clima", CZ), ("Pipeline", "Bronze, Silver\ne Gold em\narquivos locais", AZ), ("Modelos", "Treino semanal\ndo ExtraTrees", VD), ("Saídas", "Previsões, margens\ne fichas por dia", LJ), ("Aplicação", "Streamlit lê as\nsaídas e mostra\nas opções", LJ)]
    for i, (titulo, texto, cor) in enumerate(pecas):
        caixa(ax, i * 2.5, 0, 2.05, 1.4, titulo, texto, cor)
        if i:
            seta(ax, i * 2.5 - 0.45, 0.7, i * 2.5, 0.7)
    ax.set_xlim(-0.1, 12.4); ax.set_ylim(-0.1, 1.5); ax.axis("off")
    salvar(fig, "diagrama_arquitetura", "Arquitetura do artefato, das fontes de dados à aplicação")


def diagramas():
    fig, ax = plt.subplots(figsize=(12, 2.4))
    etapas = [("Bronze", "Consumo, planos,\ncardápio, calendário,\nclima e previsões\nguardadas", CZ), ("Silver", "Uma linha por dia,\nsem preencher\nausências, com\ndias classificados", AZ), ("Gold", "Uma linha por\nsexta-feira e dia\nprevisto, só com o\nque se sabia", AZ),
              ("Modelos", "ExtraTrees, com os\núltimos 12 meses\nde dados", VD), ("Decisão", "Margem de\nsegurança, contexto\ndo dia e pontos\nde atenção", LJ), ("Aplicação", "Nutricionista e\ngestora comparam\ne decidem", LJ)]
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


def diagrama_sintese_metodologica():
    fig, ax = plt.subplots(figsize=(12, 4.4))
    etapas = [("1. Fontes", "Consumo, reservas,\nquantitativo, cardápio,\ncalendário e clima", CZ, 0, 2.2), ("2. Camadas de dados", "Bronze, Silver e Gold,\ncom dias classificados\ne só o que se sabia", AZ, 4.1, 2.2), ("3. Análise exploratória", "2023 a 2025 e 2026,\nseparadamente", AZ, 8.2, 2.2),
              ("4. Validação semanal", "Treino a cada sexta-feira,\npara preparo e compras", VD, 8.2, 0), ("5. Escolha do modelo", "15 algoritmos e 4 cenários;\nExtraTrees congelado", VD, 4.1, 0), ("6. Apoio à decisão", "Margens, híbrido e\navaliação prospectiva", LJ, 0, 0)]
    for titulo, texto, cor, x, y in etapas:
        caixa(ax, x, y, 3.5, 1.7, titulo, texto, cor)
    seta(ax, 3.5, 3.05, 4.1, 3.05); seta(ax, 7.6, 3.05, 8.2, 3.05); seta(ax, 9.95, 2.2, 9.95, 1.7); seta(ax, 8.2, 0.85, 7.6, 0.85); seta(ax, 4.1, 0.85, 3.5, 0.85)
    ax.set_xlim(-0.1, 11.8); ax.set_ylim(-0.1, 4); ax.axis("off")
    salvar(fig, "diagrama_sintese_metodologica", "Síntese metodológica: das fontes de dados ao apoio à decisão")


def diagrama_fluxo_operacional():
    fig, ax = plt.subplots(figsize=(12, 3))
    passos = [("1. Atualizar", "Novo consumo e\ncalendário", CZ), ("2. Prever", "ExtraTrees prevê\npreparo e compras", VD), ("3. Contextualizar", "Margens, clima,\ncardápio e alertas", AZ), ("4. Decidir", "Nutricionista e\ngestora escolhem", LJ), ("5. Servir", "Consumo é\nregistrado", CZ)]
    for i, (titulo, texto, cor) in enumerate(passos):
        caixa(ax, i * 2.5, 0.7, 2.05, 1.2, titulo, texto, cor)
        if i:
            seta(ax, i * 2.5 - 0.45, 1.3, i * 2.5, 1.3)
    ax.annotate("", xy=(1.02, 0.7), xytext=(11.08, 0.7), arrowprops={"arrowstyle": "-|>", "color": TEXTO2, "linewidth": 1.2, "connectionstyle": "bar,fraction=-0.04"})
    ax.text(6.05, 0.08, "Toda sexta-feira o ciclo recomeça", ha="center", fontsize=9, color=TEXTO2)
    ax.set_xlim(-0.1, 12.2); ax.set_ylim(0, 2); ax.axis("off")
    salvar(fig, "diagrama_fluxo_operacional", "Fluxo operacional semanal: o artefato prevê e informa; a decisão é da nutricionista e da gestora")


def quatro_metricas(d, periodo, nome, legenda):
    """As quatro metricas da dissertacao, por alternativa."""
    f = d["final"]
    r = f[(f["periodo"] == periodo) & (f["universo"] == "sem_atipicos")]
    alternativas = {"consumo_tipico": ("Consumo de referência", TIPICO), "mesmo_dia": (NOMES_CENARIOS["mesmo_dia"], MESMO_DIA), "C2": (NOME_ARTEFATO, ARTEFATO), "plano": (QUANTITATIVO, HUMANO), "C4": (NOME_HIBRIDO, COMBINACAO)}
    preparo = r[r["janela"] == "t1"].set_index("alternativa").reindex(list(alternativas))
    compras = r[r["janela"] == "t2"].set_index("alternativa").reindex(["consumo_tipico", "mesmo_dia", "C2"])
    linhas = []
    for decisao, q in ((JANELAS["t1"], preparo), (JANELAS["t2"], compras)):
        for chave, l in q.iterrows():
            linhas.append({"Decisão": decisao, "Alternativa": alternativas[chave][0], "Datas": int(l["dias"]), "MAE": round(l["mae"], 1), "RMSE": round(l["rmse"], 1), "WAPE (%)": round(l["wape"], 1), "R²": round(l["r2"], 2)})
    tabela(pd.DataFrame(linhas), nome, legenda)
    fig, eixos = plt.subplots(2, 2, figsize=(12, 6), sharey=True)
    nomes_a = [quebrar(alternativas[c][0], 32) for c in preparo.index]
    cores = [alternativas[c][1] for c in preparo.index]
    for ax, coluna, rotulo_x, formato in ((eixos[0, 0], "mae", "MAE (refeições por dia)", "{:.1f}"), (eixos[0, 1], "rmse", "RMSE (refeições por dia)", "{:.1f}"), (eixos[1, 0], "wape", "WAPE (% do consumo)", "{:.1f}%"), (eixos[1, 1], "r2", "R² (quanto mais perto de 1, melhor)", "{:.2f}")):
        barras_horizontais(ax, nomes_a, preparo[coluna].to_numpy(), cores, formato)
        ax.set_xlabel(rotulo_x)
        if coluna == "r2":
            ax.axvline(0, color=TEXTO2, linewidth=1)
    for ax in eixos.flat:
        ax.invert_yaxis() if ax is not eixos[0, 0] else None
    fig.tight_layout()
    salvar(fig, nome, f"{legenda}. Decisão de preparo")


def metricas_por_algoritmo(d):
    j = d["janelas"]
    q = j[(j["periodo"] == "prova_2026") & (j["universo"] == "sem_atipicos") & (j["janela_treino"] == "1_ano") & (j["cenario"] == "C2") & (j["modelo"] != "extratrees")].sort_values(["janela", "mae"])
    tabela(pd.DataFrame({"Decisão": q["janela"].map(JANELAS), "Algoritmo": q["modelo"].map(ALGORITMOS), "Datas": q["dias"], "MAE": q["mae"].round(1), "RMSE": q["rmse"].round(1), "WAPE (%)": q["wape"].round(1), "R²": q["r2"].round(2)}), "metricas_por_algoritmo", "MAE, RMSE, WAPE e R² dos algoritmos de fevereiro a agosto de 2026, sem dias atípicos")


def matriz_de_cenarios(d):
    """Mostra todos os algoritmos em todos os cenarios, com as politicas de decisao."""
    m = pd.read_csv(PASTA_MODELOS / "matriz_completa.csv", sep=";", encoding="utf-8-sig")
    q = m[(m["periodo"] == "fev-ago 2026") & (m["politica"] == "Sem margem") & (m["janela"] == "t1")]
    mapa = q.pivot(index="modelo", columns="cenario", values="mae")
    mapa = mapa.loc[mapa.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(mapa, cmap="Blues_r", aspect="auto")
    ax.set_xticks(range(mapa.shape[1]), [quebrar(c, 16) for c in mapa.columns], fontsize=8)
    ax.set_yticks(range(mapa.shape[0]), mapa.index, fontsize=9)
    ax.grid(False)
    limite = mapa.to_numpy().mean()
    for (i, j), v in np.ndenumerate(mapa.to_numpy()):
        ax.text(j, i, f"{v:.1f}".replace(".", ","), ha="center", va="center", fontsize=8, color="white" if v < limite else TEXTO)
    fig.tight_layout()
    salvar(fig, "matriz_de_cenarios", "MAE de cada algoritmo em cada cenário no preparo, de fevereiro a agosto de 2026, sem margem. Quanto mais escuro, menor o erro")
    r = m[m["modelo"].isin(["Comitê", "Ridge", "Ridge forte", "CatBoost", "KNN", "ExtraTrees", "Huber"])].rename(columns={"janela": "Decisão", "cenario": "Cenário", "modelo": "Algoritmo", "politica": "Política", "periodo": "Período", "dias": "Datas", "mae": "MAE", "rmse": "RMSE", "wape": "WAPE (%)", "r2": "R²", "excesso": "Excesso potencial", "deficit": "Déficit potencial", "dias_deficit_pct": "% de dias com déficit"})
    r["Decisão"] = r["Decisão"].map(JANELAS)
    tabela(r, "matriz_de_cenarios", "Algoritmos principais em cada cenário e política. A matriz completa está em resultados/modelos/matriz_completa.csv")


def quatro_metricas_fev_ago(d):
    quatro_metricas(d, "prova_2026", "quatro_metricas_fev_ago", "MAE, RMSE, WAPE e R² de fevereiro a agosto de 2026, sem dias atípicos")


def quatro_metricas_prospectiva(d):
    quatro_metricas(d, "prospectiva_set_out", "quatro_metricas_prospectiva", "Avaliação prospectiva: MAE, RMSE, WAPE e R² em setembro e outubro de 2026, sem dias atípicos")


def prospectiva_cenarios(d):
    """Resultado de setembro e outubro, separado dos resultados ate agosto."""
    f = d["final"]
    r = f[(f["periodo"] == "prospectiva_set_out") & f["janela"].isin(["t1", "t2"])]
    fig, eixos = plt.subplots(1, 2, figsize=(12, 3.6))
    linhas = []
    for ax, universo, titulo in ((eixos[0], "sem_atipicos", "Sem dias atípicos"), (eixos[1], "todos_os_dias", "Todos os dias")):
        v = r[(r["janela"] == "t1") & (r["universo"] == universo)].set_index("alternativa").reindex(["consumo_tipico", "mesmo_dia", "C2", "plano", "C4"])
        nomes_v = ["Consumo de referência", NOMES_CENARIOS["mesmo_dia"], NOME_ARTEFATO, QUANTITATIVO, NOME_HIBRIDO]
        barras_horizontais(ax, [quebrar(n, 32) for n in nomes_v], v["mae"].to_numpy(), [TIPICO, MESMO_DIA, ARTEFATO, HUMANO, COMBINACAO], "{:.1f}")
        ax.set_title(f"{titulo} ({int(v['dias'].iloc[0])} dias)", fontsize=10, color=TEXTO2)
        ax.set_xlabel("Erro médio (refeições por dia)")
        linhas += [{"Dias considerados": titulo, "Alternativa": n, "Datas": int(l["dias"]), "Erro médio": round(l["mae"], 1), "Excesso potencial": int(l["excesso"]), "Déficit potencial": int(l["deficit"])} for n, (_, l) in zip(nomes_v, v.iterrows())]
    fig.tight_layout()
    salvar(fig, "prospectiva_erro_por_alternativa", "Avaliação prospectiva: erro médio do preparo em setembro e outubro de 2026, separado dos resultados até agosto")
    tabela(pd.DataFrame(linhas), "prospectiva_erro_por_alternativa", "Avaliação prospectiva: preparo em setembro e outubro de 2026")


def prospectiva_margens(d):
    v = d["folgas_prospectiva"]
    fv = v[v["universo"] == "sem_atipicos"]
    risco_das_margens(fv, "prospectiva_margens", f"Avaliação prospectiva: risco de déficit potencial informado antes de cada semana e o que aconteceu nos {fv['dias'].iloc[0]} dias regulares de setembro e outubro de 2026")
    nomes = {"sem_folga": "Sem margem", "moderada": "Margem moderada", "alta": "Margem alta", "nutricionista": QUANTITATIVO}
    t = v.assign(Decisão=v["decisao"].str.capitalize(), Opção=v["opcao"].map(nomes), Dias_c=v["universo"].map({"sem_atipicos": "Sem atípicos", "todos_os_dias": "Todos"}))
    t = t[["Decisão", "Dias_c", "Opção", "dias", "folga_media", "erro_medio", "a_mais", "a_menos", "prometido_dias_a_menos_pct", "dias_a_menos_pct"]].round(1)
    t.columns = ["Decisão", "Dias considerados", "Opção", "Datas", "Margem média", "Erro médio", "Excesso potencial", "Déficit potencial", "Chance informada (%)", "O que aconteceu (%)"]
    tabela(t, "prospectiva_margens", "Avaliação prospectiva: resultado das opções de margem de segurança em setembro e outubro de 2026")


def previsoes_enviadas_outubro(d):
    """Previsoes emitidas antes de cada semana de outubro, pela versao em uso na epoca."""
    e = pd.read_csv(RAIZ / "dados" / "referencia_dissertacao" / "previsoes_enviadas_outubro.csv", sep=";", parse_dates=["data", "emitida_em"])
    e = e.merge(d["preparo"][["data", "consumo", "plano", "previsao"]], on="data").sort_values("data")
    e["semana"] = (e["data"] - pd.to_timedelta(e["data"].dt.dayofweek, unit="D")).dt.strftime("%d/%m")
    opcoes = {"Modelo puro de outubro": ("modelo_puro", TIPICO, "--"), "Híbrido de outubro": ("hibrido", COMBINACAO, "-"), "Híbrido de outubro com margem de +12": ("hibrido_com_margem", COMBINACAO, ":"), QUANTITATIVO: ("plano", HUMANO, "-"), NOME_ARTEFATO + ", versão congelada": ("previsao", ARTEFATO, "-")}
    fig, ax = plt.subplots(figsize=(11, 4.4))
    posicoes = e.groupby("semana", sort=False).cumcount() + e.groupby("semana", sort=False).ngroup() * 6
    for _, g in e.assign(x=posicoes).groupby("semana", sort=False):
        ax.plot(g["x"], g["consumo"], color=CONSUMO, linewidth=2.4, marker="o", markersize=5, label="Consumo real")
        for nome, (coluna, cor, traco) in opcoes.items():
            ax.plot(g["x"], g[coluna], color=cor, linestyle=traco, linewidth=1.8, marker="o", markersize=4, label=nome)
    alcas, nomes = ax.get_legend_handles_labels()
    unicos = dict(zip(nomes, alcas))
    ax.legend(unicos.values(), unicos.keys(), ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1))
    ax.set_xticks(posicoes, e["data"].dt.strftime("%d/%m"))
    ax.set_ylabel("Refeições por dia")
    fig.tight_layout()
    salvar(fig, "previsoes_enviadas_outubro", "Previsões enviadas antes de cada semana de outubro de 2026 pela versão em uso na época: modelo puro, híbrido e híbrido com margem. Para comparar, o quantitativo final, o artefato na versão congelada e o consumo real. Registro disponível a partir de 19/10")
    linhas = []
    for semana, g in [*e.groupby("semana", sort=False), ("19/10 a 30/10", e)]:
        linhas.append({"Semana": semana, "Datas": len(g), "Emitida em": " e ".join(g["emitida_em"].dt.strftime("%d/%m").unique()), **{f"Erro médio: {n}": round(mae(g["consumo"], g[c]), 1) for n, (c, _, _) in opcoes.items()}})
    tabela(pd.DataFrame(linhas), "previsoes_enviadas_outubro", "Erro médio das previsões enviadas antes de cada semana de outubro de 2026, comparado ao quantitativo final e ao artefato na versão congelada. O híbrido de 19 a 23/10 usou a estimativa disponível em 14/10, anterior ao quantitativo final")


def com_e_sem_atipicos(d, periodo, nome, texto):
    """Mostra o mesmo resultado com e sem os dias atipicos, para a exclusao ficar a vista."""
    f = d["final"]
    r = f[f["periodo"] == periodo]
    fig, eixos = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [5, 3]})
    linhas = []
    for ax, janela, ordem in ((eixos[0], "t1", ["consumo_tipico", "mesmo_dia", "C2", "plano", "C4"]), (eixos[1], "t2", ["consumo_tipico", "mesmo_dia", "C2"])):
        q = {u: r[(r["janela"] == janela) & (r["universo"] == u)].set_index("alternativa").reindex(ordem) for u in ("sem_atipicos", "todos_os_dias")}
        posicoes = np.arange(len(ordem))
        for deslocamento, u, cor, rotulo_u in ((-0.2, "sem_atipicos", ARTEFATO, "Sem dias atípicos"), (0.2, "todos_os_dias", "#a6c8ee", "Todos os dias")):
            ax.bar(posicoes + deslocamento, q[u]["mae"], 0.38, color=cor, label=rotulo_u)
            for x_p, v in zip(posicoes + deslocamento, q[u]["mae"]):
                ax.text(x_p, v, f"{v:.1f}".replace(".", ","), ha="center", va="bottom", color=TEXTO2, fontsize=9)
        ax.set_xticks(posicoes, [quebrar(NOME_ARTEFATO if o == "C2" else QUANTITATIVO if o == "plano" else NOME_HIBRIDO if o == "C4" else NOMES_CENARIOS[o], 16) for o in ordem], fontsize=9)
        ax.set_title(f"{JANELAS[janela]}: {int(q['sem_atipicos']['dias'].iloc[0])} e {int(q['todos_os_dias']['dias'].iloc[0])} dias", fontsize=10, color=TEXTO2)
        for o in ordem:
            s, t = q["sem_atipicos"].loc[o], q["todos_os_dias"].loc[o]
            linhas.append({"Decisão": JANELAS[janela], "Alternativa": NOME_ARTEFATO if o == "C2" else QUANTITATIVO if o == "plano" else NOME_HIBRIDO if o == "C4" else NOMES_CENARIOS[o],
                           "Datas sem atípicos": int(s["dias"]), "Datas com todos os dias": int(t["dias"]), "MAE sem atípicos": round(s["mae"], 1), "MAE com todos os dias": round(t["mae"], 1),
                           "RMSE sem atípicos": round(s["rmse"], 1), "RMSE com todos os dias": round(t["rmse"], 1), "WAPE sem atípicos (%)": round(s["wape"], 1), "WAPE com todos os dias (%)": round(t["wape"], 1),
                           "R² sem atípicos": round(s["r2"], 2), "R² com todos os dias": round(t["r2"], 2)})
    eixos[0].set_ylabel("Erro médio (refeições por dia)")
    eixos[0].legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    salvar(fig, nome, f"Erro médio com e sem dias atípicos ({texto}). Os dias atípicos só são identificados depois que acontecem")
    tabela(pd.DataFrame(linhas), nome, f"Métricas com e sem dias atípicos ({texto})")


def com_e_sem_atipicos_fev_ago(d):
    com_e_sem_atipicos(d, "prova_2026", "com_e_sem_atipicos_fev_ago", "fevereiro a agosto de 2026")


def com_e_sem_atipicos_set_out(d):
    com_e_sem_atipicos(d, "prospectiva_set_out", "com_e_sem_atipicos_set_out", "setembro e outubro de 2026")


def ridge_e_extratrees(d):
    t = pd.read_csv(PASTA_MODELOS / "comparacao_ridge_extratrees.csv", sep=";", encoding="utf-8-sig")
    t = t[t["universo"] == "sem_atipicos"].assign(Decisão=lambda q: q["janela"].map(JANELAS))
    v = t.pivot_table(index=["Decisão", "politica", "periodo"], columns="modelo", values="mae", sort=False).round(1).reset_index()
    tabela(v.rename(columns={"politica": "Política", "periodo": "Período", "regressao": "MAE do Ridge", "extratrees": "MAE do ExtraTrees"}), "ridge_e_extratrees", "Ridge e ExtraTrees nas mesmas datas e políticas, sem dias atípicos")


def ganho_sobre_linhas_de_base(d):
    """Quanto o artefato e o hibrido ganham de cada referencia, com intervalo por semanas inteiras."""
    f = d["final"]
    comparacoes = (("t1_vs_tipico", "C2", "Consumo de referência"), ("t1_vs_mesmo_dia", "C2", NOMES_CENARIOS["mesmo_dia"]), ("t1_vs_mesmo_dia", "C4", NOMES_CENARIOS["mesmo_dia"]), ("t1", "C2", QUANTITATIVO), ("t1", "C4", QUANTITATIVO),
                   ("t2", "C2", "Consumo de referência"), ("t2_vs_mesmo_dia", "C2", NOMES_CENARIOS["mesmo_dia"]))
    periodos = {"prova_2026": "Fevereiro a agosto de 2026", "prospectiva_set_out": "Setembro e outubro de 2026"}
    linhas = []
    for periodo, nome_periodo in periodos.items():
        for universo, nome_universo in (("sem_atipicos", "Sem atípicos"), ("todos_os_dias", "Todos os dias")):
            for janela, alternativa, comparada in comparacoes:
                l = f[(f["periodo"] == periodo) & (f["universo"] == universo) & (f["janela"] == janela) & (f["alternativa"] == alternativa)].iloc[0]
                linhas.append({"Período": nome_periodo, "Dias considerados": nome_universo, "Decisão": JANELAS[janela[:2]], "Alternativa": NOME_ARTEFATO if alternativa == "C2" else NOME_HIBRIDO, "Comparada com": comparada,
                               "Datas": int(l["dias"]), "Diferença de erro médio": round(l["dif_referencia"], 1), "Intervalo de 95%, início": round(l["ic_inf"], 1), "Intervalo de 95%, fim": round(l["ic_sup"], 1)})
    tabela(pd.DataFrame(linhas), "ganho_sobre_linhas_de_base", "Diferença de erro médio para cada referência. Valor negativo significa erro menor que o da referência; o intervalo reamostra semanas inteiras")


def balanco_dias_regulares(d):
    so = setembro_outubro(d)
    reg = so[(so["status_qualidade"] == "REGULAR") & so["consumo"].notna()]
    opcoes = ALTERNATIVAS_PROSPECTIVA
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    posicoes = np.arange(3)
    for deslocamento, sinal, cor, nome in ((-0.2, 1, ARTEFATO, "Excesso potencial"), (0.2, -1, HUMANO, "Déficit potencial")):
        valores = [(sinal * (reg[c] - reg["consumo"])).clip(lower=0).sum() for c in opcoes.values()]
        ax.bar(posicoes + deslocamento, valores, 0.38, color=cor, label=nome)
        for x_p, valor in zip(posicoes + deslocamento, valores):
            ax.text(x_p, valor, f"{valor:.0f}", ha="center", va="bottom", color=TEXTO2, fontsize=9)
    ax.set_xticks(posicoes, [quebrar(n, 28) for n in opcoes], fontsize=9)
    ax.set_ylabel(f"Refeições somadas em {len(reg)} dias regulares")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "balanco_dias_regulares", f"Avaliação prospectiva: excesso potencial e déficit potencial nos {len(reg)} dias regulares de setembro e outubro de 2026")


def importancia_das_informacoes(d):
    colunas = CENARIOS[CENARIO_CAMPEAO]
    origem = pd.Timestamp("2025-01-01")
    fig, eixos = plt.subplots(1, 2, figsize=(12.5, 4.4))
    linhas = []
    for ax, janela in zip(eixos, ("t1", "t2")):
        gold = d[janela]
        treino = dados_de_treino(gold, origem, colunas, 12)
        alvo = gold[(gold["data"] >= origem) & (gold["data"] <= CORTE) & gold["consumo"].notna() & gold["tratamento"].isin(["normal", "previsivel_calendario", "alto_mantido"])]
        modelo_artefato = construir_modelo(CAMPEAO)
        modelo_artefato.fit(matriz(treino, colunas), treino["alvo_relativo"].clip(*LIMITES_ALVO))
        def erro(x):
            previsao = modelo_artefato.predict(x) * alvo["nivel_base"].to_numpy()
            return np.abs(previsao - alvo["consumo"].to_numpy()).mean()
        x = matriz(alvo, colunas)
        base_erro = erro(x)
        gerador = np.random.default_rng(42)
        ganho = {}
        for c in colunas:
            medidas = []
            for _ in range(20):
                embaralhado = x.copy()
                embaralhado[c] = gerador.permutation(embaralhado[c].to_numpy())
                medidas.append(erro(embaralhado) - base_erro)
            ganho[c] = np.mean(medidas)
        ordem = sorted(ganho, key=ganho.get, reverse=True)
        linhas += [{"Decisão": JANELAS[janela], "Variável": rotulo(c), "Aumento do erro (refeições por dia)": round(ganho[c], 2)} for c in ordem]
        barras_horizontais(ax, [quebrar(rotulo(c), 34) for c in ordem[:10]], [ganho[c] for c in ordem[:10]], ARTEFATO, "{:.2f}")
        ax.set_title(JANELAS[janela], fontsize=10, color=TEXTO2)
        ax.set_xlabel("Quanto o erro aumenta sem esta informação (refeições por dia)")
    fig.tight_layout()
    salvar(fig, "importancia_das_informacoes", "As 10 informações mais importantes para o ExtraTrees: treino até 2024 e erro medido em 2025, embaralhando uma informação de cada vez")
    tabela(pd.DataFrame(linhas), "importancia_das_informacoes", "Importância das informações para o ExtraTrees, com treino até 2024 e medição em 2025")


def efeito_do_clima(d):
    p = pd.read_parquet(PASTA_MODELOS / "previsoes_janelas_treino.parquet")
    p = p[(p["modelo"] == CAMPEAO) & (p["janela_treino"] == HISTORICO_CAMPEAO) & (p["periodo"] == "prova_2026") & p["regular"] & p["consumo"].notna()]
    extras = []
    for janela in ("t1", "t2"):
        origens = sorted(d[janela].loc[d[janela]["data"].between(INICIO_PROVA, FIM_PROVA), "origem"].unique())
        serie = prever_serie(d[janela], origens, CENARIOS["S_clima_perfeito"], CAMPEAO, 12)
        extras.append(d[janela].loc[serie.index, ["data", "consumo"]].assign(janela=janela, cenario="S_clima_perfeito", previsao=arredondar_refeicoes(serie)))
    p = pd.concat([p, pd.concat(extras)], ignore_index=True)
    linhas = []
    fig, eixos = plt.subplots(1, 2, figsize=(11, 3))
    for ax, janela, cenarios in ((eixos[0], "t1", {"C2": "Sem clima", "C3": "Com previsão do tempo", "S_clima_perfeito": "Com o clima que de fato ocorreu"}), (eixos[1], "t2", {"C2": "Sem clima", "S_clima_perfeito": "Com o clima que de fato ocorreu"})):
        q = p[p["janela"] == janela].pivot(index="data", columns="cenario", values="previsao")[list(cenarios)].dropna()
        consumo = p[p["janela"] == janela].drop_duplicates("data").set_index("data")["consumo"].reindex(q.index)
        erros = [(q[c] - consumo).abs().mean() for c in cenarios]
        barras_horizontais(ax, list(cenarios.values()), erros, ARTEFATO, "{:.1f}")
        ax.set_title(JANELAS[janela], fontsize=10, color=TEXTO2)
        ax.set_xlabel(f"Erro médio em {len(q)} dias (refeições por dia)")
        linhas += [{"Decisão": JANELAS[janela], "Informação de clima": n, "Datas": len(q), "Erro médio": round(e, 1)} for n, e in zip(cenarios.values(), erros)]
    fig.tight_layout()
    salvar(fig, "efeito_do_clima", "Erro do ExtraTrees com e sem clima de fevereiro a agosto de 2026, sem dias atípicos. O clima que de fato ocorreu não seria conhecido na sexta-feira e mostra o limite do ganho possível. Nas compras, ainda não há histórico suficiente de previsões guardadas")
    tabela(pd.DataFrame(linhas), "efeito_do_clima", "Erro do ExtraTrees com e sem informação de clima de fevereiro a agosto de 2026")


def compras_e_preparo(d):
    prova = d["final"][(d["final"]["periodo"] == "prova_2026") & (d["final"]["universo"] == "sem_atipicos")]
    ordem = ["consumo_tipico", "mesmo_dia", "H", "C1", "C2"]
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    posicoes = np.arange(len(ordem))
    for deslocamento, janela, cor in ((-0.2, "t1", ARTEFATO), (0.2, "t2", HUMANO)):
        v = prova[prova["janela"] == janela].set_index("alternativa").reindex(ordem)["mae"]
        ax.bar(posicoes + deslocamento, v, 0.38, color=cor, label=JANELAS[janela])
        for x_p, valor in zip(posicoes + deslocamento, v):
            ax.text(x_p, valor, f"{valor:.1f}".replace(".", ","), ha="center", va="bottom", color=TEXTO2, fontsize=9)
    ax.set_xticks(posicoes, [quebrar(NOMES_CENARIOS[o], 18) for o in ordem])
    ax.set_ylabel("Erro médio (refeições por dia)")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "compras_e_preparo", "Erro médio de fevereiro a agosto de 2026 no preparo e nas compras, por informação usada, sem dias atípicos")


def fronteira_do_hibrido(d):
    x = d["preparo"][d["preparo"]["data"].between(INICIO_PROVA, FIM_PROVA) & d["preparo"]["consumo"].notna() & d["preparo"]["regular"] & d["preparo"]["plano"].notna()]
    pesos = np.arange(0, 101, 10)
    erros = [(arredondar_refeicoes(p_ / 100 * x["previsao"] + (1 - p_ / 100) * x["plano"]) - x["consumo"]).abs().mean() for p_ in pesos]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.plot(pesos, erros, color=COMBINACAO, linewidth=2, marker="o", markersize=7, label="Peso fixo o ano inteiro")
    ax.axhline((x["combinacao"] - x["consumo"]).abs().mean(), color=TEXTO2, linestyle="--", linewidth=1, label="Híbrido usado: peso ajustado a cada semana")
    for p_, e in zip(pesos[::2], erros[::2]):
        ax.text(p_, e + 0.25, f"{e:.1f}".replace(".", ","), ha="center", color=TEXTO2, fontsize=9)
    ax.set_xticks(pesos, [f"{p_}%" for p_ in pesos])
    ax.set_xlabel("Peso do artefato (o restante é do quantitativo)")
    ax.set_ylabel("Erro médio (refeições por dia)")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "fronteira_do_hibrido", "Erro de fevereiro a agosto de 2026 conforme o peso do artefato no híbrido, sem dias atípicos. A curva é descritiva: o melhor peso só é conhecido depois")
    tabela(pd.DataFrame({"Peso do artefato (%)": pesos, "Erro médio": np.round(erros, 2)}), "fronteira_do_hibrido", "Erro de fevereiro a agosto de 2026 conforme o peso do artefato no híbrido")


def frequencia_de_deficit_e_excesso(d):
    x = d["preparo"][d["preparo"]["data"].between(INICIO_PROVA, FIM_PROVA) & d["preparo"]["consumo"].notna() & d["preparo"]["regular"] & d["preparo"]["plano"].notna()]
    fv = d["folgas"][(d["folgas"]["universo"] == "sem_atipicos") & (d["folgas"]["decisao"] == "preparo")].set_index("opcao")
    hibrido = x["combinacao"] - x["consumo"]
    nomes_op = [QUANTITATIVO, "Artefato sem margem", "Artefato com margem moderada", "Artefato com margem alta", NOME_HIBRIDO]
    chaves = ["nutricionista", "sem_folga", "moderada", "alta"]
    dias_deficit = [fv.loc[c, "dias_a_menos_pct"] for c in chaves] + [(hibrido < 0).mean() * 100]
    excesso_dia = [fv.loc[c, "a_mais"] / fv.loc[c, "dias"] for c in chaves] + [hibrido.clip(lower=0).sum() / len(x)]
    fig, eixos = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    cores = [HUMANO, ARTEFATO, ARTEFATO, ARTEFATO, COMBINACAO]
    barras_horizontais(eixos[0], [quebrar(n, 30) for n in nomes_op], dias_deficit, cores, "{:.0f}%")
    barras_horizontais(eixos[1], [quebrar(n, 30) for n in nomes_op], excesso_dia, cores, "{:.0f}")
    eixos[1].invert_yaxis()
    eixos[0].set_xlabel("Dias com déficit potencial (%)")
    eixos[1].set_xlabel("Excesso potencial médio (refeições por dia)")
    fig.tight_layout()
    salvar(fig, "frequencia_de_deficit_e_excesso", "Frequência de déficit potencial e excesso potencial médio por opção de preparo de fevereiro a agosto de 2026, sem dias atípicos")


def mae_e_rmse_por_mes(d):
    todos = d["preparo"][d["preparo"]["data"].between(INICIO_PROVA, FIM_PROVA) & d["preparo"]["consumo"].notna() & d["preparo"]["plano"].notna()].assign(mes=lambda t_: t_["data"].dt.strftime("%m/%Y"))
    cols = {QUANTITATIVO: ("plano", HUMANO, "-"), NOME_ARTEFATO: ("previsao", ARTEFATO, "-"), NOME_HIBRIDO: ("combinacao", COMBINACAO, "--")}
    fig, eixos = plt.subplots(1, 2, figsize=(12, 3.8), sharex=True)
    linhas = []
    for mes, g in todos.groupby("mes", sort=False):
        linha = {"Mês": mes, "Datas": len(g)}
        for n, (c, _, _) in cols.items():
            e = g[c] - g["consumo"]
            linha[f"Erro médio: {n}"] = round(e.abs().mean(), 1)
            linha[f"Erro com peso nos grandes desvios: {n}"] = round(np.sqrt((e ** 2).mean()), 1)
        linhas.append(linha)
    t_mes = pd.DataFrame(linhas)
    for ax, prefixo, rotulo_y in ((eixos[0], "Erro médio", "Erro médio (refeições por dia)"), (eixos[1], "Erro com peso nos grandes desvios", "Erro com peso nos grandes desvios")):
        for n, (_, cor, traco) in cols.items():
            ax.plot(t_mes["Mês"], t_mes[f"{prefixo}: {n}"], color=cor, linestyle=traco, linewidth=2, marker="o", markersize=6, label=n)
        ax.set_ylabel(rotulo_y)
        ax.tick_params(axis="x", labelsize=8)
    fig.legend(*eixos[0].get_legend_handles_labels(), loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    salvar(fig, "mae_e_rmse_por_mes", "Erro médio (MAE) e erro com peso nos grandes desvios (RMSE) por mês de fevereiro a agosto de 2026, com todos os dias")
    tabela(t_mes, "mae_e_rmse_por_mes", "MAE e RMSE por mês de fevereiro a agosto de 2026, com todos os dias")


def perfil_semanal_por_ano(d):
    b = d["base"][d["base"]["data"].dt.dayofweek < 5]
    v = b.pivot_table(index=b["data"].dt.dayofweek, columns="ano", values="consumo", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    tons = {2023: "#c9c8c1", 2024: "#a6c8ee", 2025: "#5c9ee4", 2026: ARTEFATO}
    for i, ano in enumerate(v.columns):
        ax.bar(np.arange(5) + (i - 1.5) * 0.2, v[ano], 0.18, color=tons[ano], label=str(ano))
    ax.set_xticks(range(5), ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"])
    ax.set_ylabel("Refeições por dia, em média")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    salvar(fig, "perfil_semanal_por_ano", "Consumo médio por dia da semana em cada ano. O ano de 2026 vai até agosto")


def base_e_conjuntos(d):
    g1 = d["t1"]
    avaliacao = {2023: "Só histórico", 2024: "Escolha do modelo", 2025: "Escolha do modelo", 2026: "Escolha do modelo até agosto; setembro e outubro em avaliação prospectiva"}
    tabela(pd.DataFrame([{"Ano": ano, "Datas com consumo": int(g["consumo"].notna().sum()), "Datas usadas para aprender": int(g["usar_no_treino"].sum()), "Datas fora do aprendizado": int((g["consumo"].notna() & ~g["usar_no_treino"]).sum()), "Papel na avaliação": avaliacao[ano]} for ano, g in g1.groupby("ano")]), "base_e_conjuntos", "Base histórica e conjuntos de avaliação na decisão de preparo")


def grupos_de_informacoes():
    grupos = (("Consumo recente", HISTORICO), ("Calendário", CALENDARIO), ("Cardápio", CARDAPIO), ("Previsão do tempo", CLIMA_PREVISTO))
    tabela(pd.DataFrame([{"Grupo": g, "Informação": rotulo(c)} for g, cs in grupos for c in cs]), "grupos_de_informacoes", "Grupos de informações do artefato")


def familias_de_modelos():
    familias = {"Lineares": "OLS, Ridge, Lasso, ElasticNet, Huber", "Por semelhança": "KNN", "Margem": "SVR", "Bagging": "RandomForest, ExtraTrees (modelo congelado)", "Boosting": "GradientBoosting, HistGB, LightGBM, XGBoost, CatBoost", "Redes neurais": "MLP"}
    tabela(pd.DataFrame({"Família": familias.keys(), "Algoritmos testados": familias.values()}), "familias_de_modelos", "Famílias de modelos testadas no torneio")


def ingestao_bronze():
    pastas = []
    for pasta in sorted(p_ for p_ in (RAIZ / "dados" / "bronze").iterdir() if p_.is_dir() and p_.name not in ("financeiro", "producao")):
        arquivos = list(pasta.glob("*.csv"))
        pastas.append({"Fonte": pasta.name, "Arquivos": len(arquivos), "Linhas": sum(max(len(a.read_bytes().splitlines()) - 1, 0) for a in arquivos)})
    tabela(pd.DataFrame(pastas), "ingestao_bronze", "Arquivos e linhas das fontes na camada Bronze")


HISTORIA = [
    ("1. Como o artefato funciona", ["diagrama_arquitetura", "diagrama_fluxo_do_artefato", "diagrama_cenarios", "diagrama_validacao_semanal", "diagrama_fluxo_operacional"],
     ["ingestao_bronze", "tratamento_dos_dias", "base_e_conjuntos", "grupos_de_informacoes", "familias_de_modelos"]),
    ("2. Visão geral do consumo: 2023 a agosto de 2026", ["consumo_diario", "consumo_ano_mes", "consumo_medio_por_ano", "perfil_semanal_por_ano"], ["consumo_por_ano"]),
    ("3.1. Análise de 2023 a 2025 (reservas): consumo passado", ["autocorrelacao_2023_2025", "relacao_defasagens_2023_2025", "informacoes_repetidas"], ["autocorrelacao_2023_2025", "relacao_defasagens_2023_2025", "relacao_atributos_sexta", "informacoes_repetidas"]),
    ("3.2. Análise de 2023 a 2025 (reservas): calendário", ["dia_da_semana_2023_2025", "fim_do_periodo_2023_2025", "eventos_do_calendario_2023_2025"], []),
    ("3.3. Análise de 2023 a 2025 (reservas): cardápio", ["cardapio_2023_2025"], []),
    ("3.4. Análise de 2023 a 2025 (reservas): clima", ["chuva_2023_2025", "chuva_bruta_e_ajustada_2023_2025", "temperatura_2023_2025"], ["chuva_bruta_e_ajustada_2023_2025"]),
    ("3.5. Análise de 2023 a 2025 (reservas): reservas e consumo", ["reservas_e_consumo", "excesso_deficit_reservas"], ["reservas_por_ano"]),
    ("3.6. Análise de 2023 a 2025 (reservas): síntese", ["relacao_variaveis_2023_2025"], ["efeitos_2023_2025", "relacao_variaveis_2023_2025"]),
    ("4.1. Análise de fevereiro a agosto de 2026 (quantitativo): consumo passado", ["autocorrelacao_2026", "relacao_defasagens_2026"], ["autocorrelacao_2026", "relacao_defasagens_2026"]),
    ("4.2. Análise de fevereiro a agosto de 2026 (quantitativo): calendário", ["dia_da_semana_2026", "fim_do_periodo_2026", "eventos_do_calendario_2026"], []),
    ("4.3. Análise de fevereiro a agosto de 2026 (quantitativo): cardápio", ["cardapio_2026"], []),
    ("4.4. Análise de fevereiro a agosto de 2026 (quantitativo): clima", ["chuva_2026", "chuva_bruta_e_ajustada_2026", "temperatura_2026", "qualidade_previsao_tempo"], ["chuva_bruta_e_ajustada_2026", "qualidade_previsao_tempo"]),
    ("4.5. Análise de fevereiro a agosto de 2026 (quantitativo): quantitativo e consumo", ["quantitativo_e_consumo", "excesso_deficit_quantitativo"], ["quantitativo_por_mes"]),
    ("4.6. Análise de fevereiro a agosto de 2026 (quantitativo): síntese", ["relacao_variaveis_2026"], ["efeitos_2026", "relacao_variaveis_2026"]),
    ("4.7. O que repetiu e o que mudou entre os períodos", ["repetiu_e_mudou", "diferenca_plano_consumo", "excesso_deficit_ano", "saldo_e_diferencas"], ["resumo_dos_periodos", "repetiu_e_mudou", "planos_por_ano"]),
    ("5. Treino e teste dos algoritmos", ["torneio_de_algoritmos", "historico_de_treino", "importancia_das_informacoes", "matriz_de_cenarios"], ["torneio_de_algoritmos", "ideias_testadas", "historico_de_treino", "importancia_das_informacoes", "matriz_de_cenarios", "configuracao_escolhida"]),
    ("6. Resultados de fevereiro a agosto de 2026", ["cenarios_2026", "quatro_metricas_fev_ago", "com_e_sem_atipicos_fev_ago", "compras_e_preparo", "efeito_do_clima", "algoritmos_comparados", "serie_2026", "consumo_real_e_alternativas_por_mes", "erro_mensal_2026", "mae_e_rmse_por_mes",
                                                     "mais_perto_do_consumo_real", "distribuicao_erros_2026", "planejado_contra_consumo_2026", "erro_por_dia_da_semana", "erro_por_antecedencia", "comparacao_anual"],
     ["resultado_cenarios", "quatro_metricas_fev_ago", "com_e_sem_atipicos_fev_ago", "ganho_sobre_linhas_de_base", "ridge_e_extratrees", "metricas_por_algoritmo", "efeito_do_clima", "consumo_real_e_alternativas_por_mes", "erro_mensal_2026", "mae_e_rmse_por_mes", "mais_perto_do_consumo_real", "erro_por_dia_da_semana", "erro_por_antecedencia", "comparacao_anual"]),
    ("7. Margens de segurança e híbrido", ["risco_informado_e_observado", "margem_por_situacao", "fronteira_do_hibrido", "peso_da_combinacao", "excesso_e_deficit_2026", "frequencia_de_deficit_e_excesso"],
     ["opcoes_de_margem", "margem_por_situacao", "fronteira_do_hibrido"]),
    ("8. Avaliação prospectiva: setembro e outubro de 2026", ["prospectiva_erro_por_alternativa", "quatro_metricas_prospectiva", "com_e_sem_atipicos_set_out", "setembro_outubro", "regulares_e_atipicos", "balanco_dias_regulares", "prospectiva_margens", "semana_de_retorno", "previsoes_enviadas_outubro"],
     ["prospectiva_erro_por_alternativa", "quatro_metricas_prospectiva", "com_e_sem_atipicos_set_out", "setembro_outubro", "prospectiva_margens", "semana_de_retorno", "previsoes_enviadas_outubro"]),
    ("9. Síntese do método", ["diagrama_sintese_metodologica"], []),
]


def preparar_figuras_tabelas():
    """Limpa as saidas e carrega os dados. As figuras e tabelas saem depois, parte a parte, no notebook 09."""
    lista.clear()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    TABELAS.mkdir(parents=True, exist_ok=True)
    for antigo in list(FIGURAS.glob("*.png")) + list(TABELAS.glob("*.csv")):
        antigo.unlink()
    return carregar()


def escrever_lista():
    secoes = {(tipo, n + 1): secao for tipo, indice in (("Figura", 1), ("Tabela", 2)) for n, secao in enumerate(s[0] for s in HISTORIA for _ in s[indice])}
    linhas = ["# Figuras e tabelas para a dissertação", "", "Numeração na ordem da apresentação à banca: como o artefato funciona, visão geral do consumo, análise de 2023 a 2025, análise de fevereiro a agosto de 2026, treino e teste, resultados, margens e híbrido, e avaliação prospectiva. Figuras a 300 DPI, sem título na imagem; as legendas abaixo são sugestões. Tabelas em CSV com separador `;` e decimal com vírgula.", ""]
    for tipo, titulo in (("Figura", "Figuras"), ("Tabela", "Tabelas")):
        linhas += [f"## {titulo}", ""]
        atual = None
        for (t_, n), (a, l) in sorted(i for i in lista.items() if i[0][0] == tipo):
            if secoes[tipo, n] != atual:
                atual = secoes[tipo, n]
                linhas += ["", f"### {atual}", "", "| Número | Arquivo | Legenda sugerida |", "|---:|---|---|"]
            linhas.append(f"| {n} | `{a}` | {l} |")
        linhas.append("")
    (PASTA_DISSERTACAO / "LISTA_FIGURAS_TABELAS.md").write_text("\n".join(linhas), encoding="utf-8")
    celulas = ["# 12. Galeria de figuras e tabelas\n\nAs figuras seguem a ordem da apresentação à banca. As legendas são sugestões. Arquivos em `resultados/dissertacao/`."]
    atual = None
    for (_, numero), (arquivo, legenda) in sorted(i for i in lista.items() if i[0][0] == "Figura"):
        if secoes["Figura", numero] != atual:
            atual = secoes["Figura", numero]
            celulas.append(f"## {atual}")
        celulas.append(f"**Figura {numero}.** {legenda}\n\n![Figura {numero}](../resultados/dissertacao/figuras/{arquivo})")
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(c) for c in celulas], metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    nbf.write(nb, RAIZ / "notebooks" / "12_galeria_figuras_e_tabelas.ipynb")


def fechar_figuras_tabelas():
    """Confere se toda a historia foi gerada e escreve a lista e a galeria."""
    for tipo, indice in (("Figura", 1), ("Tabela", 2)):
        previstas = sum(len(s[indice]) for s in HISTORIA)
        geradas = sum(t_ == tipo for t_, _ in lista)
        assert geradas == previstas, f"{tipo}: {geradas} de {previstas}. Rode o notebook inteiro antes de fechar"
    escrever_lista()
    print(sum(t_ == "Figura" for t_, _ in lista), "figuras e", sum(t_ == "Tabela" for t_, _ in lista), "tabelas")


OPCOES_VALOR = {QUANTITATIVO: ("plano", HUMANO), NOME_ARTEFATO: ("previsao", ARTEFATO), NOME_HIBRIDO: ("combinacao", COMBINACAO)}


def moeda(v):
    return f"R$ {v:,.2f}".replace(",", "#").replace(".", ",").replace("#", ".")


def resumir_valor(t, coluna, valor):
    e = t[coluna] - t["consumo"]
    return {"Datas": len(t), "Erro médio": round(e.abs().mean(), 1), "Excesso potencial (refeições)": int(e.clip(lower=0).sum()), "Déficit potencial (refeições)": int((-e).clip(lower=0).sum()),
            "% de dias com déficit potencial": round((e < 0).mean() * 100, 1), "Valor potencial do excesso": moeda(e.clip(lower=0).sum() * valor)}


def gerar_valor_potencial():
    PASTA_LOCAL.mkdir(exist_ok=True)
    caminho = RAIZ / "config_local.json"
    config = json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else {}
    valor = config.get("valor_refeicao")
    if not valor or not valor["2026"]:
        print("Sem valor local: etapa ignorada")
        return
    atual, anterior = valor["2026"], valor["ate_2025"]
    serie = pd.read_parquet(RAIZ / "resultados" / "comparacao" / "serie_preparo.parquet")
    margens = pd.read_csv(RAIZ / "resultados" / "planejamento" / "verificacao_folgas_2026.csv", sep=";", encoding="utf-8-sig")
    so = pd.read_csv(RAIZ / "dados" / "referencia_dissertacao" / "setembro_outubro.csv", sep=";", parse_dates=["data"]).merge(serie[["data", "previsao", "combinacao"]], on="data", how="left")
    so = so[so["consumo"].notna()].assign(plano=lambda t: t["nutricionista"])

    x = serie[serie["data"].between(INICIO_PROVA, FIM_PROVA) & serie["consumo"].notna() & serie["regular"] & serie["plano"].notna()]
    linhas = [{"Opção": n, **resumir_valor(x, c, atual)} for n, (c, _) in OPCOES_VALOR.items()]
    m = margens[(margens["decisao"] == "preparo") & (margens["universo"] == "sem_atipicos")].set_index("opcao")
    for chave, nome in (("moderada", "Artefato com margem moderada"), ("alta", "Artefato com margem alta")):
        l = m.loc[chave]
        linhas.append({"Opção": f"{nome} (+{l['folga_media']:.0f} refeições)", "Datas": int(l["dias"]), "Erro médio": round(l["erro_medio"], 1), "Excesso potencial (refeições)": int(l["a_mais"]), "Déficit potencial (refeições)": int(l["a_menos"]),
                       "% de dias com déficit potencial": round(l["dias_a_menos_pct"], 1), "Valor potencial do excesso": moeda(l["a_mais"] * atual)})
    opcoes = pd.DataFrame(linhas)
    opcoes.to_csv(PASTA_LOCAL / "tabela_opcoes_com_valor_2026.csv", sep=";", index=False, encoding="utf-8-sig")

    fig, eixos = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    nomes = [quebrar(n, 30) for n in opcoes["Opção"]]
    cores = [HUMANO, ARTEFATO, COMBINACAO, ARTEFATO, ARTEFATO]
    barras_horizontais(eixos[0], nomes, opcoes["% de dias com déficit potencial"].to_numpy(), cores, "{:.0f}%")
    barras_horizontais(eixos[1], nomes, opcoes["Excesso potencial (refeições)"].to_numpy() * atual / 1000, cores, "{:.1f} mil")
    eixos[1].invert_yaxis()
    eixos[0].set_xlabel("Dias com déficit potencial (%)")
    eixos[1].set_xlabel("Valor potencial do excesso (R$ mil)")
    fig.tight_layout()
    fig.savefig(PASTA_LOCAL / "figura_protecao_e_valor_potencial.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    anual = []
    base = serie[serie["consumo"].notna() & serie["regular"] & serie["plano"].notna() & (serie["data"] <= FIM_PROVA)]
    for ano, g in base.groupby("ano"):
        preco = atual if ano == 2026 else anterior
        anual.append({"Ano": ano, "Opção": "Quantitativo da nutricionista" if ano == 2026 else "Reservas dos alunos", **resumir_valor(g, "plano", preco)})
        if g["previsao"].notna().all():
            anual += [{"Ano": ano, "Opção": NOME_ARTEFATO, **resumir_valor(g, "previsao", preco)}, {"Ano": ano, "Opção": NOME_HIBRIDO, **resumir_valor(g.dropna(subset=["combinacao"]), "combinacao", preco)}]
    pd.DataFrame(anual).to_csv(PASTA_LOCAL / "tabela_sintese_por_ano_com_valor.csv", sep=";", index=False, encoding="utf-8-sig")

    recortes = {"28_dias_regulares": so[so["status_qualidade"] == "REGULAR"], "5_dias_atipicos": so[so["status_qualidade"] == "EVENTO_EXTREMO_EXTERNO"],
                "33_dias_com_consumo": so, "semana_de_retorno": so[so["data"].between("2026-10-19", "2026-10-23")]}
    for nome, t in recortes.items():
        pd.DataFrame([{"Opção": n, **resumir_valor(t, c, atual)} for n, (c, _) in OPCOES_VALOR.items()]).to_csv(PASTA_LOCAL / f"tabela_{nome}_com_valor.csv", sep=";", index=False, encoding="utf-8-sig")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    for n, (c, cor) in OPCOES_VALOR.items():
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
    fig.savefig(PASTA_LOCAL / "figura_acumulado_bimestre.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
