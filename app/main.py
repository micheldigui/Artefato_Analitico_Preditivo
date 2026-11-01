"""Aplicacao de apoio ao planejamento de refeicoes.

Execute a partir da raiz deste repositorio: streamlit run app/main.py
"""
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "notebooks"))
from funcoes.nomes import EVENTOS, TRATAMENTOS  # noqa: E402

PLANEJAMENTO = RAIZ / "resultados" / "planejamento"
SILVER = RAIZ / "dados" / "silver"
DECISOES = {"preparo": "Preparo (semana seguinte)", "compras": "Compras (daqui a duas semanas)"}
OPCOES = {"sem_folga": "Sem margem", "moderada": "Margem moderada", "alta": "Margem alta"}
REGISTRO = RAIZ / "dados" / "registro_decisoes" / "registro_decisoes.csv"
MOTIVOS = ["Igual à previsão", "Evento na escola", "Cardápio", "Previsão do tempo", "Prova ou período de avaliação", "Turmas ausentes", "Experiência da semana anterior", "Outro"]

st.set_page_config(page_title="Planejamento de Refeições", layout="wide", initial_sidebar_state="collapsed")


@st.cache_data
def carregar_fichas():
    arquivo = sorted(PLANEJAMENTO.glob("fichas_*.json"))[-1]
    return json.loads(arquivo.read_text(encoding="utf-8"))


@st.cache_data
def carregar_erros():
    arquivo = sorted(PLANEJAMENTO.glob("erros_recentes_*.json"))[-1]
    return json.loads(arquivo.read_text(encoding="utf-8"))


@st.cache_data
def carregar_verificacao():
    return pd.read_csv(PLANEJAMENTO / "verificacao_folgas_2026.csv", sep=";", encoding="utf-8-sig")


@st.cache_data
def carregar_historico():
    fato = pd.read_parquet(SILVER / "silver_fato_diaria.parquet", columns=["data", "consumo", "no_escopo", "evento_tipo", "ano"])
    dias = pd.read_parquet(SILVER / "silver_dias_classificados.parquet", columns=["data", "tratamento", "razao_referencia"])
    return fato[fato["no_escopo"] & fato["consumo"].notna()].merge(dias, on="data")


def rotulo_dia(ficha):
    return f"{ficha['dia'].split('-')[0]} {pd.Timestamp(ficha['data']).strftime('%d/%m')}"


def chance_de_passar(quantidade, previsao, erros):
    """Parcela dos erros recentes maior que a margem de segurança implicita na quantidade."""
    return sum(e > quantidade - previsao for e in erros) / len(erros)


def tabela_da_semana(fichas):
    linhas = []
    for f in fichas:
        linha = {"Dia": rotulo_dia(f), "Previsão": f["previsao"], "Margem moderada": f["opcoes"]["moderada"]["refeicoes"], "Margem alta": f["opcoes"]["alta"]["refeicoes"]}
        if f["decisao"] == "preparo":
            linha["Quantitativo"] = f["quantitativo_nutricionista"]
            linha["Híbrido"] = f["combinacao_com_quantitativo"]
        linha["Atenção"] = "Sim" if f["pontos_de_atencao"] else ""
        linhas.append(linha)
    return pd.DataFrame(linhas)


def mostrar_opcoes(ficha):
    for coluna, (chave, nome) in zip(st.columns(3), OPCOES.items()):
        opcao = ficha["opcoes"][chave]
        with coluna, st.container(border=True):
            st.metric(f"{nome} (+{opcao['folga']})" if opcao["folga"] else nome, f"{opcao['refeicoes']} refeições")
            st.caption(f"Chance de déficit potencial: {opcao['chance_consumo_acima_pct']}%")


def texto_calendario(ficha):
    c = ficha["calendario"]
    texto = f"{c['evento'].capitalize()}; {c['dias_de_aula_na_semana']} dias de aula na semana"
    if c["dias_de_aula_ate_fim_do_periodo"] is not None:
        texto += f"; faltam {c['dias_de_aula_ate_fim_do_periodo']} dias de aula no período"
    return texto + "."


def mostrar_contexto(ficha):
    h = ficha["historico"]
    esquerda, direita = st.columns(2)
    with esquerda, st.container(border=True):
        st.markdown("**O dia**")
        st.markdown(f"**Calendário:** {texto_calendario(ficha)}")
        st.markdown(f"**Cardápio:** {ficha['cardapio'] or 'ainda não informado.'}")
        st.markdown(f"**Clima:** {ficha['clima']['texto']}")
    with direita, st.container(border=True):
        st.markdown("**O histórico**")
        st.markdown(f"**Consumo de referência:** {h['consumo_tipico_recente']} refeições")
        st.markdown(f"**Mesmo dia nas últimas vezes:** {', '.join(map(str, h['mesmo_dia_ultimas_vezes']))}")
        if h["mesma_epoca_ano_passado_pct"] is not None:
            st.markdown(f"**Mesma época do ano passado:** {h['mesma_epoca_ano_passado_pct']}% da referência daquele momento")
        for parecido in ficha["dias_parecidos"]:
            st.markdown(f"**Dias parecidos:** {parecido}")


def mostrar_simulacao(ficha, erros):
    st.markdown("**Simule sua quantidade**")
    inicial = ficha["quantitativo_nutricionista"] or ficha["previsao"]
    quantidade = st.number_input("Quantas refeições você pretende planejar?", min_value=0, max_value=600, value=inicial, step=5, key=f"simula_{ficha['decisao']}_{ficha['data']}")
    diferenca = quantidade - ficha["previsao"]
    chance = chance_de_passar(quantidade, ficha["previsao"], erros)
    a, b = st.columns(2)
    a.metric("Diferença para a previsão", f"{diferenca:+d} refeições")
    b.metric("Chance de déficit potencial com essa quantidade", f"{chance * 100:.0f}%")
    st.caption(f"Estimativa com os erros da previsão em {len(erros)} dias recentes. Excesso e déficit são potenciais: o restaurante não pesa o que é descartado nem registra quem ficou sem refeição.")


def mostrar_ficha(ficha, erros):
    st.markdown(f"#### {ficha['dia']}, {pd.Timestamp(ficha['data']).strftime('%d/%m/%Y')}")
    st.markdown(f"**Previsão: {ficha['previsao']} refeições** · {ficha['base_da_previsao']}")
    if ficha["quantitativo_nutricionista"] is not None:
        st.markdown(f"**Quantitativo da nutricionista:** {ficha['quantitativo_nutricionista']} · **Híbrido:** {ficha['combinacao_com_quantitativo']}")
    for ponto in ficha["pontos_de_atencao"]:
        st.warning(ponto)
    mostrar_opcoes(ficha)
    mostrar_contexto(ficha)
    with st.container(border=True):
        mostrar_simulacao(ficha, erros)
    with st.container(border=True):
        registrar_decisao(ficha)


def registrar_decisao(ficha):
    st.markdown("**Registrar a decisão**")
    with st.form(f"registro_{ficha['decisao']}_{ficha['data']}", clear_on_submit=True):
        quantidade = st.number_input("Quantidade decidida", min_value=0, max_value=600, value=ficha["quantitativo_nutricionista"] or ficha["previsao"], step=5)
        motivo = st.selectbox("Principal motivo", MOTIVOS)
        observacao = st.text_input("Observação (opcional)")
        if st.form_submit_button("Salvar"):
            linha = pd.DataFrame([{"registrado_em": pd.Timestamp.now().isoformat(timespec="seconds"), "sexta_do_planejamento": ficha["sexta_do_planejamento"], "decisao": ficha["decisao"], "data": ficha["data"],
                                   "previsao": ficha["previsao"], "quantidade_decidida": quantidade, "motivo": motivo, "observacao": observacao}])
            REGISTRO.parent.mkdir(parents=True, exist_ok=True)
            linha.to_csv(REGISTRO, sep=";", index=False, encoding="utf-8-sig", mode="a", header=not REGISTRO.exists())
            st.success("Decisão registrada.")
    if REGISTRO.exists():
        st.download_button("Baixar registros", REGISTRO.read_bytes(), "registro_decisoes.csv", "text/csv")


def pagina_planejamento():
    fichas, erros = carregar_fichas(), carregar_erros()
    st.subheader("Planejamento da semana")
    st.write("Opções de quantidade para cada dia, com os dados que ajudam a escolher. A decisão é da nutricionista e da gestora.")
    st.caption(f"Sexta-feira do planejamento: {pd.Timestamp(fichas[0]['sexta_do_planejamento']).strftime('%d/%m/%Y')}")
    decisao = st.radio("Decisão", list(DECISOES), format_func=DECISOES.get, horizontal=True, label_visibility="collapsed")
    selecionadas = [f for f in fichas if f["decisao"] == decisao]
    if decisao == "compras":
        st.info("Com duas semanas de antecedência não há quantitativo, e a previsão do tempo é pouco confiável.")
    st.dataframe(tabela_da_semana(selecionadas), hide_index=True, width="stretch")
    rotulos = {rotulo_dia(f): f for f in selecionadas}
    dia = st.radio("Dia", list(rotulos), horizontal=True, key=f"dia_{decisao}")
    mostrar_ficha(rotulos[dia], erros[decisao])


def pagina_historico():
    st.subheader("Consumo registrado")
    historico = carregar_historico()
    anos = sorted(historico["ano"].unique())
    ano = st.segmented_control("Ano", anos, default=anos[-1])
    if ano is None:
        ano = anos[-1]
    d = historico[historico["ano"] == ano].copy()
    d["Situação"] = d["tratamento"].map(TRATAMENTOS)
    a, b, c = st.columns(3)
    a.metric("Dias com consumo", len(d))
    b.metric("Consumo médio por dia", f"{d['consumo'].mean():.0f} refeições")
    c.metric("Dias fora da rotina", int((d["tratamento"] != "normal").sum()))
    grafico = alt.Chart(d).mark_line(strokeWidth=2, color="#2a78d6").encode(
        x=alt.X("data:T", title=None), y=alt.Y("consumo:Q", title="Refeições por dia"),
        tooltip=[alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"), alt.Tooltip("consumo:Q", title="Refeições"), alt.Tooltip("Situação:N")])
    pontos = alt.Chart(d[d["tratamento"] != "normal"]).mark_point(filled=True, size=55).encode(
        x="data:T", y="consumo:Q", color=alt.Color("Situação:N", title="Dias fora da rotina", scale=alt.Scale(scheme="set2"), legend=alt.Legend(orient="bottom", columns=3)),
        tooltip=[alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"), alt.Tooltip("consumo:Q", title="Refeições"), alt.Tooltip("Situação:N")])
    st.altair_chart(grafico + pontos, width="stretch")
    fora = d[d["tratamento"] != "normal"].sort_values("data", ascending=False)
    with st.expander(f"Ver os {len(fora)} dias fora da rotina"):
        tabela = pd.DataFrame({"Data": fora["data"].dt.strftime("%d/%m/%Y"), "Refeições": fora["consumo"].astype(int), "Situação": fora["Situação"],
                               "Comparado à referência": (fora["razao_referencia"] * 100).round(0).astype("Int64").astype(str) + "%", "Evento": fora["evento_tipo"].map(EVENTOS).fillna(fora["evento_tipo"])})
        st.dataframe(tabela, hide_index=True, width="stretch")


def pagina_confiabilidade():
    st.subheader("Essas opções são confiáveis?")
    st.write("Comparação entre a chance informada antes de cada semana e o que aconteceu de fevereiro a agosto de 2026, sem dias atípicos. Excesso e déficit são potenciais: o restaurante não pesa o que é descartado nem registra quem ficou sem refeição.")
    v = carregar_verificacao()
    d = v[(v["universo"] == "sem_atipicos") & v["opcao"].isin(OPCOES)]
    st.dataframe(pd.DataFrame({
        "Decisão": d["decisao"].map({"preparo": "Preparo", "compras": "Compras"}), "Opção": d["opcao"].map(OPCOES),
        "Chance informada (% dos dias)": d["prometido_dias_a_menos_pct"].round(1), "O que aconteceu (% dos dias)": d["dias_a_menos_pct"].round(1),
        "Erro médio (refeições por dia)": d["erro_medio"].round(1), "Excesso potencial (refeições)": d["a_mais"].round(0).astype(int), "Déficit potencial (refeições)": d["a_menos"].round(0).astype(int),
    }), hide_index=True, width="stretch")
    st.markdown("""
**O que os testes de fevereiro a agosto de 2026 mostraram**

- O artefato usa o algoritmo ExtraTrees, escolhido com dados de 2024, 2025 e de fevereiro a agosto de 2026.
- No preparo, o artefato sozinho erra cerca de 25 refeições por dia, e o quantitativo da nutricionista, cerca de 24.
- O híbrido, que junta os dois, erra cerca de 20 refeições por dia.
- Nas compras, com duas semanas de antecedência, o erro fica em torno de 23 refeições por dia.

**Avaliação prospectiva de setembro e outubro de 2026**

- Sem dias atípicos, o erro no preparo foi de cerca de 13 refeições por dia no artefato, 22 no quantitativo e 18 no híbrido.
- São só 28 dias, então esses números ainda são pouco estáveis.

**Limites**

- A mesma margem de segurança não protege igual em todo dia. Nos últimos dias do período de aulas e em dias atípicos, a margem de segurança pode só aumentar o excesso.
- Eventos como enchentes não são previstos pelo algoritmo.
- Os números vêm de dados já registrados de um único restaurante. Não medem desperdício físico, economia nem pessoas sem refeição.
""")


pagina = st.navigation([
    st.Page(pagina_planejamento, title="Planejamento da semana", default=True),
    st.Page(pagina_historico, title="Consumo registrado", url_path="consumo"),
    st.Page(pagina_confiabilidade, title="Confiabilidade", url_path="confiabilidade"),
], position="top")
st.title("Planejamento de Refeições")
pagina.run()
st.divider()
st.caption("Autor: Michel Silva de Souza")
