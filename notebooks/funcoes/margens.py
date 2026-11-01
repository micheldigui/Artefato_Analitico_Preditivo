"""Margens de seguranca, contexto do dia e fichas de decisao."""
import json
import shutil
import math
import numpy as np
import pandas as pd

from .configuracao import BRONZE, FIM_PROSPECTIVA, FIM_PROVA, GOLD, INICIO_PROSPECTIVA, INICIO_PROVA, PASTA_MODELOS, PASTA_PLANEJAMENTO, RAIZ, SILVER
from .nomes import FOLGAS as NOMES_FOLGA
from .validacao import medir


JANELA_ERROS = 60


MINIMO_ERROS = 20


NIVEIS = {"sem_folga": None, "moderada": 0.75, "alta": 0.90}


CHUVA_FORTE_DIA = 15.0


CHUVA_FORTE_MANHA = 5.0


MINIMO_DIAS_PARECIDOS = 3


LIMITE_QUEDA = 0.95


LIMITE_DIVERGENCIA = 0.15


DIAS_SEMANA = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira"]


def calcular_folgas(tabela, coluna):
    """Folga de cada emissao, a partir dos erros de dias anteriores a ela."""
    dados = tabela.sort_values("data")
    conhecidos = dados[dados["regular"] & dados["consumo"].notna() & dados[coluna].notna()]
    linhas = []
    for origem in sorted(dados["origem"].unique()):
        erros = (conhecidos[conhecidos["data"] < origem]["consumo"] - conhecidos[conhecidos["data"] < origem][coluna]).tail(JANELA_ERROS)
        if len(erros) < MINIMO_ERROS:
            continue
        linha = {"origem": origem, "sem_folga": 0.0, "erros_usados": len(erros)}
        for nivel, quantil in NIVEIS.items():
            if quantil is not None:
                linha[nivel] = max(float(erros.quantile(quantil)), 0.0)
            linha[f"chance_faltar_{nivel}"] = float((erros > linha[nivel]).mean())
        linhas.append(linha)
    return pd.DataFrame(linhas)


def erros_na_emissao(tabela, origem):
    """Erros recentes conhecidos na emissao, usados na simulacao da aplicacao."""
    d = tabela[(tabela["data"] < origem) & tabela["regular"] & tabela["consumo"].notna()].sort_values("data")
    return (d["consumo"] - d["previsao"]).tail(JANELA_ERROS).round(1).tolist()


def aplicar_folgas(tabela, coluna):
    folgas = calcular_folgas(tabela, coluna)
    t = tabela.merge(folgas, on="origem")
    for nivel in NIVEIS:
        t[f"plano_{nivel}"] = np.ceil(t[coluna] + t[nivel])
    return t


def verificar(t, nome, referencia=None, inicio=INICIO_PROVA, fim=FIM_PROVA):
    """Compara a protecao prometida com a observada no periodo."""
    prova = t[t["data"].between(inicio, fim) & t["consumo"].notna()]
    linhas = []
    for universo, d in (("sem_atipicos", prova[prova["regular"]]), ("todos_os_dias", prova)):
        for nivel in NIVEIS:
            m = medir(d["consumo"], d[f"plano_{nivel}"])
            linhas.append(dict(decisao=nome, universo=universo, opcao=nivel, dias=m["dias"], erro_medio=m["mae"], folga_media=d[nivel].mean(),
                               a_mais=m["excesso"], a_menos=m["deficit"], dias_a_menos_pct=m["dias_deficit"],
                               prometido_dias_a_menos_pct=d[f"chance_faltar_{nivel}"].mean() * 100))
        if referencia is not None:
            m = medir(d["consumo"], d[referencia])
            linhas.append(dict(decisao=nome, universo=universo, opcao="nutricionista", dias=m["dias"], erro_medio=m["mae"], a_mais=m["excesso"], a_menos=m["deficit"], dias_a_menos_pct=m["dias_deficit"]))
    return pd.DataFrame(linhas)


def verificar_por_situacao(t, gold):
    prova = t[t["data"].between(INICIO_PROVA, FIM_PROVA) & t["consumo"].notna()].merge(gold[["data", "fim_de_bloco", "inicio_de_bloco", "pos_pausa"]], on="data")
    prova["situacao"] = np.select([prova["fim_de_bloco"], prova["inicio_de_bloco"] | prova["pos_pausa"], ~prova["regular"]],
                                  ["Últimos dias do período", "Início do período ou volta de pausa", "Dia atípico"], "Dia comum")
    linhas = []
    for situacao, d in prova.groupby("situacao"):
        linha = {"situacao": situacao, "dias": len(d), "erro_previsao": (d["previsao"] - d["consumo"]).mean()}
        for nivel in NIVEIS:
            linha[f"dias_a_menos_{nivel}"] = (d[f"plano_{nivel}"] < d["consumo"]).mean() * 100
        linhas.append(linha)
    return pd.DataFrame(linhas)


def dias_parecidos(passado, filtro, descricao):
    grupo = passado[filtro(passado)]
    if len(grupo) < MINIMO_DIAS_PARECIDOS:
        return None
    razao = grupo["razao_referencia"].median()
    return {"texto": f"{descricao}: consumo em torno de {razao * 100:.0f}% da referência ({len(grupo)} dias anteriores).", "razao": float(razao)}


def contexto_historico(g, passado):
    alertas = []
    data = g["data"]
    if g["dias_ate_fim_bloco"] <= 3:
        alertas.append(dias_parecidos(passado, lambda p: p["dias_ate_fim_bloco"] <= 3, "Últimos dias do período de aulas"))
    if g["inicio_de_bloco"]:
        alertas.append(dias_parecidos(passado, lambda p: p["dias_desde_inicio_bloco"] <= 4, "Primeiros dias do período de aulas"))
    if g["pos_pausa"]:
        alertas.append(dias_parecidos(passado, lambda p: p["pos_pausa"], "Volta depois de dia sem aula"))
    if g["vespera_de_pausa"]:
        alertas.append(dias_parecidos(passado, lambda p: p["vespera_de_pausa"], "Véspera de dia sem aula"))
    if g["evento_especial"]:
        alertas.append(dias_parecidos(passado, lambda p: p["evento_tipo"] == g["evento_tipo"], f"Evento {g['evento_tipo'].replace('_', ' ')}"))
    if g["dia_semana"] == 3:
        alertas.append(dias_parecidos(passado, lambda p: p["dia_semana"] == 3, "Quintas-feiras"))
    ano_passado = data - pd.Timedelta(days=364)
    epoca = passado[passado["data"].between(ano_passado - pd.Timedelta(days=7), ano_passado + pd.Timedelta(days=7))]
    return {"dias_parecidos": [a for a in alertas if a], "mesma_epoca_ano_passado_pct": None if epoca.empty else round(float(epoca["razao_referencia"].median()) * 100), "dias_mesma_epoca": len(epoca)}


def pontos_de_atencao(previsao, quantitativo, clima, parecidos):
    pontos = []
    if clima["alerta"]:
        pontos.append("Chuva forte prevista: o consumo pode ficar abaixo do esperado; margem de segurança pode aumentar o excesso.")
    for parecido in parecidos:
        if parecido["razao"] < LIMITE_QUEDA:
            pontos.append("Dias parecidos costumam ficar abaixo da referência: confira antes de usar margem de segurança.")
            break
    if quantitativo is not None and abs(quantitativo - previsao) / previsao >= LIMITE_DIVERGENCIA:
        lado = "acima" if quantitativo > previsao else "abaixo"
        pontos.append(f"O quantitativo está {abs(quantitativo - previsao):.0f} refeições {lado} da previsão: vale conferir o motivo.")
    return pontos


def contexto_clima(chuva, manha, temp, decisao):
    if pd.isna(chuva):
        return {"texto": "Previsão do tempo indisponível.", "alerta": False}
    alerta = chuva >= CHUVA_FORTE_DIA or (pd.notna(manha) and manha >= CHUVA_FORTE_MANHA)
    texto = f"Chuva prevista: {chuva:.0f} mm no dia" + (f", {manha:.0f} mm de manhã" if pd.notna(manha) else "") + (f"; máxima de {temp:.0f} °C." if pd.notna(temp) else ".")
    if decisao == "compras":
        texto += " Com duas semanas de antecedência, essa previsão é pouco confiável."
    return {"texto": texto, "alerta": bool(alerta and decisao == "preparo")}


def previsao_do_tempo_rodada(origem):
    """Previsao por dia na rodada da sexta-feira, para datas ainda sem Previous Runs."""
    hor = pd.read_csv(BRONZE / "clima_previsoes" / "single_runs_horario.csv", sep=";", parse_dates=["time", "rodada"])
    hor = hor[hor["rodada"] == origem].assign(data=lambda d: d["time"].dt.normalize())
    dia = hor.groupby(["modelo", "data"]).agg(chuva=("precipitation", "sum"), temp=("temperature_2m", "max")).reset_index()
    manha = hor[hor["time"].dt.hour.between(6, 13)].groupby(["modelo", "data"])["precipitation"].sum().rename("manha").reset_index()
    return dia.merge(manha, on=["modelo", "data"]).groupby("data")[["chuva", "manha", "temp"]].mean()


def montar_fichas(t, gold, classificados, decisao, origem, rodada):
    linhas = t[t["origem"] == origem].merge(gold, on=["origem", "data"], suffixes=("", "_g"))
    passado = classificados[classificados["data"] < origem]
    fichas = []
    for _, g in linhas.sort_values("data").iterrows():
        chuva, manha, temp = g["previsao_chuva_mm"], g["previsao_chuva_manha_mm"], g["previsao_temp_max"]
        if pd.isna(chuva) and g["data"] in rodada.index:
            chuva, manha, temp = rodada.loc[g["data"], ["chuva", "manha", "temp"]]
        historico = contexto_historico(g, passado)
        clima = contexto_clima(chuva, manha, temp, decisao)
        recentes = [int(g[f"mesmo_dia_ha_{k}_ocorrencias"]) for k in range(1, 5) if pd.notna(g[f"mesmo_dia_ha_{k}_ocorrencias"])]
        previsao = math.ceil(float(g["previsao"]))
        quantitativo = None if "plano" not in g or pd.isna(g.get("plano")) else int(g["plano"])
        combinacao = None if quantitativo is None else round(g["peso_modelo"] * float(g["previsao"]) + (1 - g["peso_modelo"]) * quantitativo)
        fichas.append({
            "decisao": decisao, "sexta_do_planejamento": str(origem.date()), "data": str(g["data"].date()), "dia": DIAS_SEMANA[int(g["dia_semana"])],
            "previsao": previsao, "base_da_previsao": g["base"],
            "opcoes": {n: {"refeicoes": int(g[f"plano_{n}"]), "folga": int(g[f"plano_{n}"]) - previsao, "chance_consumo_acima_pct": round(g[f"chance_faltar_{n}"] * 100)} for n in NIVEIS},
            "quantitativo_nutricionista": quantitativo,
            "combinacao_com_quantitativo": combinacao,
            "pontos_de_atencao": pontos_de_atencao(previsao, quantitativo, clima, historico["dias_parecidos"]),
            "calendario": {"evento": g["evento_tipo"].replace("_", " "), "dias_de_aula_ate_fim_do_periodo": None if g["dias_ate_fim_bloco"] >= 15 else int(g["dias_ate_fim_bloco"]), "dias_de_aula_na_semana": int(g["dias_letivos_na_semana"])},
            "cardapio": None if pd.isna(g["prato_padronizado"]) else g["prato_padronizado"].title(),
            "clima": clima,
            "historico": {"consumo_tipico_recente": round(float(g["nivel_base"])), "mesmo_dia_ultimas_vezes": recentes,
                          "mesma_epoca_ano_passado_pct": historico["mesma_epoca_ano_passado_pct"], "dias_na_mesma_epoca": historico["dias_mesma_epoca"]},
            "dias_parecidos": [p["texto"] for p in historico["dias_parecidos"]],
        })
    return fichas


def escrever_ficha(f):
    data = pd.Timestamp(f["data"]).strftime("%d/%m/%Y")
    linhas = [f"### {f['dia']}, {data}", "", f"**Previsão: {f['previsao']} refeições** ({f['base_da_previsao']})", "",
              "| Opção | Refeições | Margem de segurança | Chance de déficit potencial |", "|---|---:|---:|---:|"]
    for nivel, nome in NOMES_FOLGA.items():
        o = f["opcoes"][nivel]
        linhas.append(f"| {nome} | {o['refeicoes']} | +{o['folga']} | {o['chance_consumo_acima_pct']}% |")
    linhas.append("")
    if f["quantitativo_nutricionista"] is not None:
        linhas.append(f"- **Quantitativo da nutricionista:** {f['quantitativo_nutricionista']}. Combinado com a previsão: {f['combinacao_com_quantitativo']}.")
    ate_fim = f["calendario"]["dias_de_aula_ate_fim_do_periodo"]
    calendario = f"{f['calendario']['evento'].capitalize()}; {f['calendario']['dias_de_aula_na_semana']} dias de aula na semana"
    linhas.append(f"- **Calendário:** {calendario}" + (f"; faltam {ate_fim} dias de aula no período." if ate_fim is not None else "."))
    linhas.append(f"- **Cardápio:** {f['cardapio'] or 'ainda não informado.'}")
    linhas.append(f"- **Clima:** {f['clima']['texto']}")
    h = f["historico"]
    texto = f"Consumo de referência de {h['consumo_tipico_recente']}; mesmo dia nas últimas vezes: {', '.join(map(str, h['mesmo_dia_ultimas_vezes']))}."
    if h["mesma_epoca_ano_passado_pct"] is not None:
        texto += f" Na mesma época do ano passado, o consumo ficou em {h['mesma_epoca_ano_passado_pct']}% da referência daquele momento."
    linhas.append(f"- **Histórico:** {texto}")
    linhas += [f"- **Dias parecidos:** {p}" for p in f["dias_parecidos"]]
    linhas += [f"- **Atenção:** {p}" for p in f["pontos_de_atencao"]]
    return "\n".join(linhas) + "\n"


def escrever_documento(fichas, origem):
    partes = [f"# Planejamento de refeições: sexta-feira, {origem.strftime('%d/%m/%Y')}", "",
              "As opções e os dados abaixo apoiam a decisão. Quem define a quantidade é a nutricionista, com a gestora.", ""]
    for decisao, titulo in (("preparo", "Preparo (semana seguinte)"), ("compras", "Compras (daqui a duas semanas)")):
        partes += [f"## {titulo}", ""] + [escrever_ficha(f) for f in fichas if f["decisao"] == decisao]
    return "\n".join(partes)


def preparar_classificados():
    fato = pd.read_parquet(SILVER / "silver_fato_diaria.parquet")
    dias = pd.read_parquet(SILVER / "silver_dias_classificados.parquet")
    t1 = pd.read_parquet(GOLD / "gold_t1.parquet")
    pausas = t1[["data", "pos_pausa", "vespera_de_pausa", "evento_tipo", "dias_ate_fim_bloco", "dias_desde_inicio_bloco", "dia_semana"]]
    base = dias.merge(pausas, on="data").merge(fato[["data"]], on="data")
    return base[base["usar_no_treino"]]


def gerar_margens_e_fichas():
    PASTA_PLANEJAMENTO.mkdir(parents=True, exist_ok=True)
    preparo = pd.read_parquet(PASTA_MODELOS / "preparo_diario_final.parquet")
    compras = pd.read_parquet(PASTA_MODELOS / "compras_diario_final.parquet")
    g1 = pd.read_parquet(GOLD / "gold_t1.parquet")
    g2 = pd.read_parquet(GOLD / "gold_t2.parquet")
    preparo = preparo.merge(g1[["data", "sem_cardapio"]], on="data")
    compras = compras.merge(g2[["data", "sem_cardapio"]], on="data")
    for tabela in (preparo, compras):
        tabela["previsao"] = tabela["C1"].where(tabela["sem_cardapio"], tabela["C2"])
        tabela["base"] = np.where(tabela["sem_cardapio"], "Cenário 1: consumo e calendário", "Cenário 2: inclui o cardápio")
    t1, t2 = aplicar_margem_contexto(aplicar_folgas(preparo, "previsao"), g1), aplicar_margem_contexto(aplicar_folgas(compras, "previsao"), g2)
    verificacao = pd.concat([verificar(t1, "preparo", "plano"), verificar(t2, "compras")], ignore_index=True)
    situacao = pd.concat([verificar_por_situacao(t1, g1).assign(decisao="preparo"), verificar_por_situacao(t2, g2).assign(decisao="compras")], ignore_index=True)
    verificacao.to_csv(PASTA_PLANEJAMENTO / "verificacao_folgas_2026.csv", sep=";", index=False, encoding="utf-8-sig")
    prospectiva = pd.concat([verificar(t1, "preparo", "plano", INICIO_PROSPECTIVA, FIM_PROSPECTIVA), verificar(t2, "compras", None, INICIO_PROSPECTIVA, FIM_PROSPECTIVA)], ignore_index=True)
    prospectiva.to_csv(PASTA_PLANEJAMENTO / "verificacao_folgas_prospectiva.csv", sep=";", index=False, encoding="utf-8-sig")
    situacao.to_csv(PASTA_PLANEJAMENTO / "verificacao_por_situacao_2026.csv", sep=";", index=False, encoding="utf-8-sig")
    origem = t1["origem"].max()
    rodada = previsao_do_tempo_rodada(origem)
    classificados = preparar_classificados()
    fichas = montar_fichas(t1, g1, classificados, "preparo", origem, rodada) + montar_fichas(t2, g2, classificados, "compras", origem, rodada)
    (PASTA_PLANEJAMENTO / f"fichas_{origem.date()}.json").write_text(json.dumps(fichas, indent=2, ensure_ascii=False), encoding="utf-8")
    erros = {"preparo": erros_na_emissao(t1, origem), "compras": erros_na_emissao(t2, origem)}
    (PASTA_PLANEJAMENTO / f"erros_recentes_{origem.date()}.json").write_text(json.dumps(erros, ensure_ascii=False), encoding="utf-8")
    documento = escrever_documento(fichas, origem)
    (PASTA_PLANEJAMENTO / f"fichas_{origem.date()}.md").write_text(documento, encoding="utf-8")
    print(verificacao.round(1).to_string(index=False))
    print(documento)


MINIMO_CONTEXTO = 15


def situacao_do_dia(t):
    return np.select([t["fim_de_bloco"], t["inicio_de_bloco"] | t["pos_pausa"], t["vespera_de_pausa"]], ["fim", "inicio", "vespera"], "comum")


def margem_por_contexto(tabela, gold, coluna="previsao"):
    """Usa os erros de dias parecidos nos ultimos 12 meses; com poucos dias, usa a margem geral."""
    d = tabela.merge(gold[["data", "fim_de_bloco", "inicio_de_bloco", "pos_pausa", "vespera_de_pausa"]], on="data").sort_values("data")
    d["situacao"] = situacao_do_dia(d)
    conhecidos = d[d["regular"] & d["consumo"].notna() & d[coluna].notna()]
    saidas = []
    for origem, grupo in d.groupby("origem"):
        passado = conhecidos[(conhecidos["data"] < origem) & (conhecidos["data"] >= origem - pd.DateOffset(months=12))]
        geral = (passado["consumo"] - passado[coluna]).tail(JANELA_ERROS)
        if len(geral) < MINIMO_ERROS:
            continue
        g = grupo.copy()
        for nivel, quantil in (("moderada", 0.75), ("alta", 0.90)):
            valores = []
            for situacao in g["situacao"]:
                recorte = passado[passado["situacao"] == situacao]
                erros = (recorte["consumo"] - recorte[coluna]) if len(recorte) >= MINIMO_CONTEXTO else geral
                valores.append(max(float(erros.quantile(quantil)), 0.0))
            g[f"plano_contexto_{nivel}"] = np.ceil(g[coluna] + np.array(valores))
            g[f"plano_geral_{nivel}"] = np.ceil(g[coluna] + max(float(geral.quantile(quantil)), 0.0))
        saidas.append(g)
    return pd.concat(saidas)


def aplicar_margem_contexto(t, gold, coluna="previsao"):
    """Troca a margem geral pela margem de dias parecidos, quando ha dias suficientes."""
    d = t.merge(gold[["data", "fim_de_bloco", "inicio_de_bloco", "pos_pausa", "vespera_de_pausa"]], on="data", how="left").sort_values("data").reset_index(drop=True)
    d["situacao"] = situacao_do_dia(d.fillna({"fim_de_bloco": False, "inicio_de_bloco": False, "pos_pausa": False, "vespera_de_pausa": False}).infer_objects(copy=False))
    conhecidos = d[d["regular"] & d["consumo"].notna() & d[coluna].notna()]
    for origem, grupo in d.groupby("origem"):
        passado = conhecidos[(conhecidos["data"] < origem) & (conhecidos["data"] >= origem - pd.DateOffset(months=12))]
        for situacao, linhas in grupo.groupby("situacao"):
            recorte = passado[passado["situacao"] == situacao]
            if len(recorte) < MINIMO_CONTEXTO:
                continue
            erros = recorte["consumo"] - recorte[coluna]
            for nivel, quantil in NIVEIS.items():
                if quantil is not None:
                    d.loc[linhas.index, nivel] = max(float(erros.quantile(quantil)), 0.0)
                d.loc[linhas.index, f"chance_faltar_{nivel}"] = float((erros > d.loc[linhas.index, nivel].iloc[0]).mean())
    for nivel in NIVEIS:
        d[f"plano_{nivel}"] = np.ceil(d[coluna] + d[nivel])
    return d.drop(columns=["situacao", "fim_de_bloco", "inicio_de_bloco", "pos_pausa", "vespera_de_pausa"])


def resumir_margens(t, inicio, fim):
    q = t[t["data"].between(inicio, fim) & t["consumo"].notna() & t["regular"]]
    linhas = []
    for nome, prefixo in (("Margem geral", "plano_geral"), ("Margem por contexto", "plano_contexto")):
        for nivel in ("moderada", "alta"):
            e = q[f"{prefixo}_{nivel}"] - q["consumo"]
            linhas.append({"opcao": f"{nome}, {nivel}", "dias": len(q), "excesso": int(e.clip(lower=0).sum()), "deficit": int((-e).clip(lower=0).sum()), "dias_com_deficit_pct": round((e < 0).mean() * 100, 1), "erro_medio": round(e.abs().mean(), 1)})
    return pd.DataFrame(linhas)


def guardar_previsoes_da_semana():
    """Copia as fichas atuais para uma pasta datada, antes do servico. E a base da confirmacao real."""
    fichas = sorted(PASTA_PLANEJAMENTO.glob("fichas_*.json"))[-1]
    dados = json.loads(fichas.read_text(encoding="utf-8"))
    sexta = dados[0]["sexta_do_planejamento"]
    destino = RAIZ / "resultados" / "previsoes_guardadas" / sexta
    if destino.exists():
        print(f"Previsoes de {sexta} ja guardadas: nada foi alterado")
        return
    destino.mkdir(parents=True)
    shutil.copy(fichas, destino / fichas.name)
    (destino / "guardado_em.txt").write_text(pd.Timestamp.now().isoformat(timespec="seconds"), encoding="utf-8")
    print(f"Previsoes de {sexta} guardadas")


def avaliar_margem_por_contexto():
    """Compara a margem geral com a margem por contexto em 2025, ate agosto de 2026 e na prospectiva."""
    gold = pd.read_parquet(GOLD / "gold_t1.parquet")
    diario = pd.read_parquet(PASTA_MODELOS / "preparo_diario_final.parquet").merge(gold[["data", "sem_cardapio"]], on="data")
    diario["previsao"] = diario["C1"].where(diario["sem_cardapio"], diario["C2"])
    t = margem_por_contexto(diario, gold)
    q = t[t["data"].between(INICIO_PROVA, FIM_PROVA) & t["consumo"].notna() & t["regular"]]
    por_situacao = q.groupby("situacao").apply(lambda g: pd.Series({"dias": len(g), "deficit_geral_pct": (g["plano_geral_moderada"] < g["consumo"]).mean() * 100, "deficit_contexto_pct": (g["plano_contexto_moderada"] < g["consumo"]).mean() * 100}), include_groups=False).round(1)
    return resumir_margens(t, "2025-01-01", "2025-12-31"), resumir_margens(t, INICIO_PROVA, FIM_PROVA), resumir_margens(t, INICIO_PROSPECTIVA, FIM_PROSPECTIVA), por_situacao
