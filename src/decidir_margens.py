"""Camada de decisao: opcoes de margem de segurança e contexto de cada dia.

Em cada sexta-feira, a margem de segurança vem dos erros recentes ja conhecidos naquele dia. O
artefato mostra as opcoes e os motivos; a escolha e da nutricionista e da gestora.
No preparo usa-se o cenario 2 quando ha cardapio e o cenario 1 quando nao ha. Nas
compras usa-se o cenario 1.

Execute depois de consolidar_resultados.py: python src/decidir_margens.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from modelagem import medir
from nomes import FOLGAS as NOMES_FOLGA

RAIZ = Path(__file__).resolve().parents[1]
GOLD = RAIZ / "dados" / "gold"
SILVER = RAIZ / "dados" / "silver"
BRONZE = RAIZ / "dados" / "bronze"
MODELOS = RAIZ / "resultados" / "modelos"
SAIDA = RAIZ / "resultados" / "planejamento"
INICIO_PROVA = "2026-01-01"
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


def verificar(t, nome, referencia=None):
    """Compara a protecao prometida com a observada em 2026."""
    prova = t[(t["data"] >= INICIO_PROVA) & t["consumo"].notna()]
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
    prova = t[(t["data"] >= INICIO_PROVA) & t["consumo"].notna()].merge(gold[["data", "fim_de_bloco", "inicio_de_bloco", "pos_pausa"]], on="data")
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


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    preparo = pd.read_parquet(MODELOS / "preparo_diario_final.parquet")
    compras = pd.read_parquet(MODELOS / "compras_diario_final.parquet")
    g1 = pd.read_parquet(GOLD / "gold_t1.parquet")
    g2 = pd.read_parquet(GOLD / "gold_t2.parquet")
    preparo = preparo.merge(g1[["data", "sem_cardapio"]], on="data")
    compras = compras.merge(g2[["data", "sem_cardapio"]], on="data")
    for tabela in (preparo, compras):
        tabela["previsao"] = tabela["C1"].where(tabela["sem_cardapio"], tabela["C2"])
        tabela["base"] = np.where(tabela["sem_cardapio"], "Cenário 1: consumo e calendário", "Cenário 2: inclui o cardápio")
    t1, t2 = aplicar_folgas(preparo, "previsao"), aplicar_folgas(compras, "previsao")
    verificacao = pd.concat([verificar(t1, "preparo", "plano"), verificar(t2, "compras")], ignore_index=True)
    situacao = pd.concat([verificar_por_situacao(t1, g1).assign(decisao="preparo"), verificar_por_situacao(t2, g2).assign(decisao="compras")], ignore_index=True)
    verificacao.to_csv(SAIDA / "verificacao_folgas_2026.csv", sep=";", index=False, encoding="utf-8-sig")
    situacao.to_csv(SAIDA / "verificacao_por_situacao_2026.csv", sep=";", index=False, encoding="utf-8-sig")
    origem = t1["origem"].max()
    rodada = previsao_do_tempo_rodada(origem)
    classificados = preparar_classificados()
    fichas = montar_fichas(t1, g1, classificados, "preparo", origem, rodada) + montar_fichas(t2, g2, classificados, "compras", origem, rodada)
    (SAIDA / f"fichas_{origem.date()}.json").write_text(json.dumps(fichas, indent=2, ensure_ascii=False), encoding="utf-8")
    erros = {"preparo": erros_na_emissao(t1, origem), "compras": erros_na_emissao(t2, origem)}
    (SAIDA / f"erros_recentes_{origem.date()}.json").write_text(json.dumps(erros, ensure_ascii=False), encoding="utf-8")
    documento = escrever_documento(fichas, origem)
    (SAIDA / f"fichas_{origem.date()}.md").write_text(documento, encoding="utf-8")
    print(verificacao.round(1).to_string(index=False))
    print(documento)


if __name__ == "__main__":
    main()
