"""Camada Gold: uma linha por sexta-feira de emissao e data-alvo.

Cada linha contem apenas o que se sabia na emissao. O consumo da propria sexta
nao entra, porque a emissao ocorre antes do fechamento do registro do dia.

- gold_t1: semana seguinte (preparo), antecedencia de 3 a 7 dias;
- gold_t2: semana posterior (compras), antecedencia de 10 a 14 dias.

Dias baixos saem do calculo do nivel recente, para que uma queda passageira nao
rebaixe as previsoes seguintes. O modelo aprende o efeito do calendario sobre esse
nivel. Clima realizado fica com prefixo oraculo_ porque nao era conhecido na emissao;
as colunas previsao_ trazem a previsao de tempo disponivel naquela sexta-feira.

Execute depois de detectar_atipicos.py: python src/processar_gold.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from detectar_atipicos import EVENTOS_ROTINEIROS, RAZAO_BAIXA, classificar_dias, detectar

RAIZ = Path(__file__).resolve().parents[1]
SILVER = RAIZ / "dados" / "silver"
GOLD = RAIZ / "dados" / "gold"
HORIZONTES = {"t1": 3, "t2": 10}
JANELA_NIVEL = 10
JANELA_LONGA = 20
JANELA_MESMO_DIA = 4
OCORRENCIAS_MESMO_DIA = 4
ALFA_EXPONENCIAL = 0.2
JANELA_RECENTE = 20
MIN_NORMAIS = 5
MIN_HISTORICO = 10
PESO_PRIOR_CARDAPIO = 5
LIMITES_RAZAO = (0.3, 1.5)
JANELA_EPOCA_DIAS = 7
MIN_DIAS_EPOCA = 4
LIMITE_DISTANCIA_BLOCO = 15
ULTIMA_EMISSAO = "2026-10-30"


ATRIBUTOS_DA_EMISSAO = (
    "nivel_10d", "nivel_20d", "nivel_exponencial", "ultimo_consumo", "media_5d", "mediana_5d", "media_10d", "media_20d",
    "desvio_5d", "desvio_10d", "minimo_5d", "maximo_5d", "dispersao_recente", "ultimo_consumo_usado",
)
RELATIVOS_AO_NIVEL = ("ultimo_consumo", "media_5d", "nivel_exponencial", "nivel_bloco_atual", "mesmo_dia_ha_1_ocorrencias", "mesmo_dia_ha_2_ocorrencias")


def origem_da_janela(datas, deslocamento):
    """Sexta de emissao: `deslocamento` dias antes da segunda-feira da semana-alvo."""
    return datas - pd.to_timedelta(datas.dt.dayofweek + deslocamento, unit="D")


def resumir_historico(historico):
    """Nivel recente e efeito do cardapio com os consumos anteriores a uma emissao."""
    if len(historico) < MIN_HISTORICO:
        return None
    recentes = historico.tail(JANELA_RECENTE)
    normais = recentes[recentes["razao_referencia"].fillna(1) >= RAZAO_BAIXA]
    if len(normais) < MIN_NORMAIS:
        normais = recentes
    valores = normais["consumo"]
    resumo = {
        "nivel_10d": valores.tail(JANELA_NIVEL).median(),
        "nivel_20d": valores.tail(JANELA_LONGA).median(),
        "nivel_exponencial": valores.ewm(alpha=ALFA_EXPONENCIAL).mean().iloc[-1],
        "ultimo_consumo": valores.iloc[-1],
        "media_5d": valores.tail(5).mean(),
        "mediana_5d": valores.tail(5).median(),
        "media_10d": valores.tail(10).mean(),
        "media_20d": valores.tail(20).mean(),
        "desvio_5d": valores.tail(5).std(),
        "desvio_10d": valores.tail(10).std(),
        "minimo_5d": valores.tail(5).min(),
        "maximo_5d": valores.tail(5).max(),
        "dispersao_recente": (valores.tail(JANELA_NIVEL) - valores.tail(JANELA_NIVEL).median()).abs().median(),
        "ultimo_consumo_usado": historico["data"].iloc[-1],
    }
    semanais = historico[historico["razao_referencia"].fillna(1) >= RAZAO_BAIXA].groupby("dia_semana")["consumo"]
    resumo["mesmo_dia"] = semanais.apply(lambda s: s.tail(JANELA_MESMO_DIA).median()).to_dict()
    resumo["ocorrencias"] = semanais.apply(lambda s: s.tail(OCORRENCIAS_MESMO_DIA).tolist()[::-1]).to_dict()
    normais_bloco = historico[historico["razao_referencia"].fillna(1) >= RAZAO_BAIXA].groupby("bloco")["consumo"]
    resumo["nivel_bloco"] = normais_bloco.median().to_dict()
    resumo["dias_bloco"] = normais_bloco.size().to_dict()
    base = historico[historico["razao_referencia"].notna() & (historico["razao_referencia"] >= RAZAO_BAIXA)]
    for coluna in ("proteina", "preparo", "prato_padronizado"):
        resumo[coluna] = efeito_suavizado(base, coluna)
    # Eventos mantem os dias baixos, porque a queda e justamente o efeito medido.
    com_razao = historico[historico["razao_referencia"].notna()]
    resumo["evento_tipo"] = efeito_suavizado(com_razao, "evento_tipo")
    # Usa todos os anos, para o efeito nao depender de quantos fins de periodo cabem na janela de treino.
    limitada = com_razao.assign(razao_referencia=com_razao["razao_referencia"].clip(*LIMITES_RAZAO), faixa_bloco=faixa_do_bloco(com_razao))
    resumo["faixa_bloco"] = efeito_suavizado(limitada, "faixa_bloco")
    resumo["serie_razao"] = com_razao.set_index("data")["razao_referencia"]
    return resumo


def faixa_do_bloco(t):
    return np.select([t["dias_ate_fim_bloco"] <= 3, t["dias_ate_fim_bloco"] <= 7, t["dias_desde_inicio_bloco"] <= 4], ["ultimos_dias", "antes_do_fim", "primeiros_dias"], "meio")


def mesma_epoca_ano_passado(datas, serie):
    valores = []
    for data in datas:
        centro = data - pd.Timedelta(days=364)
        recorte = serie[centro - pd.Timedelta(days=JANELA_EPOCA_DIAS):centro + pd.Timedelta(days=JANELA_EPOCA_DIAS)]
        valores.append(recorte.median() if len(recorte) >= MIN_DIAS_EPOCA else float("nan"))
    return valores


def efeito_suavizado(tabela, coluna):
    grupo = tabela.groupby(coluna)["razao_referencia"].agg(["sum", "count"])
    return ((grupo["sum"] + PESO_PRIOR_CARDAPIO) / (grupo["count"] + PESO_PRIOR_CARDAPIO)).to_dict()


def preparar_alvos(fato):
    """Dias de segunda a sexta previstos pelo calendario, com ou sem consumo."""
    alvos = fato[fato["no_escopo"] & fato["status_legal"].eq("letivo")].copy()
    util = fato[fato["no_escopo"]].copy()
    letivo = util["status_legal"].eq("letivo")
    util["vespera_de_pausa"] = letivo & ~letivo.shift(-1, fill_value=True)
    util["pos_pausa"] = letivo & ~letivo.shift(1, fill_value=True)
    util["dias_letivos_na_semana"] = letivo.groupby(util["data"].dt.to_period("W")).transform("sum")
    sem_aula = (~letivo).astype(int)
    util["dias_sem_aula_antes"] = sem_aula.groupby(letivo.cumsum()).cumsum().shift(1).fillna(0)
    return alvos.merge(util[["data", "vespera_de_pausa", "pos_pausa", "dias_letivos_na_semana", "dias_sem_aula_antes"]], on="data")


def montar_janela(alvos, servidos, deslocamento):
    alvos = alvos.copy()
    alvos["origem"] = origem_da_janela(alvos["data"], deslocamento)
    alvos = alvos[alvos["origem"] <= ULTIMA_EMISSAO]
    linhas = []
    for origem, grupo in alvos.groupby("origem"):
        resumo = resumir_historico(servidos[servidos["data"] < origem])
        if resumo is None:
            continue
        g = grupo.copy()
        for chave in ATRIBUTOS_DA_EMISSAO:
            g[chave] = resumo[chave]
        g["nivel_mesmo_dia"] = g["dia_semana"].map(resumo["mesmo_dia"]).fillna(resumo["nivel_10d"])
        for k in range(OCORRENCIAS_MESMO_DIA):
            g[f"mesmo_dia_ha_{k + 1}_ocorrencias"] = [
                resumo["ocorrencias"].get(d, [])[k] if len(resumo["ocorrencias"].get(d, [])) > k else float("nan") for d in g["dia_semana"]]
        g["nivel_bloco_atual"] = g["bloco"].map(resumo["nivel_bloco"])
        g["dias_observados_no_bloco"] = g["bloco"].map(resumo["dias_bloco"]).fillna(0)
        g["efeito_proteina"] = g["proteina"].map(resumo["proteina"])
        g["efeito_preparo"] = g["preparo"].map(resumo["preparo"])
        g["efeito_prato"] = g["prato_padronizado"].map(resumo["prato_padronizado"])
        g["efeito_evento"] = g["evento_tipo"].map(resumo["evento_tipo"])
        g["efeito_posicao_bloco"] = pd.Series(faixa_do_bloco(g), index=g.index).map(resumo["faixa_bloco"])
        g["mesma_epoca_ano_passado"] = mesma_epoca_ano_passado(g["data"], resumo["serie_razao"])
        linhas.append(g)
    return pd.concat(linhas, ignore_index=True)


def adicionar_atributos(gold):
    gold["antecedencia_dias"] = (gold["data"] - gold["origem"]).dt.days
    gold["idade_informacao_dias"] = (gold["origem"] - gold["ultimo_consumo_usado"]).dt.days
    gold["nivel_base"] = (gold["nivel_10d"] + gold["nivel_mesmo_dia"]) / 2
    gold["tendencia_nivel"] = gold["nivel_10d"] / gold["nivel_20d"]
    gold["coef_variacao_10d"] = gold["desvio_10d"] / gold["nivel_base"]
    for coluna in RELATIVOS_AO_NIVEL:
        gold[f"rel_{coluna}"] = gold[coluna] / gold["nivel_base"]
    gold["semana_do_bloco"] = gold["dias_desde_inicio_bloco"] // 5
    gold["evento_especial"] = ~gold["evento_tipo"].isin(EVENTOS_ROTINEIROS)
    gold["fim_de_bloco"] = gold["dias_ate_fim_bloco"].le(3)
    gold["inicio_de_bloco"] = gold["dias_desde_inicio_bloco"].le(4)
    for coluna in ("dias_ate_fim_bloco", "dias_desde_inicio_bloco"):
        gold[coluna] = gold[coluna].clip(upper=LIMITE_DISTANCIA_BLOCO)
    for dia, nome in enumerate(["seg", "ter", "qua", "qui", "sex"]):
        gold[f"dia_{nome}"] = gold["dia_semana"].eq(dia)
    gold["sem_cardapio"] = gold["proteina"].isna()
    gold["alvo_relativo"] = gold["consumo"] / gold["nivel_base"]
    return gold.rename(columns={"chuva_mm": "oraculo_chuva_mm", "chuva_horas": "oraculo_chuva_horas",
                                "temp_max": "oraculo_temp_max", "vento_max": "oraculo_vento_max"})


def construir_janela(fato, servidos, dias, nome):
    gold = adicionar_atributos(montar_janela(preparar_alvos(fato), servidos, HORIZONTES[nome]))
    gold = gold.merge(dias[["data", "tratamento", "usar_no_treino", "tipo_sugerido"]], on="data", how="left")
    gold["usar_no_treino"] = gold["usar_no_treino"].astype("boolean").fillna(False).astype(bool)
    gold["tratamento"] = gold["tratamento"].fillna("sem_consumo")
    gold["horizonte"] = nome
    previsto = pd.read_parquet(SILVER / "silver_clima_previsto.parquet")
    gold = gold.merge(previsto[previsto["horizonte"] == nome].drop(columns="horizonte"), on=["origem", "data"], how="left")
    colunas = [
        "origem", "data", "horizonte", "antecedencia_dias", "ano", "mes", "consumo", "alvo_relativo", "tratamento", "usar_no_treino", "tipo_sugerido",
        "nivel_base", "nivel_10d", "nivel_20d", "nivel_mesmo_dia", "nivel_exponencial", "nivel_bloco_atual", "dias_observados_no_bloco",
        "ultimo_consumo", "media_5d", "mediana_5d", "media_10d", "media_20d", "desvio_5d", "desvio_10d", "minimo_5d", "maximo_5d",
        "mesmo_dia_ha_1_ocorrencias", "mesmo_dia_ha_2_ocorrencias", "mesmo_dia_ha_3_ocorrencias", "mesmo_dia_ha_4_ocorrencias",
        "tendencia_nivel", "dispersao_recente", "coef_variacao_10d", *[f"rel_{c}" for c in RELATIVOS_AO_NIVEL],
        "idade_informacao_dias", "ultimo_consumo_usado",
        "dia_semana", "dia_seg", "dia_ter", "dia_qua", "dia_qui", "dia_sex", "dias_ate_fim_bloco", "dias_desde_inicio_bloco", "semana_do_bloco", "fim_de_bloco", "inicio_de_bloco",
        "vespera_de_pausa", "pos_pausa", "dias_sem_aula_antes", "dias_letivos_na_semana", "evento_tipo", "evento_especial", "efeito_evento", "efeito_posicao_bloco", "mesma_epoca_ano_passado",
        "proteina", "preparo", "prato_padronizado", "sem_cardapio", "efeito_proteina", "efeito_preparo", "efeito_prato",
        "previsao_chuva_mm", "previsao_chuva_manha_mm", "previsao_chuva_acum3d_mm", "previsao_temp_max", "oraculo_chuva_mm", "oraculo_chuva_horas", "oraculo_temp_max", "oraculo_vento_max",
    ]
    if nome == "t1":
        colunas += ["reserva_discente", "plano_nutricionista"]
    return gold[colunas].sort_values(["data"]).reset_index(drop=True)


def construir_gold(fato):
    servidos, candidatos = detectar(fato)
    dias = classificar_dias(servidos, candidatos)
    return {nome: construir_janela(fato, servidos, dias, nome) for nome in HORIZONTES}


DICIONARIO = [
    ("chave", "origem", "t1 e t2", "Sexta-feira de emissao"),
    ("chave", "data", "t1 e t2", "Dia previsto"),
    ("alvo", "consumo", "resultado", "Refeicoes registradas; vazio em datas futuras ou sem registro"),
    ("alvo", "alvo_relativo", "resultado", "consumo / nivel_base"),
    ("controle", "tratamento", "resultado", "normal, previsivel_calendario, alto_mantido, imprevisivel, periodo_de_greve ou sem_consumo"),
    ("controle", "usar_no_treino", "resultado", "Falso para dias imprevisiveis, greve, fora do calendario e sem consumo"),
    ("nivel", "nivel_base", "t1 e t2", "Media entre nivel_10d e nivel_mesmo_dia"),
    ("nivel", "nivel_10d", "t1 e t2", "Mediana dos 10 dias normais anteriores a emissao"),
    ("nivel", "nivel_20d", "t1 e t2", "Mediana dos 20 dias normais anteriores a emissao"),
    ("nivel", "nivel_mesmo_dia", "t1 e t2", "Mediana das 4 ultimas ocorrencias normais do mesmo dia da semana"),
    ("nivel", "tendencia_nivel", "t1 e t2", "nivel_10d / nivel_20d"),
    ("nivel", "nivel_exponencial", "t1 e t2", "Media exponencial dos dias normais recentes, com alfa 0,2"),
    ("nivel", "nivel_bloco_atual, dias_observados_no_bloco", "t1 e t2", "Mediana e quantidade de dias normais ja observados no bloco do dia previsto"),
    ("nivel", "ultimo_consumo", "t1 e t2", "Ultimo consumo normal conhecido na emissao"),
    ("nivel", "media_5d, mediana_5d, media_10d, media_20d", "t1 e t2", "Medias e mediana moveis dos dias normais ate a emissao"),
    ("nivel", "desvio_5d, desvio_10d, minimo_5d, maximo_5d", "t1 e t2", "Dispersao e extremos dos dias normais ate a emissao"),
    ("nivel", "mesmo_dia_ha_1_ocorrencias a mesmo_dia_ha_4_ocorrencias", "t1 e t2", "Consumo normal nas ultimas ocorrencias conhecidas do mesmo dia da semana"),
    ("nivel", "rel_*", "t1 e t2", "Atributo de nivel dividido por nivel_base"),
    ("nivel", "coef_variacao_10d", "t1 e t2", "desvio_10d / nivel_base"),
    ("nivel", "dispersao_recente", "t1 e t2", "Desvio absoluto mediano nos 10 dias normais"),
    ("nivel", "idade_informacao_dias", "t1 e t2", "Dias entre o ultimo consumo usado e a emissao; alto apos recessos"),
    ("calendario", "dia_seg a dia_sex", "t1 e t2", "Dia da semana"),
    ("calendario", "dias_ate_fim_bloco", "t1 e t2", "Dias letivos ate o fim do bloco, limitado a 15"),
    ("calendario", "dias_desde_inicio_bloco", "t1 e t2", "Dias letivos desde o inicio do bloco, limitado a 15"),
    ("calendario", "fim_de_bloco, inicio_de_bloco", "t1 e t2", "Ate 3 dias do fim; ate 4 dias do inicio"),
    ("calendario", "vespera_de_pausa, pos_pausa", "t1 e t2", "Dia util seguinte ou anterior sem aula"),
    ("calendario", "dias_letivos_na_semana", "t1 e t2", "Dias letivos na semana-alvo"),
    ("calendario", "dias_sem_aula_antes", "t1 e t2", "Dias uteis seguidos sem aula imediatamente antes do dia previsto"),
    ("calendario", "semana_do_bloco", "t1 e t2", "Semana do dia previsto dentro do bloco letivo"),
    ("calendario", "evento_tipo, evento_especial", "t1 e t2", "Evento do calendario e indicador de evento fora da rotina"),
    ("calendario", "efeito_evento", "t1 e t2", "Razao media de consumo do tipo de evento ate a emissao, suavizada"),
    ("calendario", "efeito_posicao_bloco", "t1 e t2", "Razao media de consumo na mesma posicao do periodo de aulas, em todos os anos anteriores a emissao"),
    ("calendario", "mesma_epoca_ano_passado", "t1 e t2", "Razao mediana de consumo em torno da mesma data do ano anterior"),
    ("cardapio", "proteina, preparo, prato_padronizado, sem_cardapio", "t1; t2 so se o cardapio ja estiver publicado", "Categorias e nome padronizado do prato"),
    ("cardapio", "efeito_proteina, efeito_preparo, efeito_prato", "t1; t2 so se o cardapio ja estiver publicado", "Razao media de consumo ate a emissao, suavizada"),
    ("plano", "plano_nutricionista", "somente t1, 2026", "Quantitativo enviado na sexta-feira a tarde"),
    ("plano", "reserva_discente", "somente t1, ate 2025", "Reservas; momento de fechamento nao registrado"),
    ("clima previsto", "previsao_chuva_mm, previsao_chuva_manha_mm, previsao_chuva_acum3d_mm, previsao_temp_max", "t1 desde 2024; t2 desde a emissao de 03/04/2026", "Chuva do dia, das 6h as 13h e acumulada em 3 dias, e temperatura maxima previstas na emissao; Previous Runs em t1 e media GFS/ECMWF em t2"),
    ("oraculo", "oraculo_chuva_mm, oraculo_chuva_horas, oraculo_temp_max, oraculo_vento_max", "nunca na emissao", "Clima realizado; uso restrito a analise de sensibilidade"),
]


def main():
    GOLD.mkdir(parents=True, exist_ok=True)
    fato = pd.read_parquet(SILVER / "silver_fato_diaria.parquet")
    pd.DataFrame(DICIONARIO, columns=["grupo", "coluna", "disponivel_em", "descricao"]).to_csv(
        GOLD / "gold_dicionario_atributos.csv", sep=";", index=False, encoding="utf-8-sig")
    for nome, tabela in construir_gold(fato).items():
        tabela.to_parquet(GOLD / f"gold_{nome}.parquet", index=False)
        tabela.to_csv(GOLD / f"gold_{nome}.csv", sep=";", index=False, encoding="utf-8-sig")
        print(f"gold_{nome}: {tabela.shape}; {tabela['origem'].min().date()} a {tabela['origem'].max().date()}; "
              f"com consumo: {tabela['consumo'].notna().sum()}; no treino: {tabela['usar_no_treino'].sum()}")


if __name__ == "__main__":
    main()
