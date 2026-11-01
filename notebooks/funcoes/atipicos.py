"""Dias atipicos: referencia com o passado, candidatos e tratamento de cada dia."""
import pandas as pd

from .configuracao import SILVER


RAZAO_BAIXA = 0.60


RAZAO_ALTA = 1.40


JANELA_GERAL = 10


JANELA_MESMO_DIA = 4


JANELA_RECENTE = 20


MIN_HISTORICO = 8


MIN_NORMAIS = 5


DIAS_FIM_BLOCO = 4


DIAS_INICIO_BLOCO = 4


TAMANHO_SEQUENCIA = 3


CHUVA_FORTE_MM = 15.0


# Eventos em que o consumo mediano fica proximo do normal do bloco.
EVENTOS_ROTINEIROS = {"aula_regular", "encontro_institucional", "reuniao_apoio_estudantil", "prazo_administrativo", "formacao", "evento_portas_abertas"}


TIPOS_IMPREVISIVEIS = {"evento_externo_registrado", "isolado_sem_explicacao", "sequencia_de_dias"}


TRATAMENTOS_NO_TREINO = {"normal", "alto_mantido", "previsivel_calendario"}


def calcular_referencia(servidos):
    """Percorre as datas em ordem; dias ja marcados como baixos saem da janela.

    Sem isso, a queda de fim de bloco rebaixa a referencia e dias normais do bloco
    seguinte parecem picos. Se a janela recente tiver menos de MIN_NORMAIS dias
    normais, todos entram, para a referencia acompanhar uma queda real de nivel.
    """
    referencia = []
    baixo = pd.Series(False, index=servidos.index)
    for i, linha in enumerate(servidos.itertuples()):
        anteriores = servidos.iloc[:i]
        if len(anteriores) < MIN_HISTORICO:
            referencia.append(float("nan"))
            continue
        recentes = anteriores.tail(JANELA_RECENTE)
        normais = recentes[~baixo.loc[recentes.index]]
        if len(normais) < MIN_NORMAIS:
            normais = recentes
        geral = normais["consumo"].tail(JANELA_GERAL).median()
        mesmo_dia = normais.loc[normais["dia_semana"] == linha.dia_semana, "consumo"].tail(JANELA_MESMO_DIA).median()
        valor = (geral + mesmo_dia) / 2 if pd.notna(mesmo_dia) else geral
        referencia.append(valor)
        baixo.iloc[i] = linha.consumo / valor < RAZAO_BAIXA
    return pd.Series(referencia, index=servidos.index)


def sequencia_de_candidatos(servidos):
    """Tamanho do trecho consecutivo de dias com consumo na mesma direcao."""
    trecho = servidos["direcao"].ne(servidos["direcao"].shift()).cumsum()
    return servidos.groupby(trecho)["direcao"].transform("size").where(servidos["direcao"].ne(""), 0)


def sugerir_tipo(d):
    if d["status_consumo"] == "atipico":
        return "evento_externo_registrado"
    if d["status_legal"] != "letivo":
        return "nao_letivo_com_consumo"
    if d["periodo_greve"]:
        return "periodo_de_greve"
    if d["direcao"] == "baixo" and pd.notna(d["dias_ate_fim_bloco"]) and d["dias_ate_fim_bloco"] <= DIAS_FIM_BLOCO:
        return "fim_de_bloco_letivo"
    if pd.notna(d["dias_desde_inicio_bloco"]) and d["dias_desde_inicio_bloco"] <= DIAS_INICIO_BLOCO:
        return "inicio_de_bloco_letivo"
    if d["evento_tipo"] not in EVENTOS_ROTINEIROS:
        return "evento_de_calendario"
    if d["tamanho_sequencia"] >= TAMANHO_SEQUENCIA:
        return "sequencia_de_dias"
    return "isolado_sem_explicacao"


def detectar(fato):
    servidos = fato[fato["no_escopo"] & fato["tem_consumo"]].copy()
    servidos["referencia"] = calcular_referencia(servidos).round(1)
    servidos["razao_referencia"] = (servidos["consumo"] / servidos["referencia"]).round(3)
    servidos["direcao"] = ""
    servidos.loc[servidos["razao_referencia"] < RAZAO_BAIXA, "direcao"] = "baixo"
    servidos.loc[servidos["razao_referencia"] > RAZAO_ALTA, "direcao"] = "alto"
    servidos["candidato"] = servidos["direcao"].ne("") | servidos["status_consumo"].eq("atipico")
    servidos.loc[servidos["status_consumo"].eq("atipico") & servidos["direcao"].eq(""), "direcao"] = "baixo"
    servidos["tamanho_sequencia"] = sequencia_de_candidatos(servidos)
    servidos["chuva_mm_dia_anterior"] = fato.set_index("data")["chuva_mm"].shift(1).reindex(servidos["data"]).values
    servidos["chuva_forte"] = (servidos["chuva_mm"] >= CHUVA_FORTE_MM) | (servidos["chuva_mm_dia_anterior"] >= CHUVA_FORTE_MM)
    candidatos = servidos[servidos["candidato"]].copy()
    candidatos["tipo_sugerido"] = candidatos.apply(sugerir_tipo, axis=1)
    candidatos["plano"] = candidatos["reserva_discente"].fillna(candidatos["plano_nutricionista"])
    return servidos, candidatos


def definir_tratamento(d):
    if d["status_legal"] != "letivo":
        return "fora_do_calendario"
    if d["periodo_greve"]:
        return "periodo_de_greve"
    if d["direcao"] == "alto":
        return "alto_mantido"
    if d["direcao"] != "baixo":
        return "normal"
    return "imprevisivel" if d["tipo_sugerido"] in TIPOS_IMPREVISIVEIS else "previsivel_calendario"


def classificar_dias(servidos, candidatos):
    dias = servidos.merge(candidatos[["data", "tipo_sugerido"]], on="data", how="left")
    dias["tratamento"] = dias.apply(definir_tratamento, axis=1)
    dias["usar_no_treino"] = dias["tratamento"].isin(TRATAMENTOS_NO_TREINO)
    return dias[["data", "consumo", "referencia", "razao_referencia", "direcao", "tipo_sugerido", "tratamento", "usar_no_treino"]]


def dias_sem_consumo(fato):
    """Dias letivos com cardapio ou plano e sem consumo: lacuna de registro ou dia sem servico."""
    sem = fato[fato["no_escopo"] & fato["status_legal"].eq("letivo") & ~fato["tem_consumo"] & (fato["data"] <= "2026-10-30")].copy()
    sem = sem[sem["tem_cardapio"] | sem["reserva_discente"].notna() | sem["plano_nutricionista"].notna()]
    return sem[["data", "dia_nome", "evento_tipo", "periodo_greve", "reserva_discente", "plano_nutricionista", "prato", "chuva_mm"]]


def salvar_dias_classificados():
    fato = pd.read_parquet(SILVER / "silver_fato_diaria.parquet")
    servidos, candidatos = detectar(fato)
    colunas = ["data", "dia_nome", "evento_tipo", "consumo", "referencia", "razao_referencia", "direcao", "tipo_sugerido",
               "tamanho_sequencia", "dias_desde_inicio_bloco", "dias_ate_fim_bloco", "plano", "chuva_mm",
               "chuva_mm_dia_anterior", "chuva_forte", "prato", "obs_consumo"]
    tabela = candidatos[colunas].sort_values("data")
    tabela.to_parquet(SILVER / "silver_candidatos_atipicos.parquet", index=False)
    tabela.to_csv(SILVER / "silver_candidatos_atipicos.csv", sep=";", index=False, encoding="utf-8-sig")
    servidos[["data", "referencia", "razao_referencia"]].to_parquet(SILVER / "silver_referencia_consumo.parquet", index=False)
    dias = classificar_dias(servidos, candidatos)
    dias.to_parquet(SILVER / "silver_dias_classificados.parquet", index=False)
    dias.to_csv(SILVER / "silver_dias_classificados.csv", sep=";", index=False, encoding="utf-8-sig")
    dias_sem_consumo(fato).to_csv(SILVER / "silver_letivos_sem_consumo.csv", sep=";", index=False, encoding="utf-8-sig")
    print(dias.groupby(["tratamento", "usar_no_treino"]).size().to_string())
