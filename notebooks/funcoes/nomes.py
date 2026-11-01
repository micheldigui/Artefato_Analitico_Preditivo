"""Nomes usados em tudo que a gestora e a nutricionista leem.

Os nomes tecnicos ficam no codigo e nas tabelas de dados. Tabelas, figuras e telas
devem passar pelas funcoes deste modulo, para manter uma so linguagem.
"""

JANELAS = {
    "t1": "Preparo (semana seguinte)",
    "t2": "Compras (daqui a duas semanas)",
}

CENARIOS = {
    "C1": "Cenário 1: consumo e calendário",
    "C2": "Cenário 2: inclui o cardápio",
    "C3": "Cenário 3: inclui a previsão do tempo",
    "C4": "Cenário 4: inclui o quantitativo da nutricionista",
}

HISTORICOS = {
    "tudo": "Todo o histórico anterior",
    "2_anos": "Últimos 2 anos",
    "1_ano": "Últimos 12 meses",
    "6_meses": "Últimos 6 meses",
    "ano_anterior_e_atual": "Ano anterior + ano atual até agora",
}

ALTERNATIVAS = {
    "plano": "Quantitativo da nutricionista",
    "consumo_tipico": "Consumo de referência",
    "H": "Só histórico de consumo",
    "C1": CENARIOS["C1"],
    "C2": CENARIOS["C2"],
    "C3": CENARIOS["C3"],
    "C4": CENARIOS["C4"],
}

EVENTOS = {
    "aula_regular": "Aula regular", "conselho_classe": "Conselho de classe", "encerramento_etapa": "Encerramento de etapa", "inicio_etapa": "Início de etapa",
    "atividade_cultural": "Atividade cultural", "semana_consciencia_negra": "Semana da Consciência Negra", "reuniao_familias": "Reunião de famílias",
    "semana_ciencia_tecnologia": "Semana de Ciência e Tecnologia", "prazo_administrativo": "Prazo administrativo", "formacao": "Formação", "reuniao_apoio_estudantil": "Reunião de apoio ao estudante",
    "encontro_institucional": "Encontro institucional", "evento_portas_abertas": "Evento de portas abertas", "matricula": "Matrícula", "acolhimento": "Acolhimento",
    "reuniao_representantes": "Reunião de representantes", "atividade_durante_greve": "Atividade durante a greve", "jogos_interclasse": "Jogos interclasse",
    "ponto_facultativo": "Ponto facultativo", "data_comemorativa": "Data comemorativa", "planejamento": "Planejamento", "prova_final": "Prova final", "sabado_letivo": "Sábado letivo",
}

MODELOS = {
    "regressao": "Regressão (Ridge)",
    "regressao_robusta": "Regressão resistente a dias extremos (Huber)",
    "arvores": "Árvores de decisão (HistGB)",
    "media_robusta_arvores": "Comitê de Huber e HistGB",
    "catboost": "CatBoost",
    "vizinhos": "KNN (dias parecidos)",
    "extratrees": "ExtraTrees",
    "comite_catboost_knn": "Comitê de CatBoost, KNN e consumo de referência",
    "campeao": "ExtraTrees",
}

FOLGAS = {"sem_folga": "Sem margem", "moderada": "Margem moderada", "alta": "Margem alta"}

UNIVERSOS = {"sem_atipicos": "Sem dias atípicos", "todos_os_dias": "Todos os dias"}

TRATAMENTOS = {
    "normal": "Dia normal",
    "previsivel_calendario": "Queda prevista pelo calendário",
    "alto_mantido": "Consumo acima da referência",
    "imprevisivel": "Dia atípico imprevisível",
    "periodo_de_greve": "Período de greve",
    "fora_do_calendario": "Consumo fora dos dias de aula",
    "sem_consumo": "Sem consumo registrado",
}

METRICAS = {
    "mae": "Erro médio (refeições por dia)",
    "rmse": "Erro médio com peso nos grandes desvios",
    "wape": "Erro em relação ao total consumido (%)",
    "r2": "Quanto da variação foi explicada",
    "vies": "Tendência (+ acima do consumo; - abaixo do consumo)",
    "excesso": "Excesso potencial (refeições)",
    "deficit": "Déficit potencial (refeições)",
    "dias_deficit": "Dias com déficit potencial",
    "pearson": "Relação linear com o consumo",
    "spearman": "Relação com o consumo (ordem dos valores)",
    "pearson_medio_por_ano": "Relação com o consumo, média por ano",
    "dias": "Dias analisados",
    "relativo_mediano": "Consumo comparado à referência",
}

ATRIBUTOS = {
    # Consumo recente
    "consumo": "Refeições consumidas",
    "alvo_relativo": "Consumo comparado à referência",
    "nivel_base": "Consumo de referência",
    "nivel_10d": "Consumo de referência dos últimos 10 dias",
    "nivel_20d": "Consumo de referência dos últimos 20 dias",
    "nivel_mesmo_dia": "Consumo de referência neste dia da semana",
    "nivel_exponencial": "Consumo recente, com mais peso nos últimos dias",
    "nivel_bloco_atual": "Consumo de referência neste período de aulas",
    "dias_observados_no_bloco": "Dias já registrados neste período de aulas",
    "ultimo_consumo": "Último consumo conhecido",
    "media_5d": "Média dos últimos 5 dias",
    "mediana_5d": "Valor central dos últimos 5 dias",
    "media_10d": "Média dos últimos 10 dias",
    "media_20d": "Média dos últimos 20 dias",
    "desvio_5d": "Variação nos últimos 5 dias",
    "desvio_10d": "Variação nos últimos 10 dias",
    "minimo_5d": "Menor consumo nos últimos 5 dias",
    "maximo_5d": "Maior consumo nos últimos 5 dias",
    "mesmo_dia_ha_1_ocorrencias": "Mesmo dia da semana, última vez conhecida",
    "mesmo_dia_ha_2_ocorrencias": "Mesmo dia da semana, duas vezes atrás",
    "mesmo_dia_ha_3_ocorrencias": "Mesmo dia da semana, três vezes atrás",
    "mesmo_dia_ha_4_ocorrencias": "Mesmo dia da semana, quatro vezes atrás",
    "tendencia_nivel": "Consumo subindo ou caindo",
    "dispersao_recente": "Instabilidade recente do consumo",
    "coef_variacao_10d": "Instabilidade dos últimos 10 dias",
    "idade_informacao_dias": "Dias desde o último consumo conhecido",
    "rel_ultimo_consumo": "Último consumo comparado à referência",
    "rel_media_5d": "Média de 5 dias comparada à referência",
    "rel_nivel_exponencial": "Consumo recente comparado à referência",
    "rel_nivel_bloco_atual": "Consumo do período de aulas comparado à referência",
    "rel_mesmo_dia_ha_1_ocorrencias": "Mesmo dia da semana comparado à referência",
    "rel_mesmo_dia_ha_2_ocorrencias": "Mesmo dia, duas vezes atrás, comparado à referência",
    # Calendario
    "dia_seg": "Segunda-feira",
    "dia_ter": "Terça-feira",
    "dia_qua": "Quarta-feira",
    "dia_qui": "Quinta-feira",
    "dia_sex": "Sexta-feira",
    "dia_semana": "Dia da semana",
    "dias_ate_fim_bloco": "Dias de aula até o fim do período",
    "dias_desde_inicio_bloco": "Dias de aula desde o início do período",
    "semana_do_bloco": "Semana dentro do período de aulas",
    "fim_de_bloco": "Últimos dias do período de aulas",
    "inicio_de_bloco": "Primeiros dias do período de aulas",
    "vespera_de_pausa": "Véspera de dia sem aula",
    "pos_pausa": "Volta depois de dia sem aula",
    "dias_sem_aula_antes": "Dias sem aula logo antes",
    "dias_letivos_na_semana": "Dias de aula na semana",
    "evento_tipo": "Evento do calendário",
    "evento_especial": "Dia com evento fora da rotina",
    "efeito_evento": "Efeito do evento no consumo",
    # Cardapio
    "proteina": "Tipo de proteína",
    "preparo": "Forma de preparo",
    "efeito_proteina": "Efeito da proteína no consumo",
    "efeito_preparo": "Efeito do preparo no consumo",
    "efeito_prato": "Efeito do prato no consumo",
    # Clima
    "previsao_chuva_mm": "Chuva prevista no dia (mm)",
    "previsao_chuva_manha_mm": "Chuva prevista de manhã (mm)",
    "previsao_chuva_acum3d_mm": "Chuva prevista em 3 dias (mm)",
    "previsao_temp_max": "Temperatura máxima prevista",
    "oraculo_chuva_mm": "Chuva que ocorreu no dia (mm)",
    "oraculo_chuva_horas": "Horas de chuva que ocorreram",
    "oraculo_temp_max": "Temperatura máxima que ocorreu",
    "oraculo_vento_max": "Vento máximo que ocorreu",
    # Planejamento humano
    "plano_nutricionista": "Quantitativo da nutricionista",
    "reserva_discente": "Reservas dos alunos",
    # Defasagens da analise exploratoria
    "lag_1_dias_servidos": "Consumo do dia anterior",
    "lag_2_dias_servidos": "Consumo de 2 dias atrás",
    "lag_3_dias_servidos": "Consumo de 3 dias atrás",
    "lag_5_dias_servidos": "Consumo de 5 dias de serviço atrás",
    "lag_10_dias_servidos": "Consumo de 10 dias de serviço atrás",
    "mesmo_dia_1_semana(s)_antes": "Mesmo dia, semana passada",
    "mesmo_dia_2_semana(s)_antes": "Mesmo dia, há 2 semanas",
    "mesmo_dia_3_semana(s)_antes": "Mesmo dia, há 3 semanas",
    "mesmo_dia_4_semana(s)_antes": "Mesmo dia, há 4 semanas",
    "media_movel_5d": "Média dos 5 dias anteriores",
    "media_movel_10d": "Média dos 10 dias anteriores",
    "media_movel_20d": "Média dos 20 dias anteriores",
    "mediana_movel_5d": "Valor central dos 5 dias anteriores",
    "mediana_movel_10d": "Valor central dos 10 dias anteriores",
    "mediana_movel_20d": "Valor central dos 20 dias anteriores",
    "desvio_padrao_5d": "Variação nos 5 dias anteriores",
    "desvio_padrao_10d": "Variação nos 10 dias anteriores",
}

PROTEINAS = {"frango": "Frango", "bovina": "Carne bovina", "suina": "Carne suína", "peixe": "Peixe", "feijoada": "Feijoada", "mista": "Proteína mista"}
PREPAROS = {
    "ao_molho_ou_refogado": "Ao molho ou refogado", "assado": "Assado", "churrasco_ou_espeto": "Churrasco ou espeto",
    "cozido_ou_refogado": "Cozido ou refogado", "cremoso_ou_ensopado": "Cremoso ou ensopado", "empanado_ou_a_milanesa": "Empanado ou à milanesa",
    "frito": "Frito", "grelhado": "Grelhado", "massa_ou_montado": "Massa ou prato montado",
}


def rotulo(nome):
    """Devolve o nome para a gestao; nomes desconhecidos voltam sem mudanca."""
    if nome in ATRIBUTOS:
        return ATRIBUTOS[nome]
    if str(nome).startswith("proteina_"):
        return "Cardápio: " + PROTEINAS.get(nome.removeprefix("proteina_"), nome)
    if str(nome).startswith("preparo_"):
        return "Cardápio: " + PREPAROS.get(nome.removeprefix("preparo_"), nome).lower()
    return METRICAS.get(nome, nome)
