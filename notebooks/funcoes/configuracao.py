"""Caminhos e regras gerais do projeto. Mude aqui e todos os notebooks acompanham."""
import hashlib
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
BRONZE = RAIZ / "dados" / "bronze"
SILVER = RAIZ / "dados" / "silver"
GOLD = RAIZ / "dados" / "gold"
RESULTADOS = RAIZ / "resultados"
PASTA_MODELOS = RESULTADOS / "modelos"
PASTA_COMPARACAO = RESULTADOS / "comparacao"
PASTA_PLANEJAMENTO = RESULTADOS / "planejamento"
PASTA_REVISAO = RESULTADOS / "revisao"
PASTA_DISSERTACAO = RESULTADOS / "dissertacao"
FIGURAS = PASTA_DISSERTACAO / "figuras"
TABELAS = PASTA_DISSERTACAO / "tabelas"
PASTA_LOCAL = RAIZ / "resultados_locais"
FIM_ESCOLHA = "2025-12-31"
GANHO_MINIMO = 0.5

HISTORICO = ["tendencia_nivel", "coef_variacao_10d", "idade_informacao_dias", "dias_observados_no_bloco", "rel_ultimo_consumo", "rel_media_5d",
             "rel_nivel_bloco_atual", "rel_mesmo_dia_ha_1_ocorrencias", "rel_mesmo_dia_ha_2_ocorrencias"]
CALENDARIO = ["dia_seg", "dia_ter", "dia_qua", "dia_qui", "dia_sex", "dias_ate_fim_bloco", "dias_desde_inicio_bloco", "semana_do_bloco", "fim_de_bloco",
              "inicio_de_bloco", "vespera_de_pausa", "pos_pausa", "dias_sem_aula_antes", "dias_letivos_na_semana", "evento_especial", "efeito_evento"]
CARDAPIO = ["efeito_proteina", "efeito_preparo", "efeito_prato", "sem_cardapio"]
CLIMA_PREVISTO = ["previsao_chuva_mm", "previsao_chuva_manha_mm", "previsao_chuva_acum3d_mm", "previsao_temp_max"]
CLIMA_OCORRIDO = ["oraculo_chuva_mm", "oraculo_chuva_horas", "oraculo_temp_max", "oraculo_vento_max"]
SAZONALIDADE = ["efeito_posicao_bloco", "mesma_epoca_ano_passado"]
CENARIOS = {
    "C1s": HISTORICO + CALENDARIO + SAZONALIDADE,
    "C2s": HISTORICO + CALENDARIO + CARDAPIO + SAZONALIDADE,
    "H": HISTORICO,
    "C1": HISTORICO + CALENDARIO,
    "C2": HISTORICO + CALENDARIO + CARDAPIO,
    "C3": HISTORICO + CALENDARIO + CARDAPIO + CLIMA_PREVISTO,
    "S_clima_perfeito": HISTORICO + CALENDARIO + CARDAPIO + CLIMA_OCORRIDO,
}
CENARIOS_POR_JANELA = {"t1": ["C1", "C2", "C3", "S_clima_perfeito"], "t2": ["C1", "C2", "S_clima_perfeito"]}
TRATAMENTOS_REGULARES = ("normal", "previsivel_calendario", "alto_mantido")
INICIO_AVALIACAO = "2025-01-01"
INICIO_PROVA = "2026-01-01"
FIM_PROVA = "2026-08-31"
INICIO_PROSPECTIVA = "2026-09-01"
FIM_PROSPECTIVA = "2026-10-31"
LIMITES_ALVO = (0.2, 2.0)
ANO_ANTERIOR_E_ATUAL = "ano_anterior_e_atual"
# Versao congelada: ExtraTrees, escolhido com 2024, 2025 e fevereiro a agosto de 2026.
CAMPEAO = "campeao"
CENARIO_CAMPEAO = "C2"
HISTORICO_CAMPEAO = "1_ano"
MODELOS_BASE = ["regressao", "regressao_robusta", "arvores", "catboost", "vizinhos", "extratrees"]


VERSAO_CONGELADA = {
    "congelada_em": "2026-10-31",
    "modelo": "ExtraTrees, com parametros fixos, no preparo e nas compras",
    "cenario": "Cenario 2 com cardapio; cenario 1 sem cardapio",
    "historico": "Ultimos 12 meses, retreinado a cada sexta-feira",
    "hibrido": "Hibrido adaptativo: ExtraTrees com o quantitativo, peso definido pelos erros dos ultimos 40 dias",
    "margem": "Margem de seguranca por contexto, com erros dos ultimos 12 meses",
}


CODIGO = Path(__file__).parent
ENTRADAS_BASE = [GOLD / "gold_t1.parquet", GOLD / "gold_t2.parquet", CODIGO / "configuracao.py", CODIGO / "modelos.py"]
ENTRADAS = {
    "torneio": [*ENTRADAS_BASE, CODIGO / "experimentos.py"],
    "melhorias": [*ENTRADAS_BASE, CODIGO / "experimentos.py", PASTA_MODELOS / "torneio_previsoes.parquet"],
    "matriz": [*ENTRADAS_BASE, CODIGO / "experimentos.py"],
    "cenarios": [*ENTRADAS_BASE, CODIGO / "validacao.py"],
    "ridge": [*ENTRADAS_BASE, CODIGO / "experimentos.py", CODIGO / "margens.py", CODIGO / "validacao.py"],
}


def assinatura(entradas):
    resumo = hashlib.sha256()
    for arquivo in entradas:
        resumo.update(arquivo.read_bytes())
    return resumo.hexdigest()


def executar_se_preciso(etapa, recalcular, arquivos, entradas):
    """Refaz a etapa quando pedido, quando falta resultado ou quando dados ou codigo mudaram."""
    marca = arquivos[0].with_suffix(".entradas")
    atual = assinatura(entradas)
    if recalcular or not all(a.exists() for a in arquivos) or not marca.exists() or marca.read_text() != atual:
        etapa()
        marca.write_text(atual)
