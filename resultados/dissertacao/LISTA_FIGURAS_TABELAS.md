# Figuras e tabelas para a dissertação

Figuras a 300 DPI, sem título na imagem; as legendas abaixo são sugestões. Tabelas em CSV com separador `;` e decimal com vírgula.

## Figuras

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 1 | `figura_01_consumo_diario.png` | Consumo diário de refeições e dias fora da rotina (fevereiro de 2023 a outubro de 2026) |
| 2 | `figura_02_consumo_ano_mes.png` | Consumo médio diário por ano e mês |
| 3 | `figura_03_dia_da_semana.png` | Consumo comparado à referência por dia da semana |
| 4 | `figura_04_fim_do_periodo.png` | Consumo nos dias finais dos períodos de aulas, com dados até 2025 |
| 5 | `figura_05_eventos_do_calendario.png` | Consumo comparado à referência por evento do calendário |
| 6 | `figura_06_cardapio.png` | Consumo comparado à referência por proteína e forma de preparo |
| 7 | `figura_07_chuva.png` | Consumo comparado à referência por faixa de chuva |
| 8 | `figura_08_relacao_defasagens.png` | As 10 medidas de consumo passado mais relacionadas ao consumo do dia (até 2025). A lista completa está na tabela |
| 9 | `figura_09_relacao_variaveis.png` | As 10 variáveis mais relacionadas ao consumo comparado à referência (até 2025). A lista completa está na tabela |
| 10 | `figura_10_informacoes_repetidas.png` | Correlação entre seis medidas de consumo recente. Perto de 1, duas medidas contam quase a mesma história |
| 11 | `figura_11_qualidade_previsao_tempo.png` | Qualidade da previsão de chuva por antecedência |
| 12 | `figura_12_plano_contra_consumo.png` | Plano humano e consumo registrado; a linha indica plano igual ao consumo |
| 13 | `figura_13_diferenca_plano_consumo.png` | Diferença diária entre plano humano e consumo; valores positivos indicam plano acima do consumo |
| 14 | `figura_14_excesso_deficit_ano.png` | Excesso potencial e déficit potencial do plano humano, por ano |
| 15 | `figura_15_cenarios_2026.png` | Erro médio por cenário em 2026, sem dias atípicos |
| 16 | `figura_16_historico_de_treino.png` | Erro médio conforme o histórico usado para aprender |
| 17 | `figura_17_torneio_de_algoritmos.png` | Algoritmos testados com a mesma informação e os últimos 12 meses de aprendizado. Dados de 2024 e 2025, usados para escolher o comitê campeão |
| 18 | `figura_18_comparacao_anual.png` | Erro médio por ano nas mesmas datas, sem dias atípicos. Em 2023 não há histórico anterior para o artefato |
| 19 | `figura_19_serie_2026.png` | Consumo registrado e cada forma de planejamento em 2026 |
| 20 | `figura_20_consumo_real_e_alternativas_por_mes.png` | Consumo real médio e planejamento médio de cada alternativa por mês, em 2026 |
| 21 | `figura_21_mais_perto_do_consumo_real.png` | Em cada um dos 156 dias de 2026, qual opção ficou mais perto do consumo real. Empates contam para todas as opções empatadas, por isso a soma passa de 156 |
| 22 | `figura_22_algoritmos_comparados.png` | Erro médio de cada algoritmo em 2026, sem dias atípicos, com destaque para o comitê campeão, fixado antes desta medição |
| 23 | `figura_23_erro_mensal_2026.png` | Consumo real médio (barras) e erro médio de cada opção (linhas) por mês em 2026, na mesma escala de refeições, com todos os dias. Setembro: híbrido igual ao quantitativo |
| 24 | `figura_24_distribuicao_erros_2026.png` | Distribuição dos desvios em 2026, sem dias atípicos |
| 25 | `figura_25_planejado_contra_consumo_2026.png` | Planejado e consumido em 2026, sem dias atípicos; a linha indica plano igual ao consumo |
| 26 | `figura_26_excesso_e_deficit_2026.png` | Soma do excesso potencial e do déficit potencial de cada opção de preparo em 2026, sem dias atípicos. A margem muda a cada semana; entre parênteses está o acréscimo médio por dia |
| 27 | `figura_27_risco_informado_e_observado.png` | Risco de déficit potencial informado antes de cada semana e o que aconteceu em 2026, sem dias atípicos |
| 28 | `figura_28_margem_por_situacao.png` | Dias com déficit potencial por situação do dia, no preparo em 2026 |
| 29 | `figura_29_erro_por_antecedencia.png` | Erro médio do artefato por antecedência em 2026, sem dias atípicos |
| 30 | `figura_30_setembro_outubro.png` | Consumo real e planejamento em setembro e outubro de 2026. Na maior parte do período, o híbrido coincide com o quantitativo, porque o peso do artefato foi zero. As linhas são interrompidas nas datas sem registro |
| 31 | `figura_31_regulares_e_atipicos.png` | Desvios em setembro e outubro de 2026, em dias regulares e atípicos; os painéis têm escalas próprias |
| 32 | `figura_32_semana_de_retorno.png` | Consumo real e planejamento na semana de retorno, de 19 a 23 de outubro de 2026 |
| 33 | `figura_33_peso_da_combinacao.png` | Peso do artefato no híbrido, definido a cada semana pelos erros recentes. O restante é o peso do quantitativo |
| 34 | `figura_34_erro_por_dia_da_semana.png` | Erro médio por dia da semana em 2026, sem dias atípicos |
| 35 | `figura_35_diagrama_fluxo_do_artefato.png` | Fluxo do artefato, dos dados brutos à decisão da nutricionista e da gestora |
| 36 | `figura_36_diagrama_cenarios.png` | Cenários do artefato: cada um acrescenta uma fonte de dados, e o artefato usa o mais completo disponível na semana |
| 37 | `figura_37_diagrama_validacao_semanal.png` | Validação semana a semana: em cada sexta-feira, o modelo aprende só com o passado |

## Tabelas

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 1 | `tabela_01_consumo_por_ano.csv` | Consumo registrado de segunda a sexta-feira, por ano. O ano de 2026 vai até outubro |
| 2 | `tabela_02_tratamento_dos_dias.csv` | Classificação dos dias com consumo registrado e uso no aprendizado |
| 3 | `tabela_03_efeitos_categorias.csv` | Consumo comparado à referência por categoria de cardápio e evento |
| 4 | `tabela_04_relacao_defasagens.csv` | Relação das defasagens e medidas móveis com o consumo, conhecendo o consumo até a véspera (até 2025) |
| 5 | `tabela_05_relacao_atributos_sexta.csv` | Relação com o consumo das medidas realmente conhecidas na sexta-feira do planejamento (até 2025) |
| 6 | `tabela_06_relacao_variaveis.csv` | Relação das variáveis com o consumo comparado à referência (até 2025) |
| 7 | `tabela_07_informacoes_repetidas.csv` | Correlação entre as medidas de consumo recente |
| 8 | `tabela_08_qualidade_previsao_tempo.csv` | Relação entre a chuva prevista e a chuva que ocorreu, por antecedência |
| 9 | `tabela_09_planos_por_ano.csv` | Plano humano e consumo registrado por ano. O ano de 2026 vai até outubro |
| 10 | `tabela_10_resultado_cenarios.csv` | Desempenho das alternativas por período. No preparo a referência é o quantitativo; nas compras, o consumo de referência |
| 11 | `tabela_11_historico_de_treino.csv` | Erro médio por modelo e histórico usado para aprender, sem dias atípicos |
| 12 | `tabela_12_configuracao_escolhida.csv` | Comitê campeão em cada cenário. A escolha usou 2024 e 2025 |
| 13 | `tabela_13_torneio_de_algoritmos.csv` | Erro médio dos algoritmos testados, com os dados usados para a escolha |
| 14 | `tabela_14_ideias_testadas.csv` | Ideias de melhoria testadas com 2024 e 2025. Uma ideia só seria adotada com ganho de pelo menos 0,5 refeição por dia |
| 15 | `tabela_15_comparacao_anual.csv` | Comparação anual entre plano humano, consumo de referência, artefato e combinação, nas mesmas datas |
| 16 | `tabela_16_consumo_real_e_alternativas_por_mes.csv` | Consumo real médio e planejamento médio de cada alternativa por mês, em 2026 |
| 17 | `tabela_17_mais_perto_do_consumo_real.csv` | Dias em que cada alternativa ficou mais perto do consumo real em 2026. Empates contam para todas as opções empatadas |
| 18 | `tabela_18_erro_mensal_2026.csv` | Erro médio por mês em 2026, com todos os dias |
| 19 | `tabela_19_opcoes_de_margem.csv` | Resultado das opções de margem de segurança em 2026 |
| 20 | `tabela_20_margem_por_situacao.csv` | Tendência da previsão e proteção das margens de segurança por situação do dia em 2026 |
| 21 | `tabela_21_erro_por_antecedencia.csv` | Erro médio do artefato por antecedência em 2026, sem dias atípicos |
| 22 | `tabela_22_setembro_outubro.csv` | Resultado em setembro e outubro de 2026, por recorte de dias |
| 23 | `tabela_23_semana_de_retorno.csv` | Planejamento na semana de retorno, de 19 a 23 de outubro de 2026 |
| 24 | `tabela_24_erro_por_dia_da_semana.csv` | Erro médio por dia da semana em 2026, sem dias atípicos |
