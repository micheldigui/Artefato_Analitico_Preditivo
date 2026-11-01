# Figuras e tabelas para a dissertação

Numeração na ordem da apresentação à banca: como o artefato funciona, visão geral do consumo, análise de 2023 a 2025, análise de fevereiro a agosto de 2026, treino e teste, resultados, margens e híbrido, e avaliação prospectiva. Figuras a 300 DPI, sem título na imagem; as legendas abaixo são sugestões. Tabelas em CSV com separador `;` e decimal com vírgula.

## Figuras


### 1. Como o artefato funciona

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 1 | `figura_01_diagrama_arquitetura.png` | Arquitetura do artefato, das fontes de dados à aplicação |
| 2 | `figura_02_diagrama_fluxo_do_artefato.png` | Fluxo do artefato, dos dados brutos à decisão da nutricionista e da gestora |
| 3 | `figura_03_diagrama_cenarios.png` | Cenários do artefato: cada um acrescenta uma fonte de dados, e o artefato usa o mais completo disponível na semana |
| 4 | `figura_04_diagrama_validacao_semanal.png` | Validação semana a semana: em cada sexta-feira, o modelo aprende só com o passado |
| 5 | `figura_05_diagrama_fluxo_operacional.png` | Fluxo operacional semanal: o artefato prevê e informa; a decisão é da nutricionista e da gestora |

### 2. Visão geral do consumo: 2023 a agosto de 2026

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 6 | `figura_06_consumo_diario.png` | Consumo diário de refeições e dias fora da rotina (fevereiro de 2023 a agosto de 2026) |
| 7 | `figura_07_consumo_ano_mes.png` | Consumo médio diário por ano e mês |
| 8 | `figura_08_consumo_medio_por_ano.png` | Distribuição do consumo diário por ano, com todos os dias. A linha é a mediana, e a caixa reúne a metade central dos dias. O ano de 2026 vai até agosto |
| 9 | `figura_09_perfil_semanal_por_ano.png` | Consumo médio por dia da semana em cada ano. O ano de 2026 vai até agosto |

### 3.1. Análise de 2023 a 2025 (reservas): consumo passado

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 10 | `figura_10_autocorrelacao_2023_2025.png` | Quanto o consumo de um dia se relaciona com o consumo de dias anteriores (2023 a 2025, 485 dias). Média das correlações de cada ano |
| 11 | `figura_11_relacao_defasagens_2023_2025.png` | As 10 medidas de consumo passado mais relacionadas ao consumo do dia (2023 a 2025), pela média da relação em cada ano. A lista completa está na tabela |
| 12 | `figura_12_informacoes_repetidas.png` | Correlação entre seis medidas de consumo recente (2023 a 2025). Perto de 1, duas medidas contam quase a mesma história |

### 3.2. Análise de 2023 a 2025 (reservas): calendário

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 13 | `figura_13_dia_da_semana_2023_2025.png` | Consumo médio por dia da semana e diferença para o consumo de referência (2023 a 2025, sem dias atípicos). A referência já considera o mesmo dia da semana, por isso a diferença fica menor que a distância entre as médias |
| 14 | `figura_14_fim_do_periodo_2023_2025.png` | Consumo nos dias finais dos períodos de aulas (2023 a 2025, 5 fins de período) |
| 15 | `figura_15_eventos_do_calendario_2023_2025.png` | Consumo comparado à referência por evento do calendário (2023 a 2025, sem dias atípicos, eventos com pelo menos 3 dias) |

### 3.3. Análise de 2023 a 2025 (reservas): cardápio

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 16 | `figura_16_cardapio_2023_2025.png` | Consumo comparado à referência por proteína e forma de preparo (2023 a 2025, sem dias atípicos, categorias com pelo menos 5 dias) |

### 3.4. Análise de 2023 a 2025 (reservas): clima

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 17 | `figura_17_chuva_2023_2025.png` | Consumo comparado à referência por faixa de chuva (2023 a 2025, sem dias atípicos). A previsão guardada começa em 2024 |
| 18 | `figura_18_chuva_bruta_e_ajustada_2023_2025.png` | Consumo por faixa de chuva e diferença da chuva forte para dias sem chuva, na média bruta e comparando dias parecidos (2023 a 2025, com todos os dias). Dias parecidos só contam com pelo menos 3 dias sem chuva para comparar |
| 19 | `figura_19_temperatura_2023_2025.png` | Consumo comparado à referência por faixa de temperatura máxima (2023 a 2025, sem dias atípicos) |

### 3.5. Análise de 2023 a 2025 (reservas): reservas e consumo

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 20 | `figura_20_reservas_e_consumo.png` | Reservas dos alunos e consumo registrado de 2023 a 2025. À esquerda, a linha indica reserva igual ao consumo; à direita, a distribuição da diferença diária |
| 21 | `figura_21_excesso_deficit_reservas.png` | Excesso potencial e déficit potencial das reservas dos alunos, por ano |

### 3.6. Análise de 2023 a 2025 (reservas): síntese

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 22 | `figura_22_relacao_variaveis_2023_2025.png` | As 10 variáveis mais relacionadas ao consumo comparado à referência (2023 a 2025). A lista completa está na tabela |

### 4.1. Análise de fevereiro a agosto de 2026 (quantitativo): consumo passado

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 23 | `figura_23_autocorrelacao_2026.png` | Quanto o consumo de um dia se relaciona com o consumo de dias anteriores (fevereiro a agosto de 2026, 121 dias). Média das correlações de cada ano |
| 24 | `figura_24_relacao_defasagens_2026.png` | As 10 medidas de consumo passado mais relacionadas ao consumo do dia (fevereiro a agosto de 2026), pela média da relação em cada ano. A lista completa está na tabela |

### 4.2. Análise de fevereiro a agosto de 2026 (quantitativo): calendário

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 25 | `figura_25_dia_da_semana_2026.png` | Consumo médio por dia da semana e diferença para o consumo de referência (fevereiro a agosto de 2026, sem dias atípicos). A referência já considera o mesmo dia da semana, por isso a diferença fica menor que a distância entre as médias |
| 26 | `figura_26_fim_do_periodo_2026.png` | Consumo nos dias finais dos períodos de aulas (fevereiro a agosto de 2026, 1 fim de período) |
| 27 | `figura_27_eventos_do_calendario_2026.png` | Consumo comparado à referência por evento do calendário (fevereiro a agosto de 2026, sem dias atípicos, eventos com pelo menos 3 dias) |

### 4.3. Análise de fevereiro a agosto de 2026 (quantitativo): cardápio

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 28 | `figura_28_cardapio_2026.png` | Consumo comparado à referência por proteína e forma de preparo (fevereiro a agosto de 2026, sem dias atípicos, categorias com pelo menos 5 dias) |

### 4.4. Análise de fevereiro a agosto de 2026 (quantitativo): clima

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 29 | `figura_29_chuva_2026.png` | Consumo comparado à referência por faixa de chuva (fevereiro a agosto de 2026, sem dias atípicos). |
| 30 | `figura_30_chuva_bruta_e_ajustada_2026.png` | Consumo por faixa de chuva e diferença da chuva forte para dias sem chuva, na média bruta e comparando dias parecidos (fevereiro a agosto de 2026, com todos os dias). Dias parecidos só contam com pelo menos 3 dias sem chuva para comparar |
| 31 | `figura_31_temperatura_2026.png` | Consumo comparado à referência por faixa de temperatura máxima (fevereiro a agosto de 2026, sem dias atípicos) |
| 32 | `figura_32_qualidade_previsao_tempo.png` | Qualidade da previsão de chuva por antecedência. Preparo desde 2024 e compras desde abril de 2026, ambos até agosto de 2026 |

### 4.5. Análise de fevereiro a agosto de 2026 (quantitativo): quantitativo e consumo

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 33 | `figura_33_quantitativo_e_consumo.png` | Quantitativo da nutricionista e consumo registrado de fevereiro a agosto de 2026. À esquerda, a linha indica quantitativo igual ao consumo; à direita, a distribuição da diferença diária |
| 34 | `figura_34_excesso_deficit_quantitativo.png` | Excesso potencial e déficit potencial do quantitativo da nutricionista, por mês de 2026 |

### 4.6. Análise de fevereiro a agosto de 2026 (quantitativo): síntese

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 35 | `figura_35_relacao_variaveis_2026.png` | As 10 variáveis mais relacionadas ao consumo comparado à referência (fevereiro a agosto de 2026). A lista completa está na tabela |

### 4.7. O que repetiu e o que mudou entre os períodos

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 36 | `figura_36_repetiu_e_mudou.png` | O que repetiu e o que mudou entre os dois períodos, descontado o deslocamento geral de cada período, com categorias de pelo menos 10 dias em cada período e sem dias atípicos |
| 37 | `figura_37_diferenca_plano_consumo.png` | Diferença diária entre plano humano e consumo; valores positivos indicam plano acima do consumo. O ano de 2026 vai até agosto |
| 38 | `figura_38_excesso_deficit_ano.png` | Excesso potencial e déficit potencial do plano humano, por ano. O ano de 2026 vai até agosto |
| 39 | `figura_39_saldo_e_diferencas.png` | Saldo final do plano humano e soma das diferenças de cada dia. Um saldo pequeno pode esconder desvios grandes. O ano de 2026 vai até agosto |

### 5. Treino e teste dos algoritmos

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 40 | `figura_40_torneio_de_algoritmos.png` | Algoritmos testados com a mesma informação e os últimos 12 meses de aprendizado, em 2024 e 2025, primeira etapa da escolha. Em destaque, o ExtraTrees, eleito depois de somar a etapa de fevereiro a agosto de 2026 |
| 41 | `figura_41_historico_de_treino.png` | Erro médio conforme o histórico usado para aprender |
| 42 | `figura_42_importancia_das_informacoes.png` | As 10 informações mais importantes para o ExtraTrees: treino até 2024 e erro medido em 2025, embaralhando uma informação de cada vez |
| 43 | `figura_43_matriz_de_cenarios.png` | MAE de cada algoritmo em cada cenário no preparo, de fevereiro a agosto de 2026, sem margem. Quanto mais escuro, menor o erro |

### 6. Resultados de fevereiro a agosto de 2026

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 44 | `figura_44_cenarios_2026.png` | Erro médio por cenário de fevereiro a agosto de 2026, sem dias atípicos |
| 45 | `figura_45_quatro_metricas_fev_ago.png` | MAE, RMSE, WAPE e R² de fevereiro a agosto de 2026, sem dias atípicos. Decisão de preparo |
| 46 | `figura_46_com_e_sem_atipicos_fev_ago.png` | Erro médio com e sem dias atípicos (fevereiro a agosto de 2026). Os dias atípicos só são identificados depois que acontecem |
| 47 | `figura_47_compras_e_preparo.png` | Erro médio de fevereiro a agosto de 2026 no preparo e nas compras, por informação usada, sem dias atípicos |
| 48 | `figura_48_efeito_do_clima.png` | Erro do ExtraTrees com e sem clima de fevereiro a agosto de 2026, sem dias atípicos. O clima que de fato ocorreu não seria conhecido na sexta-feira e mostra o limite do ganho possível. Nas compras, ainda não há histórico suficiente de previsões guardadas |
| 49 | `figura_49_algoritmos_comparados.png` | Erro médio de cada algoritmo de fevereiro a agosto de 2026, sem dias atípicos, com destaque para o ExtraTrees |
| 50 | `figura_50_serie_2026.png` | Consumo registrado e cada forma de planejamento de fevereiro a agosto de 2026 |
| 51 | `figura_51_consumo_real_e_alternativas_por_mes.png` | Consumo real médio e planejamento médio de cada alternativa por mês, de fevereiro a agosto de 2026 |
| 52 | `figura_52_erro_mensal_2026.png` | Consumo real médio (barras) e erro médio de cada opção (linhas) por mês de fevereiro a agosto de 2026, na mesma escala de refeições, com todos os dias. |
| 53 | `figura_53_mae_e_rmse_por_mes.png` | Erro médio (MAE) e erro com peso nos grandes desvios (RMSE) por mês de fevereiro a agosto de 2026, com todos os dias |
| 54 | `figura_54_mais_perto_do_consumo_real.png` | Em cada um dos 123 dias, de fevereiro a agosto de 2026, qual opção ficou mais perto do consumo real. Empates contam para todas as opções empatadas, por isso a soma passa de 123 |
| 55 | `figura_55_distribuicao_erros_2026.png` | Distribuição dos desvios de fevereiro a agosto de 2026, sem dias atípicos |
| 56 | `figura_56_planejado_contra_consumo_2026.png` | Planejado e consumido de fevereiro a agosto de 2026, sem dias atípicos; a linha indica plano igual ao consumo |
| 57 | `figura_57_erro_por_dia_da_semana.png` | Erro médio por dia da semana de fevereiro a agosto de 2026, sem dias atípicos |
| 58 | `figura_58_erro_por_antecedencia.png` | Erro médio do artefato por antecedência de fevereiro a agosto de 2026, sem dias atípicos |
| 59 | `figura_59_comparacao_anual.png` | Erro médio por ano nas mesmas datas, sem dias atípicos. Em 2023 não há histórico anterior para o artefato |

### 7. Margens de segurança e híbrido

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 60 | `figura_60_risco_informado_e_observado.png` | Risco de déficit potencial informado antes de cada semana e o que aconteceu de fevereiro a agosto de 2026, sem dias atípicos |
| 61 | `figura_61_margem_por_situacao.png` | Dias com déficit potencial por situação do dia, no preparo de fevereiro a agosto de 2026 |
| 62 | `figura_62_fronteira_do_hibrido.png` | Erro de fevereiro a agosto de 2026 conforme o peso do artefato no híbrido, sem dias atípicos. A curva é descritiva: o melhor peso só é conhecido depois |
| 63 | `figura_63_peso_da_combinacao.png` | Peso do artefato no híbrido, definido a cada semana pelos erros recentes. O restante é o peso do quantitativo |
| 64 | `figura_64_excesso_e_deficit_2026.png` | Soma do excesso potencial e do déficit potencial de cada opção de preparo de fevereiro a agosto de 2026, sem dias atípicos. A margem muda a cada semana; entre parênteses está o acréscimo médio por dia. O artefato sem margem usa o cenário sem cardápio nos 13 dias sem cardápio registrado |
| 65 | `figura_65_frequencia_de_deficit_e_excesso.png` | Frequência de déficit potencial e excesso potencial médio por opção de preparo de fevereiro a agosto de 2026, sem dias atípicos |

### 8. Avaliação prospectiva: setembro e outubro de 2026

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 66 | `figura_66_prospectiva_erro_por_alternativa.png` | Avaliação prospectiva: erro médio do preparo em setembro e outubro de 2026, separado dos resultados até agosto |
| 67 | `figura_67_quatro_metricas_prospectiva.png` | Avaliação prospectiva: MAE, RMSE, WAPE e R² em setembro e outubro de 2026, sem dias atípicos. Decisão de preparo |
| 68 | `figura_68_com_e_sem_atipicos_set_out.png` | Erro médio com e sem dias atípicos (setembro e outubro de 2026). Os dias atípicos só são identificados depois que acontecem |
| 69 | `figura_69_setembro_outubro.png` | Avaliação prospectiva: consumo real e planejamento em setembro e outubro de 2026. O peso do artefato no híbrido variou de 20% a 100%, conforme os erros recentes. As linhas são interrompidas nas datas sem registro |
| 70 | `figura_70_regulares_e_atipicos.png` | Avaliação prospectiva: desvios em setembro e outubro de 2026, em dias regulares e atípicos; os painéis têm escalas próprias |
| 71 | `figura_71_balanco_dias_regulares.png` | Avaliação prospectiva: excesso potencial e déficit potencial nos 28 dias regulares de setembro e outubro de 2026 |
| 72 | `figura_72_prospectiva_margens.png` | Avaliação prospectiva: risco de déficit potencial informado antes de cada semana e o que aconteceu nos 28 dias regulares de setembro e outubro de 2026 |
| 73 | `figura_73_semana_de_retorno.png` | Avaliação prospectiva: consumo real e planejamento na semana de retorno, de 19 a 23 de outubro de 2026 |
| 74 | `figura_74_previsoes_enviadas_outubro.png` | Previsões enviadas antes de cada semana de outubro de 2026 pela versão em uso na época: modelo puro, híbrido e híbrido com margem. Para comparar, o quantitativo final, o artefato na versão congelada e o consumo real. Registro disponível a partir de 19/10 |

### 9. Síntese do método

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 75 | `figura_75_diagrama_sintese_metodologica.png` | Síntese metodológica: das fontes de dados ao apoio à decisão |

## Tabelas


### 1. Como o artefato funciona

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 1 | `tabela_01_ingestao_bronze.csv` | Arquivos e linhas das fontes na camada Bronze |
| 2 | `tabela_02_tratamento_dos_dias.csv` | Classificação dos dias com consumo registrado e uso no aprendizado |
| 3 | `tabela_03_base_e_conjuntos.csv` | Base histórica e conjuntos de avaliação na decisão de preparo |
| 4 | `tabela_04_grupos_de_informacoes.csv` | Grupos de informações do artefato |
| 5 | `tabela_05_familias_de_modelos.csv` | Famílias de modelos testadas no torneio |

### 2. Visão geral do consumo: 2023 a agosto de 2026

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 6 | `tabela_06_consumo_por_ano.csv` | Consumo registrado de segunda a sexta-feira, por ano. O ano de 2026 vai até agosto |

### 3.1. Análise de 2023 a 2025 (reservas): consumo passado

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 7 | `tabela_07_autocorrelacao_2023_2025.csv` | Autocorrelação do consumo (2023 a 2025) |
| 8 | `tabela_08_relacao_defasagens_2023_2025.csv` | Relação das defasagens e medidas móveis com o consumo, conhecendo o consumo até a véspera (2023 a 2025) |
| 9 | `tabela_09_relacao_atributos_sexta.csv` | Relação com o consumo das medidas realmente conhecidas na sexta-feira do planejamento (2023 a 2025) |
| 10 | `tabela_10_informacoes_repetidas.csv` | Correlação entre as medidas de consumo recente (2023 a 2025) |

### 3.4. Análise de 2023 a 2025 (reservas): clima

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 11 | `tabela_11_chuva_bruta_e_ajustada_2023_2025.csv` | Diferença de consumo entre chuva forte e dias sem chuva, na média bruta e comparando dias parecidos (2023 a 2025). Dias parecidos só contam com pelo menos 3 dias sem chuva para comparar |

### 3.5. Análise de 2023 a 2025 (reservas): reservas e consumo

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 12 | `tabela_12_reservas_por_ano.csv` | Reservas dos alunos e consumo registrado por ano |

### 3.6. Análise de 2023 a 2025 (reservas): síntese

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 13 | `tabela_13_efeitos_2023_2025.csv` | Consumo comparado à referência por dia da semana, cardápio, evento e clima, com a quantidade de dias (2023 a 2025, sem dias atípicos) |
| 14 | `tabela_14_relacao_variaveis_2023_2025.csv` | Relação das variáveis com o consumo comparado à referência (2023 a 2025) |

### 4.1. Análise de fevereiro a agosto de 2026 (quantitativo): consumo passado

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 15 | `tabela_15_autocorrelacao_2026.csv` | Autocorrelação do consumo (fevereiro a agosto de 2026) |
| 16 | `tabela_16_relacao_defasagens_2026.csv` | Relação das defasagens e medidas móveis com o consumo, conhecendo o consumo até a véspera (fevereiro a agosto de 2026) |

### 4.4. Análise de fevereiro a agosto de 2026 (quantitativo): clima

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 17 | `tabela_17_chuva_bruta_e_ajustada_2026.csv` | Diferença de consumo entre chuva forte e dias sem chuva, na média bruta e comparando dias parecidos (fevereiro a agosto de 2026). Dias parecidos só contam com pelo menos 3 dias sem chuva para comparar |
| 18 | `tabela_18_qualidade_previsao_tempo.csv` | Relação entre a chuva prevista e a chuva que ocorreu, por antecedência. Preparo desde 2024 e compras desde abril de 2026, ambos até agosto de 2026 |

### 4.5. Análise de fevereiro a agosto de 2026 (quantitativo): quantitativo e consumo

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 19 | `tabela_19_quantitativo_por_mes.csv` | Quantitativo da nutricionista e consumo registrado por mês de 2026 |

### 4.6. Análise de fevereiro a agosto de 2026 (quantitativo): síntese

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 20 | `tabela_20_efeitos_2026.csv` | Consumo comparado à referência por dia da semana, cardápio, evento e clima, com a quantidade de dias (fevereiro a agosto de 2026, sem dias atípicos) |
| 21 | `tabela_21_relacao_variaveis_2026.csv` | Relação das variáveis com o consumo comparado à referência (fevereiro a agosto de 2026) |

### 4.7. O que repetiu e o que mudou entre os períodos

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 22 | `tabela_22_resumo_dos_periodos.csv` | Resumo dos dois períodos: as mesmas perguntas respondidas em 2023 a 2025 e em fevereiro a agosto de 2026 |
| 23 | `tabela_23_repetiu_e_mudou.csv` | Consumo comparado à referência nos dois períodos, descontado o deslocamento geral de cada período. Mudou indica 5 pontos percentuais ou mais de distância; Poucos dias indica menos de 10 dias em algum período |
| 24 | `tabela_24_planos_por_ano.csv` | Plano humano e consumo registrado por ano. O ano de 2026 vai até agosto |

### 5. Treino e teste dos algoritmos

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 25 | `tabela_25_torneio_de_algoritmos.csv` | Erro médio dos algoritmos testados, com os dados usados para a escolha |
| 26 | `tabela_26_ideias_testadas.csv` | Ideias de melhoria testadas com 2024 e 2025. Uma ideia só seria adotada com ganho de pelo menos 0,5 refeição por dia |
| 27 | `tabela_27_historico_de_treino.csv` | Erro médio por modelo e histórico usado para aprender, sem dias atípicos |
| 28 | `tabela_28_importancia_das_informacoes.csv` | Importância das informações para o ExtraTrees, com treino até 2024 e medição em 2025 |
| 29 | `tabela_29_matriz_de_cenarios.csv` | Algoritmos principais em cada cenário e política. A matriz completa está em resultados/modelos/matriz_completa.csv |
| 30 | `tabela_30_configuracao_escolhida.csv` | ExtraTrees em cada cenário. A escolha usou 2024, 2025 e fevereiro a agosto de 2026 |

### 6. Resultados de fevereiro a agosto de 2026

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 31 | `tabela_31_resultado_cenarios.csv` | Desempenho das alternativas por período. No preparo a referência é o quantitativo; nas compras, o consumo de referência |
| 32 | `tabela_32_quatro_metricas_fev_ago.csv` | MAE, RMSE, WAPE e R² de fevereiro a agosto de 2026, sem dias atípicos |
| 33 | `tabela_33_com_e_sem_atipicos_fev_ago.csv` | Métricas com e sem dias atípicos (fevereiro a agosto de 2026) |
| 34 | `tabela_34_ganho_sobre_linhas_de_base.csv` | Diferença de erro médio para cada referência. Valor negativo significa erro menor que o da referência; o intervalo reamostra semanas inteiras |
| 35 | `tabela_35_ridge_e_extratrees.csv` | Ridge e ExtraTrees nas mesmas datas e políticas, sem dias atípicos |
| 36 | `tabela_36_metricas_por_algoritmo.csv` | MAE, RMSE, WAPE e R² dos algoritmos de fevereiro a agosto de 2026, sem dias atípicos |
| 37 | `tabela_37_efeito_do_clima.csv` | Erro do ExtraTrees com e sem informação de clima de fevereiro a agosto de 2026 |
| 38 | `tabela_38_consumo_real_e_alternativas_por_mes.csv` | Consumo real médio e planejamento médio de cada alternativa por mês, de fevereiro a agosto de 2026 |
| 39 | `tabela_39_erro_mensal_2026.csv` | Erro médio por mês de fevereiro a agosto de 2026, com todos os dias |
| 40 | `tabela_40_mae_e_rmse_por_mes.csv` | MAE e RMSE por mês de fevereiro a agosto de 2026, com todos os dias |
| 41 | `tabela_41_mais_perto_do_consumo_real.csv` | Dias em que cada alternativa ficou mais perto do consumo real de fevereiro a agosto de 2026. Empates contam para todas as opções empatadas |
| 42 | `tabela_42_erro_por_dia_da_semana.csv` | Erro médio por dia da semana de fevereiro a agosto de 2026, sem dias atípicos |
| 43 | `tabela_43_erro_por_antecedencia.csv` | Erro médio do artefato por antecedência de fevereiro a agosto de 2026, sem dias atípicos |
| 44 | `tabela_44_comparacao_anual.csv` | Comparação anual entre plano humano, consumo de referência, artefato e combinação, nas mesmas datas |

### 7. Margens de segurança e híbrido

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 45 | `tabela_45_opcoes_de_margem.csv` | Resultado das opções de margem de segurança de fevereiro a agosto de 2026 |
| 46 | `tabela_46_margem_por_situacao.csv` | Tendência da previsão e proteção das margens de segurança por situação do dia de fevereiro a agosto de 2026 |
| 47 | `tabela_47_fronteira_do_hibrido.csv` | Erro de fevereiro a agosto de 2026 conforme o peso do artefato no híbrido |

### 8. Avaliação prospectiva: setembro e outubro de 2026

| Número | Arquivo | Legenda sugerida |
|---:|---|---|
| 48 | `tabela_48_prospectiva_erro_por_alternativa.csv` | Avaliação prospectiva: preparo em setembro e outubro de 2026 |
| 49 | `tabela_49_quatro_metricas_prospectiva.csv` | Avaliação prospectiva: MAE, RMSE, WAPE e R² em setembro e outubro de 2026, sem dias atípicos |
| 50 | `tabela_50_com_e_sem_atipicos_set_out.csv` | Métricas com e sem dias atípicos (setembro e outubro de 2026) |
| 51 | `tabela_51_setembro_outubro.csv` | Avaliação prospectiva: resultado em setembro e outubro de 2026, por recorte de dias |
| 52 | `tabela_52_prospectiva_margens.csv` | Avaliação prospectiva: resultado das opções de margem de segurança em setembro e outubro de 2026 |
| 53 | `tabela_53_semana_de_retorno.csv` | Avaliação prospectiva: planejamento na semana de retorno, de 19 a 23 de outubro de 2026 |
| 54 | `tabela_54_previsoes_enviadas_outubro.csv` | Erro médio das previsões enviadas antes de cada semana de outubro de 2026, comparado ao quantitativo final e ao artefato na versão congelada. O híbrido de 19 a 23/10 usou a estimativa disponível em 14/10, anterior ao quantitativo final |
