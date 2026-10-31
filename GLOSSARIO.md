# Glossário

Linguagem usada nas tabelas, figuras e telas do artefato. O arquivo `src/nomes.py` guarda a lista completa, e os nomes técnicos ficam só no código.

## As duas decisões

| Como chamamos | O que significa | Termo técnico |
|---|---|---|
| **Preparo (semana seguinte)** | Na sexta-feira, prever os almoços da semana que vem, de 3 a 7 dias antes. | t+1 |
| **Compras (daqui a duas semanas)** | Na sexta-feira, prever os almoços da semana depois da próxima, de 10 a 14 dias antes. | t+2 |
| **Sexta-feira do planejamento** | Dia em que a previsão é feita. Ela só usa o que já se sabia nesse dia. | origem, emissão |

## Os cenários

Cada cenário acrescenta uma fonte de dados ao anterior. O artefato usa o mais completo que estiver disponível na semana.

| Cenário | O que usa | Quando vale |
|---|---|---|
| **Cenário 1: consumo e calendário** | Refeições consumidas e calendário de aulas | Sempre |
| **Cenário 2: inclui o cardápio** | Cenário 1 + pratos da semana | Com cardápio publicado |
| **Cenário 3: inclui a previsão do tempo** | Cenário 2 + previsão de chuva e temperatura | Só no preparo |
| **Cenário 4: inclui o quantitativo da nutricionista** | Cenário anterior + quantitativo da nutricionista ou reservas dos alunos | Quando esse número existir |

## Consumo e planejamento

| Como chamamos | O que significa | Termo técnico |
|---|---|---|
| **Refeições consumidas (consumo real)** | Total de refeições registrado no dia. | consumo, alvo |
| **Consumo de referência** | Estimativa feita antes do dia, com o consumo de dias normais recentes e do mesmo dia da semana. Não é o consumo real. | nível base, baseline |
| **Comitê** | Previsão que junta vários modelos pela média simples. O artefato usa CatBoost, KNN e o consumo de referência. | ensemble |
| **Consumo comparado à referência** | 0% é igual à referência; -10% é 10% abaixo. | consumo relativo, razão |
| **Período de aulas** | Trecho contínuo de aulas entre recessos ou férias. | bloco letivo |
| **Quantitativo da nutricionista** | Número de refeições informado por ela para cada dia. | plano, reservas (2026) |
| **Reservas dos alunos** | Refeições reservadas pelos estudantes, de 2023 a 2025. | reserva discente |

## Dias fora do padrão

| Como chamamos | O que significa | Uso pelo algoritmo |
|---|---|---|
| **Dia normal** | Consumo dentro do esperado. | Aprende com ele |
| **Queda prevista pelo calendário** | Consumo baixo em fim de período, volta de pausa ou evento. | Aprende com ele |
| **Consumo acima da referência** | Consumo bem acima da referência. | Aprende com ele |
| **Dia atípico imprevisível** | Enchente ou queda sem explicação pelo calendário. | Fica na base, sem ensinar o algoritmo |
| **Período de greve** | Dias em greve, com rotina irregular. | Fica na base, sem ensinar o algoritmo |
| **Consumo fora dos dias de aula** | Consumo em dia sem aula no calendário. | Fica na base, sem ensinar o algoritmo |

## Opções de planejamento

| Como chamamos | O que significa | Termo técnico |
|---|---|---|
| **Sem margem** | Usa a previsão como está. | previsão pontual |
| **Margem moderada** | Acréscimo para haver déficit potencial em cerca de 1 a cada 4 dias. | margem P75 |
| **Margem alta** | Acréscimo para haver déficit potencial em cerca de 1 a cada 10 dias. | margem P90 |
| **Dias parecidos** | Dias anteriores com a mesma situação do calendário. | análogos |
| **Pontos de atenção** | Avisos do dia, como chuva forte ou diferença entre quantitativo e previsão. | alertas |

## Como medimos o resultado

| Como chamamos | O que significa | Termo técnico |
|---|---|---|
| **Erro médio** | Quantas refeições a previsão erra por dia, para mais ou para menos. | MAE |
| **Erro médio com peso nos grandes desvios** | Parecido com o erro médio, mas pune mais os dias de erro grande. | RMSE |
| **Erro em relação ao total consumido** | Erro em porcentagem do que foi consumido. | WAPE |
| **Quanto da variação foi explicada** | Compara a previsão com usar sempre a média. Não é percentual de acerto. | R² |
| **Tendência** | Se a previsão costuma ficar acima (+) ou abaixo (-) do consumo. | viés |
| **Excesso potencial** | Refeições planejadas acima do consumo. É potencial porque o restaurante não pesa o que é descartado. | excesso potencial |
| **Déficit potencial** | Refeições planejadas abaixo do consumo. É potencial porque não há registro de quem ficou sem refeição. | déficit potencial |
