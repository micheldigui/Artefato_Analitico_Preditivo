# Revisão adversarial

Revisão de dados, modelos, resultados e notebooks. As verificações automáticas ficam em `notebooks/funcoes/verificacoes.py`, com o relatório em `resultados/revisao/verificacoes.csv`.

## Verificações que passaram

- **Sem informação do futuro nos dados:** o consumo a partir de cada sexta-feira foi embaralhado, e os atributos ficaram idênticos. Um vazamento inserido de propósito foi detectado.
- **Sem informação do futuro nos modelos:** consumo e rótulos futuros embaralhados não alteram nenhuma previsão, em todos os cenários e históricos testados.
- **Valores conferidos com os CSVs:** consumo, reservas e quantitativo da Gold são iguais aos dados brutos.
- **Totais conhecidos reproduzidos:** 458 datas de reservas (2.916 a mais; 9.494 a menos) e 123 datas de quantitativo até 31/08/2026 (2.561 a mais; 663 a menos).
- **Métricas recalculadas por outro caminho:** erros médios de 2026 conferem a partir das previsões diárias.
- **Modelo congelado na configuração:** a verificação confirma que os resultados vêm da versão congelada, com 12 meses de treino.
- **Projeto independente:** o pipeline rodou do zero sem depender do projeto anterior e repetiu os resultados.

## Problemas encontrados e corrigidos

| Problema | Correção |
|---|---|
| Cenário 3 comparado em condição desigual, com menos histórico. | Testado também com 12 meses. A previsão do tempo segue sem reduzir o erro. |
| Valor do calendário nunca medido. | Criada a comparação "só histórico de consumo". |
| Comparação com reservas só em 2025. | Validação estendida a 2024, além da referência em 2023. |
| Limite de fim de período definido olhando também 2026. | Conferido com dados até 2025: 0,61 contra 1,01. |
| Barras com eixo cortado exageravam diferenças. | Figuras mostram a diferença em porcentagem, a partir de 0%. |
| Previsões com números quebrados. | Previsões, margens e híbrido são arredondados para refeições inteiras: de ,5 para cima, arredonda para cima. |
| Saldo somava excesso e descontava déficit. | Saldo retirado: um dia não compensa o outro. |
| Dados que identificavam a instituição nos arquivos. | Retirados do repositório. |
| Siglas internas nos nomes de eventos do calendário. | Trocadas por nomes genéricos, da Bronze às figuras, sem alterar nenhum número. |
| Híbrido cobria o quantitativo nos gráficos quando o peso do artefato era zero. | Híbrido passou a ser desenhado em linha tracejada. |
| Pausas curtas contavam como fim de semestre. | Só pausas de 10 dias úteis ou mais contam como fim de semestre. |
| Dependência do projeto antigo. | Removida. |

## Como a versão foi escolhida e congelada

1. **Treino e primeira comparação (2024 e 2025):** 15 algoritmos, com testes de sazonalidade, recência, quantidade de preditoras, cardápio, clima, stacking, resíduo, regimes, SARIMAX e Prophet (notebooks 05 e 06).
2. **Matriz completa:** algoritmos, cenários e políticas (sem margem, margens e híbrido), no T+1 e no T+2, com MAE, RMSE, WAPE, R², excesso e déficit (`resultados/modelos/matriz_completa.csv`). No T+2 não há quantitativo, e a previsão do tempo só existe desde abril de 2026.
3. **Teste retrospectivo (fevereiro a agosto de 2026):** critério de escolha pela média do MAE de 2024-2025 e fevereiro a agosto. Opções a menos de 0,5 refeição foram tratadas como empate. O desempate usou um único algoritmo para preparo e compras, sem olhar setembro e outubro.
4. **Versão congelada em 31/10/2026:** ExtraTrees com cardápio e 12 meses de treino, híbrido adaptativo quando há quantitativo e margem de segurança por contexto (`notebooks/funcoes/configuracao.py`).

MAE em refeições por dia, sem dias atípicos:

| Opção | 2024-25 | Fev-ago | Set-out | Decisão |
|---|---:|---:|---:|---|
| ExtraTrees, preparo | 21,6 | 24,6 | 13,2 | Congelado |
| ExtraTrees, compras | 22,4 | 23,4 | 13,8 | Congelado |
| Híbrido adaptativo com ExtraTrees | 20,2 | 20,4 | 18,3 | Congelado |
| Comitê de CatBoost, KNN e referência, preparo | 19,9 | 28,0 | 12,6 | Não escolhido |
| Ridge, preparo | 22,8 | 23,1 | 22,4 | Não escolhido |
| Ridge forte, preparo | 20,5 | 25,7 | 13,5 | Empatou no critério |
| Ridge com plano corrigido, preparo | 22,3 | 20,4 | 25,7 | Não escolhido |

Resultado da versão congelada no preparo, de fevereiro a agosto (121 dias): consumo de referência 31,3, artefato 24,6, quantitativo 24,1 e híbrido 20,4. A vantagem do híbrido sobre o quantitativo é de 3,7 (-7,1 a -0,2). Nas compras, o artefato erra 23,4, contra 31,6 do consumo de referência (-8,2; -14,3 a -4,4).

Avaliação prospectiva, sem dias atípicos (28 dias): consumo de referência 13,6, artefato 13,2, quantitativo 22,1 e híbrido 18,3. Com todos os dias (33): 30,7, 30,3, 39,5 e 35,7. Nas compras, sem dias atípicos: 13,9 no consumo de referência e 13,8 no artefato.

Margem de segurança por contexto no preparo, de fevereiro a agosto: moderada com acréscimo médio de 15 refeições e déficit em 29% dos dias, contra 24% informados; alta com 32 refeições e 13%, contra 10% informados. Em setembro e outubro: moderada com déficit em 25% dos dias, contra 24%; alta sem dias de déficit, contra 10% informados.

**Fim de período corrigido:** pausas com menos de 10 dias úteis deixaram de contar como fim de semestre. A regra se sustenta com dados anteriores a setembro: nas pausas curtas de 2024, a razão de consumo foi 0,83 e 0,95, contra 0,33 a 0,75 nos fins de semestre. O defeito, porém, foi encontrado ao analisar os erros de outubro, depois de ver esses dados. A confirmação independente vem das previsões guardadas a partir de 06/11.

**Outros achados:** Ridge teve o menor erro de fevereiro a agosto (23,1), mas 22,4 em setembro e outubro e 22,8 em 2024-2025. O quantitativo como preditor reduz o erro de todos os algoritmos, mas o híbrido segue melhor que ele. O clima previsto não melhorou de forma consistente. KNN fica perto do consumo de referência: evita erros grandes, mas capta pouco a variação diária.

**Variáveis climáticas e notas antigas:** as 11 variáveis de clima dos notebooks anteriores foram testadas como previsão disponível em cada sexta-feira, com dados do Open-Meteo Previous Runs, e as notas de proteína e preparo foram recalculadas só com o passado. No ExtraTrees, de fevereiro a agosto, o erro foi de 24,6 na base, 24,5 com clima previsto, 24,4 só com notas e 24,1 com notas e clima. Em 2025, ficou entre 20,3 e 20,7, e em setembro e outubro subiu de 13,2 para 13,9 a 14,2. Nenhuma opção passou pela regra de 0,5 refeição na média de 2025 e fevereiro a agosto. No T+2, a mesma comparação não é possível, porque o histórico de previsões do Previous Runs vai até 7 dias. A pasta `comparacao_clima`, fora do repositório, guarda o resultado.

**Matriz ampliada com clima e notas:** somando as rodadas em cópia, foram 1.955 combinações no T+1 e no T+2, com todos os algoritmos, sem margem, com margens e no híbrido. Entraram as 16 variáveis climáticas previstas, as 4 mais correlacionadas e as notas de proteína, preparo e prato. Nenhuma opção superou a versão congelada com ganho de 0,5 refeição no critério até agosto. Os cenários com clima têm 2024 parcialmente limitado, porque as previsões guardadas só começam nesse ano. Os resultados ficam em `comparacao_clima`, fora do repositório.

## Limites que continuam valendo

1. **O período de desenvolvimento é retrospectivo.** 2026 foi observado nas etapas exploratórias. Limites como 60%, 140%, chuva forte e janelas foram definidos olhando os dados, inclusive 2026. Confirmação independente exige previsões guardadas antes do serviço.
2. **A vantagem sobre o consumo de referência varia por ano.** Em 2024 e 2025 ela é pequena, e em setembro e outubro há empate. A escolha olhou muitas opções, o que tende a favorecer a vencedora.
3. **O calendário usado é o atual.** Em 2024, as mudanças da greve podem não ser conhecidas nas datas simuladas.
4. **Quando o quantitativo foi escrito não está registrado.** Assume-se que estava disponível na sexta-feira e que não foi revisto depois do serviço.
5. **"Sem atípicos" é definido depois do fato.** Os resultados aparecem sempre também com todos os dias.
6. **A avaliação cobre um ano e um restaurante.** Não há medição de desperdício físico, economia ou pessoas sem refeição.
7. **Dados suspeitos seguem na base:** os registros de consumo de 2024-03 e 2024-07 merecem conferência com a instituição.
8. **O cenário 2 usa cardápio nem sempre disponível:** a data de publicação não está registrada, o que pesa mais nas compras.
