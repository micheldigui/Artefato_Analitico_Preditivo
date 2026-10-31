# Revisão adversarial

Revisão de dados, modelos, resultados e notebooks. As verificações automáticas ficam em `src/testar_gold.py`, `src/testar_modelagem.py` e `src/revisar_resultados.py`, com o relatório em `resultados/revisao/verificacoes.csv`.

## Verificações que passaram

- **Sem informação do futuro nos dados:** o consumo a partir de cada sexta-feira foi embaralhado, e os atributos ficaram idênticos. Um vazamento inserido de propósito foi detectado.
- **Sem informação do futuro nos modelos:** consumo e rótulos futuros embaralhados não alteram nenhuma previsão, em todos os cenários e históricos testados.
- **Valores conferidos com os CSVs:** consumo, reservas e quantitativo da Gold são iguais aos dados brutos.
- **Totais conhecidos reproduzidos:** 458 datas de reservas (2.916 a mais; 9.494 a menos) e 123 datas de quantitativo até 31/08/2026 (2.561 a mais; 663 a menos).
- **Métricas recalculadas por outro caminho:** erros médios de 2026 conferem a partir das previsões diárias.
- **Escolha fixada antes da medição:** o comitê campeão saiu de 2024 e 2025 e foi mantida mesmo com outra configuração melhor em 2026.
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
| Dependência do projeto antigo. | Removida. |

## Auditoria de modelos

Regra fixada antes dos testes: escolher com 2024 e 2025 e só trocar por ganho de pelo menos 0,5 refeição por dia. Entre opções a menos de 0,5, fica a mais simples.

- **Torneio:** 15 algoritmos, com e sem cardápio, com 12 meses e com todo o histórico (`src/torneio_ampliado.py`). Nenhum algoritmo sozinho venceu com folga, e os lineares erraram muito em 2024.
- **Campeã:** média simples de CatBoost, KNN e consumo de referência, com cenário 2 e os últimos 12 meses, para preparo e compras. Erro médio de 2024 e 2025: preparo 22,7 contra 24,2 da configuração anterior; compras 25,3 contra 27,6.
- **Reprovados:** medidas de posição no período e de mesma época do ano anterior, peso maior para dias recentes, quantitativo como entrada do modelo e correção de viés do plano (`src/experimentos_melhoria.py`).

Resultado em 2026, sem dias atípicos, medido depois de fixar o comitê:

| Decisão | Consumo de referência | Artefato | Quantitativo | Híbrido |
|---|---:|---:|---:|---:|
| Preparo | 28,0 | 25,6 | 23,7 | 20,9 |
| Compras | 28,3 | 25,6 | não existe | não existe |

- O artefato superou o consumo de referência nas duas decisões. Preparo: -2,4 (-4,7 a -0,5). Compras: -2,7 (-5,3 a -0,3).
- A diferença entre híbrido e quantitativo ainda pode ser acaso: -2,9 (-6,3 a 0,9).
- No preparo, a configuração anterior foi melhor em 2026: 24,2 contra 25,6. Ela não voltou, porque isso usaria a prova para escolher. Na média de 2024 a 2026, a campeã segue melhor: 23,6 contra 24,2.
- Nas compras, a campeã superou a configuração anterior em 2026: 25,6 contra 26,3. A média de regressão e árvores com cardápio, que não era a configuração escolhida, chegou a 23,6.

## Limites que continuam valendo

1. **A prova é retrospectiva.** 2026 foi observado nas etapas exploratórias. Limites como 60%, 140%, chuva forte e janelas foram definidos olhando os dados, inclusive 2026. Confirmação independente exige previsões guardadas antes do serviço.
2. **A vantagem sobre o consumo de referência varia por ano.** Em 2025 a diferença de preparo ainda pode ser acaso: -0,6 (-1,5 a 0,5). A escolha também olhou muitas opções nos mesmos dois anos, o que tende a favorecer a vencedora.
3. **O calendário usado é o atual.** Em 2024, as mudanças da greve podem não ser conhecidas nas datas simuladas.
4. **Quando o quantitativo foi escrito não está registrado.** Assume-se que estava disponível na sexta-feira e que não foi revisto depois do serviço.
5. **"Sem atípicos" é definido depois do fato.** Os resultados aparecem sempre também com todos os dias.
6. **A prova cobre um ano e um restaurante.** Não há medição de desperdício físico, economia ou pessoas sem refeição.
7. **Dados suspeitos seguem na base:** os registros de consumo de 2024-03 e 2024-07 merecem conferência com a instituição.
8. **O cenário 2 usa cardápio nem sempre disponível:** a data de publicação não está registrada, o que pesa mais nas compras.
