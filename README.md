# Artefato analítico-preditivo para planejamento de refeições

**Autor:** Michel Silva de Souza

Apoio à nutricionista e à gestora de um restaurante institucional de ensino, nas decisões de preparo (semana seguinte) e compras (daqui a duas semanas). Pipeline em arquitetura medalhão, com aplicação Streamlit.

## Como abrir a aplicação

```bash
pip install -r requirements.txt
```

```bash
streamlit run app/main.py
```

Para o Streamlit Community Cloud, o arquivo principal é `app/main.py`.

## Figuras, tabelas e revisão

- `resultados/dissertacao/LISTA_FIGURAS_TABELAS.md`: figuras e tabelas com legendas sugeridas.
- `REVISAO.md`: revisão adversarial, correções e limites.

**Estado atual:** Bronze, Silver e Gold prontas, cenários avaliados, fichas de decisão geradas e aplicativo pronto. Os CSVs da pasta de dados são a fonte dos valores.

Tudo que a gestora e a nutricionista leem segue o [GLOSSARIO.md](GLOSSARIO.md). Tabelas, figuras e telas devem usar `src/nomes.py`, e os nomes técnicos ficam só no código.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `dados/bronze/` | Cópia exata dos brutos, com hashes em `manifesto.json`. `clima_previsoes/` guarda as previsões do tempo arquivadas. |
| `dados/silver/` | Fontes padronizadas, `silver_fato_diaria` e classificação dos dias. |
| `dados/gold/` | `gold_t1` (preparo), `gold_t2` (compras) e `gold_dicionario_atributos.csv`. |
| `src/` | Scripts de cada camada. |
| `notebooks/` | `01_bronze_silver_qualidade_e_atipicos`, `02_gold_atributos_por_emissao`, `03_analise_exploratoria_variaveis` e `04_resultados_cenarios`, executados. |
| `resultados/figuras/` | Figuras geradas pelo notebook. |

## Como reproduzir

A partir da raiz do projeto, em ordem:

```bash
python src/copiar_bronze.py
```

```bash
python src/coletar_previsoes_clima.py
```

```bash
python src/processar_silver.py
```

```bash
python src/detectar_atipicos.py
```

```bash
python src/processar_gold.py
```

```bash
python src/testar_gold.py
```

```bash
python src/montar_notebook_silver.py
```

```bash
python src/montar_notebook_gold.py
```

```bash
python src/montar_notebook_eda.py
```

```bash
python src/testar_modelagem.py
```

```bash
python src/avaliar_cenarios.py
```

```bash
python src/avaliar_janelas_treino.py
```

```bash
python src/consolidar_resultados.py
```

```bash
python src/montar_notebook_resultados.py
```

```bash
python src/decidir_margens.py
```

```bash
python src/montar_notebook_decisao.py
```

```bash
python src/comparar_planejamentos.py
```

```bash
python src/revisar_resultados.py
```

```bash
python src/gerar_figuras_tabelas.py
```

A coleta de previsões do tempo precisa de `config_local.json`, criado a partir de `config_local.exemplo.json` com a localização da instituição.

## Camada de decisão

`decidir_margens.py` gera três opções de margem de segurança e uma ficha por dia em `resultados/planejamento/`. As margens de segurança usam os erros dos últimos 60 dias conhecidos em cada sexta-feira. A ficha reúne calendário, cardápio, previsão do tempo, histórico, dias parecidos e pontos de atenção. O artefato não escolhe a opção.

## Avaliação dos cenários

- A cada sexta-feira, os modelos aprendem só com dias anteriores. `testar_modelagem.py` embaralha o futuro e exige previsão idêntica.
- Modelo e histórico foram escolhidos com 2024 e 2025. 2026 só mede o resultado, mesmo quando outra combinação foi melhor na prova.
- Parâmetros fixos, sem ajuste nos dados de avaliação.
- O quantitativo e as reservas não entram nos modelos. No cenário 4, combinam-se depois com a previsão, com peso definido pelos erros recentes.
- Campeã: média simples de CatBoost, KNN e consumo de referência, com cenário 2 e os últimos 12 meses, no preparo e nas compras. Sem cardápio, usa o cenário 1. Previsões são arredondadas para refeições inteiras. Detalhes em `REVISAO.md`, `resultados/modelos/` e `notebooks/04_resultados_cenarios.ipynb`.

## Decisões da Silver

- **Nada é imputado.** Consumo ausente fica vazio, inclusive em dias letivos com cardápio.
- **Reservas e plano da nutricionista ficam separados** em `reserva_discente` (até 2025) e `plano_nutricionista` (desde 2026). A Silver original usa uma só coluna, `reservas`.
- **Escopo:** segunda a sexta (`no_escopo`). Os 3 sábados com consumo ficam na base, fora do escopo.
- **Blocos letivos:** período contínuo de aulas entre pausas de ao menos 5 dias úteis. As colunas `dias_desde_inicio_bloco` e `dias_ate_fim_bloco` dependem só do calendário.
- **Clima:** a Silver guarda o clima realizado, que não estava disponível na sexta-feira de emissão.

## Regra de dias atípicos

Para cada dia, usando somente datas anteriores:

- **Referência:** média entre a mediana dos 10 dias normais anteriores e a mediana das 4 últimas ocorrências normais do mesmo dia da semana.
- **Dia baixo:** consumo abaixo de 60% da referência. **Dia alto:** acima de 140%.
- Dias baixos saem da janela de referência. Se restarem menos de 5 dias normais entre os 20 mais recentes, todos entram, para acompanhar uma queda real de nível.
- Erros de modelo não participam da regra.

O `tipo_sugerido` é uma hipótese baseada no calendário, não uma causa comprovada. Os limites 0,60 e 1,40 e as janelas foram definidos olhando os dados, sem nenhum modelo envolvido.

## Tratamento dos dias

Todos os dias permanecem na base. O uso no treino vem de `silver_dias_classificados`:

| Tratamento | Dias | No treino | Motivo |
|---|---:|---|---|
| `normal` | 582 | Sim | Rotina do serviço. |
| `previsivel_calendario` | 32 | Sim | Queda ligada ao calendário, que o modelo pode aprender. |
| `alto_mantido` | 20 | Sim | Consumo acima da referência, sem indício de erro por si só. |
| `imprevisivel` | 11 | Não | Evento externo registrado ou dia baixo sem explicação pelo calendário. |
| `fora_do_calendario` | 8 | Não | Consumo em dia não letivo. |
| `periodo_de_greve` | 3 | Não | Rotina irregular. |

A avaliação dos modelos deve sair sempre com e sem os atípicos, para que a exclusão não melhore o resultado artificialmente.

## Camada Gold

- Cada linha combina uma sexta-feira de emissão e um dia previsto, com antecedência de 3 a 7 dias em t+1 e de 10 a 14 dias em t+2.
- O consumo da própria sexta-feira não entra, porque a emissão ocorre antes do fechamento do registro.
- O nível recente usa só dias normais; o efeito do calendário fica para o modelo aprender.
- `testar_gold.py` embaralha o consumo a partir de cada emissão e exige atributos idênticos.
- Clima realizado tem prefixo `oraculo_` e fica fora das previsões.
- As colunas `previsao_` trazem a previsão de tempo disponível na sexta-feira.
- Defasagens usam as últimas ocorrências conhecidas na emissão. "Mesmo dia há 1 ocorrência" pode ser mais antigo que sete dias, porque a sexta-feira de emissão ainda não tem consumo fechado.
- Efeitos de cardápio e de evento são médias do consumo relativo calculadas só com datas anteriores à emissão, suavizadas para grupos pequenos.
- `testar_gold.py` também rejeita valores infinitos, colunas constantes e dias de treino sem consumo.
- `gold_dicionario_atributos.csv` descreve cada coluna e quando ela está disponível.

## Previsão do tempo

| Janela | Fonte Open-Meteo | Cobertura | Uso proposto |
|---|---|---|---|
| Preparo (t+1) | Previous Runs, 1 a 7 dias antes | Desde janeiro de 2024 | Testar como atributo do modelo |
| Compras (t+2) | Single Runs, rodada de 16 dias (GFS e ECMWF) | 31 sextas-feiras, desde 03/04/2026 | Só informativo; pouca relação com a chuva real |

`dados/bronze/clima_previsoes/` guarda as rodadas. **O serviço Single Runs mantém um período curto: `coletar_previsoes_clima.py` deve rodar toda semana**, e cada execução preserva o que já foi salvo.

Em t+1, usa-se uma rodada anterior à emissão quando a antecedência permite. Como o serviço vai até 7 dias, a sexta-feira prevista usa a rodada de 7 dias antes, cujas horas finais podem superar levemente o horário de emissão.
- A base não registra quando cada cardápio foi publicado, então t+2 deve ser avaliado com e sem esses atributos.
