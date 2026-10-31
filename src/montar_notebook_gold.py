"""Monta e executa o notebook de conferencia da camada Gold.

Execute a partir da raiz do projeto: python src/montar_notebook_gold.py
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "notebooks" / "02_gold_atributos_por_emissao.ipynb"


def md(texto):
    return nbf.v4.new_markdown_cell(texto.strip())


def code(texto):
    return nbf.v4.new_code_cell(texto.strip())


CELULAS = [
    md("""
# 02. Gold: o que se sabia em cada sexta-feira

Cada linha representa uma decisão real: em uma sexta-feira de planejamento, prever um dia específico. `gold_t1` guarda o **preparo (semana seguinte)**, e `gold_t2`, as **compras (daqui a duas semanas)**. Os termos estão no `GLOSSARIO.md`.

Todos os dias continuam na tabela. A coluna `usar_no_treino` decide o que os modelos poderão aprender, e `tratamento` permite avaliar com e sem os dias atípicos. Nenhum modelo é treinado aqui.
"""),
    code("""
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 80, "display.max_columns", 30, "display.width", 180, "display.max_colwidth", 70)
GOLD = Path("..") / "dados" / "gold"
t1 = pd.read_parquet(GOLD / "gold_t1.parquet")
t2 = pd.read_parquet(GOLD / "gold_t2.parquet")
dicionario = pd.read_csv(GOLD / "gold_dicionario_atributos.csv", sep=";", encoding="utf-8-sig")
print(f"t+1: {t1.shape[0]} linhas | t+2: {t2.shape[0]} linhas")
print(f"Emissões: {t1['origem'].nunique()} | de {t1['origem'].min().date()} a {t1['origem'].max().date()}")
dicionario
"""),
    md("## 1. Exemplo de uma emissão\n\nNa emissão de 16/10/2026, o último consumo usado é anterior à própria sexta-feira, e o nível recente vem só de dias normais."),
    code("""
t1[t1["origem"] == "2026-10-16"][["origem", "data", "antecedencia_dias", "consumo", "plano_nutricionista", "nivel_base", "nivel_10d", "nivel_mesmo_dia",
                                 "idade_informacao_dias", "dias_ate_fim_bloco", "proteina", "tratamento"]].round(1)
"""),
    md("## 2. O que entra no treino\n\nOs mesmos dias são usados nas duas janelas; muda apenas o que se sabia em cada emissão."),
    code("""
pd.DataFrame({"t+1": t1["tratamento"].value_counts(), "t+2": t2["tratamento"].value_counts()}).fillna(0).astype(int)
"""),
    code("""
for nome, g in (("t+1", t1), ("t+2", t2)):
    resumo = g[g["consumo"].notna()].groupby("ano").agg(dias=("data", "size"), no_treino=("usar_no_treino", "sum"))
    print(nome); print(resumo.to_string()); print()
"""),
    md("## 3. Cobertura dos atributos"),
    code("""
atributos = ["nivel_base", "tendencia_nivel", "dias_ate_fim_bloco", "proteina", "efeito_proteina", "efeito_preparo", "reserva_discente", "plano_nutricionista"]
com_consumo = t1[t1["consumo"].notna()]
(com_consumo.groupby("ano")[atributos].apply(lambda d: d.notna().mean()) * 100).round(1).T
"""),
    md("Os dias sem bloco letivo definido e as lacunas de cardápio ficam vazios: cada modelo decidirá como tratá-los dentro do próprio treino, sem preenchimento prévio com dados do futuro."),
    code("""
ida = t1[t1["data"] <= "2026-10-30"]["idade_informacao_dias"]
print("Idade da informação em t+1 (dias):", ida.describe(percentiles=[.5, .9, .99]).round(1).to_dict())
print("Datas futuras prontas para previsão:")
futuro = t1[t1["data"] > "2026-10-30"]
print(pd.concat([futuro, t2[t2["data"] > "2026-10-30"]])[["origem", "data", "horizonte", "nivel_base", "dias_ate_fim_bloco", "proteina"]].round(1).to_string(index=False))
"""),
    md("""
## 4. Previsão do tempo disponível na emissão

As colunas `previsao_` trazem o que a previsão dizia naquela sexta-feira, recuperado dos arquivos da Open-Meteo. A tabela compara essa previsão com o clima realizado e mostra se a chuva acompanha o consumo. `consumo_vs_*` usa só dias que entram no treino.
"""),
    code("""
def avaliar_previsao(g, nome, limite=15):
    d = g[g["previsao_chuva_mm"].notna() & g["oraculo_chuva_mm"].notna()]
    evento, alerta = d["oraculo_chuva_mm"] >= limite, d["previsao_chuva_mm"] >= limite
    treino = d[d["usar_no_treino"] & d["alvo_relativo"].notna()]
    return {
        "janela": nome, "dias": len(d), "primeira_data": d["data"].min().date(),
        "corr_previsao_vs_chuva_real": round(d["previsao_chuva_mm"].corr(d["oraculo_chuva_mm"]), 2),
        "dias_com_15mm": int(evento.sum()), "alertas": int(alerta.sum()), "alertas_certos": int((evento & alerta).sum()),
        "consumo_vs_chuva_real": round(treino["oraculo_chuva_mm"].corr(treino["alvo_relativo"]), 3),
        "consumo_vs_previsao": round(treino["previsao_chuva_mm"].corr(treino["alvo_relativo"]), 3),
    }

pd.DataFrame([avaliar_previsao(t1, "Preparo, desde 2024"), avaliar_previsao(t1[t1["data"] >= t2.loc[t2["previsao_chuva_mm"].notna(), "data"].min()], "Preparo, mesmas datas de compras"), avaliar_previsao(t2, "Compras")]).set_index("janela").T
"""),
    code("""
atipicos = pd.to_datetime(["2026-09-09", "2026-09-10", "2026-09-11", "2026-09-15", "2026-09-23"])
cols = ["origem", "data", "consumo", "oraculo_chuva_mm", "previsao_chuva_mm"]
a = t1[t1["data"].isin(atipicos)][cols].rename(columns={"previsao_chuva_mm": "previsto_t1"})
b = t2[t2["data"].isin(atipicos)][["data", "previsao_chuva_mm"]].rename(columns={"previsao_chuva_mm": "previsto_t2"})
a.merge(b, on="data").round(1)
"""),
    md("""
## 5. Limites conhecidos

- **Previsão do tempo:** em t+1 há histórico desde 2024. Em t+2 há poucas rodadas, porque o serviço guarda um período curto, e a previsão quase não acompanha a chuva real nessa antecedência. `coletar_previsoes_clima.py` precisa rodar toda semana para preservar as rodadas.
- **Clima realizado:** as colunas `oraculo_` não estavam disponíveis na emissão e ficam fora das previsões.
- **Cardápio em t+2:** a base não registra quando cada cardápio foi publicado. Por isso os modelos de compras serão avaliados com e sem esses atributos.
- **Calendário:** a Gold usa a versão atual, que pode ter sido ajustada depois de publicada.
- **Plano da nutricionista:** existe só em 2026, então os modelos não o aprendem a partir das reservas discentes. As duas colunas permanecem separadas.
"""),
]


def main():
    nb = nbf.v4.new_notebook(cells=CELULAS, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(DESTINO.parent)}}).execute()
    nbf.write(nb, DESTINO)
    print(f"Notebook executado: {DESTINO}")


if __name__ == "__main__":
    main()
