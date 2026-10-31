"""Monta e executa o notebook da camada de decisao.

Execute depois de decidir_margens.py: python src/montar_notebook_decisao.py
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "notebooks" / "05_folgas_e_contexto_para_decisao.ipynb"


def md(texto):
    return nbf.v4.new_markdown_cell(texto.strip())


def code(texto):
    return nbf.v4.new_code_cell(texto.strip())


CELULAS = [
    md("""
# 05. Margens de segurança e contexto para a decisão

O artefato apresenta três opções para cada dia e os dados que ajudam a escolher. **A decisão é da nutricionista e da gestora.**

- **Sem margem:** usa a previsão como está.
- **Margem moderada:** calculada para haver déficit potencial em cerca de 1 a cada 4 dias.
- **Margem alta:** calculada para que isso ocorra em cerca de 1 a cada 10 dias.

As margens de segurança vêm dos erros dos últimos 60 dias já conhecidos em cada sexta-feira do planejamento. Excesso e déficit são potenciais: o restaurante não pesa o que é descartado nem registra quem ficou sem refeição.
"""),
    code("""
import json
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

sys.path.append("../src")
from nomes import FOLGAS, UNIVERSOS

pd.set_option("display.max_rows", 60, "display.max_columns", 20, "display.width", 180)
SAIDA = Path("..") / "resultados" / "planejamento"
v = pd.read_csv(SAIDA / "verificacao_folgas_2026.csv", sep=";", encoding="utf-8-sig")
s = pd.read_csv(SAIDA / "verificacao_por_situacao_2026.csv", sep=";", encoding="utf-8-sig")
NOMES_OPCAO = {**FOLGAS, "nutricionista": "Quantitativo da nutricionista"}
"""),
    md("## 1. As margens de segurança protegem como prometem?\n\nA coluna **prometido** é a estimativa feita antes de cada semana; **observado** é o que aconteceu em 2026, sem dias atípicos."),
    code("""
d = v[v["universo"] == "sem_atipicos"].copy()
d["Opção"] = d["opcao"].map(NOMES_OPCAO)
d = d.rename(columns={"decisao": "Decisão", "folga_media": "Margem média", "prometido_dias_a_menos_pct": "Déficit potencial, prometido (% dos dias)",
                      "dias_a_menos_pct": "Déficit potencial, observado (% dos dias)", "a_mais": "Excesso potencial (refeições)", "a_menos": "Déficit potencial (refeições)"})
d[["Decisão", "Opção", "Margem média", "Déficit potencial, prometido (% dos dias)", "Déficit potencial, observado (% dos dias)", "Excesso potencial (refeições)", "Déficit potencial (refeições)"]].round(1)
"""),
    md("## 2. Por que a margem de segurança precisa de contexto\n\nA mesma margem de segurança não protege igual em todo dia. Em dia atípico, o consumo fica muito abaixo do previsto e qualquer margem de segurança só aumenta a diferença. `Tendência da previsão` positiva indica previsão acima do consumo."),
    code("""
t = s.rename(columns={"decisao": "Decisão", "situacao": "Situação do dia", "dias": "Dias", "erro_previsao": "Tendência da previsão (refeições)",
                      "dias_a_menos_sem_folga": "Sem margem", "dias_a_menos_moderada": "Margem moderada", "dias_a_menos_alta": "Margem alta"})
t[["Decisão", "Situação do dia", "Dias", "Tendência da previsão (refeições)", "Sem margem", "Margem moderada", "Margem alta"]].round(1)
"""),
    md("## 3. Fichas da próxima emissão\n\nExemplo do que a nutricionista e a gestora receberiam na sexta-feira mais recente."),
    code("""
arquivo = sorted(SAIDA.glob("fichas_*.md"))[-1]
display(Markdown(arquivo.read_text(encoding="utf-8")))
"""),
]


def main():
    nb = nbf.v4.new_notebook(cells=CELULAS, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(DESTINO.parent)}}).execute()
    nbf.write(nb, DESTINO)
    print(f"Notebook executado: {DESTINO}")


if __name__ == "__main__":
    main()
