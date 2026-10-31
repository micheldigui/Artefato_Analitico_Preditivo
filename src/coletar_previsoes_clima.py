"""Salva na Bronze as previsoes de tempo arquivadas pela Open-Meteo.

- Previous Runs: previsao por hora feita 1 a 7 dias antes, desde janeiro de 2024.
  Atende o preparo (t+1).
- Single Runs: rodada completa de 16 dias. O arquivo e curto, entao cada execucao
  guarda as rodadas de sexta-feira ainda disponiveis e preserva as ja salvas.
  Atende as compras (t+2).

Usa a rodada das 00:00 UTC de sexta-feira, que ja estava pronta na emissao da tarde.
Execute a partir da raiz do projeto: python src/coletar_previsoes_clima.py
"""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "dados" / "bronze" / "clima_previsoes"
LOCAL = json.loads((RAIZ / "config_local.json").read_text(encoding="utf-8"))
URL_PREVIOUS = "https://previous-runs-api.open-meteo.com/v1/forecast"
URL_SINGLE = "https://single-runs-api.open-meteo.com/v1/forecast"
INICIO_PREVIOUS = "2024-01-01"
INICIO_BUSCA_SINGLE = "2026-01-02"
MODELOS_SINGLE = ["gfs_global", "ecmwf_ifs025"]
ANTECEDENCIAS = range(1, 8)
VARIAVEIS = ["precipitation", "temperature_2m"]


def coletar_previous_runs(fim):
    colunas = VARIAVEIS + [f"{v}_previous_day{d}" for v in VARIAVEIS for d in ANTECEDENCIAS]
    params = {**LOCAL, "hourly": ",".join(colunas), "start_date": INICIO_PREVIOUS, "end_date": fim}
    resposta = requests.get(URL_PREVIOUS, params=params, timeout=180)
    resposta.raise_for_status()
    tabela = pd.DataFrame(resposta.json()["hourly"])
    tabela.to_csv(DESTINO / "previous_runs_horario.csv", sep=";", index=False)
    return tabela


def coletar_rodada(sexta, modelo):
    params = {**LOCAL, "hourly": ",".join(VARIAVEIS), "run": f"{sexta}T00:00", "forecast_days": 16, "models": modelo}
    resposta = requests.get(URL_SINGLE, params=params, timeout=60)
    if resposta.status_code != 200:
        return None
    tabela = pd.DataFrame(resposta.json()["hourly"])
    tabela.columns = [c.replace(f"_{modelo}", "") for c in tabela.columns]
    return tabela.assign(rodada=sexta, modelo=modelo)


def coletar_single_runs(hoje):
    arquivo = DESTINO / "single_runs_horario.csv"
    salvo = pd.read_csv(arquivo, sep=";") if arquivo.exists() else pd.DataFrame(columns=["rodada", "modelo"])
    ja_salvas = set(zip(salvo["rodada"], salvo["modelo"]))
    novas = []
    disponiveis = {}
    for sexta in pd.date_range(INICIO_BUSCA_SINGLE, hoje, freq="W-FRI").strftime("%Y-%m-%d"):
        for modelo in MODELOS_SINGLE:
            if (sexta, modelo) in ja_salvas:
                disponiveis.setdefault(modelo, []).append(sexta)
                continue
            rodada = coletar_rodada(sexta, modelo)
            time.sleep(0.15)
            if rodada is not None:
                novas.append(rodada)
                disponiveis.setdefault(modelo, []).append(sexta)
    tabela = pd.concat([salvo, *novas], ignore_index=True) if novas else salvo
    tabela.to_csv(arquivo, sep=";", index=False)
    return tabela, disponiveis


def sha256(caminho):
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    hoje = datetime.now().strftime("%Y-%m-%d")
    previous = coletar_previous_runs(hoje)
    single, disponiveis = coletar_single_runs(hoje)
    resumo = {m: {"sextas": len(d), "primeira": d[0], "ultima": d[-1]} for m, d in disponiveis.items()}
    manifesto = {
        "coletado_em": datetime.now().isoformat(timespec="seconds"),
        "previous_runs_linhas": len(previous),
        "single_runs_linhas": len(single),
        "single_runs_cobertura": resumo,
        "arquivos": {p.name: sha256(p) for p in sorted(DESTINO.glob("*.csv"))},
    }
    (DESTINO / "manifesto_coleta.json").write_text(json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifesto, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
