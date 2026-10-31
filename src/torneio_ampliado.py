"""Torneio amplo de algoritmos, semana a semana, desde 2024.

Compara familias de modelos com os mesmos dados e parametros fixos. A selecao usa
apenas 2024 e 2025; 2026 fica guardado ate a escolha ser fixada. Por isso, este
script so imprime resultados ate 2025.

Execute: python src/torneio_ampliado.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from modelagem import CENARIOS, LIMITES_ALVO, TRATAMENTOS_REGULARES, dados_de_treino, matriz

RAIZ = Path(__file__).resolve().parents[1]
GOLD = RAIZ / "dados" / "gold"
SAIDA = RAIZ / "resultados" / "modelos"
INICIO = "2024-01-01"
FIM_ESCOLHA = "2025-12-31"
SEMENTE = 42
HISTORICOS = {"1_ano": 12, "tudo": None}


def linear(modelo):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), modelo)


def imputado(modelo):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), modelo)


ALGORITMOS = {
    "OLS": lambda: linear(LinearRegression()),
    "Ridge": lambda: linear(Ridge(alpha=30.0)),
    "Lasso": lambda: linear(Lasso(alpha=0.005, max_iter=5000)),
    "ElasticNet": lambda: linear(ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)),
    "Huber": lambda: linear(HuberRegressor(alpha=1.0, epsilon=1.35, max_iter=1000)),
    "KNN": lambda: linear(KNeighborsRegressor(n_neighbors=15, weights="distance")),
    "SVR": lambda: linear(SVR(C=1.0, epsilon=0.05, gamma="scale")),
    "RandomForest": lambda: imputado(RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5, random_state=SEMENTE, n_jobs=1)),
    "ExtraTrees": lambda: imputado(ExtraTreesRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5, random_state=SEMENTE, n_jobs=1)),
    "GradientBoosting": lambda: imputado(GradientBoostingRegressor(loss="absolute_error", n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, min_samples_leaf=10, random_state=SEMENTE)),
    "HistGB": lambda: HistGradientBoostingRegressor(loss="absolute_error", max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=20, l2_regularization=1.0, random_state=SEMENTE),
    "LightGBM": lambda: LGBMRegressor(objective="l1", n_estimators=150, learning_rate=0.05, max_depth=3, num_leaves=7, min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEMENTE, verbose=-1, n_jobs=1),
    "XGBoost": lambda: XGBRegressor(objective="reg:absoluteerror", n_estimators=150, learning_rate=0.05, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=SEMENTE, verbosity=0, n_jobs=1),
    "CatBoost": lambda: CatBoostRegressor(loss_function="MAE", iterations=200, learning_rate=0.05, depth=4, random_seed=SEMENTE, verbose=0, allow_writing_files=False, thread_count=1),
    "MLP": lambda: linear(MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0, max_iter=800, early_stopping=False, random_state=SEMENTE)),
}


def rodar(gold, janela, cenario, algoritmo, historico):
    colunas = CENARIOS[cenario]
    meses = HISTORICOS[historico]
    origens = sorted(gold.loc[gold["data"] >= INICIO, "origem"].unique())
    saidas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for origem in origens:
            alvo = gold[gold["origem"] == origem]
            treino = dados_de_treino(gold, origem, colunas, meses)
            assert treino["data"].max() < origem
            modelo = ALGORITMOS[algoritmo]()
            modelo.fit(matriz(treino, colunas), treino["alvo_relativo"].clip(*LIMITES_ALVO))
            previsao = np.maximum(modelo.predict(matriz(alvo, colunas)), 0) * alvo["nivel_base"].to_numpy()
            saidas.append(pd.DataFrame({"origem": origem, "data": alvo["data"].to_numpy(), "previsao": previsao}))
    return pd.concat(saidas).assign(janela=janela, cenario=cenario, algoritmo=algoritmo, historico=historico)


def main():
    golds = {j: pd.read_parquet(GOLD / f"gold_{j}.parquet") for j in ("t1", "t2")}
    tarefas = [(j, c, a, h) for j in golds for c in ("C1", "C2") for a in ALGORITMOS for h in HISTORICOS]
    print(len(tarefas), "combinacoes")
    resultados = Parallel(n_jobs=-2, verbose=1)(delayed(rodar)(golds[j], j, c, a, h) for j, c, a, h in tarefas)
    previsoes = pd.concat(resultados, ignore_index=True)
    previsoes.to_parquet(SAIDA / "torneio_previsoes.parquet", index=False)
    base = {j: g[["data", "consumo", "tratamento", "ano"]] for j, g in golds.items()}
    linhas = []
    for (j, c, a, h), g in previsoes.groupby(["janela", "cenario", "algoritmo", "historico"]):
        d = g.merge(base[j], on="data")
        d = d[d["consumo"].notna() & d["tratamento"].isin(TRATAMENTOS_REGULARES) & (d["data"] <= FIM_ESCOLHA)]
        e = (d["previsao"] - d["consumo"]).abs()
        linhas.append({"janela": j, "cenario": c, "algoritmo": a, "historico": h, "erro_2024": e[d["ano"] == 2024].mean(), "erro_2025": e[d["ano"] == 2025].mean()})
    t = pd.DataFrame(linhas)
    t["erro_medio_2024_2025"] = (t["erro_2024"] + t["erro_2025"]) / 2
    t.to_csv(SAIDA / "torneio_ate_2025.csv", sep=";", index=False, encoding="utf-8-sig")
    print(t.sort_values(["janela", "erro_medio_2024_2025"]).groupby("janela").head(14).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
