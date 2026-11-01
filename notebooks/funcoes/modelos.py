"""Algoritmos, modelo congelado e previsao de uma sexta-feira."""
import numpy as np
import pandas as pd
import warnings
from catboost import CatBoostRegressor
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

from .configuracao import ANO_ANTERIOR_E_ATUAL, CAMPEAO, LIMITES_ALVO


SEMENTE = 42


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


def construir_modelo(nome):
    if nome == "regressao":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=30.0))
    if nome == "regressao_robusta":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), HuberRegressor(alpha=1.0, epsilon=1.35, max_iter=1000))
    if nome == "arvores":
        return HistGradientBoostingRegressor(loss="absolute_error", max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=20, l2_regularization=1.0, random_state=42)
    if nome == "catboost":
        return CatBoostRegressor(loss_function="MAE", iterations=200, learning_rate=0.05, depth=4, random_seed=42, verbose=0, allow_writing_files=False, thread_count=1)
    if nome in ("extratrees", CAMPEAO):
        return ALGORITMOS["ExtraTrees"]()
    if nome == "vizinhos":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), KNeighborsRegressor(n_neighbors=15, weights="distance"))
    raise ValueError(nome)


def arredondar_refeicoes(valores):
    """Refeicao e unidade inteira: de ,5 para cima arredonda para cima."""
    return np.floor(valores + 0.5)


def matriz(tabela, colunas):
    return tabela[colunas].astype(float)


def dados_de_treino(gold, origem, colunas, meses=None):
    treino = gold[(gold["data"] < origem) & gold["usar_no_treino"] & gold["alvo_relativo"].notna()]
    if meses == ANO_ANTERIOR_E_ATUAL:
        treino = treino[treino["data"] >= pd.Timestamp(year=pd.Timestamp(origem).year - 1, month=1, day=1)]
    elif meses is not None:
        treino = treino[treino["data"] >= origem - pd.DateOffset(months=meses)]
    clima = [c for c in colunas if c.startswith("previsao_")]
    return treino.dropna(subset=clima) if clima else treino


def prever_emissao(gold, origem, colunas, nome_modelo, meses=None):
    """Treina com o passado da emissao e devolve a previsao, em refeicoes."""
    alvo = gold[gold["origem"] == origem]
    treino = dados_de_treino(gold, origem, colunas, meses)
    assert treino["data"].max() < origem
    modelo = construir_modelo(nome_modelo)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo.fit(matriz(treino, colunas), treino["alvo_relativo"].clip(*LIMITES_ALVO))
    return pd.Series(np.maximum(modelo.predict(matriz(alvo, colunas)), 0) * alvo["nivel_base"].to_numpy(), index=alvo.index)


def prever_serie(gold, origens, colunas, nome, meses):
    return pd.concat([prever_emissao(gold, origem, colunas, nome, meses) for origem in origens])


def peso_adaptativo(passado, coluna_modelo, coluna_plano, janela_dias=40, minimo_dias=10):
    """Peso do modelo que teria dado o menor erro nos dias recentes ja conhecidos."""
    recente = passado.tail(janela_dias)
    if len(recente) < minimo_dias:
        return 0.5
    pesos = np.arange(0, 1.01, 0.1)
    erros = [(p * recente[coluna_modelo] + (1 - p) * recente[coluna_plano] - recente["consumo"]).abs().mean() for p in pesos]
    return float(pesos[int(np.argmin(erros))])
