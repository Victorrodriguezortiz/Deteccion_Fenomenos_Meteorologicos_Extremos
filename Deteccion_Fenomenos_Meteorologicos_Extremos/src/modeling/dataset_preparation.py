"""
Dataset preparation helpers for tabular extreme-event models.

This module keeps notebook-style dataset construction out of the model
pipeline classes. Pipelines train models; these functions prepare the
train/test tables that feed those pipelines.
"""

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler


REPO_DIR_NAME = "Deteccion_Fenomenos_Meteorologicos_Extremos"


def get_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == REPO_DIR_NAME:
            return parent
    return current.parents[2]


REPO_ROOT = get_repo_root()
DATA_DIR = REPO_ROOT / "data" / "datos_tfg"


def resolver_ruta(ruta: Union[str, Path]) -> str:
    ruta = Path(ruta)
    if ruta.is_absolute():
        return str(ruta)
    return str(DATA_DIR / ruta)


def estandarizar_coords(ds: xr.Dataset) -> xr.Dataset:
    renombres = {}
    for nombre in list(ds.dims) + list(ds.coords):
        nombre_lower = nombre.lower()
        if nombre_lower in {"t", "time_utc", "date", "valid_time"}:
            renombres[nombre] = "time"
        elif nombre_lower in {"latitude", "y"}:
            renombres[nombre] = "lat"
        elif nombre_lower in {"longitude", "x"}:
            renombres[nombre] = "lon"

    if renombres:
        ds = ds.rename(renombres)

    if "pressure_level" in ds.dims:
        ds = ds.squeeze("pressure_level", drop=True)
    if "pressure_level" in ds.coords:
        ds = ds.drop_vars("pressure_level", errors="ignore")

    return ds


def _variables_numericas_lageables(
    ds: xr.Dataset,
    exclude: Iterable[str],
    variables_candidatas: Optional[Iterable[str]] = None,
) -> list:
    if variables_candidatas is None:
        variables = [v for v in ds.data_vars if v not in exclude]
    else:
        variables = [v for v in variables_candidatas if v in ds.data_vars and v not in exclude]

    return [v for v in variables if np.issubdtype(ds[v].dtype, np.number)]


def seleccionar_lags_por_correlacion_espacial(
    ds: xr.Dataset,
    var_target: str,
    variables_candidatas: Iterable[str],
    max_lags: int = 30,
    umbral_corr_min: float = 0.20,
    umbral_corr_alto: float = 0.60,
    ventana_exclusion: int = 5,
    umbral_corr_local: float = 0.30,
) -> Dict[str, list]:
    """
    Select informative lags by measuring pixel-wise temporal correlation.

    For each candidate variable and lag, the score is the fraction of pixels
    whose absolute correlation with the target is above ``umbral_corr_local``.
    Strong variables keep one best lag; moderate variables keep up to three
    separated lags.
    """
    seleccion = {}

    print("\nAnalizando correlaciones temporales espaciales...")
    for var in variables_candidatas:
        print(f"\n  > Analizando variable: {var}")
        candidatos = []

        for lag in range(1, max_lags + 1):
            shifted = ds[var].shift(time=lag)
            mapa_corr = xr.corr(shifted, ds[var_target], dim="time")
            score = (abs(mapa_corr) > umbral_corr_local).mean().values
            score = float(score) if not np.isnan(score) else 0.0
            candidatos.append((lag, score))

        ranking = sorted(candidatos, key=lambda x: x[1], reverse=True)
        mejor_score = ranking[0][1] if ranking else 0.0

        if mejor_score < umbral_corr_min:
            print(f"    descartada: score max {mejor_score:.4f} < {umbral_corr_min}")
            continue

        seleccionados = []
        if mejor_score >= umbral_corr_alto:
            seleccionados = [ranking[0][0]]
            print(f"    senal alta ({mejor_score:.4f}) -> lag unico: {seleccionados[0]}")
        else:
            for lag, score in ranking:
                if score >= umbral_corr_min and all(
                    abs(lag - elegido) > ventana_exclusion for elegido in seleccionados
                ):
                    seleccionados.append(lag)
                if len(seleccionados) == 3:
                    break

            detalle = ", ".join(
                f"{lag}({next(score for lag_ref, score in ranking if lag_ref == lag):.4f})"
                for lag in seleccionados
            )
            print(f"    senal media ({mejor_score:.4f}) -> lags: [{detalle}]")

        seleccion[var] = seleccionados

    return seleccion


def eliminar_multicolinealidad(
    df: pd.DataFrame,
    features: list,
    target_col: str = "target_regresion",
    corr_threshold: float = 0.95,
) -> Tuple[pd.DataFrame, list, list]:
    """
    Drop highly correlated features, keeping the one most correlated with target.
    """
    features = list(dict.fromkeys(features))
    corr_matrix = df[features].corr()
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

    print(f"  Pares altamente correlacionados: {high_corr_pairs}")
    variables_a_descartar = set()

    for var1, var2 in high_corr_pairs:
        if var1 in variables_a_descartar or var2 in variables_a_descartar:
            continue

        corr_var1 = float(abs(df[var1].corr(df[target_col])))
        corr_var2 = float(abs(df[var2].corr(df[target_col])))

        if corr_var1 >= corr_var2:
            descartar, ganadora = var2, var1
        else:
            descartar, ganadora = var1, var2

        variables_a_descartar.add(descartar)
        print(f"    {descartar} eliminada; redundante con {ganadora}")

    features_finales = [f for f in features if f not in variables_a_descartar]
    df = df.drop(columns=list(variables_a_descartar), errors="ignore")

    return df, features_finales, high_corr_pairs


def generar_dataset_final_ml_limpio_v3(
    ruta_nc: Union[str, Path],
    var_target: str = "mx2t",
    p_umbral: float = 0.95,
    max_lags: int = 30,
    dias_test: int = 365,
    variables_candidatas: Optional[Iterable[str]] = None,
    umbral_corr_min: float = 0.20,
    umbral_corr_alto: float = 0.60,
    ventana_exclusion: int = 5,
    umbral_corr_local: float = 0.30,
    corr_multicolinealidad: float = 0.95,
    escalar: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, list, Optional[StandardScaler]]:
    """
    Build a clean train/test table with optimized spatial-temporal lags.

    The target threshold is calculated only on the train period to avoid
    leakage. The binary label is 1 when ``var_target`` is above that train
    percentile.
    """
    ruta_nc = resolver_ruta(ruta_nc)
    print(f"Iniciando seleccion optimizada para: {var_target}")
    print(f"Cargando dataset: {ruta_nc}")

    ds = xr.open_dataset(ruta_nc)
    ds = estandarizar_coords(ds)

    if var_target not in ds.data_vars:
        raise ValueError(f"No existe {var_target}. Variables disponibles: {list(ds.data_vars)}")
    if not {"time", "lat", "lon"}.issubset(set(ds.dims) | set(ds.coords)):
        raise ValueError("El dataset debe contener time, lat y lon.")

    ds["target_regresion"] = ds[var_target]

    fecha_corte_target = pd.to_datetime(ds.time.max().values) - pd.Timedelta(days=dias_test)
    ds_train_ref = ds.sel(time=ds.time < np.datetime64(fecha_corte_target))

    dims_umbral = [d for d in ds_train_ref[var_target].dims if d in ["time", "lat", "lon"]]
    umbral_global = ds_train_ref[var_target].quantile(p_umbral, dim=dims_umbral)
    ds["target_binario"] = (ds[var_target] > umbral_global).astype("int8")

    print(f"Umbral global P{int(p_umbral * 100)} calculado solo en train: {float(umbral_global.values):.4f}")

    month = ds.time.dt.month
    week = ds.time.dt.isocalendar().week.astype(float)

    ds["month_sin"] = np.sin(2 * np.pi * month / 12)
    ds["month_cos"] = np.cos(2 * np.pi * month / 12)
    ds["week_sin"] = np.sin(2 * np.pi * week / 52)
    ds["week_cos"] = np.cos(2 * np.pi * week / 52)

    exclude = {
        "target_regresion",
        "target_binario",
        "month_sin",
        "month_cos",
        "week_sin",
        "week_cos",
        "lat",
        "lon",
        "time",
        "pressure_level",
    }

    variables_para_lags = _variables_numericas_lageables(ds, exclude, variables_candidatas)

    final_features = ["month_sin", "month_cos", "week_sin", "week_cos", "lat", "lon"]
    ds_new = ds[
        [
            "target_regresion",
            "target_binario",
            "month_sin",
            "month_cos",
            "week_sin",
            "week_cos",
        ]
    ].copy()

    lags_por_variable = seleccionar_lags_por_correlacion_espacial(
        ds=ds,
        var_target="target_regresion",
        variables_candidatas=variables_para_lags,
        max_lags=max_lags,
        umbral_corr_min=umbral_corr_min,
        umbral_corr_alto=umbral_corr_alto,
        ventana_exclusion=ventana_exclusion,
        umbral_corr_local=umbral_corr_local,
    )

    for var, lags in lags_por_variable.items():
        for lag in lags:
            nombre = f"{var}_lag{lag}"
            ds_new[nombre] = ds[var].shift(time=lag)
            final_features.append(nombre)

    print("\nConvirtiendo dataset a tabla...")
    df = ds_new.isel(time=slice(max_lags, None)).to_dataframe().reset_index()

    cols_mantener = [
        "time",
        "lat",
        "lon",
        "target_regresion",
        "target_binario",
    ] + [f for f in final_features if f not in ["lat", "lon"]]

    df = df[cols_mantener].dropna()
    df = df.loc[:, ~df.columns.duplicated()].copy()
    final_features = list(dict.fromkeys(final_features))

    print("\nVerificando multicolinealidad...")
    df, final_features, _ = eliminar_multicolinealidad(
        df=df,
        features=final_features,
        target_col="target_regresion",
        corr_threshold=corr_multicolinealidad,
    )

    print("\nRealizando split temporal...")
    df = df.sort_values("time")
    fecha_corte = df["time"].max() - pd.Timedelta(days=dias_test)
    df_train = df[df["time"] < fecha_corte].copy()
    df_test = df[df["time"] >= fecha_corte].copy()

    scaler = None
    if escalar:
        print("\nEscalando variables...")
        scaler = StandardScaler()
        df_train[final_features] = scaler.fit_transform(df_train[final_features])
        df_test[final_features] = scaler.transform(df_test[final_features])

    print(f"\nDataset finalizado con {len(final_features)} variables predictoras.")
    print("Train:", df_train.shape, df_train["target_binario"].value_counts(normalize=True).to_dict())
    print("Test:", df_test.shape, df_test["target_binario"].value_counts(normalize=True).to_dict())

    ds.close()
    return df_train, df_test, final_features, scaler


def separar_X_y(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: Iterable[str],
    target_col: str = "target_binario",
):
    """
    Convenience helper to feed the modeling pipelines.
    """
    features = list(features)
    X_train = df_train[features]
    y_train = df_train[target_col].astype(int)
    X_test = df_test[features]
    y_test = df_test[target_col].astype(int)
    return X_train, y_train, X_test, y_test


__all__ = [
    "DATA_DIR",
    "generar_dataset_final_ml_limpio_v3",
    "seleccionar_lags_por_correlacion_espacial",
    "eliminar_multicolinealidad",
    "separar_X_y",
]

