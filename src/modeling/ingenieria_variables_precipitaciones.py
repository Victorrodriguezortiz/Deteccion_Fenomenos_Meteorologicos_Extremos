import numpy as np
import pandas as pd
import xarray as xr

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


VARIABLES_BASE_PRECIPITACION = [
    "tp_mm", "cp_mm", "lsp_mm",
    "tcwv", "msl",
    "u10", "v10",
    "u_850", "v_850",
    "tcc", "z", "skt",
]


def _estandarizar_dataset(ds):
    renombres = {}

    if "valid_time" in ds.coords:
        renombres["valid_time"] = "time"
    if "latitude" in ds.coords:
        renombres["latitude"] = "lat"
    if "longitude" in ds.coords:
        renombres["longitude"] = "lon"

    if renombres:
        ds = ds.rename(renombres)

    if "pressure_level" in ds.dims:
        ds = ds.squeeze("pressure_level", drop=True)

    return ds


def _validar_dimensiones(ds):
    requeridas = {"time", "lat", "lon"}
    faltantes = requeridas.difference(ds.dims)

    if faltantes:
        raise ValueError(
            "El dataset debe tener dimensiones time, lat y lon. "
            f"Faltan: {sorted(faltantes)}"
        )


def _filtrar_variables_base(ds, variables_base):
    if variables_base is None:
        variables_base = VARIABLES_BASE_PRECIPITACION

    variables = [v for v in variables_base if v in ds.data_vars]

    if not variables:
        raise ValueError("No se encontro ninguna variable base en el dataset.")

    return variables


def crear_features_horarias_precipitacion(ds, variables_base=None):
    """
    Anade ingenieria de variables horaria para precipitacion extrema.

    Las variables usan solo informacion pasada: lags, ventanas rolling
    desplazadas, diferencias temporales e interacciones meteorologicas simples.
    """
    ds = _estandarizar_dataset(ds)
    _validar_dimensiones(ds)

    variables_base = _filtrar_variables_base(ds, variables_base)
    features = []

    print("Variables base usadas:")
    print(variables_base)

    lags_cortos = [1, 2, 3, 4, 6, 8, 12, 24]
    lags_largos = [36, 48, 72, 96, 120, 144, 168]
    lags = sorted(set(lags_cortos + lags_largos))

    print("Creando lags simples...")
    for var in variables_base:
        for lag in lags:
            nombre = f"{var}_lag{lag}h"
            ds[nombre] = ds[var].shift(time=lag)
            features.append(nombre)

    print("Creando rolling means/acumulados/maximos...")
    ventanas = [2, 4, 6, 8, 12, 24, 48, 72]

    for var in variables_base:
        base_pasada = ds[var].shift(time=1)

        for w in ventanas:
            min_periods = max(1, w // 2)

            nombre_mean = f"{var}_mean_{w}h"
            ds[nombre_mean] = base_pasada.rolling(
                time=w,
                min_periods=min_periods,
            ).mean()
            features.append(nombre_mean)

            if var in ["tp_mm", "cp_mm", "lsp_mm", "tp", "cp", "lsp"]:
                nombre_accum = f"{var}_accum_{w}h"
                ds[nombre_accum] = base_pasada.rolling(
                    time=w,
                    min_periods=min_periods,
                ).sum()
                features.append(nombre_accum)

                nombre_max = f"{var}_max_{w}h"
                ds[nombre_max] = base_pasada.rolling(
                    time=w,
                    min_periods=min_periods,
                ).max()
                features.append(nombre_max)

    print("Creando diferencias y tendencias...")
    pares_diff = [
        (1, 3), (1, 6), (1, 8), (1, 12),
        (1, 24), (6, 24), (24, 72), (24, 168),
    ]

    variables_tendencia = [
        v for v in ["msl", "tcwv", "tcc", "z", "u_850", "v_850", "skt"]
        if v in ds.data_vars
    ]

    for var in variables_tendencia:
        for lag_reciente, lag_antiguo in pares_diff:
            nombre = f"{var}_diff_{lag_antiguo}h_to_{lag_reciente}h"
            ds[nombre] = (
                ds[var].shift(time=lag_reciente)
                - ds[var].shift(time=lag_antiguo)
            )
            features.append(nombre)

    if "msl" in ds.data_vars:
        for lag_antiguo in [3, 6, 8, 12, 24, 48]:
            nombre = f"msl_drop_{lag_antiguo}h"
            ds[nombre] = ds["msl"].shift(time=lag_antiguo) - ds["msl"].shift(time=1)
            features.append(nombre)

    print("Creando variables de viento y humedad...")
    if "u_850" in ds.data_vars and "v_850" in ds.data_vars:
        ds["wind850_speed_lag1h"] = np.sqrt(
            ds["u_850"].shift(time=1) ** 2 + ds["v_850"].shift(time=1) ** 2
        )
        ds["wind850_speed_mean_6h"] = ds["wind850_speed_lag1h"].rolling(
            time=6,
            min_periods=3,
        ).mean()
        features += ["wind850_speed_lag1h", "wind850_speed_mean_6h"]

    if "u10" in ds.data_vars and "v10" in ds.data_vars:
        ds["wind10_speed_lag1h"] = np.sqrt(
            ds["u10"].shift(time=1) ** 2 + ds["v10"].shift(time=1) ** 2
        )
        features.append("wind10_speed_lag1h")

    if "tcwv" in ds.data_vars and "u_850" in ds.data_vars and "v_850" in ds.data_vars:
        ds["moisture_flux_u_lag1h"] = (
            ds["tcwv"].shift(time=1) * ds["u_850"].shift(time=1)
        )
        ds["moisture_flux_v_lag1h"] = (
            ds["tcwv"].shift(time=1) * ds["v_850"].shift(time=1)
        )
        ds["moisture_flux_v_mean_6h"] = ds["moisture_flux_v_lag1h"].rolling(
            time=6,
            min_periods=3,
        ).mean()
        features += [
            "moisture_flux_u_lag1h",
            "moisture_flux_v_lag1h",
            "moisture_flux_v_mean_6h",
        ]

    print("Creando interacciones fisicas...")
    if "tcwv" in ds.data_vars and "tcc" in ds.data_vars:
        ds["tcwv_x_tcc_lag1h"] = ds["tcwv"].shift(time=1) * ds["tcc"].shift(time=1)
        features.append("tcwv_x_tcc_lag1h")

    if "tcwv" in ds.data_vars and "msl" in ds.data_vars:
        ds["tcwv_x_msl_drop_6h"] = ds["tcwv"].shift(time=1) * (
            ds["msl"].shift(time=6) - ds["msl"].shift(time=1)
        )
        features.append("tcwv_x_msl_drop_6h")

    var_tp = "tp_mm" if "tp_mm" in ds.data_vars else "tp"
    if var_tp in ds.data_vars and "tcwv" in ds.data_vars:
        ds["tp_accum_6h_x_tcwv"] = (
            ds[var_tp].shift(time=1).rolling(time=6, min_periods=3).sum()
            * ds["tcwv"].shift(time=1)
        )
        features.append("tp_accum_6h_x_tcwv")

    if "tcc" in ds.data_vars and "z" in ds.data_vars:
        ds["tcc_x_z_lag1h"] = ds["tcc"].shift(time=1) * ds["z"].shift(time=1)
        features.append("tcc_x_z_lag1h")

    print("Creando variables temporales...")
    hour = ds["time"].dt.hour
    month = ds["time"].dt.month
    dayofyear = ds["time"].dt.dayofyear

    ds["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    ds["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    ds["month_sin"] = np.sin(2 * np.pi * month / 12)
    ds["month_cos"] = np.cos(2 * np.pi * month / 12)
    ds["doy_sin"] = np.sin(2 * np.pi * dayofyear / 365)
    ds["doy_cos"] = np.cos(2 * np.pi * dayofyear / 365)

    features += [
        "hour_sin", "hour_cos",
        "month_sin", "month_cos",
        "doy_sin", "doy_cos",
    ]

    return ds, features


def _samplear_indices_target(
    ds,
    target_col="target_binario",
    max_rows=300_000,
    neg_por_pos=8,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    print("Localizando positivos y negativos para muestreo controlado...")
    target = ds[target_col].load().values.astype(bool)

    n_time, n_lat, n_lon = target.shape
    total = target.size

    pos_flat = np.flatnonzero(target.ravel())
    n_pos_total = len(pos_flat)

    if n_pos_total == 0:
        raise ValueError("No hay eventos positivos con el umbral indicado.")

    if n_pos_total > max_rows:
        pos_flat = rng.choice(pos_flat, size=max_rows, replace=False)
        n_neg_obj = 0
    else:
        n_neg_obj = min(total - n_pos_total, n_pos_total * neg_por_pos)
        n_neg_obj = min(n_neg_obj, max_rows - n_pos_total)

    if n_neg_obj > 0:
        neg_sample = []
        vistos = set()
        lote = max(10_000, n_neg_obj * 2)

        while len(neg_sample) < n_neg_obj:
            candidatos = rng.integers(0, total, size=lote)
            candidatos = candidatos[~target.ravel()[candidatos]]

            for idx in candidatos:
                idx = int(idx)
                if idx not in vistos:
                    vistos.add(idx)
                    neg_sample.append(idx)
                    if len(neg_sample) >= n_neg_obj:
                        break

        flat_indices = np.concatenate([pos_flat, np.array(neg_sample, dtype=np.int64)])
    else:
        flat_indices = pos_flat

    rng.shuffle(flat_indices)
    time_idx, lat_idx, lon_idx = np.unravel_index(flat_indices, (n_time, n_lat, n_lon))

    print(f"Positivos disponibles: {n_pos_total}")
    print(f"Filas muestreadas: {len(flat_indices)}")

    return time_idx, lat_idx, lon_idx


def crear_dataset_horario_precipitacion_fe(
    ruta_nc,
    var_prec="tp_mm",
    variables_base=None,
    horizonte=24,
    umbral=4.0,
    max_rows=300_000,
    neg_por_pos=8,
    random_state=42,
    chunks=None,
    fecha_inicio=None,
    fecha_fin=None,
):
    """
    Crea un dataset horario con ingenieria fuerte para precipitacion extrema.

    La extraccion es deliberadamente conservadora con memoria: primero crea el
    target y selecciona posiciones positivas/negativas; despues calcula las
    features solo sobre esas posiciones.
    """
    if chunks is None:
        chunks = "auto"

    print("Cargando dataset horario...")
    ds = xr.open_dataset(ruta_nc, chunks=chunks)
    ds = _estandarizar_dataset(ds)
    _validar_dimensiones(ds)

    if fecha_inicio is not None or fecha_fin is not None:
        ds = ds.sel(time=slice(fecha_inicio, fecha_fin))
        print(
            "Ventana temporal aplicada:",
            str(ds["time"].min().values),
            "->",
            str(ds["time"].max().values),
        )

    if var_prec not in ds.data_vars:
        raise ValueError(f"La variable de precipitacion {var_prec!r} no existe.")

    print("Creando target futuro...")
    ds["target_regresion"] = ds[var_prec].shift(time=-horizonte)
    ds["target_binario"] = (ds["target_regresion"] >= umbral).astype("int8")

    ds, features = crear_features_horarias_precipitacion(ds, variables_base)

    lags = [1, 2, 3, 4, 6, 8, 12, 24, 36, 48, 72, 96, 120, 144, 168]
    ventanas = [2, 4, 6, 8, 12, 24, 48, 72]
    max_lag = max(max(lags), max(ventanas))
    ds = ds.isel(time=slice(max_lag, -horizonte))

    time_idx, lat_idx, lon_idx = _samplear_indices_target(
        ds,
        target_col="target_binario",
        max_rows=max_rows,
        neg_por_pos=neg_por_pos,
        random_state=random_state,
    )

    print("Extrayendo DataFrame final de muestras...")
    vars_finales = features + ["target_regresion", "target_binario"]
    ds_sample = ds[vars_finales].isel(
        time=xr.DataArray(time_idx, dims="sample"),
        lat=xr.DataArray(lat_idx, dims="sample"),
        lon=xr.DataArray(lon_idx, dims="sample"),
    )

    df = ds_sample.to_dataframe().reset_index().dropna()

    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)

    print("\nDistribucion tras muestreo:")
    print(df["target_binario"].value_counts())
    print(df["target_binario"].value_counts(normalize=True))
    print(f"Filas finales: {len(df)}")
    print(f"Features creadas: {len(features)}")

    ds.close()

    return df, features


def split_temporal_df(df, dias_test=365, time_col="time", target_col="target_binario"):
    df = df.sort_values(time_col).copy()

    fecha_max = df[time_col].max()
    fecha_corte = fecha_max - pd.Timedelta(days=dias_test)

    df_train = df[df[time_col] < fecha_corte].copy()
    df_test = df[df[time_col] >= fecha_corte].copy()

    print("Fecha maxima:", fecha_max)
    print("Fecha corte:", fecha_corte)

    print("\nTRAIN:")
    print(df_train[target_col].value_counts())
    print(df_train[target_col].value_counts(normalize=True))

    print("\nTEST:")
    print(df_test[target_col].value_counts())
    print(df_test[target_col].value_counts(normalize=True))

    return df_train, df_test


def _samplear_df_balanceado(
    df,
    target_col,
    max_rows,
    random_state,
):
    if len(df) <= max_rows:
        return df

    df_pos = df[df[target_col] == 1]
    df_neg = df[df[target_col] == 0]

    n_pos = len(df_pos)
    n_neg = max_rows - n_pos

    if n_neg > 0:
        df_neg = df_neg.sample(n=min(n_neg, len(df_neg)), random_state=random_state)
        return pd.concat([df_pos, df_neg], axis=0)

    return df_pos.sample(n=max_rows, random_state=random_state)


def seleccionar_features_mrmr_aprox(
    df_train,
    features,
    target_col="target_binario",
    top_k=70,
    max_rows=80_000,
    random_state=42,
):
    """
    mRMR aproximado:
    - relevancia: mutual information con target
    - redundancia: correlacion media absoluta con variables ya seleccionadas
    """
    df = df_train[features + [target_col]].dropna().copy()
    df = _samplear_df_balanceado(df, target_col, max_rows, random_state)

    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target_col].astype(int)

    print(f"Filas usadas para mRMR: {len(X)}")
    print(f"Features iniciales: {len(features)}")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=features,
        index=X.index,
    )

    print("Calculando mutual information...")
    mi = mutual_info_classif(
        X_scaled,
        y,
        random_state=random_state,
        discrete_features=False,
    )

    relevancia = pd.Series(mi, index=features).sort_values(ascending=False)
    corr_abs = X_scaled.corr().abs()

    seleccionadas = []
    candidatas = list(relevancia.index)
    objetivo = min(top_k, len(candidatas))

    while len(seleccionadas) < objetivo:
        mejor_feature = None
        mejor_score = -np.inf

        for f in candidatas:
            rel = relevancia[f]
            red = corr_abs.loc[f, seleccionadas].mean() if seleccionadas else 0.0
            score = rel - red

            if score > mejor_score:
                mejor_score = score
                mejor_feature = f

        seleccionadas.append(mejor_feature)
        candidatas.remove(mejor_feature)

        if len(seleccionadas) % 10 == 0:
            print(f"Seleccionadas: {len(seleccionadas)}")

    df_resumen = pd.DataFrame({
        "feature": seleccionadas,
        "mi_relevance": [relevancia[f] for f in seleccionadas],
    })

    return seleccionadas, df_resumen, relevancia


def seleccionar_features_rfe_extratrees(
    df_train,
    features,
    target_col="target_binario",
    n_features_to_select=35,
    max_rows=120_000,
    random_state=42,
):
    df = df_train[features + [target_col]].dropna().copy()
    df = _samplear_df_balanceado(df, target_col, max_rows, random_state)

    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target_col].astype(int)

    print(f"Filas usadas para RFE: {len(X)}")
    print(f"Features entrada RFE: {len(features)}")

    estimator = ExtraTreesClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    selector = RFE(
        estimator=estimator,
        n_features_to_select=n_features_to_select,
        step=0.15,
    )
    selector.fit(X, y)

    features_sel = list(np.array(features)[selector.support_])

    ranking = (
        pd.DataFrame({
            "feature": features,
            "ranking": selector.ranking_,
            "selected": selector.support_,
        })
        .sort_values(["ranking", "feature"])
        .reset_index(drop=True)
    )

    print(f"Features seleccionadas: {len(features_sel)}")

    return features_sel, ranking, selector


def preparar_experimento_horario_precipitacion(
    ruta_nc="train_precipitaciones_horario/X_train_precipitaciones_horario.nc",
    var_prec="tp_mm",
    variables_base=None,
    horizonte=24,
    umbral=4.0,
    max_rows=300_000,
    neg_por_pos=8,
    dias_test=365,
    top_k_mrmr=70,
    n_features_rfe=35,
    max_rows_mrmr=80_000,
    max_rows_rfe=120_000,
    random_state=42,
    chunks=None,
    fecha_inicio=None,
    fecha_fin=None,
):
    """
    Ejecuta el flujo completo de preparacion:
    dataset horario FE -> split temporal -> mRMR aproximado -> RFE.
    """
    df, features = crear_dataset_horario_precipitacion_fe(
        ruta_nc=ruta_nc,
        var_prec=var_prec,
        variables_base=variables_base,
        horizonte=horizonte,
        umbral=umbral,
        max_rows=max_rows,
        neg_por_pos=neg_por_pos,
        random_state=random_state,
        chunks=chunks,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
    )

    df_train, df_test = split_temporal_df(
        df,
        dias_test=dias_test,
        time_col="time",
        target_col="target_binario",
    )

    features_mrmr, resumen_mrmr, relevancia_mi = seleccionar_features_mrmr_aprox(
        df_train=df_train,
        features=features,
        target_col="target_binario",
        top_k=top_k_mrmr,
        max_rows=max_rows_mrmr,
        random_state=random_state,
    )

    features_final, ranking_rfe, selector_rfe = seleccionar_features_rfe_extratrees(
        df_train=df_train,
        features=features_mrmr,
        target_col="target_binario",
        n_features_to_select=n_features_rfe,
        max_rows=max_rows_rfe,
        random_state=random_state,
    )

    X_train = df_train[features_final].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = df_train["target_binario"].astype(int)
    X_test = df_test[features_final].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = df_test["target_binario"].astype(int)

    return {
        "df": df,
        "df_train": df_train,
        "df_test": df_test,
        "features": features,
        "features_mrmr": features_mrmr,
        "features_final": features_final,
        "resumen_mrmr": resumen_mrmr,
        "relevancia_mi": relevancia_mi,
        "ranking_rfe": ranking_rfe,
        "selector_rfe": selector_rfe,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def entrenar_pipeline_precipitacion_horaria(
    experimento,
    modelos=None,
    optimize=False,
    threshold_metric="f2",
    beta=1.5,
    random_state=42,
):
    """
    Entrena PrecipitationEventModelPipeline usando la salida de
    preparar_experimento_horario_precipitacion.
    """
    from pipeline_modelos_precipitaciones import PrecipitationEventModelPipeline

    if modelos is None:
        modelos = [
            "Logistic Regression",
            "Random Forest",
            "Extra Trees",
            "HistGradientBoosting",
            "Neural Network",
        ]

    pipeline = PrecipitationEventModelPipeline(
        X_train=experimento["X_train"],
        y_train=experimento["y_train"],
        X_test=experimento["X_test"],
        y_test=experimento["y_test"],
        feature_names=experimento["features_final"],
        threshold_metric=threshold_metric,
        beta=beta,
        random_state=random_state,
    )

    pipeline.train(models=modelos, optimize=optimize)
    resumen = pipeline.get_summary_df()

    return pipeline, resumen


def crear_variables_secuenciales_precipitacion(
    ds,
    variables_base=None,
    incluir_derivadas=True,
):
    """
    Prepara variables para modelos secuenciales.

    Para LSTM/atencion conviene usar la serie horaria reciente de variables
    meteorologicas y unas pocas derivadas fisicas por paso temporal. No se
    crean cientos de lags, porque la ventana secuencial ya contiene el pasado.
    """
    ds = _estandarizar_dataset(ds)
    _validar_dimensiones(ds)

    variables = _filtrar_variables_base(ds, variables_base)
    features = list(variables)

    if incluir_derivadas:
        if "u_850" in ds.data_vars and "v_850" in ds.data_vars:
            ds["wind850_speed"] = np.sqrt(ds["u_850"] ** 2 + ds["v_850"] ** 2)
            features.append("wind850_speed")

        if "u10" in ds.data_vars and "v10" in ds.data_vars:
            ds["wind10_speed"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
            features.append("wind10_speed")

        if "tcwv" in ds.data_vars and "u_850" in ds.data_vars:
            ds["moisture_flux_u"] = ds["tcwv"] * ds["u_850"]
            features.append("moisture_flux_u")

        if "tcwv" in ds.data_vars and "v_850" in ds.data_vars:
            ds["moisture_flux_v"] = ds["tcwv"] * ds["v_850"]
            features.append("moisture_flux_v")

        if "msl" in ds.data_vars:
            ds["msl_drop_6h_inst"] = ds["msl"].shift(time=6) - ds["msl"]
            ds["msl_drop_12h_inst"] = ds["msl"].shift(time=12) - ds["msl"]
            features += ["msl_drop_6h_inst", "msl_drop_12h_inst"]

        if "tcwv" in ds.data_vars:
            ds["tcwv_diff_6h_inst"] = ds["tcwv"] - ds["tcwv"].shift(time=6)
            features.append("tcwv_diff_6h_inst")

        if "tcc" in ds.data_vars and "tcwv" in ds.data_vars:
            ds["tcwv_x_tcc"] = ds["tcwv"] * ds["tcc"]
            features.append("tcwv_x_tcc")

        var_tp = "tp_mm" if "tp_mm" in ds.data_vars else "tp"
        if var_tp in ds.data_vars:
            ds["tp_accum_6h_inst"] = ds[var_tp].rolling(time=6, min_periods=3).sum()
            ds["tp_max_6h_inst"] = ds[var_tp].rolling(time=6, min_periods=3).max()
            features += ["tp_accum_6h_inst", "tp_max_6h_inst"]

    hour = ds["time"].dt.hour
    month = ds["time"].dt.month
    dayofyear = ds["time"].dt.dayofyear

    ds["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    ds["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    ds["month_sin"] = np.sin(2 * np.pi * month / 12)
    ds["month_cos"] = np.cos(2 * np.pi * month / 12)
    ds["doy_sin"] = np.sin(2 * np.pi * dayofyear / 365)
    ds["doy_cos"] = np.cos(2 * np.pi * dayofyear / 365)
    features += ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"]

    return ds, features


def _samplear_indices_secuenciales(
    target,
    max_samples=80_000,
    positive_ratio=0.20,
    random_state=42,
):
    rng = np.random.default_rng(random_state)
    target_flat = target.ravel().astype(bool)

    pos_flat = np.flatnonzero(target_flat)
    neg_flat = np.flatnonzero(~target_flat)

    if len(pos_flat) == 0:
        raise ValueError("No hay eventos extremos para el umbral indicado.")

    if not 0 < positive_ratio < 1:
        raise ValueError("positive_ratio debe estar entre 0 y 1.")

    n_pos_obj = min(len(pos_flat), int(max_samples * positive_ratio))
    n_neg_obj = min(len(neg_flat), max_samples - n_pos_obj)

    if len(pos_flat) < int(max_samples * positive_ratio):
        n_pos_obj = len(pos_flat)
        n_neg_obj = min(len(neg_flat), int(round(n_pos_obj * (1 - positive_ratio) / positive_ratio)))

    pos_sel = rng.choice(pos_flat, size=n_pos_obj, replace=False)
    neg_sel = rng.choice(neg_flat, size=n_neg_obj, replace=False)

    flat_indices = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(flat_indices)

    return flat_indices, len(pos_flat), len(neg_flat)


def _crear_target_futuro_precipitacion(
    ds,
    var_prec,
    horizonte,
    umbral,
    horizonte_inicio=None,
    horizonte_fin=None,
):
    if horizonte_inicio is None and horizonte_fin is None:
        ds["target_regresion"] = ds[var_prec].shift(time=-horizonte)
        ds["target_binario"] = (ds["target_regresion"] >= umbral).astype("int8")
        return ds, horizonte

    if horizonte_inicio is None or horizonte_fin is None:
        raise ValueError(
            "Para target por ventana futura debes indicar horizonte_inicio "
            "y horizonte_fin."
        )

    if horizonte_inicio < 1 or horizonte_fin < horizonte_inicio:
        raise ValueError(
            "La ventana futura debe cumplir: 1 <= horizonte_inicio <= horizonte_fin."
        )

    print(
        "Creando target por ventana futura:",
        f"max({var_prec}) entre t+{horizonte_inicio}h y t+{horizonte_fin}h",
    )

    horizontes = range(horizonte_inicio, horizonte_fin + 1)
    futuros = xr.concat(
        [ds[var_prec].shift(time=-h) for h in horizontes],
        dim=pd.Index(list(horizontes), name="horizonte"),
    )

    ds["target_regresion"] = futuros.max(dim="horizonte")
    ds["target_binario"] = (ds["target_regresion"] >= umbral).astype("int8")

    return ds, horizonte_fin


def crear_dataset_secuencial_precipitacion(
    ruta_nc,
    var_prec="tp_mm",
    variables_base=None,
    horizonte=24,
    horizonte_inicio=None,
    horizonte_fin=None,
    umbral=4.0,
    seq_len=72,
    max_samples=80_000,
    positive_ratio=0.20,
    dias_test=365,
    random_state=42,
    chunks=None,
    fecha_inicio=None,
    fecha_fin=None,
    incluir_derivadas=True,
):
    """
    Crea un dataset para modelos LSTM/atencion.

    X tiene forma (muestras, seq_len, n_features). Cada muestra contiene las
    ultimas seq_len horas en una celda de rejilla, y la etiqueta indica si la
    precipitacion en t + horizonte supera el umbral.
    """
    if chunks is None:
        chunks = "auto"

    print("Cargando dataset horario para modelo secuencial...")
    ds = xr.open_dataset(ruta_nc, chunks=chunks)
    ds = _estandarizar_dataset(ds)
    _validar_dimensiones(ds)

    if fecha_inicio is not None or fecha_fin is not None:
        ds = ds.sel(time=slice(fecha_inicio, fecha_fin))
        print(
            "Ventana temporal aplicada:",
            str(ds["time"].min().values),
            "->",
            str(ds["time"].max().values),
        )

    if var_prec not in ds.data_vars:
        raise ValueError(f"La variable de precipitacion {var_prec!r} no existe.")

    ds, max_horizonte_target = _crear_target_futuro_precipitacion(
        ds=ds,
        var_prec=var_prec,
        horizonte=horizonte,
        umbral=umbral,
        horizonte_inicio=horizonte_inicio,
        horizonte_fin=horizonte_fin,
    )
    ds, features = crear_variables_secuenciales_precipitacion(
        ds,
        variables_base=variables_base,
        incluir_derivadas=incluir_derivadas,
    )

    margen_derivadas = 12 if incluir_derivadas else 0
    inicio = max(seq_len - 1, margen_derivadas)
    fin = -max_horizonte_target if max_horizonte_target > 0 else None
    ds_valid = ds.isel(time=slice(inicio, fin))

    print("Localizando muestras secuenciales...")
    target = ds_valid["target_binario"].load().values.astype("int8")
    flat_indices, n_pos_total, n_neg_total = _samplear_indices_secuenciales(
        target,
        max_samples=max_samples,
        positive_ratio=positive_ratio,
        random_state=random_state,
    )

    n_time, n_lat, n_lon = target.shape
    time_rel, lat_idx, lon_idx = np.unravel_index(flat_indices, (n_time, n_lat, n_lon))
    time_idx = time_rel + inicio

    y = target.ravel()[flat_indices].astype("int8")
    y_reg = ds_valid["target_regresion"].load().values.ravel()[flat_indices].astype("float32")
    times = ds["time"].values[time_idx]
    lats = ds["lat"].values[lat_idx]
    lons = ds["lon"].values[lon_idx]

    print(f"Extremos disponibles: {n_pos_total}")
    print(f"No extremos disponibles: {n_neg_total}")
    print(f"Muestras seleccionadas: {len(y)}")
    print("Distribucion final:")
    print(pd.Series(y).value_counts(normalize=True).rename("proporcion"))

    X = np.empty((len(y), seq_len, len(features)), dtype="float32")
    offsets = np.arange(seq_len - 1, -1, -1)
    seq_time_idx = time_idx[:, None] - offsets[None, :]
    lat_seq = lat_idx[:, None]
    lon_seq = lon_idx[:, None]

    print("Extrayendo secuencias...")
    for j, feature in enumerate(features):
        arr = ds[feature].load().values
        if arr.ndim == 1:
            X[:, :, j] = arr[seq_time_idx].astype("float32")
        else:
            X[:, :, j] = arr[seq_time_idx, lat_seq, lon_seq].astype("float32")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    metadata = pd.DataFrame({
        "time": times,
        "lat": lats,
        "lon": lons,
        "target_regresion": y_reg,
        "target_binario": y,
    }).sort_values("time")

    orden = metadata.index.to_numpy()
    X = X[orden]
    y = y[orden]
    y_reg = y_reg[orden]
    metadata = metadata.reset_index(drop=True)

    fecha_max = metadata["time"].max()
    fecha_corte = fecha_max - pd.Timedelta(days=dias_test)
    train_mask = metadata["time"] < fecha_corte
    test_mask = ~train_mask

    X_train = X[train_mask.values]
    y_train = y[train_mask.values]
    X_test = X[test_mask.values]
    y_test = y[test_mask.values]

    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])

    X_train = scaler.fit_transform(X_train_2d).reshape(X_train.shape).astype("float32")
    X_test = scaler.transform(X_test_2d).reshape(X_test.shape).astype("float32")

    print("Fecha maxima:", fecha_max)
    print("Fecha corte:", fecha_corte)
    print("Train:", X_train.shape, pd.Series(y_train).value_counts(normalize=True).to_dict())
    print("Test:", X_test.shape, pd.Series(y_test).value_counts(normalize=True).to_dict())

    ds.close()

    return {
        "X": X,
        "y": y,
        "y_reg": y_reg,
        "metadata": metadata,
        "features": features,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "seq_len": seq_len,
        "horizonte": horizonte,
        "horizonte_inicio": horizonte_inicio,
        "horizonte_fin": horizonte_fin,
        "umbral": umbral,
        "positive_ratio": positive_ratio,
    }


def _buscar_mejor_umbral(y_true, proba, beta=1.5):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision = precision[:-1]
    recall = recall[:-1]

    if len(thresholds) == 0:
        return 0.5

    beta2 = beta ** 2
    scores = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-10)
    return float(thresholds[int(np.argmax(scores))])


def _evaluar_predicciones_secuenciales(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, pred),
        "Precision Extremo": precision_score(y_true, pred, zero_division=0),
        "Recall Extremo": recall_score(y_true, pred, zero_division=0),
        "F1 Extremo": f1_score(y_true, pred, zero_division=0),
        "F2 Extremo": fbeta_score(y_true, pred, beta=2.0, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, proba),
        "PR-AUC": average_precision_score(y_true, proba),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def crear_modelo_lstm_atencion(
    seq_len,
    n_features,
    lstm_units=96,
    dense_units=64,
    dropout=0.30,
    learning_rate=1e-3,
):
    """
    Crea una red LSTM bidireccional con atencion temporal.
    TensorFlow se importa aqui para que el resto del modulo funcione sin el.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as exc:
        raise ImportError(
            "TensorFlow no esta instalado en este entorno. "
            "Instalalo o ejecuta esta parte en el entorno del notebook que lo tenga."
        ) from exc

    inputs = keras.Input(shape=(seq_len, n_features))

    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True)
    )(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout)(x)

    attention = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=max(8, lstm_units // 4),
        dropout=dropout,
    )(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)

    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])

    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="roc_auc", curve="ROC"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.Precision(name="precision"),
        ],
    )

    return model


def entrenar_lstm_atencion_precipitacion(
    experimento,
    lstm_units=96,
    dense_units=64,
    dropout=0.30,
    learning_rate=1e-3,
    epochs=40,
    batch_size=512,
    beta=1.5,
    validation_split=0.2,
):
    """
    Entrena el modelo LSTM + atencion y devuelve modelo, resumen y probabilidades.
    """
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "TensorFlow no esta instalado en este entorno. "
            "Ejecuta esta funcion en el entorno del notebook con TensorFlow."
        ) from exc

    X_train = experimento["X_train"]
    y_train = experimento["y_train"]
    X_test = experimento["X_test"]
    y_test = experimento["y_test"]

    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    class_weight = {
        0: len(y_train) / (2.0 * neg) if neg > 0 else 1.0,
        1: len(y_train) / (2.0 * pos) if pos > 0 else 1.0,
    }

    model = crear_modelo_lstm_atencion(
        seq_len=X_train.shape[1],
        n_features=X_train.shape[2],
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout=dropout,
        learning_rate=learning_rate,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=8,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc",
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    proba_train = model.predict(X_train, batch_size=batch_size, verbose=0).ravel()
    threshold = _buscar_mejor_umbral(y_train, proba_train, beta=beta)

    proba_test = model.predict(X_test, batch_size=batch_size, verbose=0).ravel()
    metrics = _evaluar_predicciones_secuenciales(y_test, proba_test, threshold)
    summary = pd.DataFrame(metrics, index=["LSTM Attention"]).round(4)

    return {
        "model": model,
        "history": history,
        "threshold": threshold,
        "proba_test": proba_test,
        "summary": summary,
    }


def calcular_shap_lstm_atencion(
    modelo,
    experimento,
    n_background=100,
    n_explain=300,
    set_explain="test",
    random_state=42,
):
    """
    Calcula valores SHAP para el modelo LSTM + atencion.

    Devuelve:
    - shap_values: array (muestras, seq_len, n_features)
    - X_explain: muestras explicadas
    - resumen_features: importancia media absoluta por variable
    - resumen_temporal: importancia media absoluta por posicion temporal
    - resumen_feature_lag: importancia media absoluta por variable y lag

    Nota: requiere shap instalado en el entorno del notebook.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP no esta instalado en este entorno. "
            "Instalalo con: pip install shap"
        ) from exc

    rng = np.random.default_rng(random_state)
    X_train = experimento["X_train"]
    X_base = experimento["X_test"] if set_explain == "test" else experimento["X_train"]
    features = experimento["features"]
    seq_len = X_train.shape[1]

    n_background = min(n_background, len(X_train))
    n_explain = min(n_explain, len(X_base))

    idx_background = rng.choice(len(X_train), size=n_background, replace=False)
    idx_explain = rng.choice(len(X_base), size=n_explain, replace=False)

    background = X_train[idx_background].astype("float32")
    X_explain = X_base[idx_explain].astype("float32")

    print(f"Fondo SHAP: {background.shape}")
    print(f"Muestras explicadas: {X_explain.shape}")

    try:
        explainer = shap.GradientExplainer(modelo, background)
        shap_values = explainer.shap_values(X_explain)
    except Exception as err_gradient:
        print("GradientExplainer fallo, probando DeepExplainer...")
        try:
            explainer = shap.DeepExplainer(modelo, background)
            shap_values = explainer.shap_values(X_explain)
        except Exception as err_deep:
            raise RuntimeError(
                "No se pudieron calcular SHAP con GradientExplainer ni "
                "DeepExplainer. Prueba a reducir n_background/n_explain o "
                "usa una version compatible de TensorFlow/SHAP."
            ) from err_deep

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]

    if shap_values.shape != X_explain.shape:
        raise ValueError(
            "La forma de los valores SHAP no coincide con X_explain: "
            f"{shap_values.shape} vs {X_explain.shape}"
        )

    abs_shap = np.abs(shap_values)

    resumen_features = (
        pd.DataFrame({
            "feature": features,
            "mean_abs_shap": abs_shap.mean(axis=(0, 1)),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    lags = np.arange(seq_len - 1, -1, -1)
    resumen_temporal = (
        pd.DataFrame({
            "lag_horas": lags,
            "mean_abs_shap": abs_shap.mean(axis=(0, 2)),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    matriz_feature_lag = abs_shap.mean(axis=0)
    resumen_feature_lag = (
        pd.DataFrame(
            matriz_feature_lag,
            index=[f"t-{lag}h" for lag in lags],
            columns=features,
        )
        .reset_index()
        .rename(columns={"index": "lag"})
    )

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "X_explain": X_explain,
        "idx_explain": idx_explain,
        "resumen_features": resumen_features,
        "resumen_temporal": resumen_temporal,
        "resumen_feature_lag": resumen_feature_lag,
    }


def plot_shap_lstm_resumen(shap_result, top_n=20, figsize=(10, 6)):
    """
    Grafica la importancia SHAP media absoluta por variable.
    """
    import matplotlib.pyplot as plt

    df = shap_result["resumen_features"].head(top_n).iloc[::-1]

    plt.figure(figsize=figsize)
    plt.barh(df["feature"], df["mean_abs_shap"])
    plt.xlabel("mean(|SHAP|)")
    plt.title("Importancia SHAP por variable")
    plt.tight_layout()
    plt.show()


def plot_shap_lstm_temporal(shap_result, figsize=(10, 4)):
    """
    Grafica la importancia SHAP media absoluta por lag temporal.
    """
    import matplotlib.pyplot as plt

    df = shap_result["resumen_temporal"].sort_values("lag_horas", ascending=False)

    plt.figure(figsize=figsize)
    plt.plot(df["lag_horas"], df["mean_abs_shap"], marker="o", linewidth=1.5)
    plt.gca().invert_xaxis()
    plt.xlabel("Horas antes del instante t")
    plt.ylabel("mean(|SHAP|)")
    plt.title("Importancia SHAP por posicion temporal")
    plt.tight_layout()
    plt.show()


__all__ = [
    "VARIABLES_BASE_PRECIPITACION",
    "crear_features_horarias_precipitacion",
    "crear_dataset_horario_precipitacion_fe",
    "split_temporal_df",
    "seleccionar_features_mrmr_aprox",
    "seleccionar_features_rfe_extratrees",
    "preparar_experimento_horario_precipitacion",
    "entrenar_pipeline_precipitacion_horaria",
    "crear_variables_secuenciales_precipitacion",
    "crear_dataset_secuencial_precipitacion",
    "crear_modelo_lstm_atencion",
    "entrenar_lstm_atencion_precipitacion",
    "calcular_shap_lstm_atencion",
    "plot_shap_lstm_resumen",
    "plot_shap_lstm_temporal",
]
