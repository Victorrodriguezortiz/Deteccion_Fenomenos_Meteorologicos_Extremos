"""
Data Processing Module for TFG: Climate and Air Quality Analysis
================================================================================
This module contains all data cleaning, processing, and aggregation functions
for satellite data, ERA5 reanalysis, and quality control variables.

Author: Victor Rodriguez
Date: 2024-2025
"""

import xarray as xr
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
import glob
import os
from pathlib import Path


# ============================================================================
# REPOSITORY PATHS
# ============================================================================

REPO_DIR_NAME = "Deteccion_Fenomenos_Meteorologicos_Extremos"


def get_repo_root() -> Path:
    """
    Locate the project root regardless of where this module is executed from.
    """
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == REPO_DIR_NAME:
            return parent
    return current.parents[2]


REPO_ROOT = get_repo_root()
DATA_DIR = REPO_ROOT / "data" / "datos_tfg"
CLEAN_DATA_DIR = DATA_DIR / "dataset_final_limpio"
TRAIN_DATA_DIR = DATA_DIR / "train"


def resolver_ruta(ruta: Union[str, Path]) -> str:
    """
    Resolve a path/glob pattern relative to repo data if it is not absolute.

    Examples:
        resolver_ruta("satelite_s5p/S5P_NO2*.nc")
        resolver_ruta("dataset_final_limpio/no2_final_limpio.nc")
    """
    ruta = Path(ruta)
    if ruta.is_absolute():
        return str(ruta)
    return str(DATA_DIR / ruta)


def asegurar_directorio_salida(ruta_salida: Union[str, Path]) -> str:
    """
    Resolve an output path and create its parent directory.
    """
    ruta = Path(ruta_salida)
    if not ruta.is_absolute():
        ruta = DATA_DIR / ruta
    ruta.parent.mkdir(parents=True, exist_ok=True)
    return str(ruta)


def resolver_rutas_dict(rutas: Dict[str, Union[str, Path]]) -> Dict[str, str]:
    """
    Resolve a dictionary of input paths relative to repo data.
    """
    return {nombre: resolver_ruta(ruta) for nombre, ruta in rutas.items()}


# ============================================================================
# PART 1: UTILITY FUNCTIONS
# ============================================================================

def estandarizar_nombres(ds: xr.Dataset) -> xr.Dataset:
    """
    Standardize dimension names in xarray datasets.
    
    Converts common naming variations to a standard format:
    - Time dimensions: 't', 'time_utc', 'date', 'valid_time' -> 'time'
    - Latitude: 'latitude', 'y' -> 'lat'
    - Longitude: 'longitude', 'x' -> 'lon'
    
    Args:
        ds: xarray Dataset to standardize
        
    Returns:
        Dataset with standardized dimension names
    """
    renombres = {}
    for nombre in list(ds.dims) + list(ds.coords):
        nombre_lower = nombre.lower()
        if nombre_lower in ['t', 'time_utc', 'date', 'valid_time']:
            renombres[nombre] = 'time'
        elif nombre_lower in ['latitude', 'y']:
            renombres[nombre] = 'lat'
        elif nombre_lower in ['longitude', 'x']:
            renombres[nombre] = 'lon'
    
    if renombres:
        return ds.rename(renombres)
    return ds


def limpiar_dimensiones_duplicadas(ds: xr.Dataset) -> xr.Dataset:
    """
    Clean duplicate dimensions in xarray datasets.
    
    Solves: "ValueError" when resample fails due to duplicate dimensions
    like ('time', 'time', 'lat', 'lon')
    
    This happens with some CAMS/ERA5 files that have coordinate/dimension conflicts.
    
    Also handles the case where concatenation creates multiple 'time' dimensions.
    In that case, keeps the longest one (actual time points) and removes the file index.
    
    Args:
        ds: xarray Dataset to clean
        
    Returns:
        Dataset with cleaned dimensions
    """
    print("🛠️ Verificando dimensiones...")
    
    # Check for duplicate dimension names
    dim_names = list(ds.dims.keys())
    
    # Special case: Multiple 'time' dimensions
    # This happens when concatenating files: (num_files, num_timepoints, lat, lon)
    # but both are called 'time'
    if 'time' in dim_names and dim_names.count('time') > 1:
        print("⚠️  Detectadas múltiples dimensiones 'time'. Analizando estructura...")
        
        # Find all 'time' dimensions and their sizes
        time_dims = {i: (name, ds.dims[name]) for i, name in enumerate(dim_names) if name == 'time'}
        print(f"   Dimensiones 'time' encontradas: {time_dims}")
        
        if len(time_dims) == 2:
            # Get sizes
            time_sizes = list(time_dims.values())
            print(f"   Tamaños: {time_sizes}")
            
            # Keep the larger one (actual time dimension) and stack/remove the smaller
            if time_sizes[0][1] > time_sizes[1][1]:
                # First 'time' is larger - need to flatten it with second
                print(f"   Combinando {time_sizes[1][1]} archivos con {time_sizes[0][1]} puntos de tiempo...")
                # Stack the two time dimensions into one
                first_time_name = 'time_file'
                ds = ds.rename({dim_names[dim_names.index('time')]: first_time_name})
                ds = ds.stack(time=(first_time_name, 'time'))
            else:
                # Second 'time' is larger - need to flatten it with first
                print(f"   Combinando {time_sizes[0][1]} archivos con {time_sizes[1][1]} puntos de tiempo...")
                second_time_name = 'time_file'
                # Find position of second 'time' in dim_names
                second_time_pos = [i for i, n in enumerate(dim_names) if n == 'time'][1]
                ds = ds.rename({dim_names[second_time_pos]: second_time_name})
                ds = ds.stack(time=('time', second_time_name))
            
            print(f"   ✅ Dimensiones combinadas exitosamente")
        else:
            print("   ⚠️  Más de 2 dimensiones 'time' detectadas, intentando squeeze...")
            ds = ds.squeeze()
    
    elif len(dim_names) != len(set(dim_names)):
        print("⚠️  Dimensiones duplicadas detectadas. Limpiando...")
        
        # Try to clean with squeeze (removes dimensions of size 1)
        ds = ds.squeeze()
        
        # If still problematic, remove duplicate indices
        for dim in set(dim_names):
            if dim in ds.coords and dim in ds.dims:
                # Remove duplicated index values
                try:
                    if hasattr(ds.indexes[dim], 'duplicated'):
                        ds = ds.loc[{dim: ~ds.get_index(dim).duplicated()}]
                except:
                    pass
        
        print("✅ Dimensiones limpiadas correctamente")
    else:
        print("✅ Dimensiones verificadas (sin duplicados)")
    
    return ds


def _audit_variable(var: xr.DataArray, name: str) -> None:
    """
    Print audit statistics for a variable.
    
    Args:
        var: Variable to audit
        name: Name for display
    """
    nulos_totales = var.isnull().sum().values
    negativos = (var < 0).sum().values
    total_puntos = var.size
    
    print(f"""
    📊 AUDITORÍA: {name}
    -------------------------
    - Total píxeles: {total_puntos}
    - Nulos (NaNs): {nulos_totales} ({(nulos_totales/total_puntos)*100:.2f}%)
    - Valores negativos: {negativos}
    - Valor Máximo: {var.max().values}
    - Valor Mínimo: {var.min().values}
    """)


# ============================================================================
# PART 2: SATELLITE DATA CLEANING FUNCTIONS
# ============================================================================

def limpiar_rellenar_no2(no2_raw: xr.DataArray) -> xr.DataArray:
    """
    Clean and fill NO2 data from Sentinel-5P.
    
    Cleaning pipeline:
    1. Remove negative values (sensor noise)
    2. Spatial filling (neighbors 3x3)
    3. Temporal interpolation (linear, max 2 days)
    4. Temporal neighbors (forward/backward fill, max 1 day)
    5. Rolling mean (11 days)
    6. Historical mean (final fallback)
    
    Args:
        no2_raw: Raw NO2 DataArray
        
    Returns:
        Cleaned NO2 DataArray with no null values
    """
    print("🛠️ Iniciando limpieza de NO2...")
    _audit_variable(no2_raw, "NO2 INICIAL")
    
    # Step 1: Remove negative values
    no2_original = no2_raw.where(no2_raw >= 0, np.nan)
    print("✅ Paso 1: Valores negativos eliminados")
    _audit_variable(no2_original, "NO2 SIN NEGATIVOS")
    
    # Step 2: Spatial filling (3x3 neighbors)
    no2_medias_vecinos = no2_original.rolling(lat=3, lon=3, center=True, min_periods=1).mean()
    no2_espacial = xr.where(no2_original.isnull(), no2_medias_vecinos, no2_original)
    print("✅ Paso 2: Rellenado espacial completado")
    _audit_variable(no2_espacial, "NO2 RELLENADO ESPACIAL")
    
    # Step 3: Temporal interpolation (linear, limit=2)
    no2_espacial = no2_espacial.chunk({'time': -1})
    no2_interp = no2_espacial.interpolate_na(dim='time', method='linear', limit=2)
    print("✅ Paso 3: Interpolación temporal lineal completada")
    _audit_variable(no2_interp, "NO2 INTERPOLADO TEMPORAL")
    
    # Step 4: Temporal neighbors (forward/backward)
    no2_vecinos = no2_interp.bfill(dim='time', limit=1).ffill(dim='time', limit=1)
    print("✅ Paso 4: Rellenado por vecinos temporales completado")
    _audit_variable(no2_vecinos, "NO2 VECINOS TEMPORALES")
    
    # Step 5: Rolling mean (11 days)
    no2_rolling = no2_vecinos.rolling(time=11, center=True, min_periods=1).mean()
    no2_intermedio = no2_vecinos.fillna(no2_rolling)
    print("✅ Paso 5: Media móvil (11 días) completada")
    _audit_variable(no2_intermedio, "NO2 MEDIA MÓVIL")
    
    # Step 6: Historical mean (final safety net)
    media_total_pixel = no2_intermedio.mean(dim='time')
    no2_final = no2_intermedio.fillna(media_total_pixel)
    print("✅ Paso 6: Media histórica completada")
    _audit_variable(no2_final, "NO2 FINAL")
    
    print("🚀 NO2 limpio y completado al 100%.")
    return no2_final


def limpiar_rellenar_MODIS_AOD(aod_raw: xr.DataArray) -> xr.DataArray:
    """
    Clean and fill AOD (Aerosol Optical Depth) data from MODIS.
    
    Cleaning pipeline:
    1. Correct negative values to 0 and remove outliers
    2. Spatial filling (5x5 neighbors)
    3. Temporal interpolation (linear, max 2 days)
    4. Rolling mean (7 days)
    5. Historical mean (final fallback)
    
    Args:
        aod_raw: Raw AOD DataArray
        
    Returns:
        Cleaned AOD DataArray with no null values
    """
    print("🛠️ Iniciando limpieza de MODIS AOD...")
    _audit_variable(aod_raw, "AOD INICIAL")
    
    # Step 1: Physical correction (AOD >= 0)
    aod_original = aod_raw.where((aod_raw >= 0) | (aod_raw.isnull()), 0)
    print("✅ Paso 1: Valores negativos corregidos a 0")
    _audit_variable(aod_original, "AOD SIN NEGATIVOS")
    
    # Step 2: Spatial filling (5x5 neighbors)
    print("⏳ Aplicando rellenado espacial 5x5...")
    aod_medias_vecinos = aod_original.rolling(lat=5, lon=5, center=True, min_periods=1).mean()
    aod_espacial = xr.where(aod_original.isnull(), aod_medias_vecinos, aod_original)
    print("✅ Paso 2: Rellenado espacial completado")
    _audit_variable(aod_espacial, "AOD RELLENADO ESPACIAL")
    
    # Step 3: Temporal interpolation (linear, limit=2)
    aod_espacial = aod_espacial.chunk({'time': -1})
    print("⏳ Interpolación temporal...")
    aod_interp = aod_espacial.interpolate_na(dim='time', method='linear', limit=2)
    print("✅ Paso 3: Interpolación temporal completada")
    _audit_variable(aod_interp, "AOD INTERPOLADO TEMPORAL")
    
    # Step 4: Rolling mean (7 days)
    print("⏳ Calculando ventana móvil de 7 días...")
    aod_rolling = aod_interp.rolling(time=7, center=True, min_periods=1).mean()
    aod_casi_listo = aod_interp.fillna(aod_rolling)
    print("✅ Paso 4: Media móvil completada")
    _audit_variable(aod_casi_listo, "AOD MEDIA MÓVIL")
    
    # Step 5: Historical mean (final safety net)
    media_hist = aod_casi_listo.mean(dim='time')
    aod_final = aod_casi_listo.fillna(media_hist)
    print("✅ Paso 5: Media histórica completada")
    _audit_variable(aod_final, "AOD FINAL")
    
    print("🚀 MODIS AOD limpio y completado al 100%.")
    return aod_final


def limpiar_rellenar_CO(co_raw: xr.DataArray) -> xr.DataArray:
    """
    Clean and fill Sentinel-5P CO.

    CO is usually smoother and more persistent than NO2, so the temporal
    fallback uses an 11-day rolling window after local interpolation.
    """
    print("🛠️ Iniciando limpieza de CO...")
    _audit_variable(co_raw, "CO INICIAL")

    co_original = co_raw.where(co_raw >= 0, np.nan)
    print("✅ Paso 1: Valores negativos eliminados")
    _audit_variable(co_original, "CO SIN NEGATIVOS")

    co_medias_vecinos = co_original.rolling(lat=3, lon=3, center=True, min_periods=1).mean()
    co_espacial = xr.where(co_original.isnull(), co_medias_vecinos, co_original)
    print("✅ Paso 2: Rellenado espacial 3x3 completado")
    _audit_variable(co_espacial, "CO RELLENADO ESPACIAL")

    co_espacial = co_espacial.chunk({"time": -1})
    co_interp = co_espacial.interpolate_na(dim="time", method="linear", limit=2)
    print("✅ Paso 3: Interpolación temporal lineal completada")
    _audit_variable(co_interp, "CO INTERPOLADO TEMPORAL")

    co_vecinos = co_interp.ffill(dim="time", limit=1).bfill(dim="time", limit=1)
    print("✅ Paso 4: Rellenado por vecinos temporales completado")
    _audit_variable(co_vecinos, "CO VECINOS TEMPORALES")

    co_rolling = co_vecinos.rolling(time=11, center=True, min_periods=1).mean()
    co_casi_listo = co_vecinos.fillna(co_rolling)
    print("✅ Paso 5: Media móvil de 11 días completada")
    _audit_variable(co_casi_listo, "CO MEDIA MÓVIL")

    media_hist = co_casi_listo.mean(dim="time")
    co_final = co_casi_listo.fillna(media_hist)
    print("✅ Paso 6: Media histórica completada")
    _audit_variable(co_final, "CO FINAL")

    print("🚀 CO limpio y completado al 100%.")
    return co_final


def limpiar_rellenar_aerosoles(aer_raw: xr.DataArray) -> xr.DataArray:
    """
    Clean and fill Sentinel-5P aerosol index (UVAI).

    Negative values are physically meaningful in UVAI, so they are kept.
    Only clearly impossible high values are treated as missing.
    """
    print("🛠️ Iniciando limpieza de aerosoles Sentinel-5P...")
    _audit_variable(aer_raw, "AEROSOLES INICIAL")

    aer_original = aer_raw.where((aer_raw <= 10.0) | aer_raw.isnull(), np.nan)
    print("✅ Paso 1: Valores superiores a 10 eliminados; negativos preservados")
    _audit_variable(aer_original, "AEROSOLES FILTRADO FÍSICO")

    aer_medias_vecinos = aer_original.rolling(lat=5, lon=5, center=True, min_periods=1).mean()
    aer_espacial = xr.where(aer_original.isnull(), aer_medias_vecinos, aer_original)
    print("✅ Paso 2: Rellenado espacial 5x5 completado")
    _audit_variable(aer_espacial, "AEROSOLES RELLENADO ESPACIAL")

    aer_espacial = aer_espacial.chunk({"time": -1})
    aer_interp = aer_espacial.interpolate_na(dim="time", method="linear", limit=2)
    print("✅ Paso 3: Interpolación temporal lineal completada")
    _audit_variable(aer_interp, "AEROSOLES INTERPOLADO TEMPORAL")

    aer_vecinos = aer_interp.ffill(dim="time", limit=1).bfill(dim="time", limit=1)
    print("✅ Paso 4: Rellenado por vecinos temporales completado")
    _audit_variable(aer_vecinos, "AEROSOLES VECINOS TEMPORALES")

    aer_rolling = aer_vecinos.rolling(time=7, center=True, min_periods=1).mean()
    aer_casi_listo = aer_vecinos.fillna(aer_rolling)
    print("✅ Paso 5: Media móvil de 7 días completada")
    _audit_variable(aer_casi_listo, "AEROSOLES MEDIA MÓVIL")

    media_hist = aer_casi_listo.mean(dim="time")
    aer_final = aer_casi_listo.fillna(media_hist)
    print("✅ Paso 6: Media histórica completada")
    _audit_variable(aer_final, "AEROSOLES FINAL")

    print("🚀 Aerosoles limpios y completados al 100%.")
    return aer_final


LIMPIEZAS_SATELITALES = {
    "no2": {
        "patron": "satelite_s5p/S5P_NO2*.nc",
        "variable_entrada": "NO2",
        "variable_salida": "no2_limpio",
        "salida": "dataset_final_limpio/no2_final_limpio.nc",
        "funcion": limpiar_rellenar_no2,
    },
    "co": {
        "patron": "satelite_s5p/S5P_CO*.nc",
        "variable_entrada": "CO",
        "variable_salida": "co_limpio",
        "salida": "dataset_final_limpio/co_final_limpio.nc",
        "funcion": limpiar_rellenar_CO,
    },
    "aerosoles": {
        "patron": "satelite_s5p/S5P_Aerosoles*.nc",
        "variable_entrada": "AER_AI_354_388",
        "variable_salida": "aerosoles_limpio",
        "salida": "dataset_final_limpio/aerosoles_final_limpio.nc",
        "funcion": limpiar_rellenar_aerosoles,
    },
    "aod": {
        "patron": "modis_diario_full/MODIS*.nc",
        "variable_entrada": "aod",
        "variable_salida": "aod_limpio",
        "salida": "dataset_final_limpio/aod_final_limpio.nc",
        "funcion": limpiar_rellenar_MODIS_AOD,
    },
}


def procesar_producto_satelital_limpio(
    producto: str,
    ruta_entrada: Optional[Union[str, Path]] = None,
    ruta_salida: Optional[Union[str, Path]] = None,
    variable_entrada: Optional[str] = None,
    chunks: Dict = None,
) -> xr.DataArray:
    """
    Load, clean and save one satellite product.

    producto:
        "no2", "co", "aerosoles" or "aod".
    """
    producto = producto.lower()
    if chunks is None:
        chunks = {"time": -1}

    if producto not in LIMPIEZAS_SATELITALES:
        disponibles = ", ".join(sorted(LIMPIEZAS_SATELITALES))
        raise ValueError(f"Producto no soportado: {producto}. Disponibles: {disponibles}")

    cfg = LIMPIEZAS_SATELITALES[producto]
    ruta_entrada = resolver_ruta(ruta_entrada or cfg["patron"])
    ruta_salida = asegurar_directorio_salida(ruta_salida or cfg["salida"])
    variable_entrada = variable_entrada or cfg["variable_entrada"]

    print(f"📂 Cargando {producto} desde: {ruta_entrada}")
    ds = xr.open_mfdataset(ruta_entrada, chunks=chunks, combine="by_coords")
    ds = estandarizar_nombres(ds)

    if variable_entrada not in ds.data_vars:
        raise ValueError(
            f"No existe la variable {variable_entrada}. "
            f"Variables disponibles: {list(ds.data_vars)}"
        )

    da_raw = ds[variable_entrada].chunk({"time": -1})
    da_limpio = cfg["funcion"](da_raw)
    da_limpio.name = cfg["variable_salida"]

    print(f"💾 Guardando {producto} limpio en: {ruta_salida}")
    da_limpio.to_netcdf(ruta_salida)
    ds.close()

    print(f"✅ {producto} procesado correctamente.")
    return da_limpio


def procesar_satelites_calidad_aire(
    productos=("no2", "co", "aerosoles", "aod"),
    chunks: Dict = None,
) -> Dict[str, xr.DataArray]:
    """
    Clean all air-quality satellite products while keeping each product separate.
    """
    resultados = {}
    for producto in productos:
        resultados[producto] = procesar_producto_satelital_limpio(producto, chunks=chunks)
    return resultados


def limpieza_xarray_lst_celsius_solo_superficie_lst(
    ruta_lst: Union[str, Path] = "satelite_s3_lst/*.nc",
    ruta_era: Union[str, Path] = "olas_calor/*.nc",
    ruta_salida: Union[str, Path] = "dataset_final_limpio/lst_final_limpio_celsius_mascara_lst.nc",
    var_lst: str = "LST",
    var_skt: str = "skt",
    umbral_frio_c: float = 0.0,
    diferencia_nube_c: float = 10.0,
    horas_satelite=None,
    min_obs_pixel: int = 1,
) -> xr.DataArray:
    """
    Clean Sentinel-3 LST in Celsius while preserving the original LST coverage.

    The function uses ERA5 SKT only as a physically consistent fallback for
    missing/cloud-suspicious LST pixels inside the original Sentinel-3 mask.
    Pixels outside the original LST footprint remain NaN.
    """
    if horas_satelite is None:
        horas_satelite = [9, 10, 11, 21, 22, 23]

    ruta_lst = resolver_ruta(ruta_lst)
    ruta_era = resolver_ruta(ruta_era)
    ruta_salida = asegurar_directorio_salida(ruta_salida)

    print("📂 1. Cargando datos LST satelital...")
    ds_sat = xr.open_mfdataset(ruta_lst, chunks={"time": 10}, combine="by_coords")
    ds_sat = estandarizar_nombres(ds_sat)

    if var_lst not in ds_sat.data_vars:
        raise ValueError(f"No existe {var_lst}. Variables disponibles: {list(ds_sat.data_vars)}")

    lst = ds_sat[var_lst]

    print("📂 2. Cargando datos ERA5/SKT...")
    ds_era = xr.open_mfdataset(ruta_era, chunks={"time": 10}, combine="by_coords")
    ds_era = estandarizar_nombres(ds_era)

    if var_skt not in ds_era.data_vars:
        raise ValueError(f"No existe {var_skt}. Variables disponibles: {list(ds_era.data_vars)}")

    skt = ds_era[var_skt]

    print("🛠️ 3. Preparando datos en ºC...")
    media_lst = float(lst.mean(skipna=True).compute().values)
    media_skt = float(skt.mean(skipna=True).compute().values)

    if media_lst > 100:
        print("🌡️ LST parece estar en Kelvin. Convirtiendo a ºC...")
        lst = lst - 273.15
    else:
        print("✅ LST ya parece estar en ºC.")

    if media_skt > 100:
        print("🌡️ SKT parece estar en Kelvin. Convirtiendo a ºC...")
        skt = skt - 273.15
    else:
        print("✅ SKT ya parece estar en ºC.")

    lst = lst.sortby(["lat", "lon"])
    skt = skt.sortby(["lat", "lon"])

    print("🗺️ 4. Creando máscara espacial de cobertura válida de LST...")
    mascara_lst_original = lst.notnull().sum(dim="time") >= min_obs_pixel
    print(
        "   Píxeles válidos en máscara LST original:",
        int(mascara_lst_original.sum().compute().values),
    )

    print("📆 5. Agregando LST y SKT a escala diaria...")
    lst_diaria = lst.resample(time="1D").max(skipna=True)
    skt_pasadas = skt.sel(time=skt.time.dt.hour.isin(horas_satelite))
    skt_diaria = skt_pasadas.resample(time="1D").max(skipna=True)

    print("🔗 6. Alineando fechas comunes...")
    fechas_lst = pd.to_datetime(lst_diaria.time.values).normalize()
    fechas_skt = pd.to_datetime(skt_diaria.time.values).normalize()
    fechas_comunes = np.intersect1d(
        fechas_lst.values.astype("datetime64[ns]"),
        fechas_skt.values.astype("datetime64[ns]"),
    )

    if len(fechas_comunes) == 0:
        raise ValueError("No hay fechas comunes entre LST diaria y SKT diaria.")

    lst_diaria = lst_diaria.sel(time=fechas_comunes)
    skt_diaria = skt_diaria.sel(time=fechas_comunes)
    print(f"   Fechas comunes: {len(fechas_comunes)}")
    print(f"   LST grid original: lat={len(lst_diaria.lat)}, lon={len(lst_diaria.lon)}")
    print(f"   SKT grid destino:  lat={len(skt_diaria.lat)}, lon={len(skt_diaria.lon)}")

    print("🗺️ 7. Rejillando LST y máscara a la malla ERA5/SKT...")
    lst_rejilla_era = lst_diaria.interp(lat=skt_diaria.lat, lon=skt_diaria.lon, method="nearest")

    mascara_lst_era = (
        mascara_lst_original.astype("float32")
        .interp(lat=skt_diaria.lat, lon=skt_diaria.lon, method="nearest")
        >= 0.5
    )

    lst_rejilla_era = lst_rejilla_era.reindex_like(skt_diaria, method="nearest")
    mascara_lst_era = mascara_lst_era.reindex_like(skt_diaria, method="nearest")

    nulos_antes_total = int(lst_rejilla_era.isnull().sum().compute().values)
    nulos_antes_dentro = int(lst_rejilla_era.where(mascara_lst_era).isnull().sum().compute().values)
    sospechosos_antes = int(((lst_rejilla_era < umbral_frio_c) & mascara_lst_era).sum().compute().values)
    pixeles_mascara_era = int(mascara_lst_era.sum().compute().values)

    print("📊 Antes de limpieza:")
    print(f"   Píxeles dentro de máscara LST en rejilla ERA5: {pixeles_mascara_era}")
    print(f"   Nulos totales LST reescalada: {nulos_antes_total}")
    print(f"   Nulos dentro de máscara LST: {nulos_antes_dentro}")
    print(f"   Sospechosos dentro de máscara < {umbral_frio_c} ºC: {sospechosos_antes}")

    print("☁️ 8. Detectando nulos y posibles nubes dentro de la máscara LST...")
    mask_nube = (
        mascara_lst_era
        & (lst_rejilla_era < umbral_frio_c)
        & ((skt_diaria - lst_rejilla_era) > diferencia_nube_c)
    )
    mask_nulos_dentro = mascara_lst_era & lst_rejilla_era.isnull()
    mask_relleno = mask_nube | mask_nulos_dentro

    print("🔧 9. Rellenando solo dentro de la cobertura original de LST...")
    lst_rellenada = xr.where(mask_relleno, skt_diaria, lst_rejilla_era)
    lst_final = lst_rellenada.where(mascara_lst_era)

    lst_final.name = "lst_celsius_limpia"
    lst_final.attrs["units"] = "°C"
    lst_final.attrs["long_name"] = "Land Surface Temperature cleaned in Celsius"
    lst_final.attrs["description"] = (
        "LST satelital diaria reescalada a la rejilla ERA5. "
        "Los valores nulos y sospechosamente fríos se sustituyen por SKT "
        "únicamente dentro de la máscara espacial original de cobertura válida de LST."
    )

    nulos_despues_total = int(lst_final.isnull().sum().compute().values)
    nulos_despues_dentro = int(lst_final.where(mascara_lst_era).isnull().sum().compute().values)
    pixeles_nube = int(mask_nube.sum().compute().values)
    pixeles_nulos_rellenados = int(mask_nulos_dentro.sum().compute().values)
    sospechosos_despues = int(((lst_final < umbral_frio_c) & mascara_lst_era).sum().compute().values)

    print("\n📊 Auditoría después de limpieza:")
    print(f"   Píxeles corregidos por posible nube: {pixeles_nube}")
    print(f"   Píxeles rellenados por nulo dentro de máscara: {pixeles_nulos_rellenados}")
    print(f"   Nulos totales después: {nulos_despues_total}")
    print(f"   Nulos dentro de máscara LST después: {nulos_despues_dentro}")
    print(f"   Sospechosos después < {umbral_frio_c} ºC: {sospechosos_despues}")

    print("\n🌡️ Resumen LST final limpia:")
    print(f"   Media: {float(lst_final.mean(skipna=True).compute().values):.2f} ºC")
    print(f"   Mín:   {float(lst_final.min(skipna=True).compute().values):.2f} ºC")
    print(f"   Máx:   {float(lst_final.max(skipna=True).compute().values):.2f} ºC")

    print(f"\n💾 Guardando resultado en: {ruta_salida}")
    lst_final.to_netcdf(ruta_salida)

    ds_sat.close()
    ds_era.close()
    print("✅ LST limpia guardada correctamente manteniendo máscara espacial LST.")
    return lst_final


def procesar_nasa_gpm_mensual(ruta_archivos: str, ruta_salida: str) -> xr.Dataset:
    """
    Une y procesa los archivos mensuales de NASA GPM.
    Aplica un check para garantizar resolución diaria (sumando precipitaciones)
    y previene desajustes de coordenadas entre diferentes años.
    """
    ruta_archivos = resolver_ruta(ruta_archivos)
    ruta_salida = asegurar_directorio_salida(ruta_salida)

    print(f"🌧️ Iniciando procesamiento de NASA GPM...")
    archivos = sorted(glob.glob(ruta_archivos))
    
    if not archivos:
        print(f"❌ No se encontraron archivos en: {ruta_archivos}")
        return None

    datasets_diarios = []
    ref_ds = None # El molde para el grid espacial
    
    print(f"⏳ Procesando {len(archivos)} archivos de precipitación...")
    
    for archivo in archivos:
        nombre = os.path.basename(archivo)
        try:
            ds = xr.open_dataset(archivo)
            
            # 1. ESTANDARIZAR NOMBRES (Por si NASA usa nombres raros)
            renames = {}
            for dim in ds.dims:
                if dim.lower() in ['latitude', 'y']: renames[dim] = 'lat'
                if dim.lower() in ['longitude', 'x']: renames[dim] = 'lon'
                if dim.lower() in ['t', 'time_utc', 'date', 'valid_time']: renames[dim] = 'time'
            
            for coord in ds.coords:
                if coord.lower() in ['latitude', 'y'] and coord not in renames: renames[coord] = 'lat'
                if coord.lower() in ['longitude', 'x'] and coord not in renames: renames[coord] = 'lon'
                
            if renames:
                ds = ds.rename(renames)

            # 2. EL MOLDE ESPACIAL (Evita el error de monotonic indexes)
            if ref_ds is None:
                ref_ds = ds.isel(time=0).drop_vars('time', errors='ignore')
                print(f"✅ Molde de referencia GPM establecido (lat: {len(ref_ds.lat)}, lon: {len(ref_ds.lon)})")

            # 3. INTERPOLACIÓN DE SEGURIDAD (Por si algún mes/año viene movido)
            if len(ds.lon) != len(ref_ds.lon) or len(ds.lat) != len(ref_ds.lat):
                print(f"   ⚠️ Reajustando grid en {nombre}...")
                ds = ds.interp(lat=ref_ds.lat, lon=ref_ds.lon, method="linear")
                ds = ds.bfill(dim='lon').ffill(dim='lon').bfill(dim='lat').ffill(dim='lat')

            # 4. CHECK DIARIO Y AGREGACIÓN
            # Si hay varios datos el mismo día, los suma (precipitación acumulada). 
            # min_count=1 evita que los días 100% nulos se conviertan en 0 (falsos positivos de "no lluvia").
            if 'time' in ds.coords:
                ds.time.encoding = {}
                ds_diario = ds.resample(time='1D').sum(min_count=1)
            else:
                ds_diario = ds
            
            datasets_diarios.append(ds_diario)
            ds.close()
            
        except Exception as e:
            print(f"   ❌ Error en {nombre}: {e}")

    # 5. UNIÓN Y GUARDADO
    if datasets_diarios:
        print("\n🔗 Uniendo todas las piezas de GPM...")
        ds_final = xr.concat(datasets_diarios, dim='time')
        
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        
        # Blindaje de fechas
        enc = {'time': {'units': 'days since 2020-01-01', 'calendar': 'standard'}}
        for v in ds_final.data_vars: 
            enc[v] = {'zlib': True, 'complevel': 5}
        
        print(f"💾 Guardando archivo final en: {ruta_salida}")
        ds_final.to_netcdf(ruta_salida, encoding=enc)
        
        print(f"✅ ¡TERMINADO! GPM guardado con {len(ds_final.time)} días diarios.")
        return ds_final
    else:
        return None

# ============================================================================
# PART 3: ERA5 REANALYSIS PROCESSING FUNCTIONS
# ============================================================================

def procesar_olas_calor(ds_era5: xr.Dataset, convert_kelvin: bool = True) -> xr.Dataset:
    """
    Process ERA5 data for heat wave analysis.
    
    Variables and aggregations:
    - t2m (2m temperature): Daily mean. General "tone" of the day.
    - mx2t (Max temperature): Daily maximum. TARGET VARIABLE.
    - mn2t (Min temperature): Daily minimum. Detects tropical nights.
    - skt (Skin temperature): Daily maximum. Soil heat flux driver.
    - ssrd (Solar radiation): Daily sum. Total energy input.
    - sshf (Sensible heat flux): Daily mean. Soil-to-air heat transfer.
    - tcc (Total cloud cover): Daily mean. Usually near 0 in heat waves.
    
    Args:
        ds_era5: ERA5 dataset (must have time dimension)
        convert_kelvin: If True, convert temperatures from K to °C
        
    Returns:
        Dataset with processed variables at daily resolution
    """
    print("🌡️ Procesando datos de olas de calor...")
    
    # Ensure standardized dimensions
    ds_era5 = estandarizar_nombres(ds_era5)
    
    # Clean duplicate dimensions
    ds_era5 = limpiar_dimensiones_duplicadas(ds_era5)
    
    # Convert time to datetime if it's numeric
    if 'time' in ds_era5.coords:
        time_index = ds_era5.time
        if not isinstance(time_index.values[0], (np.datetime64, pd.Timestamp)):
            try:
                ds_era5['time'] = pd.to_datetime(ds_era5.time.values, unit='D', origin='2020-01-01')
                print("  ℹ️  Convertido indice 'time' a datetime")
            except:
                ds_era5['time'] = pd.date_range(start='2020-01-01', periods=len(ds_era5.time))
                print("  ⚠️  Creado indice de tiempo estimado")
    
    result = xr.Dataset()
    
    # Temperature variables (convert if needed)
    for var_name in ['t2m', 'mx2t', 'mn2t', 'skt']:
        if var_name in ds_era5:
            var = ds_era5[var_name].copy()
            if convert_kelvin:
                var = var - 273.15
            
            # Daily aggregation
            if var_name == 'skt':
                # Maximum for skin temperature
                result[var_name] = var.resample(time='1D').max()
            else:
                # Mean for other temperature variables (except mx2t)
                if var_name == 'mx2t':
                    result[var_name] = var.resample(time='1D').max()
                else:
                    result[var_name] = var.resample(time='1D').mean()
            
            print(f"  ✅ {var_name}: procesado")
    
    # Solar radiation (daily sum)
    if 'ssrd' in ds_era5:
        result['ssrd'] = ds_era5['ssrd'].resample(time='1D').sum()
        print(f"  ✅ ssrd: suma diaria")
    
    # Sensible heat flux (daily mean)
    if 'sshf' in ds_era5:
        result['sshf'] = ds_era5['sshf'].resample(time='1D').mean()
        print(f"  ✅ sshf: media diaria")
    
    # Cloud cover (daily mean)
    if 'tcc' in ds_era5:
        result['tcc'] = ds_era5['tcc'].resample(time='1D').mean()
        print(f"  ✅ tcc: media diaria")
    
    # Geopotential at 500 hPa (daily mean)
    if 'z' in ds_era5:
        result['z'] = ds_era5['z'].resample(time='1D').mean()
        print(f"  ✅ z: media diaria")
    
    # Soil moisture (daily mean)
    if 'swvl1' in ds_era5:
        result['swvl1'] = ds_era5['swvl1'].resample(time='1D').mean()
        print(f"  ✅ swvl1: media diaria")
    
    print("🚀 Procesamiento de olas de calor completado.")
    return result


def procesar_precipitaciones(ds_era5: xr.Dataset) -> xr.Dataset:
    """
    Process ERA5 data for precipitation analysis.
    
    Variables and aggregations:
    - tp (Total Precipitation): Daily sum.
    - cp (Convective Precipitation): Daily sum.
    - lsp (Large-scale/Stratiform Precipitation): Daily sum.
    - tcwv (Total Column Water Vapour): Daily mean.
    - msl (Mean Sea Level Pressure): Daily mean.
    - u10/v10/u850/v850 (Wind components): Daily mean.
    
    Args:
        ds_era5: ERA5 dataset (must have time dimension)
        
    Returns:
        Dataset with processed variables at daily resolution
    """
    print("🌧️ Procesando datos de precipitaciones...")
    
    # Ensure standardized dimensions
    ds_era5 = estandarizar_nombres(ds_era5)
    
    # Clean duplicate dimensions
    ds_era5 = limpiar_dimensiones_duplicadas(ds_era5)
    
    # Convert time to datetime if it's numeric
    if 'time' in ds_era5.coords:
        time_index = ds_era5.time
        if not isinstance(time_index.values[0], (np.datetime64, pd.Timestamp)):
            try:
                ds_era5['time'] = pd.to_datetime(ds_era5.time.values, unit='D', origin='2020-01-01')
                print("  ℹ️  Convertido indice 'time' a datetime")
            except:
                ds_era5['time'] = pd.date_range(start='2020-01-01', periods=len(ds_era5.time))
                print("  ⚠️  Creado indice de tiempo estimado")
    
    result = xr.Dataset()
    
    # Precipitation variables (daily sum)
    for var_name in ['tp', 'cp', 'lsp']:
        if var_name in ds_era5:
            result[var_name] = ds_era5[var_name].resample(time='1D').sum()
            print(f"  ✅ {var_name}: suma diaria")
    
    # Atmospheric variables (daily mean)
    for var_name in ['tcwv', 'msl']:
        if var_name in ds_era5:
            result[var_name] = ds_era5[var_name].resample(time='1D').mean()
            print(f"  ✅ {var_name}: media diaria")
    
    # Wind components (daily mean)
    for var_name in ['u10', 'v10', 'u850', 'v850', 'u_850', 'v_850']:
        if var_name in ds_era5:
            result[var_name] = ds_era5[var_name].resample(time='1D').mean()
            print(f"  ✅ {var_name}: media diaria")
    
    print("🚀 Procesamiento de precipitaciones completado.")
    return result


def procesar_calidad_aire_diario_final_v4(ruta_archivos: str, ruta_salida: str):
    """
    Procesa CAMS arreglando el descuadre de dimensiones de 2025 (17 vs 18 lon).
    Fuerza el relleno espacial en los bordes para no perder ni un píxel del mapa.
    """
    ruta_archivos = resolver_ruta(ruta_archivos)
    ruta_salida = asegurar_directorio_salida(ruta_salida)

    print(f"🚀 Iniciando procesamiento HÍBRIDO (Interpolación + Relleno de Bordes para 2025)...")
    archivos = sorted(glob.glob(ruta_archivos))
    
    if not archivos:
        print("❌ No hay archivos.")
        return None

    datasets_diarios = []
    ref_ds = None 
    
    for archivo in archivos:
        nombre = os.path.basename(archivo)
        try:
            ds = xr.open_dataset(archivo)
            
            # 1. Limpieza básica
            if 'expver' in ds.dims or 'expver' in ds.coords:
                try: ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
                except: ds = ds.isel(expver=0)
                ds = ds.drop_vars('expver', errors='ignore')
            
            ds = ds.rename({'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon'}) if 'valid_time' in ds.coords else ds
            
            # 2. EL MOLDE (18 puntos)
            if ref_ds is None:
                ref_ds = ds.isel(time=0).drop_vars('time', errors='ignore')
                print(f"✅ Molde de referencia establecido (lat: {len(ref_ds.lat)}, lon: {len(ref_ds.lon)})")

            # 3. INTERPOLACIÓN Y RELLENO (El truco para 2025)
            if len(ds.lon) != len(ref_ds.lon) or len(ds.lat) != len(ref_ds.lat):
                print(f"   ⚠️ Reajustando y rellenando bordes en {nombre}...")
                
                # Paso A: Interpolamos. (Los bordes que se salgan quedarán como NaN)
                ds = ds.interp(lat=ref_ds.lat, lon=ref_ds.lon, method="linear")
                
                # Paso B: "Inventamos" el dato de la esquina estirando el píxel vecino
                # ffill = arrastra de izquierda a derecha / bfill = de derecha a izquierda
                ds = ds.bfill(dim='lon').ffill(dim='lon').bfill(dim='lat').ffill(dim='lat')

            # 4. Agregación diaria
            ds.time.encoding = {}
            ds_diario = xr.Dataset()
            for var in ds.data_vars:
                if var in ['total_column_ozone', 'gtco3']:
                    ds_diario[var] = ds[var].resample(time='1D').max()
                else:
                    ds_diario[var] = ds[var].resample(time='1D').mean()
            
            datasets_diarios.append(ds_diario)
            ds.close()
            
        except Exception as e:
            print(f"   ❌ Error en {nombre}: {e}")

    # 5. Unión y Guardado
    print("\n🔗 Uniendo todas las piezas...")
    ds_final = xr.concat(datasets_diarios, dim='time')
    
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    enc = {'time': {'units': 'days since 2020-01-01', 'calendar': 'standard'}}
    for v in ds_final.data_vars: enc[v] = {'zlib': True, 'complevel': 5}
    
    ds_final.to_netcdf(ruta_salida, encoding=enc)
    print(f"✅ ¡TERMINADO! Archivo guardado con {len(ds_final.time)} días. Mapa 100% completo.")
    return ds_final

# ============================================================================
# PART 4: SATELLITE-REANALYSIS ANALYSIS
# ============================================================================

def correlacion_satelite_reanalisis(
    var_reanalisis: xr.DataArray,
    var_satelite: xr.DataArray,
    resample_freq: str = '1D'
) -> Tuple[xr.DataArray, float]:
    """
    Calculate pixel-wise correlation between satellite and reanalysis data.
    
    Args:
        var_reanalisis: Reanalysis variable (e.g., ERA5 t2m)
        var_satelite: Satellite variable (e.g., Sentinel-3 LST)
        resample_freq: Resampling frequency (default '1D' for daily)
        
    Returns:
        Tuple of (correlation map, global mean correlation)
    """
    print("📊 Calculando correlación píxel a píxel...")
    
    # Resample to same frequency
    var_r = var_reanalisis.resample(time=resample_freq).mean()
    var_s = var_satelite.resample(time=resample_freq).mean() if 'time' in var_satelite.dims else var_satelite
    
    # Align grids
    var_r_alineado, var_s_alineado = xr.align(var_r, var_s, join='inner')
    
    # Calculate correlation
    mapa_corr = xr.corr(var_r_alineado, var_s_alineado, dim='time')
    mapa_corr = mapa_corr.compute()
    
    # Global mean
    corr_media = mapa_corr.mean().item()
    
    print(f"✅ Correlación media global: r = {corr_media:.3f}")
    
    return mapa_corr, corr_media


def analizar_nubosidad_satelite(
    ds_era5: xr.Dataset,
    ds_satelite: xr.Dataset,
    fecha_inicio: Optional[str] = None,
    fecha_fin: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze impact of cloud cover on satellite data availability.
    
    Args:
        ds_era5: ERA5 dataset with tcc (cloud cover)
        ds_satelite: Satellite dataset (e.g., LST)
        fecha_inicio: Start date (optional)
        fecha_fin: End date (optional)
        
    Returns:
        DataFrame with analysis results
    """
    print("⏳ Analizando impacto de nubosidad...")
    
    # Ensure standardized names
    ds_era5 = estandarizar_nombres(ds_era5)
    ds_satelite = estandarizar_nombres(ds_satelite)
    
    # Select date range if provided
    if fecha_inicio and fecha_fin:
        ds_era5 = ds_era5.sel(time=slice(fecha_inicio, fecha_fin))
        ds_satelite = ds_satelite.sel(time=slice(fecha_inicio, fecha_fin))
    
    # Process cloud cover
    tcc_diario = ds_era5['tcc'].resample(time='1D').mean()
    
    # Get first satellite variable
    sat_var_name = list(ds_satelite.data_vars)[0]
    sat_diario = ds_satelite[sat_var_name]
    
    # Align grids
    tcc_alineado = tcc_diario.interp_like(sat_diario, method='nearest')
    
    # Convert to DataFrame
    ds_analisis = xr.Dataset({'tcc': tcc_alineado, 'sat': sat_diario})
    df = ds_analisis.to_dataframe().reset_index()
    
    # Add null indicator
    df['Satelite_Nulo'] = df['sat'].isna()
    
    print("✅ Análisis completado")
    
    return df


# ============================================================================
# PART 5: DATA LOADING AND SAVING
# ============================================================================

def cargar_datos_era5_seguro(
    ruta: str, 
    chunks: Dict = None, 
    fix_coords: bool = True
) -> xr.Dataset:
    """
    Load ERA5 dataset with automatic coordinate fixing.
    
    Solves: "ValueError: Resulting object does not have monotonic global 
    indexes along dimension longitude"
    
    This error occurs when netCDF files have unordered coordinates.
    
    Args:
        ruta: Path to data (supports wildcards for glob patterns)
        chunks: Dask chunking strategy (default: auto)
        fix_coords: If True, automatically sort coordinates if needed
        
    Returns:
        Loaded dataset with standardized names
    """
    if chunks is None:
        chunks = 'auto'

    ruta = resolver_ruta(ruta)
    
    print(f"📂 Cargando datos desde: {ruta}")
    
    try:
        # First attempt: standard loading
        ds = xr.open_mfdataset(ruta, chunks=chunks)
        print(f"✅ Datos cargados exitosamente")
        
    except ValueError as e:
        if "monotonic global indexes" in str(e):
            print(f"⚠️  Detectado error de índices no monótonos, intentando solución...")
            
            # Fallback: load without combining
            import glob
            files = sorted(glob.glob(ruta))
            print(f"📋 Cargando {len(files)} archivo(s) por separado...")
            
            datasets = []
            for i, f in enumerate(files):
                ds_temp = xr.open_dataset(f, chunks=chunks)
                if fix_coords:
                    # Sort coordinates
                    for coord in ['lat', 'lon', 'latitude', 'longitude']:
                        if coord in ds_temp.coords:
                            ds_temp = ds_temp.sortby(coord)
                
                # Add file index as coordinate to track which file each time point comes from
                ds_temp = ds_temp.assign_coords({'file_idx': i})
                datasets.append(ds_temp)
            
            # Concatenate along time dimension
            # Use combine='by_coords' to properly handle multiple files
            try:
                ds = xr.concat(datasets, dim='time')
            except ValueError as concat_err:
                # If concat fails, try with join='override'
                if "duplicate" in str(concat_err).lower():
                    print(f"⚠️  Intentando concatenación alternativa...")
                    ds = xr.concat(datasets, dim='time', join='override')
                else:
                    raise
            
            print(f"✅ Datos cargados y combinados exitosamente")
        else:
            raise
    
    # Ensure standardized dimension names
    ds = estandarizar_nombres(ds)
    
    # Clean duplicate dimensions (CAMS/ERA5 may have these)
    ds = limpiar_dimensiones_duplicadas(ds)
    
    # Final coordinate check
    if fix_coords:
        try:
            for coord in ['lat', 'lon', 'time']:
                if coord in ds.coords:
                    if not ds.indexes[coord].is_monotonic_increasing:
                        ds = ds.sortby(coord)
        except:
            pass  # Silent fail if checking fails
    
    return ds


def cargar_datos_era5(rutas: Dict[str, str], chunks: Dict = None) -> Dict[str, xr.Dataset]:
    """
    Load ERA5 datasets from multiple paths.
    
    Use cargar_datos_era5_seguro() instead if you get monotonic index errors.
    
    Args:
        rutas: Dictionary mapping dataset names to file paths
               (supports wildcards for glob patterns)
        chunks: Dask chunking strategy (default: auto)
        
    Returns:
        Dictionary mapping names to loaded datasets
    """
    if chunks is None:
        chunks = 'auto'
    
    datasets = {}
    rutas = resolver_rutas_dict(rutas)
    for nombre, ruta in rutas.items():
        print(f"📂 Cargando {nombre} desde: {ruta}")
        ds = xr.open_mfdataset(ruta, chunks=chunks)
        ds = estandarizar_nombres(ds)
        datasets[nombre] = ds
        print(f"✅ {nombre} cargado")
    
    return datasets


def guardar_datos(var: xr.DataArray, ruta: str, nombre_var: str = None) -> None:
    """
    Save processed variable to netCDF file.
    
    Args:
        var: Variable to save
        ruta: Output file path
        nombre_var: Variable name (if None, uses var.name)
    """
    if nombre_var:
        var.name = nombre_var

    ruta = asegurar_directorio_salida(ruta)
    
    print(f"💾 Guardando en: {ruta}")
    var.to_netcdf(ruta)
    print(f"✅ Archivo guardado")


######## FUNCIONES DE REREJILLADO Y CREACIÓN DE DATASETS X E Y PARA ENTRENAMIENTO ########


def crear_dataset_X_olas_calor_final(rutas_procesadas: dict, ruta_salida: str):
    """
    Crea el Data Cube definitivo para ML:
    1. Grid de 0.75º (CAMS).
    2. Máscara de tierra estricta basada en LST de Sentinel-3.
    3. Aplicación de máscara a TODAS las variables (ERA5, CAMS, GPM).
    """
    print(f"🏗️  Construyendo Dataset X con Máscara de Tierra Estricta...")
    rutas_procesadas = resolver_rutas_dict(rutas_procesadas)
    ruta_salida = asegurar_directorio_salida(ruta_salida)

    # 1. CARGAR CAMS COMO MOLDE ESPACIAL (0.75)
    ds_ref = xr.open_dataset(rutas_procesadas['cams']).compute()
    ds_ref = ds_ref.sortby(['lat', 'lon'])
    grid_lat = ds_ref.lat
    grid_lon = ds_ref.lon

    # 2. CARGAR LST Y CREAR LA MÁSCARA MAESTRA
    print("🌍 Extrayendo máscara de tierra de Sentinel-3 (LST)...")
    ds_lst = xr.open_dataset(rutas_procesadas['s3lst'])
    ds_lst = ds_lst.rename({'latitude': 'lat', 'longitude': 'lon'}) if 'latitude' in ds_lst.coords else ds_lst
    ds_lst = ds_lst.sortby(['lat', 'lon'])
    
    # Interpolamos LST al grid de 0.75 primero
    lst_075 = ds_lst['lst_celsius_limpia'].interp(lat=grid_lat, lon=grid_lon, method='linear')
    
    # Definimos la máscara: Donde LST tiene datos, es tierra (True/1). El resto es Mar (False/NaN).
    mask_tierra = lst_075.notnull()

    # 3. FILTRAR CAMS CON LA MÁSCARA
    # Ahora CAMS ya no tendrá datos sobre el mar
    ds_ref = ds_ref.where(mask_tierra)
    
    datasets_list = [ds_ref, lst_075.to_dataset(name='lst_celsius_limpia')]

    # 4. ALINEAR Y ENMASCARAR EL RESTO (ERA5, GPM, etc.)
    for nombre, ruta in rutas_procesadas.items():
        if nombre in ['cams', 's3lst']: continue 
        
        print(f"🔄 Aplicando máscara terrestre a {nombre}...")
        ds_var = xr.open_dataset(ruta)
        ds_var = ds_var.rename({'latitude': 'lat', 'longitude': 'lon'}) if 'latitude' in ds_var.coords else ds_var
        ds_var = ds_var.sortby(['lat', 'lon'])
        
        # Interpolar y aplicar máscara
        ds_interp = ds_var.interp(lat=grid_lat, lon=grid_lon, method='linear')
        ds_interp = ds_interp.where(mask_tierra)
        
        datasets_list.append(ds_interp)

    # 5. FUSIÓN FINAL
    print("🔗 Fusionando variables sobre el perfil terrestre...")
    ds_X = xr.merge(datasets_list, join='inner')
    
    # Rellenado de huecos internos (Padding de 1 píxel para esquinas)
    # Solo rellenamos si el píxel es "tierra" según la máscara pero está vacío
    ds_X = ds_X.interpolate_na(dim='lon', method='nearest', limit=1)
    ds_X = ds_X.interpolate_na(dim='lat', method='nearest', limit=1)
    
    # Aseguramos que tras el relleno NO nos hayamos salido al mar de nuevo
    ds_X = ds_X.where(mask_tierra)

    # 6. GUARDAR
    print(f"💾 Guardando cubo de datos en: {ruta_salida}")
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    encoding = {v: {'zlib': True, 'complevel': 5} for v in ds_X.data_vars}
    ds_X.to_netcdf(ruta_salida, encoding=encoding)
    
    print(f"✅ ¡PROCESO FINALIZADO! Dataset X listo con {len(ds_X.time)} días.")
    return ds_X


def crear_dataset_X_precipitaciones_extremas(rutas_procesadas: dict, ruta_salida: str):
    """
    Crea el dataset X alineado a 0.75 de CAMS pero recortado al área común
    más pequeña (S5P) para evitar bandas blancas sin datos.
    """

    print(f"🏗️  Construyendo Dataset X optimizado (Recorte por área común)...")
    rutas_procesadas = resolver_rutas_dict(rutas_procesadas)
    ruta_salida = asegurar_directorio_salida(ruta_salida)

    def limpiar_pressure_level(ds, nombre="dataset"):
        """
        Elimina de forma segura la dimensión pressure_level si existe.
        Evita que xr.merge(join='inner') deje una dimensión vacía.
        """
        if "pressure_level" in ds.dims:
            size_pl = ds.sizes["pressure_level"]

            if size_pl == 1:
                ds = ds.isel(pressure_level=0, drop=True)
                print(f"   ✅ {nombre}: pressure_level eliminado (size=1)")

            elif size_pl > 1:
                # Si hubiera varios niveles, nos quedamos con el primero.
                # Si quieres uno concreto, aquí podríamos seleccionar 850, 500, etc.
                ds = ds.isel(pressure_level=0, drop=True)
                print(f"   ⚠️ {nombre}: varios pressure_level, seleccionado el primero")

            else:
                # Caso raro: dimensión vacía
                ds = ds.drop_dims("pressure_level")
                print(f"   ⚠️ {nombre}: pressure_level vacío eliminado")

        if "pressure_level" in ds.coords:
            ds = ds.drop_vars("pressure_level", errors="ignore")

        return ds

    def estandarizar_coords(ds):
        """
        Estandariza nombres de coordenadas espaciales.
        """
        rename_dict = {}

        if "latitude" in ds.coords:
            rename_dict["latitude"] = "lat"

        if "longitude" in ds.coords:
            rename_dict["longitude"] = "lon"

        if rename_dict:
            ds = ds.rename(rename_dict)

        return ds

    # =====================================================
    # 1. CARGAR CAMS COMO MOLDE
    # =====================================================

    ds_cams = xr.open_dataset(rutas_procesadas["cams"]).compute()
    ds_cams = estandarizar_coords(ds_cams)
    ds_cams = limpiar_pressure_level(ds_cams, nombre="cams")
    ds_cams = ds_cams.sortby(["lat", "lon"])

    # =====================================================
    # 2. CARGAR SENTINEL-5 PARA DEFINIR ÁREA REAL
    # =====================================================

    ds_s5 = xr.open_dataset(rutas_procesadas["no2"])
    ds_s5 = estandarizar_coords(ds_s5)
    ds_s5 = limpiar_pressure_level(ds_s5, nombre="no2_ref")
    ds_s5 = ds_s5.sortby(["lat", "lon"])

    lat_min = float(ds_s5.lat.min().values)
    lat_max = float(ds_s5.lat.max().values)
    lon_min = float(ds_s5.lon.min().values)
    lon_max = float(ds_s5.lon.max().values)

    print(
        f"📏 Área común detectada (S5P): "
        f"Lat[{lat_min:.2f}, {lat_max:.2f}] "
        f"Lon[{lon_min:.2f}, {lon_max:.2f}]"
    )

    # =====================================================
    # 3. RECORTAR CAMS AL ÁREA COMÚN
    # =====================================================

    ds_ref = ds_cams.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max)
    )

    if ds_ref.sizes.get("lat", 0) == 0 or ds_ref.sizes.get("lon", 0) == 0:
        raise ValueError(
            "❌ El recorte espacial ha dejado ds_ref vacío. "
            "Revisa si las lat/lon de CAMS y S5P están en el mismo rango "
            "y ordenadas correctamente."
        )

    grid_lat = ds_ref.lat
    grid_lon = ds_ref.lon

    print(f"✅ Molde recortado: {len(grid_lat)} lat x {len(grid_lon)} lon")

    datasets_list = [ds_ref]

    # =====================================================
    # 4. ALINEAR VARIABLES
    # =====================================================

    for nombre, ruta in rutas_procesadas.items():

        if nombre == "cams":
            continue

        print(f"🔄 Interpolando y recortando {nombre}...")

        ds_var = xr.open_dataset(ruta)
        ds_var = estandarizar_coords(ds_var)
        ds_var = limpiar_pressure_level(ds_var, nombre=nombre)
        ds_var = ds_var.sortby(["lat", "lon"])

        ds_interp = ds_var.interp(
            lat=grid_lat,
            lon=grid_lon,
            method="linear"
        )

        datasets_list.append(ds_interp)

    # =====================================================
    # 5. FUSIÓN FINAL
    # =====================================================

    print("🔗 Fusionando variables...")

    ds_X = xr.merge(
        datasets_list,
        join="inner",
        compat="override"
    )

    # Seguridad extra por si quedara pressure_level
    if "pressure_level" in ds_X.dims:
        if ds_X.sizes["pressure_level"] == 1:
            ds_X = ds_X.isel(pressure_level=0, drop=True)
        elif ds_X.sizes["pressure_level"] == 0:
            ds_X = ds_X.drop_dims("pressure_level")
        else:
            ds_X = ds_X.isel(pressure_level=0, drop=True)

    if "pressure_level" in ds_X.coords:
        ds_X = ds_X.drop_vars("pressure_level", errors="ignore")

    # =====================================================
    # 6. CHECK FINAL
    # =====================================================

    print("\n📊 Dimensiones finales:")
    print(dict(ds_X.sizes))

    if ds_X.sizes.get("time", 0) == 0:
        raise ValueError("❌ El dataset final tiene time=0. Revisa el solape temporal.")

    if ds_X.sizes.get("lat", 0) == 0 or ds_X.sizes.get("lon", 0) == 0:
        raise ValueError("❌ El dataset final tiene lat/lon vacío. Revisa el recorte espacial.")

    # =====================================================
    # 7. GUARDADO
    # =====================================================

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    encoding = {
        v: {"zlib": True, "complevel": 5}
        for v in ds_X.data_vars
    }

    ds_X.to_netcdf(ruta_salida, encoding=encoding)

    print(
        f"✅ Dataset X finalizado: "
        f"{len(ds_X.lat)}x{len(ds_X.lon)} píxeles "
        f"(Sin bordes vacíos)."
    )

    return ds_X

def construir_cubo_entrenamiento(rutas_procesadas: dict, ruta_salida: str, variables_interp_lineal=None):
    """
    Construye un cubo de datos unificado (X o Y) para cualquier modelo.
    - Molde espacial: CAMS (0.75º).
    - Recorte: Área común mínima entre todos los datasets (evita bandas blancas).
    - Interpolación: Bilineal para variables continuas, Nearest para el resto.
    """
    if variables_interp_lineal is None:
        variables_interp_lineal = [] # Lista de variables que prefieres interpolación suave

    rutas_procesadas = resolver_rutas_dict(rutas_procesadas)
    ruta_salida = asegurar_directorio_salida(ruta_salida)

    print(f"🏗️  Iniciando construcción del Cubo de Datos en: {ruta_salida}")

    # 1. CARGAR TODOS LOS DATASETS PARA CALCULAR EL ÁREA COMÚN (Bounding Box)
    ds_dict = {}
    lat_min, lat_max = -90, 90
    lon_min, lon_max = -180, 180

    for nombre, ruta in rutas_procesadas.items():
        ds = xr.open_dataset(ruta)
        ds = estandarizar_nombres(ds).sortby(['lat', 'lon'])
        ds_dict[nombre] = ds
        
        # Actualizamos los límites para encontrar el "recuadro" donde todos tienen datos
        lat_min = max(lat_min, ds.lat.min().values)
        lat_max = min(lat_max, ds.lat.max().values)
        lon_min = max(lon_min, ds.lon.min().values)
        lon_max = min(lon_max, ds.lon.max().values)

    print(f"📏 Área de recorte común calculada: Lat[{lat_min:.2f}, {lat_max:.2f}] Lon[{lon_min:.2f}, {lon_max:.2f}]")

    # 2. DEFINIR EL MOLDE DE CAMS DENTRO DE ESE RECUADRO
    # Usamos CAMS como referencia de resolución (0.75)
    ds_ref_full = ds_dict['cams'] if 'cams' in ds_dict else list(ds_dict.values())
    ds_molde = ds_ref_full.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    
    grid_lat = ds_molde.lat
    grid_lon = ds_molde.lon

    datasets_list = []

    # 3. INTERPOLAR TODAS LAS VARIABLES AL MOLDE RECORTADO
    for nombre, ds in ds_dict.items():
        print(f"🔄 Alineando {nombre}...")
        
        # Decidir método de interpolación
        # Si la variable está en la lista de 'lineal', se ve más suave; si no, 'nearest'
        metodo = 'linear' if any(v in variables_interp_lineal for v in ds.data_vars) else 'linear'
        
        ds_interp = ds.interp(lat=grid_lat, lon=grid_lon, method=metodo)
        
        # Limpieza de nulos internos por seguridad (máximo 1 píxel)
        ds_interp = ds_interp.interpolate_na(dim='lon', method='nearest', limit=1)
        ds_interp = ds_interp.interpolate_na(dim='lat', method='nearest', limit=1)
        
        datasets_list.append(ds_interp)

    # 4. FUSIÓN Y GUARDADO
    print("🔗 Fusionando todas las variables en un solo archivo...")
    ds_final = xr.merge(datasets_list, join='inner')
    
    # Quitar dimensiones vacías (como pressure_level si solo hay 1)
    if 'pressure_level' in ds_final.dims:
        ds_final = ds_final.squeeze('pressure_level', drop=True)

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    encoding = {v: {'zlib': True, 'complevel': 5} for v in ds_final.data_vars}
    
    ds_final.to_netcdf(ruta_salida, encoding=encoding)
    print(f"✅ Cubo guardado con éxito. Dimensiones: {dict(ds_final.sizes)}")
    
    return ds_final
