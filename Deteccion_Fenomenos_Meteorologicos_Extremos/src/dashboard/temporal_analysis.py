#!/usr/bin/env python3
"""
Script simple para analizar la frecuencia temporal del dataset nasa_gpm_mensual.
"""

import xarray as xr
import pandas as pd
from pathlib import Path


REPO_DIR_NAME = "Deteccion_Fenomenos_Meteorologicos_Extremos"


def get_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == REPO_DIR_NAME:
            return parent
    return current.parent


def analyze_nasa_gpm_temporal():
    """Analiza la dimensión temporal del dataset nasa_gpm_mensual"""

    # Path al dataset
    data_dir = get_repo_root() / "data" / "datos_tfg" / "nasa_gpm_mensual"

    if not data_dir.exists():
        print(f"Error: Directorio {data_dir} no encontrado")
        return

    # Buscar archivos NetCDF
    files = sorted(data_dir.glob("*.nc"))
    if not files:
        print(f"No se encontraron archivos .nc en {data_dir}")
        return

    print(f"Encontrados {len(files)} archivos NetCDF")
    print("Primeros 5 archivos:")
    for f in files[:5]:
        print(f"  {f.name}")

    # Cargar dataset
    print("\nCargando dataset...")
    try:
        ds = xr.open_mfdataset([str(f) for f in files], combine='by_coords', engine='netcdf4')
        print("✓ Dataset cargado exitosamente")
    except Exception as e:
        print(f"Error cargando dataset: {e}")
        return

    # Analizar dimensión temporal
    if 'time' not in ds.dims:
        print("Error: No se encontró dimensión 'time' en el dataset")
        return

    times = ds.time.values
    print(f"\nDimensión temporal: {len(times)} puntos")
    print(f"Rango temporal: {times[0]} a {times[-1]}")

    # Primeras fechas
    print("\nPrimeras 10 fechas:")
    for i, t in enumerate(times[:10]):
        print(f"  {i+1}: {t}")

    # Inferir frecuencia
    try:
        time_series = pd.to_datetime(times)
        freq = pd.infer_freq(time_series)
        if freq:
            print(f"\nFrecuencia inferida: {freq}")
            if 'D' in freq:
                print("→ Resolución DIARIA")
            elif 'M' in freq or 'MS' in freq:
                print("→ Resolución MENSUAL")
            elif 'H' in freq:
                print("→ Resolución HORARIA")
            else:
                print(f"→ Frecuencia: {freq}")
        else:
            print("\nNo se pudo inferir frecuencia automáticamente")
            # Calcular diferencias manuales
            diffs = pd.Series(time_series).diff().dropna()
            if len(diffs) > 0:
                mean_diff = diffs.mean()
                print(f"Diferencia media entre fechas: {mean_diff}")
                if mean_diff.days == 1:
                    print("→ Parece ser DIARIO")
                elif mean_diff.days >= 28 and mean_diff.days <= 31:
                    print("→ Parece ser MENSUAL")
    except Exception as e:
        print(f"Error analizando frecuencia: {e}")

    # Verificar fecha específica
    target_date = '2024-10-29'
    print(f"\nVerificando datos para fecha específica: {target_date}")
    try:
        ds_date = ds.sel(time=target_date)
        if ds_date.time.size > 0:
            print(f"✓ Hay datos para {target_date}")
            print(f"  Forma del array para esa fecha: {ds_date[list(ds.data_vars.keys())[0]].shape}")
        else:
            print(f"✗ No hay datos para {target_date}")
    except Exception as e:
        print(f"Error verificando fecha {target_date}: {e}")

    # Mostrar variables disponibles
    print("\nVariables disponibles:")
    for var in ds.data_vars:
        print(f"  {var}: {ds[var].dims}")

if __name__ == "__main__":
    analyze_nasa_gpm_temporal()
