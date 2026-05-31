"""
Pipeline limpio de extraccion de datos climaticos y satelitales para el TFG.

El modulo aglutina las descargas que estaban en Creacion_BBDD.ipynb, sin
registro en base de datos. Permite descargar todos los bloques o solo productos
concretos.

Ejemplos:
    python extraccion_datos_pipeline.py --producto olas_calor --years 2025 --months 8 9
    python extraccion_datos_pipeline.py --producto sentinel5p --s5p-producto NO2 --years 2025
    python extraccion_datos_pipeline.py --producto todo --years 2024 2025
"""

from __future__ import annotations

import argparse
import calendar
import glob
import os
import shutil
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import xarray as xr


Producto = Literal[
    "todo",
    "olas_calor",
    "precipitacion",
    "calidad_aire",
    "sentinel5p",
    "gpm",
    "modis_mensual",
    "modis_diario",
]


AREA_ERA5 = [44.0, -9.5, 36.0, 3.5]  # north, west, south, east
BBOX_SPAIN_OPENEO = {"west": -9.5, "south": 36.0, "east": 3.5, "north": 44.0}
BOX_SPAIN_GPM = {"lon": slice(-10, 5), "lat": slice(35, 45)}
REPO_DIR_NAME = "Deteccion_Fenomenos_Meteorologicos_Extremos"


def _repo_root() -> Path:
    """
    Devuelve la raiz del repo aunque este archivo viva en src/extraccion.

    Si el archivo aun se ejecuta desde otra ubicacion durante la reorganizacion,
    cae al directorio donde esta el propio archivo.
    """
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == REPO_DIR_NAME:
            return parent
        repo_child = parent / REPO_DIR_NAME
        if repo_child.is_dir():
            return repo_child
    return current.parent


DEFAULT_DATA_DIR = _repo_root() / "data" / "datos_tfg"
DEFAULT_TEMP_DIR = _repo_root() / "data" / "temp_extraccion_datos"


@dataclass
class PipelineConfig:
    base_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    temp_dir: Path = field(default_factory=lambda: DEFAULT_TEMP_DIR)
    years: list[int] = field(default_factory=lambda: list(range(2020, 2026)))
    months: list[int] = field(default_factory=lambda: list(range(1, 13)))
    overwrite: bool = False
    area_era5: list[float] = field(default_factory=lambda: AREA_ERA5.copy())

    # ADS/CAMS. Si no se pasa key, cdsapi usara ~/.cdsapirc cuando pueda.
    ads_url: str = "https://ads.atmosphere.copernicus.eu/api"
    ads_key: str | None = None

    # MODIS/GPM.
    modis_lat_min: float = 35.0
    modis_lat_max: float = 45.0
    modis_lon_min: float = -10.0
    modis_lon_max: float = 5.0
    modis_step: float = 0.1

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.temp_dir = Path(self.temp_dir)
        self.ads_key = self.ads_key or os.getenv("ADS_KEY")


TABLA_VARIABLES_OLAS_CALOR = pd.DataFrame(
    [
        ("ERA5 single-levels", "2m_temperature", "t2m", "Temperatura a 2 m"),
        ("ERA5 single-levels", "skin_temperature", "skt", "Temperatura de superficie"),
        ("ERA5 single-levels", "total_cloud_cover", "tcc", "Cobertura nubosa total"),
        (
            "ERA5 single-levels",
            "volumetric_soil_water_layer_1",
            "swvl1",
            "Humedad volumetrica del suelo capa 1",
        ),
        (
            "ERA5 single-levels",
            "surface_sensible_heat_flux",
            "sshf",
            "Flujo de calor sensible superficial",
        ),
        (
            "ERA5 single-levels",
            "surface_solar_radiation_downwards",
            "ssrd",
            "Radiacion solar descendente superficial",
        ),
        (
            "ERA5 single-levels",
            "maximum_2m_temperature_since_previous_post_processing",
            "mx2t",
            "Temperatura maxima 2 m",
        ),
        (
            "ERA5 single-levels",
            "minimum_2m_temperature_since_previous_post_processing",
            "mn2t",
            "Temperatura minima 2 m",
        ),
        ("ERA5 pressure-levels", "geopotential 500 hPa", "z", "Geopotencial en 500 hPa"),
    ],
    columns=["fuente", "variable_api", "variable_netcdf", "descripcion"],
)

TABLA_VARIABLES_PRECIPITACION = pd.DataFrame(
    [
        ("ERA5 single-levels", "total_precipitation", "tp", "Precipitacion total"),
        ("ERA5 single-levels", "convective_precipitation", "cp", "Precipitacion convectiva"),
        ("ERA5 single-levels", "large_scale_precipitation", "lsp", "Precipitacion estratiforme"),
        (
            "ERA5 single-levels",
            "total_column_water_vapour",
            "tcwv",
            "Vapor de agua total en columna",
        ),
        ("ERA5 single-levels", "mean_sea_level_pressure", "msl", "Presion media al nivel del mar"),
        ("ERA5 single-levels", "10m_u_component_of_wind", "u10", "Viento zonal a 10 m"),
        ("ERA5 single-levels", "10m_v_component_of_wind", "v10", "Viento meridional a 10 m"),
        ("ERA5 pressure-levels", "u_component_of_wind 850 hPa", "u_850", "Viento zonal 850 hPa"),
        ("ERA5 pressure-levels", "v_component_of_wind 850 hPa", "v_850", "Viento meridional 850 hPa"),
    ],
    columns=["fuente", "variable_api", "variable_netcdf", "descripcion"],
)

TABLA_VARIABLES_CALIDAD_AIRE_CAMS = pd.DataFrame(
    [
        ("CAMS EAC4", "particulate_matter_2.5um", "pm2p5", "PM2.5"),
        ("CAMS EAC4", "particulate_matter_10um", "pm10", "PM10"),
        ("CAMS EAC4", "total_aerosol_optical_depth_550nm", "aod550", "AOD 550 nm"),
        ("CAMS EAC4", "total_column_carbon_monoxide", "tcco", "CO total en columna"),
        ("CAMS EAC4", "total_column_nitrogen_dioxide", "tcno2", "NO2 total en columna"),
        ("CAMS EAC4", "total_column_ozone", "gtco3", "Ozono total en columna"),
    ],
    columns=["fuente", "variable_api", "variable_netcdf", "descripcion"],
)

TABLA_VARIABLES_SENTINEL5P = pd.DataFrame(
    [
        ("Sentinel-5P", "NO2", "S5P_NO2", "Dioxido de nitrogeno"),
        ("Sentinel-5P", "CO", "S5P_CO", "Monoxido de carbono"),
        ("Sentinel-5P", "AER_AI_354_388", "S5P_Aerosoles", "Indice de aerosoles"),
    ],
    columns=["fuente", "variable_api", "variable_netcdf", "descripcion"],
)

TABLA_VARIABLES_MODIS = pd.DataFrame(
    [
        ("MODIS", "Optical_Depth_Land_And_Ocean", "aod", "Aerosol Optical Depth"),
    ],
    columns=["fuente", "variable_api", "variable_netcdf", "descripcion"],
)

TABLA_VARIABLES_GPM = pd.DataFrame(
    [
        ("GPM IMERG Final", "precipitation", "precipitation", "Precipitacion diaria GPM"),
    ],
    columns=["fuente", "variable_api", "variable_netcdf", "descripcion"],
)


S5P_PRODUCTS = {
    "NO2": "NO2",
    "CO": "CO",
    "Aerosoles": "AER_AI_354_388",
}

MODIS_CONCEPT_IDS = ["C1443420430-LAADS", "C1443528505-LAADS"]  # Terra y Aqua 3 km.
GPM_CONCEPT_ID = "C2723754864-GES_DISC"  # GPM IMERG Final v07.


def tabla_variables(fenomeno: str | None = None) -> pd.DataFrame:
    """Devuelve las tablas de variables usadas por cada bloque."""
    tablas = {
        "olas_calor": TABLA_VARIABLES_OLAS_CALOR,
        "precipitacion": TABLA_VARIABLES_PRECIPITACION,
        "calidad_aire": TABLA_VARIABLES_CALIDAD_AIRE_CAMS,
        "cams": TABLA_VARIABLES_CALIDAD_AIRE_CAMS,
        "sentinel5p": TABLA_VARIABLES_SENTINEL5P,
        "modis": TABLA_VARIABLES_MODIS,
        "gpm": TABLA_VARIABLES_GPM,
    }
    if fenomeno is None:
        return pd.concat(
            [df.assign(fenomeno=nombre) for nombre, df in tablas.items()],
            ignore_index=True,
        )
    if fenomeno not in tablas:
        raise ValueError(f"Fenomeno no reconocido: {fenomeno}")
    return tablas[fenomeno].copy()


def _as_years(years: Iterable[int | str]) -> list[str]:
    return [str(y) for y in years]


def _as_months(months: Iterable[int | str]) -> list[str]:
    return [str(int(m)).zfill(2) for m in months]


def _days_for_month(year: int | str, month: int | str) -> list[str]:
    _, last_day = calendar.monthrange(int(year), int(month))
    return [str(d).zfill(2) for d in range(1, last_day + 1)]


def _hours_hourly() -> list[str]:
    return [f"{h:02d}:00" for h in range(24)]


def _compress_encoding(ds: xr.Dataset, complevel: int = 5) -> dict:
    return {var: {"zlib": True, "complevel": complevel} for var in ds.data_vars}


def _ensure_clean(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _safe_close(datasets: Iterable[xr.Dataset]) -> None:
    for ds in datasets:
        try:
            ds.close()
        except Exception:
            pass


def crear_cliente_cds():
    try:
        import cdsapi
    except ImportError as exc:
        raise ImportError("Falta cdsapi. Instala con: pip install cdsapi") from exc
    return cdsapi.Client()


def crear_cliente_ads(config: PipelineConfig):
    try:
        import cdsapi
    except ImportError as exc:
        raise ImportError("Falta cdsapi. Instala con: pip install cdsapi") from exc

    if config.ads_key:
        return cdsapi.Client(url=config.ads_url, key=config.ads_key)
    return cdsapi.Client(url=config.ads_url)


def descargar_y_extraer_cds(client, dataset: str, request: dict, temp_name: Path) -> Path:
    """
    Descarga robusta de CDS/ADS.

    Algunos endpoints devuelven ZIP y otros NetCDF directo; esta funcion maneja
    ambos casos y devuelve la ruta del primer NetCDF encontrado.
    """
    zip_path = temp_name.with_suffix(".zip")
    folder = temp_name
    nc_direct = temp_name.with_suffix(".nc")

    _ensure_clean(zip_path)
    _ensure_clean(folder)
    _ensure_clean(nc_direct)

    client.retrieve(dataset, request, str(zip_path))

    folder.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(folder)
        nc_files = sorted(folder.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"ZIP descargado sin .nc: {zip_path}")
        return nc_files[0]
    except zipfile.BadZipFile:
        zip_path.rename(nc_direct)
        return nc_direct


def descargar_olas_calor(
    config: PipelineConfig | None = None,
    client=None,
) -> list[Path]:
    """Descarga ERA5 mensual para olas de calor."""
    config = config or PipelineConfig()
    client = client or crear_cliente_cds()
    output_folder = config.base_dir / "olas_calor"
    output_folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for year in _as_years(config.years):
        for month in _as_months(config.months):
            output = output_folder / f"ERA5_Calor_{year}_{month}.nc"
            if output.exists() and not config.overwrite:
                print(f"Saltando {output}, ya existe.")
                paths.append(output)
                continue

            print(f"\nOlas de calor {year}-{month}")
            month_temp = config.temp_dir / "olas_calor" / f"{year}_{month}"
            month_temp.mkdir(parents=True, exist_ok=True)

            try:
                common = {
                    "product_type": "reanalysis",
                    "data_format": "netcdf",
                    "year": year,
                    "month": month,
                    "day": _days_for_month(year, month),
                    "time": _hours_hourly(),
                    "area": config.area_era5,
                }
                nc_estado = descargar_y_extraer_cds(
                    client,
                    "reanalysis-era5-single-levels",
                    {
                        **common,
                        "variable": [
                            "2m_temperature",
                            "skin_temperature",
                            "total_cloud_cover",
                            "volumetric_soil_water_layer_1",
                        ],
                    },
                    month_temp / "estado",
                )
                nc_flujos = descargar_y_extraer_cds(
                    client,
                    "reanalysis-era5-single-levels",
                    {
                        **common,
                        "variable": [
                            "surface_sensible_heat_flux",
                            "surface_solar_radiation_downwards",
                        ],
                    },
                    month_temp / "flujos",
                )
                nc_extremos = descargar_y_extraer_cds(
                    client,
                    "reanalysis-era5-single-levels",
                    {
                        **common,
                        "variable": [
                            "maximum_2m_temperature_since_previous_post_processing",
                            "minimum_2m_temperature_since_previous_post_processing",
                        ],
                    },
                    month_temp / "extremos",
                )
                nc_altura = descargar_y_extraer_cds(
                    client,
                    "reanalysis-era5-pressure-levels",
                    {
                        **common,
                        "variable": ["geopotential"],
                        "pressure_level": ["500"],
                    },
                    month_temp / "altura",
                )

                datasets = [xr.open_dataset(p) for p in [nc_estado, nc_flujos, nc_extremos, nc_altura]]
                ds_final = xr.merge(datasets, join="override")
                _ensure_clean(output)
                ds_final.to_netcdf(output, encoding=_compress_encoding(ds_final))
                _safe_close([ds_final, *datasets])
                paths.append(output)
                print(f"Guardado: {output}")
            except Exception:
                print(f"Error descargando olas_calor {year}-{month}")
                traceback.print_exc()
            finally:
                shutil.rmtree(month_temp, ignore_errors=True)

    return paths


def descargar_precipitacion(
    config: PipelineConfig | None = None,
    client=None,
) -> list[Path]:
    """Descarga ERA5 mensual para precipitaciones extremas."""
    config = config or PipelineConfig()
    client = client or crear_cliente_cds()
    output_folder = config.base_dir / "precipitacion"
    output_folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for year in _as_years(config.years):
        for month in _as_months(config.months):
            output = output_folder / f"ERA5_Precip_{year}_{month}.nc"
            if output.exists() and not config.overwrite:
                print(f"Saltando {output}, ya existe.")
                paths.append(output)
                continue

            print(f"\nPrecipitacion {year}-{month}")
            month_temp = config.temp_dir / "precipitacion" / f"{year}_{month}"
            month_temp.mkdir(parents=True, exist_ok=True)

            try:
                common = {
                    "product_type": "reanalysis",
                    "data_format": "netcdf",
                    "year": year,
                    "month": month,
                    "day": _days_for_month(year, month),
                    "time": _hours_hourly(),
                    "area": config.area_era5,
                }
                nc_lluvia = descargar_y_extraer_cds(
                    client,
                    "reanalysis-era5-single-levels",
                    {
                        **common,
                        "variable": [
                            "total_precipitation",
                            "convective_precipitation",
                            "large_scale_precipitation",
                        ],
                    },
                    month_temp / "lluvia",
                )
                nc_dinamica = descargar_y_extraer_cds(
                    client,
                    "reanalysis-era5-single-levels",
                    {
                        **common,
                        "variable": [
                            "total_column_water_vapour",
                            "mean_sea_level_pressure",
                            "10m_u_component_of_wind",
                            "10m_v_component_of_wind",
                        ],
                    },
                    month_temp / "dinamica",
                )
                nc_altura = descargar_y_extraer_cds(
                    client,
                    "reanalysis-era5-pressure-levels",
                    {
                        **common,
                        "variable": ["u_component_of_wind", "v_component_of_wind"],
                        "pressure_level": ["850"],
                    },
                    month_temp / "altura",
                )

                ds1 = xr.open_dataset(nc_lluvia)
                ds2 = xr.open_dataset(nc_dinamica)
                ds3 = xr.open_dataset(nc_altura)
                renames = {}
                if "u" in ds3.data_vars:
                    renames["u"] = "u_850"
                if "v" in ds3.data_vars:
                    renames["v"] = "v_850"
                if renames:
                    ds3 = ds3.rename(renames)

                ds_final = xr.merge([ds1, ds2, ds3], join="override")
                _ensure_clean(output)
                ds_final.to_netcdf(output, encoding=_compress_encoding(ds_final))
                _safe_close([ds_final, ds1, ds2, ds3])
                paths.append(output)
                print(f"Guardado: {output}")
            except Exception:
                print(f"Error descargando precipitacion {year}-{month}")
                traceback.print_exc()
            finally:
                shutil.rmtree(month_temp, ignore_errors=True)

    return paths


def descargar_calidad_aire_cams(
    config: PipelineConfig | None = None,
    client=None,
) -> list[Path]:
    """Descarga CAMS EAC4 mensual para calidad del aire."""
    config = config or PipelineConfig()
    client = client or crear_cliente_ads(config)
    output_folder = config.base_dir / "calidad_aire"
    output_folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for year in _as_years(config.years):
        for month in _as_months(config.months):
            output = output_folder / f"CAMS_Aire_{year}_{month}.nc"
            if output.exists() and not config.overwrite:
                print(f"Saltando {output}, ya existe.")
                paths.append(output)
                continue

            _, last_day = calendar.monthrange(int(year), int(month))
            print(f"\nCAMS calidad aire {year}-{month}")
            month_temp = config.temp_dir / "calidad_aire" / f"{year}_{month}"
            month_temp.mkdir(parents=True, exist_ok=True)

            request = {
                "format": "netcdf",
                "variable": [
                    "particulate_matter_2.5um",
                    "particulate_matter_10um",
                    "total_aerosol_optical_depth_550nm",
                    "total_column_carbon_monoxide",
                    "total_column_nitrogen_dioxide",
                    "total_column_ozone",
                ],
                "date": f"{year}-{month}-01/{year}-{month}-{last_day}",
                "time": ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"],
                "area": config.area_era5,
            }

            try:
                nc_cams = descargar_y_extraer_cds(
                    client,
                    "cams-global-reanalysis-eac4",
                    request,
                    month_temp / "cams",
                )
                ds = xr.open_dataset(nc_cams)
                _ensure_clean(output)
                ds.to_netcdf(output, encoding=_compress_encoding(ds))
                ds.close()
                paths.append(output)
                print(f"Guardado: {output}")
            except Exception:
                print(f"Error descargando CAMS {year}-{month}")
                traceback.print_exc()
            finally:
                shutil.rmtree(month_temp, ignore_errors=True)

    return paths


def crear_cliente_openeo(authenticate: bool = True):
    try:
        import openeo
    except ImportError as exc:
        raise ImportError("Falta openeo. Instala con: pip install openeo") from exc
    con = openeo.connect("https://openeo.dataspace.copernicus.eu")
    if authenticate:
        con.authenticate_oidc()
    return con


def descargar_sentinel5p(
    config: PipelineConfig | None = None,
    producto: str = "todos",
    con=None,
    resolution: float = 0.05,
    authenticate: bool = True,
) -> list[Path]:
    """Descarga productos Sentinel-5P mensuales mediante openEO."""
    config = config or PipelineConfig()
    con = con or crear_cliente_openeo(authenticate=authenticate)
    output_folder = config.base_dir / "satelite_s5p"
    output_folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if producto.lower() == "todos":
        products = S5P_PRODUCTS
    else:
        canonical = next((p for p in S5P_PRODUCTS if p.lower() == producto.lower()), None)
        if canonical is None:
            raise ValueError(f"Producto Sentinel-5P no reconocido: {producto}")
        products = {canonical: S5P_PRODUCTS[canonical]}

    for year in config.years:
        for month in config.months:
            last_day = calendar.monthrange(int(year), int(month))[1]
            date_start = f"{int(year)}-{int(month):02d}-01"
            date_end = f"{int(year)}-{int(month):02d}-{last_day:02d}"

            for product_name, band_id in products.items():
                output = output_folder / f"S5P_{product_name}_{int(year)}_{int(month):02d}.nc"
                if output.exists() and not config.overwrite:
                    print(f"Saltando {output}, ya existe.")
                    paths.append(output)
                    continue

                print(f"\nSentinel-5P {product_name} {date_start}/{date_end}")
                for attempt in range(1, 4):
                    try:
                        cube = con.load_collection(
                            "SENTINEL_5P_L2",
                            spatial_extent=BBOX_SPAIN_OPENEO,
                            temporal_extent=[date_start, date_end],
                            bands=[band_id],
                        )
                        cube = cube.resample_spatial(
                            resolution=resolution,
                            projection=4326,
                            method="bilinear",
                        )
                        _ensure_clean(output)
                        cube.download(str(output), format="NetCDF")
                        paths.append(output)
                        print(f"Guardado: {output}")
                        time.sleep(5)
                        break
                    except Exception as exc:
                        print(f"Error Sentinel-5P intento {attempt}: {exc}")
                        if attempt == 3:
                            traceback.print_exc()
                        time.sleep(10 * attempt)

    return paths


def login_earthaccess(strategy: str = "interactive") -> None:
    try:
        import earthaccess
    except ImportError as exc:
        raise ImportError("Falta earthaccess. Instala con: pip install earthaccess") from exc
    earthaccess.login(strategy=strategy)


def descargar_gpm(
    config: PipelineConfig | None = None,
    login_strategy: str = "interactive",
) -> list[Path]:
    """Descarga y consolida GPM IMERG Final diario en archivos mensuales."""
    try:
        import earthaccess
    except ImportError as exc:
        raise ImportError("Falta earthaccess. Instala con: pip install earthaccess") from exc

    config = config or PipelineConfig()
    login_earthaccess(login_strategy)
    output_folder = config.base_dir / "nasa_gpm_mensual"
    output_folder.mkdir(parents=True, exist_ok=True)
    temp_folder = config.temp_dir / "gpm"
    temp_folder.mkdir(parents=True, exist_ok=True)
    paths_out: list[Path] = []

    for year in config.years:
        for month in config.months:
            output = output_folder / f"GPM_Spain_{int(year)}_{int(month):02d}.nc"
            if output.exists() and not config.overwrite:
                print(f"Saltando {output}, ya existe.")
                paths_out.append(output)
                continue

            print(f"\nGPM {int(year)}-{int(month):02d}")
            last_day = calendar.monthrange(int(year), int(month))[1]
            datasets = []

            for day in range(1, last_day + 1):
                date_str = f"{int(year)}-{int(month):02d}-{day:02d}"
                try:
                    results = earthaccess.search_data(
                        concept_id=GPM_CONCEPT_ID,
                        temporal=(date_str, date_str),
                    )
                    if not results:
                        print(f"Sin datos para {date_str}")
                        continue
                    downloaded = earthaccess.download(results, str(temp_folder))
                    if not downloaded:
                        continue

                    raw_file = Path(downloaded[0])
                    try:
                        with xr.open_dataset(raw_file) as ds:
                            var_name = "precipitation" if "precipitation" in ds.data_vars else "precipitationCal"
                            ds_crop = ds[[var_name]].sel(lon=BOX_SPAIN_GPM["lon"], lat=BOX_SPAIN_GPM["lat"])
                            if set(["time", "lat", "lon"]).issubset(ds_crop.dims):
                                ds_crop = ds_crop.transpose("time", "lat", "lon")
                            ds_crop.load()
                            datasets.append(ds_crop)
                    finally:
                        _ensure_clean(raw_file)
                except Exception as exc:
                    print(f"Error GPM {date_str}: {exc}")

            if datasets:
                ds_final = xr.concat(datasets, dim="time")
                var = list(ds_final.data_vars)[0]
                _ensure_clean(output)
                ds_final.to_netcdf(
                    output,
                    engine="netcdf4",
                    encoding={var: {"zlib": True, "complevel": 5}},
                )
                ds_final.close()
                paths_out.append(output)
                print(f"Guardado: {output}")
            else:
                print("Mes GPM vacio.")

    shutil.rmtree(temp_folder, ignore_errors=True)
    return paths_out


def _modis_grid(config: PipelineConfig):
    lat_grid = np.arange(config.modis_lat_min, config.modis_lat_max, config.modis_step)
    lon_grid = np.arange(config.modis_lon_min, config.modis_lon_max, config.modis_step)
    return lat_grid, lon_grid


def _acumular_modis_hdf(path: Path, grid_sum: np.ndarray, grid_count: np.ndarray, config: PipelineConfig) -> bool:
    try:
        from pyhdf.SD import SD, SDC
    except ImportError as exc:
        raise ImportError("Falta pyhdf. Instala pyhdf en el entorno donde descargues MODIS.") from exc

    hdf = None
    try:
        hdf = SD(str(path.resolve()), SDC.READ)
        lat_data = hdf.select("Latitude").get()
        lon_data = hdf.select("Longitude").get()
        aod_obj = hdf.select("Optical_Depth_Land_And_Ocean")
        aod_data = aod_obj.get().astype(float)

        attrs = aod_obj.attributes()
        fill_val = attrs.get("_FillValue", -9999)
        scale = attrs.get("scale_factor", 1.0)
        offset = attrs.get("add_offset", 0.0)

        aod_data[aod_data == fill_val] = np.nan
        aod_data = (aod_data - offset) * scale

        mask = (
            ~np.isnan(aod_data)
            & (lat_data >= config.modis_lat_min)
            & (lat_data < config.modis_lat_max)
            & (lon_data >= config.modis_lon_min)
            & (lon_data < config.modis_lon_max)
        )
        if not np.any(mask):
            return False

        lat_idx = ((lat_data[mask] - config.modis_lat_min) / config.modis_step).astype(int)
        lon_idx = ((lon_data[mask] - config.modis_lon_min) / config.modis_step).astype(int)
        np.add.at(grid_sum, (lat_idx, lon_idx), aod_data[mask])
        np.add.at(grid_count, (lat_idx, lon_idx), 1)
        return True
    finally:
        if hdf is not None:
            hdf.end()


def descargar_modis_mensual(
    config: PipelineConfig | None = None,
    login_strategy: str = "interactive",
) -> list[Path]:
    """Descarga MODIS AOD 3 km y genera medias mensuales."""
    try:
        import earthaccess
    except ImportError as exc:
        raise ImportError("Falta earthaccess. Instala con: pip install earthaccess") from exc

    config = config or PipelineConfig()
    login_earthaccess(login_strategy)
    output_folder = config.base_dir / "modis_mensual_procesado"
    output_folder.mkdir(parents=True, exist_ok=True)
    temp_folder = config.temp_dir / "modis_mensual"
    temp_folder.mkdir(parents=True, exist_ok=True)
    lat_grid, lon_grid = _modis_grid(config)
    n_lat, n_lon = len(lat_grid), len(lon_grid)
    paths_out: list[Path] = []

    for year in config.years:
        for month in config.months:
            output = output_folder / f"MODIS_AOD_Spain_{int(year)}_{int(month):02d}.nc"
            if output.exists() and not config.overwrite:
                print(f"Saltando {output}, ya existe.")
                paths_out.append(output)
                continue

            grid_sum = np.zeros((n_lat, n_lon))
            grid_count = np.zeros((n_lat, n_lon))
            files_used = 0
            last_day = calendar.monthrange(int(year), int(month))[1]
            blocks = [
                (f"{int(year)}-{int(month):02d}-01", f"{int(year)}-{int(month):02d}-15"),
                (f"{int(year)}-{int(month):02d}-16", f"{int(year)}-{int(month):02d}-{last_day:02d}"),
            ]
            print(f"\nMODIS mensual {int(year)}-{int(month):02d}")

            for start, end in blocks:
                for concept_id in MODIS_CONCEPT_IDS:
                    try:
                        results = earthaccess.search_data(
                            concept_id=concept_id,
                            bounding_box=(
                                config.modis_lon_min,
                                config.modis_lat_min,
                                config.modis_lon_max,
                                config.modis_lat_max,
                            ),
                            temporal=(start, end),
                        )
                        if not results:
                            continue
                        downloaded = earthaccess.download(results, str(temp_folder))
                        for downloaded_path in downloaded:
                            path = Path(downloaded_path)
                            try:
                                if _acumular_modis_hdf(path, grid_sum, grid_count, config):
                                    files_used += 1
                            except Exception as exc:
                                print(f"Error procesando MODIS {path.name}: {exc}")
                            finally:
                                _ensure_clean(path)
                    except Exception as exc:
                        print(f"Error bloque MODIS {start}/{end}: {exc}")

            if files_used > 0:
                with np.errstate(invalid="ignore"):
                    aod_mean = grid_sum / grid_count
                ds = xr.Dataset(
                    data_vars={"aod": (("lat", "lon"), aod_mean)},
                    coords={"lat": lat_grid, "lon": lon_grid},
                    attrs={"description": "MODIS AOD 3km Monthly Mean (Terra+Aqua)"},
                ).sortby("lat", ascending=False)
                _ensure_clean(output)
                ds.to_netcdf(output)
                ds.close()
                paths_out.append(output)
                print(f"Guardado: {output} ({files_used} barridos)")
            else:
                print("Mes MODIS vacio.")

    shutil.rmtree(temp_folder, ignore_errors=True)
    return paths_out


def descargar_modis_diario(
    config: PipelineConfig | None = None,
    login_strategy: str = "interactive",
) -> list[Path]:
    """Descarga MODIS AOD 3 km y genera cubos diarios mensuales."""
    try:
        import earthaccess
    except ImportError as exc:
        raise ImportError("Falta earthaccess. Instala con: pip install earthaccess") from exc

    config = config or PipelineConfig()
    login_earthaccess(login_strategy)
    output_folder = config.base_dir / "modis_diario_full"
    output_folder.mkdir(parents=True, exist_ok=True)
    temp_folder = config.temp_dir / "modis_diario"
    temp_folder.mkdir(parents=True, exist_ok=True)
    lat_grid, lon_grid = _modis_grid(config)
    n_lat, n_lon = len(lat_grid), len(lon_grid)
    paths_out: list[Path] = []

    for year in config.years:
        for month in config.months:
            output = output_folder / f"MODIS_AOD_Daily_Spain_{int(year)}_{int(month):02d}.nc"
            if output.exists() and not config.overwrite:
                print(f"Saltando {output}, ya existe.")
                paths_out.append(output)
                continue

            print(f"\nMODIS diario {int(year)}-{int(month):02d}")
            last_day = calendar.monthrange(int(year), int(month))[1]
            month_data = np.full((last_day, n_lat, n_lon), np.nan)

            for day in range(1, last_day + 1):
                date_str = f"{int(year)}-{int(month):02d}-{day:02d}"
                day_sum = np.zeros((n_lat, n_lon))
                day_count = np.zeros((n_lat, n_lon))
                files_used = 0

                for concept_id in MODIS_CONCEPT_IDS:
                    try:
                        results = earthaccess.search_data(
                            concept_id=concept_id,
                            bounding_box=(
                                config.modis_lon_min,
                                config.modis_lat_min,
                                config.modis_lon_max,
                                config.modis_lat_max,
                            ),
                            temporal=(date_str, date_str),
                        )
                        if not results:
                            continue
                        downloaded = earthaccess.download(results, str(temp_folder))
                        for downloaded_path in downloaded:
                            path = Path(downloaded_path)
                            try:
                                if _acumular_modis_hdf(path, day_sum, day_count, config):
                                    files_used += 1
                            except Exception as exc:
                                print(f"Error procesando MODIS {date_str}: {exc}")
                            finally:
                                _ensure_clean(path)
                    except Exception as exc:
                        print(f"Error descarga MODIS {date_str}: {exc}")

                if files_used > 0:
                    with np.errstate(invalid="ignore"):
                        month_data[day - 1, :, :] = day_sum / day_count

            time_coords = [np.datetime64(f"{int(year)}-{int(month):02d}-{d:02d}") for d in range(1, last_day + 1)]
            ds = xr.Dataset(
                data_vars={"aod": (("time", "lat", "lon"), month_data)},
                coords={"time": time_coords, "lat": lat_grid, "lon": lon_grid},
                attrs={"description": "MODIS AOD 3km Daily Grid"},
            ).sortby("lat", ascending=False)
            _ensure_clean(output)
            ds.to_netcdf(
                output,
                encoding={"aod": {"zlib": True, "complevel": 5, "_FillValue": np.nan}},
            )
            ds.close()
            paths_out.append(output)
            print(f"Guardado: {output}")

    shutil.rmtree(temp_folder, ignore_errors=True)
    return paths_out


def ejecutar_pipeline(
    producto: Producto = "todo",
    config: PipelineConfig | None = None,
    s5p_producto: str = "todos",
    login_strategy: str = "interactive",
    openeo_authenticate: bool = True,
) -> dict[str, list[Path]]:
    """Ejecuta uno o varios bloques de descarga."""
    config = config or PipelineConfig()
    config.base_dir.mkdir(parents=True, exist_ok=True)
    config.temp_dir.mkdir(parents=True, exist_ok=True)

    resultados: dict[str, list[Path]] = {}

    if producto in ("todo", "olas_calor"):
        resultados["olas_calor"] = descargar_olas_calor(config)
    if producto in ("todo", "precipitacion"):
        resultados["precipitacion"] = descargar_precipitacion(config)
    if producto in ("todo", "calidad_aire"):
        resultados["calidad_aire"] = descargar_calidad_aire_cams(config)
    if producto in ("todo", "sentinel5p"):
        resultados["sentinel5p"] = descargar_sentinel5p(
            config,
            producto=s5p_producto,
            authenticate=openeo_authenticate,
        )
    if producto in ("todo", "gpm"):
        resultados["gpm"] = descargar_gpm(config, login_strategy=login_strategy)
    if producto in ("todo", "modis_mensual"):
        resultados["modis_mensual"] = descargar_modis_mensual(config, login_strategy=login_strategy)
    if producto in ("todo", "modis_diario"):
        resultados["modis_diario"] = descargar_modis_diario(config, login_strategy=login_strategy)

    return resultados


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de extraccion de datos TFG.")
    parser.add_argument(
        "--producto",
        choices=[
            "todo",
            "olas_calor",
            "precipitacion",
            "calidad_aire",
            "sentinel5p",
            "gpm",
            "modis_mensual",
            "modis_diario",
        ],
        default="todo",
    )
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2020, 2026)))
    parser.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--base-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--temp-dir", default=str(DEFAULT_TEMP_DIR))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ads-key", default=None)
    parser.add_argument("--s5p-producto", default="todos", help="todos, NO2, CO o Aerosoles")
    parser.add_argument(
        "--earthaccess-login",
        default="interactive",
        help="Estrategia earthaccess.login: interactive, netrc, environment...",
    )
    parser.add_argument("--no-openeo-auth", action="store_true")
    parser.add_argument("--mostrar-tablas", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mostrar_tablas:
        with pd.option_context("display.max_rows", None, "display.max_colwidth", 80):
            print("\nOLAS DE CALOR")
            print(tabla_variables("olas_calor"))
            print("\nPRECIPITACION")
            print(tabla_variables("precipitacion"))
            print("\nCALIDAD DEL AIRE - CAMS")
            print(tabla_variables("calidad_aire"))
            print("\nSENTINEL-5P")
            print(tabla_variables("sentinel5p"))
            print("\nMODIS")
            print(tabla_variables("modis"))
            print("\nGPM")
            print(tabla_variables("gpm"))

    config = PipelineConfig(
        base_dir=Path(args.base_dir),
        temp_dir=Path(args.temp_dir),
        years=args.years,
        months=args.months,
        overwrite=args.overwrite,
        ads_key=args.ads_key,
    )
    ejecutar_pipeline(
        producto=args.producto,
        config=config,
        s5p_producto=args.s5p_producto,
        login_strategy=args.earthaccess_login,
        openeo_authenticate=not args.no_openeo_auth,
    )


if __name__ == "__main__":
    main()
