"""
Data loader optimizado para:
- Lazy loading con Dask (sin cargar todo a memoria)
- Persistencia en Zarr (más rápido que NetCDF)
- Cálculo on-demand de agregaciones
- Estandarizar dimensiones automáticamente
"""
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import zarr
import shutil


REPO_DIR_NAME = "Deteccion_Fenomenos_Meteorologicos_Extremos"


def get_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == REPO_DIR_NAME:
            return parent
    return current.parent


REPO_ROOT = get_repo_root()
DATA_DIR = REPO_ROOT / "data" / "datos_tfg"


def estandarizar_nombres(ds):
    """Estandariza nombres raros de dimensiones a time, lat, lon"""
    renombres = {}
    for nombre in list(ds.dims):
        nombre_lower = nombre.lower()
        if nombre_lower in ['t', 'time_utc', 'date', 'valid_time']:
            renombres[nombre] = 'time'
        elif nombre_lower in ['latitude', 'y']:
            renombres[nombre] = 'lat'
        elif nombre_lower in ['longitude', 'x']:
            renombres[nombre] = 'lon'
    
    # También renombrar coordenadas para mantener consistencia
    for coord in list(ds.coords):
        coord_lower = coord.lower()
        if coord not in renombres:  # No sobreescribir si ya está en renombres
            if coord_lower in ['t', 'time_utc', 'date', 'valid_time']:
                renombres[coord] = 'time'
            elif coord_lower in ['latitude', 'y']:
                renombres[coord] = 'lat'
            elif coord_lower in ['longitude', 'x']:
                renombres[coord] = 'lon'
    
    if renombres:
        return ds.rename(renombres)
    return ds


class DataLoader:
    """Cargador con lazy loading, Zarr cache, y agregaciones on-demand"""
    
    def __init__(self, data_dir: str | Path = DATA_DIR, use_cache: bool = True):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(data_dir) / ".cache_zarr"
        self.use_cache = use_cache
        self.datasets = {}  # Datos base (lazy con Dask)
        self.metadata = {}
        
        if use_cache and not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_all_datasets(self, start_date: Optional[str] = None, 
                         end_date: Optional[str] = None,
                         variables_to_load: Optional[Dict[str, List[str]]] = None,
                         pre_cache_aggregations: bool = True) -> Dict[str, xr.Dataset]:
        """
        Carga datasets con soporte a carga selectiva de variables.
        
        Args:
            start_date: Fecha inicio (ej: '2020-01-01')
            end_date: Fecha fin (ej: '2020-01-31')
            variables_to_load: Dict {dataset_name: [var1, var2, ...]}
                             Si None, carga todas las variables
                             Ej: {"calidad_aire": ["NO2", "O3"]}
            pre_cache_aggregations: Si True, pre-cachea agregaciones W y MS en Zarr
        
        Returns:
            Dict con referencias a los datasets (lazy-loaded)
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directorio {self.data_dir} no encontrado")
        
        dataset_mapping = {
            "calidad_aire": self._load_calidad_aire,
            "modis_diario_full": self._load_modis_diario,
            "nasa_gpm_mensual": self._load_nasa_gpm,
            "satelite_s3_lst": self._load_satelite_lst,
            "olas_calor": self._load_olas_calor,
            "precipitacion": self._load_precipitacion,
            "satelite_s5p_aerosoles": self._load_satelite_s5p_aerosoles,
            "satelite_s5p_no2": self._load_satelite_s5p_no2,
            "satelite_s5p_co": self._load_satelite_s5p_co
        }
        
        for name, loader_func in dataset_mapping.items():
            # Handle S5P datasets specially
            if name.startswith("satelite_s5p"):
                dataset_path = self.data_dir / "satelite_s5p"
            else:
                dataset_path = self.data_dir / name
            
            if dataset_path.exists():
                try:
                    ds = loader_func(dataset_path, start_date, end_date)
                    if ds is not None:
                        ds = estandarizar_nombres(ds)
                        # Añade esta limpieza en load_all_datasets justo después de estandarizar_nombres(ds)
                        # Buscas esta línea: ds = estandarizar_nombres(ds)
                        # Y añades debajo:
                        if 'crs' in ds.data_vars:
                            ds = ds.drop_vars('crs')
                        
                        # Filtrar variables si se especificaron
                        if variables_to_load and name in variables_to_load:
                            requested_vars = variables_to_load[name]
                            available_vars = list(ds.data_vars.keys())
                            vars_to_keep = [v for v in requested_vars if v in available_vars]
                            
                            if vars_to_keep:
                                print(f"  Filtrando variables de {name}: {vars_to_keep}")
                                ds = ds[vars_to_keep]
                            else:
                                print(f"  ⚠️  Ninguna variable solicitada encontrada en {name}")
                                continue
                        
                        # Intentar cargar desde Zarr cache si existe
                        zarr_path = self.cache_dir / f"{name}.zarr"
                        if zarr_path.exists():
                            print(f"  Cargando {name} desde cache Zarr...")
                            ds = xr.open_zarr(zarr_path)
                        else:
                            # Guardar en Zarr para futuros accesos
                            if self.use_cache:
                                print(f"  Guardando {name} en Zarr cache...")
                                try:
                                    ds.to_zarr(zarr_path, mode='w')
                                except:
                                    pass  # Si falla, continuamos sin cache
                        
                        # Pre-cachear agregaciones (W, MS) si está habilitado
                        if pre_cache_aggregations and self.use_cache:
                            self._pre_cache_aggregations(name, ds)
                        
                        self.datasets[name] = ds
                        print(f"✓ Cargado: {name}")
                except Exception as e:
                    print(f"⚠ Error cargando {name}: {str(e)[:80]}")
        
        return self.datasets
    
    # def _concat_all_files(self, files: List[Path], 
    #                      dim_name: str = 'time',
    #                      start_date: Optional[str] = None,
    #                      end_date: Optional[str] = None) -> Optional[xr.Dataset]:
    #     """Concatena todos los archivos de un dataset"""
    #     if not files:
    #         return None
        
    #     datasets = []
    #     for f in sorted(files):
    #         try:
    #             ds = xr.open_dataset(f)
    #             datasets.append(ds)
    #         except:
    #             continue
        
    #     if not datasets:
    #         return None
        
    #     # Concatenar
    #     try:
    #         result = xr.concat(datasets, dim=dim_name)
    #     except:
    #         return datasets[0]
        
    #     # Filtrar por fechas
    #     if (start_date or end_date) and dim_name in result.dims:
    #         result = result.sel({dim_name: slice(start_date, end_date)})
        
    #     return result

    def _concat_all_files(self, files: List[Path], 
                         dim_name: str = 'time',
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Optional[xr.Dataset]:
        """Abre múltiples archivos, con o sin Dask"""
        if not files:
            return None
        
        # Convertimos los objetos Path a strings
        files_str = [str(f) for f in sorted(files)]
        
        # Intentar con Dask si está disponible
        try:
            result = xr.open_mfdataset(
                files_str, 
                combine='by_coords',
                chunks='auto',
                engine='netcdf4'
            )
            # Renombrar dimensiones después de cargar
            result = estandarizar_nombres(result)
            
            # Usar el nombre estandarizado para la dimensión temporal
            if 'time' in result.dims:
                dim_name = 'time'
            
            if (start_date or end_date) and dim_name in result.dims:
                result = result.sel({dim_name: slice(start_date, end_date)})
            return result
        except Exception:
            pass
        
        # Fallback: abrir sin Dask (sin lazy loading)
        try:
            result = xr.open_mfdataset(
                files_str,
                combine='by_coords',
                engine='netcdf4'
            )
            # Renombrar dimensiones
            result = estandarizar_nombres(result)
            
            if 'time' in result.dims:
                dim_name = 'time'
            
            if (start_date or end_date) and dim_name in result.dims:
                result = result.sel({dim_name: slice(start_date, end_date)})
            return result
        except Exception:
            pass
        
        # Último fallback: abrir archivo por archivo y concatenar
        try:
            datasets = []
            for f in sorted(files):
                try:
                    ds = xr.open_dataset(f, engine='netcdf4')
                    # Renombrar antes de concatenar
                    ds = estandarizar_nombres(ds)
                    datasets.append(ds)
                except:
                    continue
            
            if not datasets:
                return None
            
            # Encontrar la dimensión temporal después de renombrar
            time_dim = 'time' if 'time' in datasets[0].dims else dim_name
            
            if len(datasets) == 1:
                result = datasets[0]
            else:
                result = xr.concat(datasets, dim=time_dim)
            
            if (start_date or end_date) and time_dim in result.dims:
                result = result.sel({time_dim: slice(start_date, end_date)})
            
            return result
        except Exception as e:
            print(f"⚠️  Error al concatenar archivos: {str(e)[:80]}")
            return None
    
    def _load_calidad_aire(self, path, start_date=None, end_date=None):
        files = sorted(path.glob("*.nc"))
        ds = self._concat_all_files(files, 'time', start_date, end_date)
        # Ensure renaming is applied before returning
        if ds is not None:
            ds = estandarizar_nombres(ds)
        return ds
    
    def _load_modis_diario(self, path, start_date=None, end_date=None):
        files = sorted(path.glob("*.nc"))
        return self._concat_all_files(files, 'time', start_date, end_date)
    
    def _load_modis_mensual(self, path, start_date=None, end_date=None):
        files = sorted(path.glob("*.nc"))
        if files:
            return xr.open_dataset(files[0])
        return None
    
    def _load_nasa_gpm(self, path, start_date=None, end_date=None):
        files = sorted(path.glob("*.nc"))
        return self._concat_all_files(files, 'time', start_date, end_date)
    
    # Modifica esta función para cargar todo el dataset de LST
    def _load_satelite_lst(self, path, start_date=None, end_date=None):
        """Carga TODOS los archivos de LST, no solo el primero"""
        files = sorted(path.glob("*.nc"))
        if not files:
            return None
        # Usamos la función de concatenación que ya tienes para que cargue toda la serie temporal
        return self._concat_all_files(files, 'time', start_date, end_date)
    
    def _load_olas_calor(self, path, start_date=None, end_date=None):
        files = sorted(path.glob("*.nc"))
        return self._concat_all_files(files, 'time', start_date, end_date)
    
    def _load_precipitacion(self, path, start_date=None, end_date=None):
        files = sorted(path.glob("*.nc"))
        return self._concat_all_files(files, 'time', start_date, end_date)
    
    def _load_satelite_s5p_aerosoles(self, path, start_date=None, end_date=None):
        """Carga datos S5P Aerosoles (AER_AI_354_388)"""
        files = sorted(path.glob("S5P_Aerosoles_*.nc"))
        ds = self._concat_all_files(files, 'time', start_date, end_date)
        return ds
    
    def _load_satelite_s5p_no2(self, path, start_date=None, end_date=None):
        """Carga datos S5P NO2"""
        files = sorted(path.glob("S5P_NO2_*.nc"))
        ds = self._concat_all_files(files, 'time', start_date, end_date)
        return ds
    
    def _load_satelite_s5p_co(self, path, start_date=None, end_date=None):
        """Carga datos S5P CO"""
        files = sorted(path.glob("S5P_CO_*.nc"))
        ds = self._concat_all_files(files, 'time', start_date, end_date)
        return ds
    
    def aggregate_temporal(self, dataset: xr.Dataset, frequency: str = 'D') -> xr.Dataset:
        """Reagrega dataset a diferente granularidad temporal de forma robusta"""
        if frequency == 'D':
            return dataset
            
        # 1. Asegurar que 'time' existe y es una coordenada válida
        if 'time' not in dataset.coords and 'time' not in dataset.dims:
            return dataset
        
        try:
            # 2. SELECCIONAR SOLO NUMÉRICOS: Esto mata el error de "dtypes compatible with add"
            ds_numeric = dataset.select_dtypes(include=[np.number])
            
            # 3. ELIMINAR CRS Y BASURA: Si 'crs' se coló como variable, la quitamos
            if 'crs' in ds_numeric.data_vars:
                ds_numeric = ds_numeric.drop_vars('crs')
                
            # 4. Verificar que quedan variables con la dimensión tiempo
            vars_validas = [v for v in ds_numeric.data_vars if 'time' in ds_numeric[v].dims]
            if not vars_validas:
                return dataset
                
            ds_to_process = ds_numeric[vars_validas]

            # 5. RESAMPLE: Usamos min_count=1 para que si hay algún dato en la semana, lo saque.
            # Si todo es nulo, sacará nulo pero no dará error de tipos.
            result = ds_to_process.resample(time=frequency).mean(dim='time', skipna=True)
            
            return result
        except Exception as e:
            print(f"  ⚠️ Error crítico en resample ({frequency}): {str(e)}")
            return dataset

    
    def interpolate_to_common_grid(self, dataset: xr.Dataset, 
                                  resolution: float = 0.1) -> xr.Dataset:
        """
        Interpola a una rejilla común
        
        Args:
            dataset: Dataset a interpolar
            resolution: Resolución en grados
        
        Returns:
            Dataset interpolado
        """
        if 'lat' not in dataset.dims or 'lon' not in dataset.dims:
            return dataset
        
        lat_min = float(dataset.lat.min())
        lat_max = float(dataset.lat.max())
        lon_min = float(dataset.lon.min())
        lon_max = float(dataset.lon.max())
        
        new_lat = np.arange(lat_max, lat_min, -resolution)
        new_lon = np.arange(lon_min, lon_max, resolution)
        
        try:
            return dataset.interp(lat=new_lat, lon=new_lon, method='linear')
        except:
            return dataset
    
    def get_aggregated_dataset(self, dataset_name: str, frequency: str = 'D') -> Optional[xr.Dataset]:
        """
        Obtiene dataset agregado bajo demanda (lazy loading).
        Primero intenta cargar desde cache, luego calcula si es necesario.
        
        Args:
            dataset_name: Nombre del dataset
            frequency: 'D' (diario), 'W' (semanal), 'MS' (mensual)
        
        Returns:
            Dataset agregado (calculado al momento, sin cargar todo a memoria)
        """
        if dataset_name not in self.datasets:
            return None
        
        # Si es diario, devolver como está
        if frequency == 'D':
            return self.datasets[dataset_name]
        
        # Intentar cargar desde cache pre-generado
        if self.use_cache:
            cache_key = f"{dataset_name}_{frequency}"
            cached_path = self.cache_dir / f"{cache_key}.zarr"
            if cached_path.exists():
                print(f"  Cargando {cache_key} desde cache pre-generado...")
                try:
                    return xr.open_zarr(cached_path)
                except:
                    pass
        
        # Calcular agregación on-demand (lazy)
        base_ds = self.datasets[dataset_name]
        agg_ds = self.aggregate_temporal(base_ds, frequency)
        return agg_ds
    
    def _pre_cache_aggregations(self, dataset_name: str, dataset: xr.Dataset):
        """
        Pre-cachea agregaciones semanales (W) y mensuales (MS) en Zarr.
        Solo se ejecuta una vez por dataset (al cargar).
        
        Args:
            dataset_name: Nombre del dataset
            dataset: Dataset a pre-cachear
        """
        if 'time' not in dataset.dims:
            return
        
        frequencies = ['W', 'MS']
        
        for freq in frequencies:
            cache_key = f"{dataset_name}_{freq}"
            cached_path = self.cache_dir / f"{cache_key}.zarr"
            
            # Solo crear si no existe
            if cached_path.exists():
                continue
            
            try:
                print(f"  📦 Pre-cacheando {cache_key}...")
                agg_ds = self.aggregate_temporal(dataset, freq)
                agg_ds.to_zarr(cached_path, mode='w')
                print(f"    ✓ {cache_key} guardado en Zarr")
            except Exception as e:
                print(f"    ⚠️  Error pre-cacheando {cache_key}: {str(e)[:50]}")

    
    # def get_timeseries(self, dataset: xr.Dataset, variable: str,
    #                   spatial_agg: str = 'mean') -> Optional[pd.Series]:
    #     """Obtiene serie temporal agregada espacialmente"""
    #     if variable not in dataset.data_vars:
    #         return None
        
    #     data = dataset[variable]
        
    #     # Agregación espacial
    #     spatial_dims = [d for d in data.dims if d not in ['time']]
    #     if spatial_dims:
    #         if spatial_agg == 'mean':
    #             ts = data.mean(dim=spatial_dims)
    #         elif spatial_agg == 'max':
    #             ts = data.max(dim=spatial_dims)
    #         else:
    #             ts = data.mean(dim=spatial_dims)
    #     else:
    #         ts = data
        
    #     ts_pd = ts.to_pandas()
    #     if isinstance(ts_pd, pd.DataFrame):
    #         ts_pd = ts_pd.iloc[:, 0]
        
    #     return ts_pd
    
    # def get_spatial_field(self, dataset: xr.Dataset, variable: str,
    #                      time_index: int = 0) -> Optional[np.ndarray]:
    #     """Obtiene campo espacial en un tiempo específico"""
    #     if variable not in dataset.data_vars:
    #         return None
        
    #     data = dataset[variable]
        
    #     # Seleccionar primer tiempo
    #     if 'time' in data.dims:
    #         data = data.isel(time=time_index)
        
    #     return data.values

    def get_regional_timeseries(self, dataset: xr.Dataset, variable: str,
                                lat_bounds: Tuple[float, float] = None,
                                lon_bounds: Tuple[float, float] = None,
                                spatial_agg: str = 'mean') -> Optional[pd.Series]:
        """Obtiene serie temporal para una región geográfica rectángular.

        Args:
            dataset: xr.Dataset con coordenadas lat/lon/time
            variable: nombre de la variable de interés
            lat_bounds: (lat_min, lat_max) en grados
            lon_bounds: (lon_min, lon_max) en grados
            spatial_agg: 'mean' | 'max' | 'min' | 'median'

        Returns:
            pd.Series de tiempo con la agregación espacial en la región.
        """
        if variable not in dataset.data_vars:
            return None

        ds = dataset

        # Normalizar nombres si no vienen en lat/lon
        coord_renames = {}
        if 'lat' not in ds.coords:
            if 'latitude' in ds.coords:
                coord_renames['latitude'] = 'lat'
            elif 'y' in ds.coords:
                coord_renames['y'] = 'lat'
        if 'lon' not in ds.coords:
            if 'longitude' in ds.coords:
                coord_renames['longitude'] = 'lon'
            elif 'x' in ds.coords:
                coord_renames['x'] = 'lon'
        if coord_renames:
            ds = ds.rename(coord_renames)

        if lat_bounds is not None and lon_bounds is not None:
            lat_min, lat_max = sorted(lat_bounds)
            lon_min, lon_max = sorted(lon_bounds)

            if 'lat' in ds.coords and 'lon' in ds.coords:
                lat_coord = ds['lat']
                lon_coord = ds['lon']

                # Coordenadas 1D por dimensión
                if ('lat' in ds.dims and 'lon' in ds.dims):
                    if lat_coord.size > 1:
                        lat_asc = float(lat_coord[0]) < float(lat_coord[-1])
                    else:
                        lat_asc = True

                    if lat_asc:
                        ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                    else:
                        ds = ds.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))

                else:
                    # Coordenadas curvilíneas (lat/lon como 2D). Usar máscara.
                    mask = (
                        (lat_coord >= lat_min) & (lat_coord <= lat_max) &
                        (lon_coord >= lon_min) & (lon_coord <= lon_max)
                    )
                    ds = ds.where(mask, drop=True)
            else:
                return None

        data = ds[variable]

        # Agregación espacial
        spatial_dims = [d for d in data.dims if d != 'time']
        if spatial_dims:
            if spatial_agg == 'mean':
                data = data.mean(dim=spatial_dims)
            elif spatial_agg == 'max':
                data = data.max(dim=spatial_dims)
            elif spatial_agg == 'min':
                data = data.min(dim=spatial_dims)
            elif spatial_agg == 'median':
                data = data.median(dim=spatial_dims)
            else:
                data = data.mean(dim=spatial_dims)

        try:
            ts_pd = data.to_pandas()
        except Exception:
            return None

        if isinstance(ts_pd, pd.DataFrame):
            ts_pd = ts_pd.iloc[:, 0]

        return ts_pd

    def get_timeseries(self, dataset: xr.Dataset, variable: str,
                      spatial_agg: str = 'mean') -> Optional[pd.Series]:
        """Obtiene serie temporal agregada espacialmente"""
        if variable not in dataset.data_vars:
            return None
        
        data = dataset[variable]
        
        # Agregación espacial
        spatial_dims = [d for d in data.dims if d not in ['time']]
        if spatial_dims:
            if spatial_agg == 'mean':
                ts = data.mean(dim=spatial_dims)
            elif spatial_agg == 'max':
                ts = data.max(dim=spatial_dims)
            else:
                ts = data.mean(dim=spatial_dims)
        else:
            ts = data
        
        ts_pd = ts.to_pandas()
        if isinstance(ts_pd, pd.DataFrame):
            ts_pd = ts_pd.iloc[:, 0]
        
        return ts_pd
    
    def get_spatial_field(self, dataset: xr.Dataset, variable: str,
                         time_index: int = 0) -> Optional[np.ndarray]:
        """Obtiene campo espacial en un tiempo específico"""
        if variable not in dataset.data_vars:
            return None
        
        data = dataset[variable]
        
        # Seleccionar primer tiempo
        if 'time' in data.dims:
            data = data.isel(time=time_index)
        
        return data.values
