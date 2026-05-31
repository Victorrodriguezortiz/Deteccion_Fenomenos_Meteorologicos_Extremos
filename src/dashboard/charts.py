"""
Generador de gráficas mejorado con:
- Soporte a múltiples granularidades temporales
- Mapas de correlación lageada
- Diagramas de barras con lags
- Interpolación espacial automática
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Optional, Dict, List, Tuple
import folium
import xarray as xr
from scipy.stats import pearsonr, spearmanr
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class ChartGenerator:
    """Generador de gráficas con soporte a granularidades temporales precomputadas"""
    
    def __init__(self, datasets: Dict[str, xr.Dataset], data_loader=None):
        self.datasets = datasets
        self.data_loader = data_loader  # Para acceder a datos preagregados
        sns.set_style("darkgrid")
        plt.rcParams['figure.figsize'] = (14, 6)
        plt.rcParams['font.size'] = 10
    
    def _get_data_for_frequency(self, dataset_name: str, variable: str, frequency: str = 'D'):
        """Obtiene datos ya agregados del data_loader si está disponible"""
        if self.data_loader and hasattr(self.data_loader, 'get_aggregated_dataset'):
            try:
                ds = self.data_loader.get_aggregated_dataset(dataset_name, frequency)
                if ds is not None and variable in ds.data_vars:
                    return ds[variable]
            except:
                pass  # Fallback si falla carga desde data_loader
        
        # Fallback: usar datos base
        try:
            data = self.datasets[dataset_name][variable]
        except:
            # Si no existe la variable, intentar obtener el dataset completo
            data = self.datasets[dataset_name]
        
        # Convertir a DataArray si es necesario
        if isinstance(data, xr.Dataset):
            if variable in data.data_vars:
                data = data[variable]
            else:
                # Usar la primera variable disponible
                first_var = list(data.data_vars)[0]
                data = data[first_var]
        
        # Si se pide granularidad semanal, usar 'W' que es estándar en xarray
        if frequency != 'D':
            # Verificar si time existe como dimensión o coordenada
            time_dim = None
            if 'time' in data.dims:
                time_dim = 'time'
            elif 'time' in data.coords and 'time' not in data.dims:
                # time está como coordenada pero no es dimensión, necesitamos usar set_index si es posible
                pass
            
            if time_dim:
                try:
                    if frequency == 'W':
                        data = data.resample(**{time_dim: '1W'}).mean()
                    elif frequency == 'MS':
                        data = data.resample(**{time_dim: 'MS'}).mean()
                except Exception as e:
                    print(f"⚠️  No se pudo hacer resample para {frequency}: {str(e)[:50]}")
                    pass  # Si falla resample, usar datos sin agregar
        
        return data
    
    def plot_time_series(self, 
                        dataset1: str, 
                        variable1: str,
                        frequency: str = 'D',
                        dataset2: Optional[str] = None,
                        variable2: Optional[str] = None,
                        extreme_threshold: float = 0.95,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> plt.Figure:
        """
        Serie temporal con granularidad configurable
        
        Args:
            frequency: 'D' (diario), 'W' (semanal), 'MS' (mensual)
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Usar datos preagregados si existen
        data1 = self._get_data_for_frequency(dataset1, variable1, frequency)
        
        # Filtrar por rango de fechas si se especifica
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            data1 = data1.sel(time=slice(start_dt, end_dt))
        
        # Agregación espacial (sin resample, ya está agregado temporalmente)
        spatial_dims = [d for d in data1.dims if d not in ['time']]
        if spatial_dims:
            ts1 = data1.mean(dim=spatial_dims)
        else:
            ts1 = data1
        
        ts1_pd = ts1.to_pandas()
        if isinstance(ts1_pd, pd.DataFrame):
            ts1_pd = ts1_pd.iloc[:, 0]
        
        # Calcular extremos
        threshold1 = np.nanpercentile(ts1_pd.dropna(), extreme_threshold * 100)
        extremes1 = ts1_pd >= threshold1
        
        # Plotear
        ax.plot(ts1_pd.index, ts1_pd.values, label=variable1, linewidth=2, color='#1f77b4')
        
        extreme_indices = np.where(extremes1)[0]
        if len(extreme_indices) > 0:
            ax.scatter(ts1_pd.index[extreme_indices], 
                      ts1_pd.values[extreme_indices],
                      color='red', s=100, alpha=0.6, label=f'{variable1} (extremo)', zorder=5)
        
        # Segunda variable si existe
        if dataset2 and variable2:
            data2 = self._get_data_for_frequency(dataset2, variable2, frequency)
            
            # Filtrar por rango de fechas si se especifica
            if start_date and end_date:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                data2 = data2.sel(time=slice(start_dt, end_dt))
            
            spatial_dims = [d for d in data2.dims if d not in ['time']]
            if spatial_dims:
                ts2 = data2.mean(dim=spatial_dims)
            else:
                ts2 = data2
            
            ts2_pd = ts2.to_pandas()
            if isinstance(ts2_pd, pd.DataFrame):
                ts2_pd = ts2_pd.iloc[:, 0]
            
            common_index = ts1_pd.index.intersection(ts2_pd.index)
            ts1_pd = ts1_pd[common_index]
            ts2_pd = ts2_pd[common_index]
            
            ax2 = ax.twinx()
            ax2.plot(ts2_pd.index, ts2_pd.values, label=variable2, linewidth=2, 
                    color='#ff7f0e', linestyle='--')
            
            threshold2 = np.nanpercentile(ts2_pd.dropna(), extreme_threshold * 100)
            extremes2 = ts2_pd >= threshold2
            extreme_indices2 = np.where(extremes2)[0]
            
            if len(extreme_indices2) > 0:
                ax2.scatter(ts2_pd.index[extreme_indices2], 
                           ts2_pd.values[extreme_indices2],
                           color='darkred', s=100, alpha=0.6, label=f'{variable2} (extremo)', zorder=5)
            
            ax.set_ylabel(variable1, color='#1f77b4', fontsize=11)
            ax2.set_ylabel(variable2, color='#ff7f0e', fontsize=11)
            ax.tick_params(axis='y', labelcolor='#1f77b4')
            ax2.tick_params(axis='y', labelcolor='#ff7f0e')
            
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax.set_ylabel(variable1, fontsize=11)
            ax.legend(loc='upper left')
        
        ax.set_xlabel('Tiempo', fontsize=11)
        ax.set_title(f'Serie Temporal ({frequency}): {variable1}' + 
                    (f' vs {variable2}' if variable2 else ''),
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_lagged_correlation(self,
                               dataset1: str,
                               variable1: str,
                               dataset2: str,
                               variable2: str,
                               frequency: str = 'D',
                               max_lag: int = 12,
                               min_periods: int = 10,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> plt.Figure:
        """
        Gráfico de barras con correlación lageada
        
        Args:
            frequency: 'D' (diario), 'W' (semanal), 'MS' (mensual)
        """
        # Usar datos preagregados
        data1 = self._get_data_for_frequency(dataset1, variable1, frequency)
        data2 = self._get_data_for_frequency(dataset2, variable2, frequency)
        
        # Filtrar por rango de fechas si se especifica
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            data1 = data1.sel(time=slice(start_dt, end_dt))
            data2 = data2.sel(time=slice(start_dt, end_dt))
        
        # Agregación espacial
        spatial_dims1 = [d for d in data1.dims if d not in ['time']]
        spatial_dims2 = [d for d in data2.dims if d not in ['time']]
        
        if spatial_dims1:
            ts1 = data1.mean(dim=spatial_dims1)
        else:
            ts1 = data1
        
        if spatial_dims2:
            ts2 = data2.mean(dim=spatial_dims2)
        else:
            ts2 = data2
        
        ts1_pd = ts1.to_pandas()
        ts2_pd = ts2.to_pandas()
        
        if isinstance(ts1_pd, pd.DataFrame):
            ts1_pd = ts1_pd.iloc[:, 0]
        if isinstance(ts2_pd, pd.DataFrame):
            ts2_pd = ts2_pd.iloc[:, 0]
        
        common_index = ts1_pd.index.intersection(ts2_pd.index)
        ts1_pd = ts1_pd[common_index].values
        ts2_pd = ts2_pd[common_index].values
        
        # Normalizar
        ts1_norm = (ts1_pd - np.nanmean(ts1_pd)) / np.nanstd(ts1_pd)
        ts2_norm = (ts2_pd - np.nanmean(ts2_pd)) / np.nanstd(ts2_pd)
        
        # Calcular correlaciones con lags
        correlations = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                ts1_shifted = ts1_norm[-lag:]
                ts2_shifted = ts2_norm[:lag]
            elif lag > 0:
                ts1_shifted = ts1_norm[:-lag]
                ts2_shifted = ts2_norm[lag:]
            else:
                ts1_shifted = ts1_norm
                ts2_shifted = ts2_norm
            
            valid_mask = ~(np.isnan(ts1_shifted) | np.isnan(ts2_shifted))
            if np.sum(valid_mask) >= min_periods:
                corr = np.corrcoef(ts1_shifted[valid_mask], ts2_shifted[valid_mask])[0, 1]
            else:
                corr = np.nan
            
            correlations.append(corr)
        
        # Plotear
        fig, ax = plt.subplots(figsize=(12, 6))
        lags = range(-max_lag, max_lag + 1)
        
        colors = ['red' if c < 0 else 'green' if c > 0 else 'gray' for c in correlations]
        ax.bar(lags, correlations, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.axvline(x=0, color='blue', linestyle='--', linewidth=1.5, label='Lag=0')
        
        ax.set_xlabel('Lag (períodos)', fontsize=11)
        ax.set_ylabel('Correlación de Pearson', fontsize=11)
        ax.set_title(f'Correlación Lageada ({frequency}): {variable1} vs {variable2}',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def _filter_to_spain(self, data: xr.DataArray) -> xr.DataArray:
        """Filtra datos a coordenadas de España (aprox 35-43.5°N, -9.5 a 3°E)"""
        if 'lat' in data.dims and 'lon' in data.dims:
            try:
                # España: 35°N a 43.5°N, -9.5°E a 3°E
                filtered = data.sel(lat=slice(43.5, 35), lon=slice(-9.5, 3))
                # Marcar que se aplicó filtro de España
                filtered.attrs['region'] = 'España'
                return filtered
            except:
                pass
        return data

    def _densify_spatial_grid(self, data: xr.DataArray, target_resolution: float = 0.25) -> xr.DataArray:
        """Interpolar datos de baja resolución a una rejilla más fina para visualización."""
        if 'lat' not in data.dims or 'lon' not in data.dims:
            return data

        if data.lat.size <= 1 or data.lon.size <= 1:
            return data

        # Normalizar orden de coordenadas para interpolación
        if not np.all(np.diff(data.lon.values) > 0):
            data = data.sortby('lon')
        if not np.all(np.diff(data.lat.values) > 0):
            data = data.sortby('lat')

        lat = data.lat.values
        lon = data.lon.values
        lat_step = abs(float(lat[1] - lat[0]))
        lon_step = abs(float(lon[1] - lon[0]))

        if lat_step <= target_resolution and lon_step <= target_resolution and data.size >= 2000:
            return data

        lat_min, lat_max = float(lat.min()), float(lat.max())
        lon_min, lon_max = float(lon.min()), float(lon.max())

        new_lat = np.arange(lat_min, lat_max + target_resolution / 2, target_resolution)
        new_lon = np.arange(lon_min, lon_max + target_resolution / 2, target_resolution)

        try:
            return data.interp(lat=new_lat, lon=new_lon, method='nearest')
        except Exception:
            return data

    def _plot_dataarray_grid(self, ax, data: xr.DataArray, cmap: str, vmin: float, vmax: float, cbar_label: str):
        """Plota un DataArray 2D lat/lon como malla usando matplotlib directamente."""
        if 'lat' not in data.dims or 'lon' not in data.dims:
            raise ValueError('Los datos no tienen coordenadas lat/lon para pintar.')

        data = data.dropna(dim='lat', how='all').dropna(dim='lon', how='all')
        if data.size == 0:
            raise ValueError('No hay datos numéricos válidos para pintar.')

        lat = data.lat.values
        lon = data.lon.values
        values = np.asarray(data.values, dtype=float)
        values[~np.isfinite(values)] = np.nan

        if lat.ndim != 1 or lon.ndim != 1:
            raise ValueError('Las coordenadas lat/lon deben ser unidimensionales.')

        if values.ndim == 3:
            values = np.squeeze(values)

        if values.size == 0 or np.all(np.isnan(values)):
            raise ValueError('No hay datos numéricos válidos para pintar.')

        lon_grid, lat_grid = np.meshgrid(lon, lat)
        mesh = ax.pcolormesh(
            lon_grid,
            lat_grid,
            values,
            cmap=cmap,
            shading='auto',
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree()
        )
        cbar = plt.colorbar(mesh, ax=ax, shrink=0.7)
        cbar.set_label(cbar_label)
        return mesh
    
    def plot_lagged_correlation_map(self,
                                   dataset1: str,
                                   variable1: str,
                                   dataset2: str,
                                   variable2: str,
                                   frequency: str = 'D',
                                   lag: int = 0,
                                   resolution: float = 0.5,
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None) -> plt.Figure:
        """
        Mapa de correlación lageada entre dos variables (solo España)
        
        Args:
            frequency: 'D' (diario), 'W' (semanal), 'MS' (mensual)
            lag: Número de períodos de retraso
            resolution: Resolución espacial en grados
        """
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
        except ImportError:
            raise ImportError("Se requiere cartopy para los mapas. Instala con: pip install cartopy")
        
        # Usar datos preagregados
        ds1 = self._get_data_for_frequency(dataset1, variable1, frequency)
        ds2 = self._get_data_for_frequency(dataset2, variable2, frequency)
        
        # Filtrar por rango de fechas si se especifica
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            ds1 = ds1.sel(time=slice(start_dt, end_dt))
            ds2 = ds2.sel(time=slice(start_dt, end_dt))
        
        # Aplicar lag
        if lag != 0:
            ds1 = ds1.shift(time=lag)

        # Alinear tiempos antes de correlacionar
        if 'time' in ds1.dims and 'time' in ds2.dims:
            common_times = np.intersect1d(ds1.time.values, ds2.time.values)
            ds1 = ds1.sel(time=common_times)
            ds2 = ds2.sel(time=common_times)

        # Interpolar a rejilla común y filtrar a España
        if 'lat' in ds1.dims and 'lon' in ds1.dims and 'lat' in ds2.dims and 'lon' in ds2.dims:
            # Normalizar coordenadas a orden ascendente para interpolación
            if not np.all(np.diff(ds1.lon.values) > 0):
                ds1 = ds1.sortby('lon')
            if not np.all(np.diff(ds1.lat.values) > 0):
                ds1 = ds1.sortby('lat')
            if not np.all(np.diff(ds2.lon.values) > 0):
                ds2 = ds2.sortby('lon')
            if not np.all(np.diff(ds2.lat.values) > 0):
                ds2 = ds2.sortby('lat')

            size1 = ds1.lat.size * ds1.lon.size
            size2 = ds2.lat.size * ds2.lon.size

            if size1 < 1500 or size2 < 1500:
                ds1 = self._densify_spatial_grid(ds1, target_resolution=resolution)
                ds2 = self._densify_spatial_grid(ds2, target_resolution=resolution)
                size1 = ds1.lat.size * ds1.lon.size
                size2 = ds2.lat.size * ds2.lon.size

            if size1 < 1000 or size2 < 1000:
                if size1 >= size2:
                    base_ds = ds1
                    interp_ds = ds2
                else:
                    base_ds = ds2
                    interp_ds = ds1
                try:
                    interp_ds = interp_ds.interp(lat=base_ds.lat, lon=base_ds.lon, method='linear')
                    if size1 >= size2:
                        ds2 = interp_ds
                    else:
                        ds1 = interp_ds
                except:
                    pass
            else:
                lat_min = max(float(ds1.lat.min()), float(ds2.lat.min()))
                lat_max = min(float(ds1.lat.max()), float(ds2.lat.max()))
                lon_min = max(float(ds1.lon.min()), float(ds2.lon.min()))
                lon_max = min(float(ds1.lon.max()), float(ds2.lon.max()))
                lat_max = min(lat_max, 43.5)
                lat_min = max(lat_min, 35)
                lon_min = max(lon_min, -9.5)
                lon_max = min(lon_max, 3)
                if lat_min < lat_max and lon_min < lon_max:
                    new_lat = np.arange(lat_min, lat_max + resolution / 2, resolution)
                    new_lon = np.arange(lon_min, lon_max + resolution / 2, resolution)
                    try:
                        ds1 = ds1.interp(lat=new_lat, lon=new_lon, method='linear')
                        ds2 = ds2.interp(lat=new_lat, lon=new_lon, method='linear')
                    except:
                        pass
        
        # Calcular correlación
        try:
            corr = xr.corr(ds1, ds2, dim='time')
        except:
            corr = ds1.copy()
            corr.values = np.full_like(corr.values, np.nan)
        
        # Eliminar latitudes/longitudes completamente vacías para evitar huecos
        if hasattr(corr, 'lat') and hasattr(corr, 'lon'):
            corr = corr.dropna(dim='lat', how='all').dropna(dim='lon', how='all')

        # Plotear mapa
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)

        if isinstance(corr, xr.Dataset):
            corr = corr[list(corr.data_vars)[0]]

        corr.plot(
            ax=ax,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            cbar_kwargs={'label': 'Correlación (r)', 'shrink': 0.8}
        )

        # Ajustar extent dinámicamente a los datos disponibles
        if hasattr(corr, 'lat') and hasattr(corr, 'lon'):
            lat_min = float(corr.lat.min())
            lat_max = float(corr.lat.max())
            lon_min = float(corr.lon.min())
            lon_max = float(corr.lon.max())
            
            # Expandir extent para mostrar más contexto (Península + alrededor)
            # Margen más grande para capturar Portugal completo y contexto al este
            margin = 1.0
            ax.set_extent([lon_min - margin, lon_max + margin, 
                          lat_min - margin, lat_max + margin], crs=ccrs.PlateCarree())
        else:
            # Fallback a Península Ibérica + contexto
            ax.set_extent([-12, 6, 32, 46], crs=ccrs.PlateCarree())
        
        ax.coastlines()
        
        title = f'Correlación Espacial ({frequency}): {variable1} vs {variable2}'
        if lag != 0:
            title += f'\n(Lag: {lag})'
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    
    
    def plot_distribution(self,
                         dataset: str,
                         variable: str,
                         frequency: str = 'D',
                         bins: int = 50,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> plt.Figure:
        """Histograma con granularidad temporal configurable"""
        data = self._get_data_for_frequency(dataset, variable, frequency)
        
        # Filtrar por rango de fechas si se especifica
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            data = data.sel(time=slice(start_dt, end_dt))
        
        # Obtener valores
        values = data.values.flatten()
        values_clean = values[~np.isnan(values)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(values_clean, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
        
        mean = np.mean(values_clean)
        median = np.median(values_clean)
        std = np.std(values_clean)
        
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Media: {mean:.2f}')
        ax.axvline(median, color='orange', linestyle='--', linewidth=2, label=f'Mediana: {median:.2f}')
        
        ax.set_xlabel(f'{variable}', fontsize=11)
        ax.set_ylabel('Frecuencia', fontsize=11)
        ax.set_title(f'Distribución ({frequency}): {variable}',
                    fontsize=13, fontweight='bold')
        
        textstr = f'Desv. Est.: {std:.2f}\nMín: {np.min(values_clean):.2f}\nMáx: {np.max(values_clean):.2f}'
        ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_scatter(self,
                    dataset1: str,
                    variable1: str,
                    dataset2: str,
                    variable2: str,
                    frequency: str = 'D',
                    extreme_threshold: float = 0.95,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> plt.Figure:
        """Scatter plot con granularidad temporal configurable"""
        # Usar datos preagregados
        data1 = self._get_data_for_frequency(dataset1, variable1, frequency)
        data2 = self._get_data_for_frequency(dataset2, variable2, frequency)
        
        # Filtrar por rango de fechas si se especifica
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            data1 = data1.sel(time=slice(start_dt, end_dt))
            data2 = data2.sel(time=slice(start_dt, end_dt))
        
        # Agregación espacial
        spatial_dims1 = [d for d in data1.dims if d not in ['time']]
        spatial_dims2 = [d for d in data2.dims if d not in ['time']]
        
        if spatial_dims1:
            ts1 = data1.mean(dim=spatial_dims1)
        else:
            ts1 = data1
        
        if spatial_dims2:
            ts2 = data2.mean(dim=spatial_dims2)
        else:
            ts2 = data2
        
        ts1_pd = ts1.to_pandas()
        ts2_pd = ts2.to_pandas()
        
        if isinstance(ts1_pd, pd.DataFrame):
            ts1_pd = ts1_pd.iloc[:, 0]
        if isinstance(ts2_pd, pd.DataFrame):
            ts2_pd = ts2_pd.iloc[:, 0]
        
        common_index = ts1_pd.index.intersection(ts2_pd.index)
        ts1_aligned = ts1_pd[common_index].values
        ts2_aligned = ts2_pd[common_index].values
        
        # Calcular extremos
        threshold1 = np.nanpercentile(ts1_aligned, extreme_threshold * 100)
        threshold2 = np.nanpercentile(ts2_aligned, extreme_threshold * 100)
        
        extremes1 = ts1_aligned >= threshold1
        extremes2 = ts2_aligned >= threshold2
        
        # Plotear
        fig, ax = plt.subplots(figsize=(10, 8))
        
        normal = ~(extremes1 | extremes2)
        ax.scatter(ts1_aligned[normal], ts2_aligned[normal], 
                  alpha=0.6, s=50, color='blue', label='Normal')
        
        only_extremes1 = extremes1 & ~extremes2
        ax.scatter(ts1_aligned[only_extremes1], ts2_aligned[only_extremes1],
                  alpha=0.8, s=100, color='orange', label=f'{variable1} (extremo)')
        
        only_extremes2 = ~extremes1 & extremes2
        ax.scatter(ts1_aligned[only_extremes2], ts2_aligned[only_extremes2],
                  alpha=0.8, s=100, color='red', label=f'{variable2} (extremo)')
        
        both_extremes = extremes1 & extremes2
        ax.scatter(ts1_aligned[both_extremes], ts2_aligned[both_extremes],
                  alpha=0.9, s=150, color='darkred', marker='*', 
                  label='Ambas (extremo)')
        
        valid_mask = ~(np.isnan(ts1_aligned) | np.isnan(ts2_aligned))
        if np.sum(valid_mask) > 2:
            corr = np.corrcoef(ts1_aligned[valid_mask], ts2_aligned[valid_mask])[0, 1]
        else:
            corr = np.nan
        
        ax.set_xlabel(f'{variable1}', fontsize=11)
        ax.set_ylabel(f'{variable2}', fontsize=11)
        ax.set_title(f'Dispersión ({frequency}): {variable1} vs {variable2}\n(r={corr:.3f})',
                    fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_extremes_timeline(self,
                              dataset: str,
                              variable: str,
                              frequency: str = 'D',
                              extreme_threshold: float = 0.95,
                              window: int = 3,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> plt.Figure:
        """Timeline de eventos extremos con granularidad configurable"""
        data = self._get_data_for_frequency(dataset, variable, frequency)
        
        # Filtrar por rango de fechas si se especifica
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            data = data.sel(time=slice(start_dt, end_dt))
        
        # Agregación espacial
        spatial_dims = [d for d in data.dims if d not in ['time']]
        if spatial_dims:
            ts = data.mean(dim=spatial_dims)
        else:
            ts = data
        
        ts_pd = ts.to_pandas()
        if isinstance(ts_pd, pd.DataFrame):
            ts_pd = ts_pd.iloc[:, 0]
        
        # Calcular extremos
        threshold = np.nanpercentile(ts_pd.dropna(), extreme_threshold * 100)
        extremes = (ts_pd >= threshold).astype(int)
        
        # Suavizar
        extremes_smooth = extremes.rolling(window=window, center=True).max()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(ts_pd.index, ts_pd.values, label='Variable', linewidth=2, color='steelblue', zorder=2)
        
        # Sombrear extremos
        extremes_periods = []
        start = None
        for i, (idx, val) in enumerate(zip(ts_pd.index, extremes_smooth)):
            if val > 0 and start is None:
                start = idx
            elif val == 0 and start is not None:
                extremes_periods.append((start, idx))
                start = None
        
        if start is not None:
            extremes_periods.append((start, ts_pd.index[-1]))
        
        for start, end in extremes_periods:
            ax.axvspan(start, end, alpha=0.3, color='red')
        
        ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, 
                  label=f'Threshold ({extreme_threshold*100:.0f} percentil)')
        
        ax.set_xlabel('Tiempo', fontsize=11)
        ax.set_ylabel(f'{variable}', fontsize=11)
        ax.set_title(f'Timeline de Eventos Extremos ({frequency}): {variable}',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig

    def plot_timeseries_custom(self, ts_pandas, title="Serie Temporal"):
        """Grafica una serie temporal de pandas (ej: región seleccionada)"""
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.plot(ts_pandas.index, ts_pandas.values, linewidth=2, marker='o', markersize=3, color='#1f77b4')
        ax.fill_between(ts_pandas.index, ts_pandas.values, alpha=0.3, color='#1f77b4')
        
        ax.set_xlabel('Tiempo', fontsize=11)
        ax.set_ylabel('Valor', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig

    def plot_map_for_date(self,
                          dataset: str,
                          variable: str,
                          date: str,
                          frequency: str = 'D'):
        """Genera un mapa folium para una fecha concreta (sin Plotly)."""
        data = self._get_data_for_frequency(dataset, variable, frequency)

        date_idx = pd.to_datetime(date)
        try:
            data_date = data.sel(time=date_idx)
        except Exception:
            data_date = data.sel(time=date_idx, method='nearest')

        data_date = self._filter_to_spain(data_date)

        if 'lat' not in data_date.dims or 'lon' not in data_date.dims:
            raise ValueError('Los datos no están en formato lat/lon para mapa.')

        df = data_date.to_dataframe().reset_index().dropna(subset=[variable])
        if df.empty:
            raise ValueError('No hay datos para la fecha seleccionada.')

        m = folium.Map(location=[40.0, -3.5], zoom_start=6, tiles='OpenStreetMap')

        vmin = df[variable].min()
        vmax = df[variable].max()
        if vmin == vmax:
            vmax = vmin + 1

        for _, row in df.iterrows():
            val_scaled = (row[variable] - vmin) / (vmax - vmin)
            rgba = plt.cm.RdYlBu_r(val_scaled)
            color = mcolors.rgb2hex(rgba)
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=2,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"{variable}: {row[variable]:.2f}"
            ).add_to(m)

        return m

    def plot_map_raster_for_date(self,
                                 dataset: str,
                                 variable: str,
                                 date: str,
                                 frequency: str = 'D') -> plt.Figure:
        """Genera un mapa de calor de malla (no puntos) para una fecha concreta."""
        data = self._get_data_for_frequency(dataset, variable, frequency)

        date_idx = pd.to_datetime(date)
        try:
            data_date = data.sel(time=date_idx)
        except Exception:
            data_date = data.sel(time=date_idx, method='nearest')

        # Intentar primero filtrar a España
        data_filtered = self._filter_to_spain(data_date)

        # Verificar si hay datos válidos después del filtro
        if isinstance(data_filtered, xr.Dataset):
            data_filtered = data_filtered[variable]

        if 'lat' not in data_filtered.dims or 'lon' not in data_filtered.dims:
            raise ValueError('No hay malla lat-lon en la fecha seleccionada.')

        if data_filtered.size < 1500:
            data_filtered = self._densify_spatial_grid(data_filtered, target_resolution=0.25)

        data_filtered = self._densify_spatial_grid(data_filtered, target_resolution=0.1)
        data_filtered = data_filtered.dropna(dim='lat', how='all').dropna(dim='lon', how='all')

        # Verificar si hay datos no-NaN
        data_values = data_filtered.values
        valid_data = data_values[~np.isnan(data_values)]

        if valid_data.size == 0:
            # Si no hay datos en España, intentar sin filtro (global)
            if isinstance(data_date, xr.Dataset):
                data_global = data_date[variable]
            else:
                data_global = data_date

            if 'lat' in data_global.dims and 'lon' in data_global.dims:
                data_values_global = data_global.values
                valid_data_global = data_values_global[~np.isnan(data_values_global)]

                if valid_data_global.size > 0:
                    data_filtered = data_global
                    print(f"⚠️  No hay datos en España para {date}, mostrando vista global")
                else:
                    raise ValueError(f"No hay datos válidos para la fecha {date} (ni en España ni globalmente)")
            else:
                raise ValueError(f"No hay datos válidos para la fecha {date}")

        # Calcular min/max solo con datos válidos
        vmin = float(np.nanmin(data_filtered.values))
        vmax = float(np.nanmax(data_filtered.values))
        if vmin == vmax:
            vmax = vmin + 1

        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        if 'lat' in data_filtered.dims and 'lon' in data_filtered.dims:
            lat_min = float(data_filtered.lat.min())
            lat_max = float(data_filtered.lat.max())
            lon_min = float(data_filtered.lon.min())
            lon_max = float(data_filtered.lon.max())
            
            # Mantener vista general si el dataset es pequeño (poco dato)
            data_size = data_filtered.lat.size * data_filtered.lon.size
            if data_size < 500:
                # Mostrar vista península para datasets con poco dato
                ax.set_extent([-11, 5, 33, 45], crs=ccrs.PlateCarree())
            else:
                # Para datasets con buena cobertura, mostrar justo el área de datos
                margin = 0.2
                ax.set_extent([lon_min - margin, lon_max + margin, lat_min - margin, lat_max + margin], crs=ccrs.PlateCarree())
        else:
            ax.set_extent([-11, 5, 33, 45], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)

        data_filtered.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='RdYlBu_r',
            vmin=vmin,
            vmax=vmax,
            cbar_kwargs={'label': variable, 'shrink': 0.7}
        )

        title_region = "España" if 'España' in str(data_filtered.attrs.get('region', '')) else "Global"
        ax.set_title(f'{variable} - {date_idx.strftime("%Y-%m-%d")} ({title_region})', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_animated_map(self,
                         dataset: str,
                         variable: str,
                         frequency: str = 'D',
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None):
        """
        Función de compatibilidad (Plotly desactivado). Usa `plot_map_for_date` con slider en el dashboard.
        """
        raise NotImplementedError('Plotly-based animated map no está disponible. Usa `plot_map_for_date` con slider.')
