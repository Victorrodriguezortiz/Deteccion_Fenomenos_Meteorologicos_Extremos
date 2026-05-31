"""
Dashboard Interactivo para Detección de Fenómenos Meteorológicos Extremos
Versión 2.0: Con soporte a múltiples granularidades temporales y mapas de correlación
"""
import streamlit as st
import matplotlib.pyplot as plt
from data_loader import DataLoader
from charts import ChartGenerator
import warnings
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import pandas as pd
import time

warnings.filterwarnings('ignore')

# Configuración
st.set_page_config(
    page_title="Dashboard TFG - Fenómenos Extremos v2.0",
    layout="wide",
    page_icon="🌍"
)

st.markdown("# 🌍 Dashboard: Detección de Fenómenos Meteorológicos Extremos")
st.markdown("**Versión 2.0**: Análisis multitemporal con mapas de correlación espacial")

# ========================================
# FUNCIONES DE SELECCIÓN REGIONAL
# ========================================
def create_regional_selector(title="Seleccionar región (arrastra para dibujar)"):
    """Crea mapa interactivo para seleccionar región por arrastre"""
    # Centro aproximado de España
    center_lat, center_lon = 40.0, -3.5
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="OpenStreetMap"
    )
    
    # Plugin Draw para permitir dibujar rectangulares
    Draw(
        export=False,
        position='topleft',
        draw_options={
            'polyline': False,
            'polygon': True,      # Permitir polígonos (que usen como rectangulares)
            'circle': False,
            'rectangle': True,    # Permitir rectángulos directamente
            'marker': False,
            'circlemarker': False
        }
    ).add_to(m)
    
    # Mostrar mapa y capturar dibujos
    map_output = st_folium(m, width=700, height=400, key=f"mapa_{title}")
    
    # Extraer coordenadas del rectángulo/polígono dibujado
    lat_bounds = None
    lon_bounds = None
    
    if map_output and map_output.get("all_drawings"):
        try:
            drawings = map_output["all_drawings"]
            if len(drawings) > 0:
                # Tomar el primer dibujo
                geom = drawings[0].get("geometry", {})
                coords = geom.get("coordinates", [])
                
                if coords:
                    if geom["type"] == "Polygon":
                        # coords[0] es el anillo exterior
                        lats = [c[1] for c in coords[0]]
                        lons = [c[0] for c in coords[0]]
                    else:
                        lats = [c[1] for c in coords]
                        lons = [c[0] for c in coords]
                    
                    if lats and lons:
                        lat_bounds = (min(lats), max(lats))
                        lon_bounds = (min(lons), max(lons))
                        
                        st.success(f"✅ Región seleccionada: Lat {lat_bounds[0]:.2f}° a {lat_bounds[1]:.2f}°, Lon {lon_bounds[0]:.2f}° a {lon_bounds[1]:.2f}°")
        except Exception as e:
            st.warning(f"⚠️  Error extrayendo coordenadas: {e}")
    
    return lat_bounds, lon_bounds


# ========================================
# CARGA DE DATOS
# ========================================
# Cambiamos a cache_data y le decimos que ignore el hash del objeto de la clase DataLoader
@st.cache_data(hash_funcs={DataLoader: lambda _: None})
def load_datasets():
    loader = DataLoader()
    # Usar carga selectiva para reducir tiempo
    variables_to_load = {
        "olas_calor": ["t2m", "mx2t", "mn2t", "swvl1", "tcc"],
        "precipitacion": ["tp", "cp", "lsp", "tcwv"],
        "calidad_aire": ["pm2p5", "pm10", "aod550", "tcno2", "tcco"],
        "satelite_s3_lst": ["LST"],
        "nasa_gpm_mensual": ["precipitation"],
        "modis_diario_full": ["aod"],
        "satelite_s5p_aerosoles": ["AER_AI_354_388"],
        "satelite_s5p_no2": ["NO2"],
        "satelite_s5p_co": ["CO"],
    }
    datasets = loader.load_all_datasets(
        variables_to_load=variables_to_load,
        pre_cache_aggregations=False  # Pre-cachea W y MS
    )
    return loader, datasets

with st.spinner("📂 Cargando datos y preagregando granularidades..."):
    loader, datasets = load_datasets()

if not datasets:
    st.error("❌ No se pudieron cargar los datasets")
    st.stop()

chart_gen = ChartGenerator(datasets, loader)
available_vars = list(datasets.keys())

# ========================================
# SIDEBAR - CONFIGURACIÓN
# ========================================
with st.sidebar:
    st.header("⚙️ Configuración Global")
    
    st.markdown(f"**Datasets cargados:** {len(datasets)}")
    
    # Selector de granularidad temporal GLOBAL
    frequency_map = {
        'Diario': 'D',
        'Semanal': 'W',
        'Mensual': 'MS'
    }
    freq_display = st.selectbox(
        "📅 Granularidad Temporal",
        options=list(frequency_map.keys()),
        help="Agregación temporal aplicable a todas las gráficas"
    )
    frequency = frequency_map[freq_display]
    
    st.markdown("---")
    
    # Selector de rango de fechas GLOBAL
    all_times = []
    for ds in datasets.values():
        if 'time' in ds.dims:
            all_times.extend(ds.time.values)
    
    if all_times:
        min_date = pd.to_datetime(min(all_times)).date()
        max_date = pd.to_datetime(max(all_times)).date()
        
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date = st.date_input(
                "📅 Fecha Inicio",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                help="Fecha de inicio para filtrar datos"
            )
        with col_date2:
            end_date = st.date_input(
                "📅 Fecha Fin",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                help="Fecha de fin para filtrar datos"
            )
    else:
        start_date = None
        end_date = None
    
    st.markdown("---")
    
    # Control de extremos
    extreme_threshold = st.slider(
        "⚡ Percentil para eventos extremos",
        min_value=80,
        max_value=99,
        value=95,
        step=1
    ) / 100
    
    st.markdown("---")
    
    with st.expander("📊 Datasets disponibles"):
        for name in sorted(available_vars):
            ds = datasets[name]
            vars_list = list(ds.data_vars.keys())
            st.write(f"**{name}:** {', '.join(vars_list[:3])}{'...' if len(vars_list) > 3 else ''}")

# ========================================
# PESTAÑAS
# ========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Series Temporales",
    "🗺️ Correlación Lageada (Mapa+Barras)",
    "📊 Distribuciones",
    "🔴 Scatter",
    "⚡ Timeline Extremos",
    "Selección Regional (Mapa)"
])

# ========================================
# TAB 1: SERIES TEMPORALES
# ========================================
with tab1:
    st.subheader("📈 Series Temporales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Variable 1**")
        dataset1 = st.selectbox("Dataset 1", sorted(available_vars), key="ts_ds1")
        vars1 = [v for v in datasets[dataset1].data_vars.keys() if v != 'crs']
        variable1 = st.selectbox("Variable 1", vars1, key="ts_var1")
    
    with col2:
        st.markdown("**Variable 2 (Opcional)**")
        dataset2 = st.selectbox("Dataset 2", ["Ninguno"] + sorted(available_vars), key="ts_ds2")
        
        if dataset2 != "Ninguno":
            vars2 = [v for v in datasets[dataset2].data_vars.keys() if v != 'crs']
            variable2 = st.selectbox("Variable 2", vars2, key="ts_var2")
        else:
            variable2 = None
    
    try:
        fig = chart_gen.plot_time_series(
            dataset1=dataset1,
            variable1=variable1,
            frequency=frequency,
            dataset2=dataset2 if dataset2 != "Ninguno" else None,
            variable2=variable2,
            extreme_threshold=extreme_threshold,
            start_date=start_date,
            end_date=end_date
        )
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"❌ Error: {str(e)[:100]}")

# ========================================
# TAB 2: CORRELACIÓN LAGEADA (MAPA + BARRAS)
# ========================================
with tab2:
    st.subheader("🗺️ Correlación Lageada Espacial")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dataset1 = st.selectbox("Dataset 1", sorted(available_vars), key="lag_ds1")
        vars1 = [v for v in datasets[dataset1].data_vars.keys() if v != 'crs']
        variable1 = st.selectbox("Variable 1", vars1, key="lag_var1")
    
    with col2:
        dataset2 = st.selectbox("Dataset 2", sorted(available_vars), key="lag_ds2")
        vars2 = [v for v in datasets[dataset2].data_vars.keys() if v != 'crs']
        variable2 = st.selectbox("Variable 2", vars2, key="lag_var2")
    
    with col3:
        max_lag = st.slider("Máximo lag", 1, 30, 14, key="max_lag")
        current_lag = st.slider("Lag actual (mapa)", -max_lag, max_lag, 0, key="current_lag")
        resolution = st.slider("Resolución espacial (°)", 0.1, 2.0, 0.5, step=0.1)
    
    try:
        # MAPA DE CORRELACIÓN
        st.markdown("### Mapa de Correlación Espacial")
        fig_map = chart_gen.plot_lagged_correlation_map(
            dataset1=dataset1,
            variable1=variable1,
            dataset2=dataset2,
            variable2=variable2,
            frequency=frequency,
            lag=current_lag,
            resolution=resolution,
            start_date=start_date,
            end_date=end_date
        )
        st.pyplot(fig_map)
        plt.close(fig_map)
        
        # GRÁFICO DE BARRAS CON TODOS LOS LAGS
        st.markdown("### Correlación Media por Lag (Temporal)")
        fig_bars = chart_gen.plot_lagged_correlation(
            dataset1=dataset1,
            variable1=variable1,
            dataset2=dataset2,
            variable2=variable2,
            frequency=frequency,
            max_lag=max_lag,
            start_date=start_date,
            end_date=end_date
        )
        st.pyplot(fig_bars)
        plt.close(fig_bars)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)[:100]}")

# ========================================
# TAB 3: DISTRIBUCIONES
# ========================================
with tab3:
    st.subheader("📊 Distribuciones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dataset = st.selectbox("Dataset", sorted(available_vars), key="dist_ds")
        vars_list = list(v for v in datasets[dataset].data_vars.keys() if v != 'crs')
        variable = st.selectbox("Variable", vars_list, key="dist_var")
    
    with col2:
        bins = st.slider("Número de bins", 20, 100, 50)
    
    try:
        fig = chart_gen.plot_distribution(
            dataset=dataset,
            variable=variable,
            frequency=frequency,
            bins=bins,
            start_date=start_date,
            end_date=end_date
        )
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"❌ Error: {str(e)[:100]}")

# ========================================
# TAB 4: SCATTER
# ========================================
with tab4:
    st.subheader("🔴 Análisis de Dispersión")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dataset1 = st.selectbox("Dataset 1", sorted(available_vars), key="scatter_ds1")
        vars1 = [v for v in datasets[dataset1].data_vars.keys() if v != 'crs']
        variable1 = st.selectbox("Variable 1", vars1, key="scatter_var1")
    
    with col2:
        dataset2 = st.selectbox("Dataset 2", sorted(available_vars), key="scatter_ds2")
        vars2 = [v for v in datasets[dataset2].data_vars.keys() if v != 'crs']
        variable2 = st.selectbox("Variable 2", vars2, key="scatter_var2")
    
    try:
        fig = chart_gen.plot_scatter(
            dataset1=dataset1,
            variable1=variable1,
            dataset2=dataset2,
            variable2=variable2,
            frequency=frequency,
            extreme_threshold=extreme_threshold,
            start_date=start_date,
            end_date=end_date
        )
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"❌ Error: {str(e)[:100]}")

# ========================================
# TAB 5: TIMELINE EXTREMOS
# ========================================
with tab5:
    st.subheader("⚡ Timeline de Eventos Extremos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dataset = st.selectbox("Dataset", sorted(available_vars), key="extreme_ds")
        vars_list = list(v for v in datasets[dataset].data_vars.keys() if v != 'crs')
        variable = st.selectbox("Variable", vars_list, key="extreme_var")
    
    with col2:
        window = st.slider("Ventana móvil", 1, 12, 3)
    
    try:
        fig = chart_gen.plot_extremes_timeline(
            dataset=dataset,
            variable=variable,
            frequency=frequency,
            extreme_threshold=extreme_threshold,
            window=window,
            start_date=start_date,
            end_date=end_date
        )
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"❌ Error: {str(e)[:100]}")

# ========================================
# TAB 6: Selección Regional (MAPA)
# ========================================
with tab6:
    st.subheader("🗺️ Análisis Univariante - Selecciona una Región")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Paso 1: Dibuja una región en el mapa (arrastra para crear un rectángulo)**")
        lat_bounds, lon_bounds = create_regional_selector()
    
    with col2:
        st.write("**Paso 2: Selecciona variable a analizar**")
        # Filtrar datasets diarios para mapas (excluir solo mensuales reales)
        monthly_datasets = ['modis_mensual_procesado']  # nasa_gpm_mensual es diario según usuario
        daily_datasets = [d for d in available_vars if d not in monthly_datasets]
        dataset_uv = st.selectbox("Dataset", sorted(daily_datasets), key="uv_ds")
        vars_list = list(v for v in datasets[dataset_uv].data_vars.keys() if v != 'crs')
        variable_uv = st.selectbox("Variable", vars_list, key="uv_var")
    
    st.markdown("---")
    
    if lat_bounds and lon_bounds:
        try:
            # Recortar espacialmente primero (preserva coordenadas), LUEGO agregar temporalmente
            ts = loader.get_regional_timeseries(
                datasets[dataset_uv], variable_uv,
                lat_bounds=lat_bounds,
                lon_bounds=lon_bounds
            )
            
            if ts is not None:
                # Filtrar por rango de fechas si se especifica
                if start_date and end_date:
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    ts = ts[(ts.index >= start_dt) & (ts.index <= end_dt)]
                
                # Luego agregamos temporalmente la serie
                if frequency != 'D':
                    if frequency == 'W':
                        ts = ts.groupby(ts.index.isocalendar().week).mean()
                    elif frequency == 'MS':
                        ts = ts.groupby(ts.index.to_period('M')).mean()
                
                st.success("✅ Serie temporal regional calculada")
                
                # Mostrar estadísticas
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Media", f"{ts.mean():.2f}")
                with col2:
                    st.metric("Máximo", f"{ts.max():.2f}")
                with col3:
                    st.metric("Mínimo", f"{ts.min():.2f}")
                with col4:
                    st.metric("Desv. Est.", f"{ts.std():.2f}")
                
                # Graficar
                fig = chart_gen.plot_timeseries_custom(ts, title=f"{variable_uv} en región seleccionada")
                st.pyplot(fig)
                plt.close(fig)
                
                # ANIMACIÓN TEMPORAL DEL MAPA (solo raster con play)
                st.markdown("### 🎬 Animación Temporal del Mapa")
                st.write("Pulsa Play para ver la evolución día a día del raster en el período filtrado:")

                if start_date and end_date:
                    period_dates = pd.date_range(start=start_date, end=end_date, freq='D')
                    
                    if st.button('▶️ Play evolución diaria'):
                        placeholder = st.empty()
                        for day in period_dates:
                            try:
                                fig_raster = chart_gen.plot_map_raster_for_date(
                                    dataset=dataset_uv,
                                    variable=variable_uv,
                                    date=day,
                                    frequency=frequency
                                )
                                placeholder.pyplot(fig_raster)
                                plt.close(fig_raster)
                                time.sleep(0.4)
                            except Exception as er:
                                st.warning(f"Sin datos en {day.date()}: {str(er)[:50]}")
                                continue
                else:
                    st.warning("Selecciona un rango de fechas en la barra lateral para activar la secuencia temporal.")
                
            else:
                st.warning("⚠️  No se pudo obtener datos para la región seleccionada")
        except Exception as e:
            st.error(f"❌ Error: {str(e)[:200]}")
    else:
        st.info("📍 Dibuja un rectángulo en el mapa para seleccionar una región")
# ========================================
# FOOTER
# ========================================
st.markdown("---")
