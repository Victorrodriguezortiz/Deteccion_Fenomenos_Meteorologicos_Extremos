"""
Evaluation plots for extreme-event models.

The functions here are notebook-friendly: they receive DataFrames and return
figures or summary tables without depending on trained pipeline internals.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False


CONFUSION_COLORS = {
    "TP - Extremo detectado": "#2ca02c",
    "FP - Falsa alarma": "#ff9800",
    "FN - Extremo no detectado": "#d62728",
    "TN - No extremo correcto": "#cfcfcf",
}

CONFUSION_ORDER = [
    "TN - No extremo correcto",
    "FP - Falsa alarma",
    "FN - Extremo no detectado",
    "TP - Extremo detectado",
]


def anadir_tipo_confusion(
    df: pd.DataFrame,
    y_real_col: str = "y_real",
    y_pred_col: str = "y_pred",
) -> pd.DataFrame:
    """
    Add a human-readable confusion category to each row.
    """
    df = df.copy()

    condiciones = [
        (df[y_real_col] == 1) & (df[y_pred_col] == 1),
        (df[y_real_col] == 0) & (df[y_pred_col] == 1),
        (df[y_real_col] == 1) & (df[y_pred_col] == 0),
        (df[y_real_col] == 0) & (df[y_pred_col] == 0),
    ]

    categorias = [
        "TP - Extremo detectado",
        "FP - Falsa alarma",
        "FN - Extremo no detectado",
        "TN - No extremo correcto",
    ]

    df["tipo_confusion"] = np.select(condiciones, categorias, default="Sin clasificar")
    return df


# Alias with Spanish spelling used in the notebooks.
añadir_tipo_confusion = anadir_tipo_confusion


def _add_spain_base_map(ax, extent=(-10.0, 4.0, 35.0, 44.5), transform=None):
    if CARTOPY_AVAILABLE and hasattr(ax, "add_feature"):
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor="#dceaf7", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#f3efe3", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#333333", zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.45, edgecolor="#555555", zorder=1)
        try:
            grid = ax.gridlines(
                draw_labels=True,
                linewidth=0.35,
                color="gray",
                alpha=0.35,
                linestyle="--",
            )
            grid.top_labels = False
            grid.right_labels = False
        except Exception:
            ax.grid(alpha=0.25, linewidth=0.35)
        return

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_facecolor("#dceaf7")
    ax.grid(alpha=0.25, linewidth=0.4)


def _crear_eje_mapa(
    figsize=(9, 7),
    extent=(-10.0, 4.0, 35.0, 44.5),
    pintar_mapa=True,
):
    if CARTOPY_AVAILABLE and pintar_mapa:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        transform = ccrs.PlateCarree()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        transform = None

    if pintar_mapa:
        _add_spain_base_map(ax, extent=extent, transform=transform)
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(alpha=0.25, linewidth=0.4)

    return fig, ax, transform


def _filtrar_dia(df_evento, fecha, time_col):
    df = df_evento.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    fecha = pd.to_datetime(fecha).date()
    return df[df[time_col].dt.date == fecha].copy(), fecha


def _resumen_confusion(df_dia: pd.DataFrame, fecha) -> pd.DataFrame:
    resumen = (
        df_dia["tipo_confusion"]
        .value_counts()
        .rename_axis("Categoria")
        .reset_index(name="Numero de pixeles")
    )
    resumen.insert(0, "Fecha", pd.to_datetime(fecha))
    return resumen


def plot_mapa_confusion_evento(
    df_evento: pd.DataFrame,
    fecha,
    time_col: str = "time",
    lat_col: str = "lat",
    lon_col: str = "lon",
    y_real_col: str = "y_real",
    y_pred_col: str = "y_pred",
    mostrar_tn: bool = True,
    title: Optional[str] = None,
    guardar: bool = False,
    nombre_archivo: str = "mapa_confusion_evento.png",
    mostrar: bool = True,
    figsize=(9, 7),
    extent=(-10.0, 4.0, 35.0, 44.5),
    pintar_mapa: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Plot the spatial confusion matrix for one day.
    """
    df_dia, fecha = _filtrar_dia(df_evento, fecha, time_col)

    if df_dia.empty:
        print(f"No hay datos para la fecha {fecha}")
        return None

    df_dia = anadir_tipo_confusion(df_dia, y_real_col=y_real_col, y_pred_col=y_pred_col)
    orden = CONFUSION_ORDER if mostrar_tn else CONFUSION_ORDER[1:]

    fig, ax, transform = _crear_eje_mapa(figsize=figsize, extent=extent, pintar_mapa=pintar_mapa)

    for categoria in orden:
        subset = df_dia[df_dia["tipo_confusion"] == categoria]
        if subset.empty:
            continue

        alpha = 0.30 if categoria.startswith("TN") else 0.92
        size = 42 if categoria.startswith("TN") else 115
        zorder = 3 if categoria.startswith("TN") else 5
        scatter_kwargs = {"transform": transform} if transform is not None else {}

        ax.scatter(
            subset[lon_col],
            subset[lat_col],
            c=CONFUSION_COLORS[categoria],
            label=f"{categoria} ({len(subset)})",
            s=size,
            marker="s",
            edgecolors="black",
            linewidths=0.35,
            alpha=alpha,
            zorder=zorder,
            **scatter_kwargs,
        )

    ax.set_title(title or f"Matriz de confusion espacial - {fecha}")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    if guardar:
        fig.savefig(nombre_archivo, dpi=300, bbox_inches="tight")

    if mostrar:
        plt.show()
    else:
        plt.close(fig)

    return _resumen_confusion(df_dia, fecha)


def plot_mapa_confusion_eventos_rango(
    df_evento: pd.DataFrame,
    fecha_inicio="2025-08-03",
    fecha_fin="2025-08-18",
    time_col: str = "time",
    lat_col: str = "lat",
    lon_col: str = "lon",
    y_real_col: str = "y_real",
    y_pred_col: str = "y_pred",
    mostrar_tn: bool = True,
    guardar: bool = True,
    carpeta_salida: str = "mapas_confusion",
    mostrar: bool = False,
    figsize=(9, 7),
    extent=(-10.0, 4.0, 35.0, 44.5),
    pintar_mapa: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Save or display one spatial confusion map for each date in a range.
    """
    fechas = pd.date_range(fecha_inicio, fecha_fin, freq="D")
    carpeta = Path(carpeta_salida) if guardar else None
    if carpeta is not None:
        carpeta.mkdir(parents=True, exist_ok=True)

    resumenes = []
    for fecha in fechas:
        nombre_archivo = None
        if carpeta is not None:
            nombre_archivo = carpeta / f"mapa_confusion_{fecha.strftime('%Y_%m_%d')}.png"

        resumen = plot_mapa_confusion_evento(
            df_evento=df_evento,
            fecha=fecha,
            time_col=time_col,
            lat_col=lat_col,
            lon_col=lon_col,
            y_real_col=y_real_col,
            y_pred_col=y_pred_col,
            mostrar_tn=mostrar_tn,
            title=f"Matriz de confusion espacial - {fecha.strftime('%d/%m/%Y')}",
            guardar=guardar,
            nombre_archivo=nombre_archivo,
            mostrar=mostrar,
            figsize=figsize,
            extent=extent,
            pintar_mapa=pintar_mapa,
        )
        if resumen is not None:
            resumenes.append(resumen)

    if not resumenes:
        return None

    resumen_total = pd.concat(resumenes, ignore_index=True)
    if carpeta is not None:
        resumen_total.to_csv(carpeta / "resumen_mapas_confusion.csv", index=False)
        print(f"Mapas guardados en: {carpeta.resolve()}")

    return resumen_total


def plot_mapa_confusion_evento_en_ax(
    ax,
    df_evento: pd.DataFrame,
    fecha,
    time_col: str = "time",
    lat_col: str = "lat",
    lon_col: str = "lon",
    y_real_col: str = "y_real",
    y_pred_col: str = "y_pred",
    mostrar_tn: bool = True,
    title: Optional[str] = None,
    extent=(-10.0, 4.0, 35.0, 44.5),
    pintar_mapa: bool = True,
) -> Optional[Tuple[pd.DataFrame, dict]]:
    """
    Draw one daily confusion map inside an existing axis.
    """
    transform = ccrs.PlateCarree() if CARTOPY_AVAILABLE and hasattr(ax, "projection") else None

    if pintar_mapa:
        _add_spain_base_map(ax, extent=extent, transform=transform)
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(alpha=0.25, linewidth=0.4)

    df_dia, fecha = _filtrar_dia(df_evento, fecha, time_col)
    if df_dia.empty:
        ax.set_title(title or str(fecha), fontsize=9)
        ax.text(0.5, 0.5, "Sin datos", transform=ax.transAxes, ha="center", va="center", fontsize=8)
        return None

    df_dia = anadir_tipo_confusion(df_dia, y_real_col=y_real_col, y_pred_col=y_pred_col)
    orden = CONFUSION_ORDER if mostrar_tn else CONFUSION_ORDER[1:]
    handles = {}

    for categoria in orden:
        subset = df_dia[df_dia["tipo_confusion"] == categoria]
        if subset.empty:
            continue

        alpha = 0.35 if categoria.startswith("TN") else 0.9
        size = 18 if categoria.startswith("TN") else 38
        scatter_kwargs = {"transform": transform} if transform is not None else {}

        sc = ax.scatter(
            subset[lon_col],
            subset[lat_col],
            c=CONFUSION_COLORS[categoria],
            label=categoria,
            s=size,
            marker="s",
            edgecolors="black",
            linewidths=0.25,
            alpha=alpha,
            zorder=5,
            **scatter_kwargs,
        )
        handles[categoria] = sc

    ax.set_title(title or pd.to_datetime(fecha).strftime("%d/%m/%Y"), fontsize=9)
    return _resumen_confusion(df_dia, fecha), handles


def plot_panel_mapa_confusion_eventos_rango(
    df_evento: pd.DataFrame,
    fecha_inicio="2025-08-03",
    fecha_fin="2025-08-18",
    time_col: str = "time",
    lat_col: str = "lat",
    lon_col: str = "lon",
    y_real_col: str = "y_real",
    y_pred_col: str = "y_pred",
    mostrar_tn: bool = True,
    ncols: int = 4,
    figsize=(18, 14),
    extent=(-10.0, 4.0, 35.0, 44.5),
    pintar_mapa: bool = True,
    guardar: bool = False,
    nombre_archivo: str = "panel_mapas_confusion_eventos.png",
) -> Optional[pd.DataFrame]:
    """
    Plot a compact panel of daily spatial confusion maps.
    """
    fechas = pd.date_range(fecha_inicio, fecha_fin, freq="D")
    n = len(fechas)
    nrows = int(np.ceil(n / ncols))

    subplot_kw = {"projection": ccrs.PlateCarree()} if CARTOPY_AVAILABLE and pintar_mapa else {}
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw=subplot_kw)
    axes = np.array(axes).reshape(-1)

    resumenes = []
    handles_global = {}

    for i, fecha in enumerate(fechas):
        ax = axes[i]
        resultado = plot_mapa_confusion_evento_en_ax(
            ax=ax,
            df_evento=df_evento,
            fecha=fecha,
            time_col=time_col,
            lat_col=lat_col,
            lon_col=lon_col,
            y_real_col=y_real_col,
            y_pred_col=y_pred_col,
            mostrar_tn=mostrar_tn,
            title=fecha.strftime("%d/%m/%Y"),
            extent=extent,
            pintar_mapa=pintar_mapa,
        )

        if resultado is not None:
            resumen, handles = resultado
            resumenes.append(resumen)
            for categoria, handle in handles.items():
                handles_global.setdefault(categoria, handle)

        if not (CARTOPY_AVAILABLE and pintar_mapa):
            if i % ncols == 0:
                ax.set_ylabel("Latitud", fontsize=8)
            else:
                ax.set_yticklabels([])

            if i >= (nrows - 1) * ncols:
                ax.set_xlabel("Longitud", fontsize=8)
            else:
                ax.set_xticklabels([])

        ax.tick_params(axis="both", labelsize=7)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    if handles_global:
        fig.legend(
            handles_global.values(),
            handles_global.keys(),
            loc="lower center",
            ncol=min(4, len(handles_global)),
            frameon=True,
            fontsize=10,
        )

    fig.suptitle(
        f"Matriz de confusion espacial diaria ({fecha_inicio} a {fecha_fin})",
        fontsize=16,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    if guardar:
        fig.savefig(nombre_archivo, dpi=300, bbox_inches="tight")

    plt.show()

    if not resumenes:
        return None
    return pd.concat(resumenes, ignore_index=True)


def restaurar_lat_lon_desde_scaler(
    df: pd.DataFrame,
    scaler,
    feature_names,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """
    Restore scaled lat/lon columns using a fitted sklearn scaler.

    Useful when coordinates were included among model features and therefore
    standardized before creating spatial plots.
    """
    df = df.copy()
    feature_names = list(feature_names)

    for col in [lat_col, lon_col]:
        if col not in feature_names:
            raise ValueError(f"{col!r} no esta en feature_names; no se puede desescalar.")
        if col not in df.columns:
            raise ValueError(f"{col!r} no esta en el DataFrame.")

    means = dict(zip(feature_names, scaler.mean_))
    scales = dict(zip(feature_names, scaler.scale_))

    df[lat_col] = df[lat_col] * scales[lat_col] + means[lat_col]
    df[lon_col] = df[lon_col] * scales[lon_col] + means[lon_col]
    return df


__all__ = [
    "anadir_tipo_confusion",
    "añadir_tipo_confusion",
    "plot_mapa_confusion_evento",
    "plot_mapa_confusion_eventos_rango",
    "plot_mapa_confusion_evento_en_ax",
    "plot_panel_mapa_confusion_eventos_rango",
    "restaurar_lat_lon_desde_scaler",
]

