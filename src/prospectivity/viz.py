"""Figures and the interactive map."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from pyproj import Transformer  # noqa: E402

from .grid import Grid  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.facecolor": "white",
    }
)


def _extent(grid: Grid) -> tuple[float, float, float, float]:
    b = grid.bounds
    return (b.left / 1000.0, b.right / 1000.0, b.bottom / 1000.0, b.top / 1000.0)


def _deposit_xy(deposits: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return deposits["x"].to_numpy() / 1000.0, deposits["y"].to_numpy() / 1000.0


def plot_study_area(
    dem: np.ndarray, grid: Grid, deposits: pd.DataFrame, path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(dem, extent=_extent(grid), cmap="terrain", origin="upper")
    x, y = _deposit_xy(deposits)
    prod = deposits.get("confidence", pd.Series(1, index=deposits.index)) >= 3
    ax.scatter(x[~prod], y[~prod], s=9, c="#222222", alpha=0.5, label="Au/Ag occurrence")
    ax.scatter(x[prod], y[prod], s=22, c="#d62728", edgecolor="white",
               linewidth=0.4, label="Producer / past producer")
    ax.set_title(f"Study area: {grid.width} x {grid.height} cells at {grid.res[0]:.0f} m")
    ax.set_xlabel("UTM 11N easting (km)")
    ax.set_ylabel("UTM 11N northing (km)")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Elevation (m)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_feature_maps(cube, grid: Grid, path: Path, names: list[str] | None = None) -> Path:
    names = names or cube.names[:9]
    n = len(names)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.1 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, names, strict=False):
        arr = cube.layers[name]
        im = ax.imshow(arr, extent=_extent(grid), origin="upper", cmap="viridis")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Feature layers on the common 500 m lattice", y=1.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_prospectivity(
    prob: np.ndarray,
    grid: Grid,
    deposits: pd.DataFrame,
    path: Path,
    targets: pd.DataFrame | None = None,
    model_label: str = "random forest",
) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    vmax = float(np.nanpercentile(prob, 99.8)) or 1.0
    im = ax.imshow(prob, extent=_extent(grid), origin="upper", cmap="magma",
                   norm=Normalize(0, vmax))
    x, y = _deposit_xy(deposits)
    ax.scatter(x, y, s=10, facecolor="none", edgecolor="#39d0ff", linewidth=0.6,
               label="Known Au/Ag deposits")
    if targets is not None and len(targets):
        tx = (grid.bounds.left + (targets["col"] + 0.5) * grid.res[0]) / 1000.0
        ty = (grid.bounds.top - (targets["row"] + 0.5) * grid.res[1]) / 1000.0
        ax.scatter(tx, ty, s=70, marker="^", facecolor="#ffe14d",
                   edgecolor="black", linewidth=0.6, label="Top-ranked targets")
    ax.set_title(f"Prospectivity probability ({model_label}), 500 m grid")
    ax.set_xlabel("UTM 11N easting (km)")
    ax.set_ylabel("UTM 11N northing (km)")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.75, label="P(deposit | features)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_metrics(
    metrics: dict[str, dict], curves: dict[str, tuple], path: Path,
    final_key: str = "naive",
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    ax = axes[0]
    for label, m in metrics.items():
        ax.bar(label, m["roc_auc"], color="#3b6ea5", width=0.5)
        ax.text(label, m["roc_auc"] + 0.01, f"{m['roc_auc']:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("ROC AUC vs unlabelled")
    ax.set_title("Discrimination")

    ax = axes[1]
    for label, (area, cum) in curves.items():
        ax.plot(area * 100, cum * 100, label=label)
    ax.plot([0, 100], [0, 100], ls="--", color="#999999", lw=0.8)
    ax.set_xlabel("Area ranked (%)")
    ax.set_ylabel("Deposits captured (%)")
    ax.set_title("Capture curve")
    ax.legend(fontsize=8)

    ax = axes[2]
    m = metrics[final_key]
    keys = [k for k in m if k.startswith("capture_at")]
    vals = [m[k] * 100 for k in keys]
    ax.bar([k.replace("capture_at_", "").replace("pct", "%") for k in keys], vals,
           color="#c1553b")
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_ylabel("Deposits captured")
    ax.set_title("Capture at area budgets (final model)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_shap_importance(importance: pd.DataFrame, path: Path, top: int = 18) -> Path:
    data = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(data) + 1.6))
    ax.barh(data["feature"], data["mean_abs_shap"], color="#4c72b0")
    ax.set_xlabel("mean |SHAP value| (probability units)")
    ax.set_title("Global feature importance")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_shap_beeswarm(shap_result, path: Path) -> Path:
    import shap

    fig = plt.figure(figsize=(8, 6.5))
    shap.summary_plot(
        shap_result.values, shap_result.data, show=False, max_display=16
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_shap_dependence(shap_result, features: list[str], path: Path) -> Path:
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, features, strict=False):
        i = shap_result.feature_names.index(name)
        shap_i = shap_result.values[:, i]
        # Colour by the feature whose value interacts most strongly with SHAP_i.
        best_j, best_r = i, 0.0
        for k in range(shap_result.values.shape[1]):
            if k == i:
                continue
            xk = shap_result.data.iloc[:, k].to_numpy(dtype="float64")
            if np.std(xk) == 0 or np.std(shap_i) == 0:
                continue
            r = abs(float(np.corrcoef(xk, shap_i)[0, 1]))
            if r > best_r:
                best_j, best_r = k, r
        j = best_j
        color = shap_result.data.iloc[:, j].to_numpy()
        sc = ax.scatter(shap_result.data.iloc[:, i], shap_result.values[:, i],
                        c=color, s=6, cmap="coolwarm", alpha=0.7)
        ax.set_xlabel(name)
        ax.set_ylabel("SHAP value")
        fig.colorbar(sc, ax=ax, shrink=0.8, label=shap_result.feature_names[j])
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_local_explanation(shap_result, row: int, path: Path, label: str = "target") -> Path:
    import shap

    fig = plt.figure(figsize=(7.5, 5))
    explanation = shap.Explanation(
        values=shap_result.values[row],
        base_values=shap_result.base_value,
        data=shap_result.data.iloc[row].to_numpy(),
        feature_names=shap_result.feature_names,
    )
    shap.plots.waterfall(explanation, show=False, max_display=12)
    plt.title(f"Why this cell is prospective ({label})", fontsize=10)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _overlay_png(prob: np.ndarray, grid: Grid) -> str:
    vmax = float(np.nanpercentile(prob, 99.8)) or 1.0
    fig = plt.figure(figsize=(6, 6 * grid.height / grid.width))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(prob, origin="upper", cmap="magma", norm=Normalize(0, vmax),
              interpolation="bilinear", aspect="auto")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=160)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def folium_map(
    prob: np.ndarray,
    grid: Grid,
    deposits: pd.DataFrame,
    path: Path,
    targets: pd.DataFrame | None = None,
    title: str = "Prospectivity map",
) -> Path:
    import folium
    from folium import plugins

    tr = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    b = grid.bounds
    lons, lats = tr.transform([b.left, b.right, b.left, b.right],
                              [b.bottom, b.bottom, b.top, b.top])
    bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
    centre = [(min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2]

    carto_key = os.environ["CARTO_API_KEY"]

    m = folium.Map(location=centre, zoom_start=9, tiles=None,
                   control_scale=True)

    folium.TileLayer(
        tiles=(
            "https://basemaps.cartocdn.com/rastertiles/"
            f"dark_all/{{z}}/{{x}}/{{y}}.png?key={carto_key}"
        ),
        attr='© <a href="https://www.openstreetmap.org/copyright">'
                'OpenStreetMap</a> contributors © '
                '<a href="https://carto.com/attributions/">CARTO</a>',
        name="CARTO Dark Matter",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{_overlay_png(prob, grid)}",
        bounds=bounds,
        opacity=0.75,
        name="Prospectivity probability",
        interactive=True,
    ).add_to(m)

    fg = folium.FeatureGroup(name="Known Au/Ag deposits")
    for row in deposits.itertuples():
        colour = "#ff4d4d" if row.confidence >= 3 else "#ffd24d"
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=3.2,
            color=colour,
            fill=True,
            fill_opacity=0.85,
            weight=0.6,
            popup=f"{row.site_name}<br>{row.dev_stat}",
        ).add_to(fg)
    fg.add_to(m)

    if targets is not None and len(targets):
        tg = folium.FeatureGroup(name="Top-ranked targets")
        for rank, row in enumerate(targets.itertuples(), start=1):
            lat = row.lat
            lon = row.lon
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(
                    html=(
                        f'<div style="font-size:10px;font-weight:700;color:#111;'
                        f'background:#ffe14d;border:1px solid #333;border-radius:9px;'
                        f'width:18px;height:18px;text-align:center;line-height:18px;">'
                        f'{rank}</div>'
                    ),
                    icon_size=(18, 18),
                ),
                popup=(f"Rank {rank}<br>P = {row.probability:.3f}<br>"
                       f"Nearest known deposit: {row.nearest_deposit_km:.1f} km"),
            ).add_to(tg)
        tg.add_to(m)

    legend = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: rgba(20,20,20,0.86); color: #eee; padding: 10px 12px;
                border-radius: 6px; font-size: 12px; line-height: 1.5;
                font-family: system-ui, sans-serif;">
      <b>Prospectivity probability</b><br>
      <div style="height:10px;width:180px;margin:4px 0;
                  background:linear-gradient(90deg,#000004,#51127c,#b73779,#fc8961,#fcfdbf);"></div>
      0.0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.5 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1.0<br>
      <span style="color:#ff4d4d">&#9679;</span> producer / past producer &nbsp;
      <span style="color:#ffd24d">&#9679;</span> occurrence<br>
      <span style="color:#ffe14d">&#9650;</span> top-ranked targets
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(bounds)
    m.save(str(path))
    return path
