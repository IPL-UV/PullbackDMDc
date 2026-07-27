from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _filter_inlier_coords(coords: np.ndarray, fence_scale: float = 1.5) -> np.ndarray:
    if coords.shape[0] < 4:
        return coords

    q1 = np.percentile(coords, 25.0, axis=0)
    q3 = np.percentile(coords, 75.0, axis=0)
    iqr = q3 - q1
    lower = q1 - fence_scale * iqr
    upper = q3 + fence_scale * iqr
    mask = np.all((coords >= lower) & (coords <= upper), axis=1)

    if np.count_nonzero(mask) >= 3:
        return coords[mask]
    return coords


class TaylorDiagramPlot:
    """Creates a Taylor diagram on standard Cartesian axes."""

    def __init__(
        self,
        ref_point: float,
        fig: Optional[plt.Figure] = None,
        subplot: Optional[int] = 111,
        ax: Optional[plt.Axes] = None,
        extend_angle: bool = False,
        corr_labels: Optional[np.ndarray] = None,
        ref_range: Tuple[float, float] = (0, 10),
        std_range: Optional[Tuple[float, float]] = None,
        ref_label: str = "Reference Point",
        angle_label: str = "Correlation",
        var_label: str = "Standard Deviation",
        corr_label_pad: float = 0.0,
        corr_label_fontsize: float = 10.0,
    ) -> None:
        self.angle_label = angle_label
        self.ref_label = ref_label
        self.var_label = var_label
        self.extend_angle = extend_angle
        self.ref_point = ref_point
        self.corr_label_pad = corr_label_pad
        self.corr_label_fontsize = corr_label_fontsize

        if corr_labels is None:
            corr_labels = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])

        if extend_angle:
            corr_labels = np.concatenate((-corr_labels[:0:-1], corr_labels))
            self.tmin = 0.0
            self.tmax = np.pi
        else:
            self.tmin = 0.0
            self.tmax = np.pi / 2.0

        if fig is None:
            fig = plt.figure(figsize=(8, 8))

        if std_range is None:
            self.smin = ref_range[0] * ref_point
            self.smax = (ref_range[1] / 100) * ref_point + ref_point
        else:
            self.smin, self.smax = std_range

        self.corr_labels = np.clip(np.asarray(corr_labels, dtype=float), -1.0, 1.0)

        if ax is None:
            ax = fig.add_subplot(subplot)
        self.graph_axes = ax
        self.polar_axes = ax
        self.sample_points = []
        self.point_coords = []
        self.corr_tick_artists = []
        self._updating_corr_ticklabels = False
        self._xlim_callback_id = None
        self._ylim_callback_id = None

        self.reset_axes()
        self._register_limit_callbacks()

    def _to_cartesian(self, std_value: float, corr_value: float) -> Tuple[float, float]:
        corr_value = float(np.clip(corr_value, -1.0, 1.0))
        theta = np.arccos(corr_value)
        return std_value * np.cos(theta), std_value * np.sin(theta)

    def _theta_values(self) -> np.ndarray:
        return np.linspace(self.tmin, self.tmax, 400)

    def _std_ticks(self) -> np.ndarray:
        return np.linspace(self.smin, self.smax, 7)

    def add_reference_point(self, ref_point: float, *args, **kwargs) -> None:
        line = self.graph_axes.plot([ref_point], [0.0], *args, **kwargs)
        self.sample_points.append(line[0])
        self.point_coords.append((ref_point, 0.0))

    def add_reference_line(self, ref_point: float, *args, **kwargs) -> None:
        theta = self._theta_values()
        x = ref_point * np.cos(theta)
        y = ref_point * np.sin(theta)
        self.graph_axes.plot(x, y, *args, **kwargs)

    def add_point(self, var_point: float, corr_point: float, *args, **kwargs) -> None:
        x, y = self._to_cartesian(var_point, corr_point)
        line = self.graph_axes.plot([x], [y], *args, **kwargs)
        self.sample_points.append(line[0])
        self.point_coords.append((x, y))

    def add_scatter(
        self, var_points: np.ndarray, corr_points: np.ndarray, *args, **kwargs
    ) -> None:
        corr_points = np.clip(np.asarray(corr_points, dtype=float), -1.0, 1.0)
        var_points = np.asarray(var_points, dtype=float)
        theta = np.arccos(corr_points)
        x = var_points * np.cos(theta)
        y = var_points * np.sin(theta)
        pts = self.graph_axes.scatter(x, y, *args, **kwargs)
        self.sample_points.append(pts)
        self.point_coords.extend(zip(x.tolist(), y.tolist()))

    def add_grid(self, *args, **kwargs):
        grid_color = kwargs.pop("color", "0.7")
        grid_alpha = kwargs.pop("alpha", 1.0)
        grid_lw = kwargs.pop("linewidth", 1.0)
        theta = self._theta_values()

        for std_tick in self._std_ticks():
            x = std_tick * np.cos(theta)
            y = std_tick * np.sin(theta)
            self.graph_axes.plot(x, y, color=grid_color, alpha=grid_alpha, linewidth=grid_lw, zorder=0)

        for corr_tick in self.corr_labels:
            if not self.extend_angle and corr_tick < 0:
                continue
            x = np.array([0.0, self.smax * corr_tick])
            y = np.array([0.0, self.smax * np.sqrt(max(0.0, 1.0 - corr_tick ** 2))])
            self.graph_axes.plot(x, y, color=grid_color, alpha=grid_alpha, linewidth=grid_lw, zorder=0)

        self._draw_corr_ticklabels()

    def add_contours(self, ref_point: float, levels: int = 4, **kwargs) -> None:
        x_min = -self.smax if self.extend_angle else 0.0
        x_vals = np.linspace(x_min, self.smax, 400)
        y_vals = np.linspace(0.0, self.smax, 400)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)

        std_grid = np.sqrt(x_grid ** 2 + y_grid ** 2)
        valid = (std_grid >= self.smin) & (std_grid <= self.smax)
        if not self.extend_angle:
            valid &= x_grid >= 0.0

        dist = np.sqrt((x_grid - ref_point) ** 2 + y_grid ** 2)
        dist = np.ma.masked_where(~valid, dist)
        self.contours = self.graph_axes.contour(x_grid, y_grid, dist, levels=levels, **kwargs)

    def add_legend(self, fig, *args, **kwargs):
        handles_labels = [
            (point, point.get_label())
            for point in self.sample_points
            if point.get_label() is not None and not str(point.get_label()).startswith("_")
        ]
        if handles_labels:
            handles, labels = zip(*handles_labels)
            legend = fig.legend(handles, labels, *args, **kwargs)
            marker_sizes = [handle.get_markersize() for handle in handles if hasattr(handle, "get_markersize")]
            if marker_sizes:
                max_size = max(marker_sizes)
                for handle in legend.legend_handles:
                    if hasattr(handle, "set_markersize"):
                        handle.set_markersize(max_size)

    def autoscale_to_points(self, padding: float = 0.03, percentage: float = 98.0):
        extent = self.get_point_extent(padding=padding, percentage=percentage)
        if extent is None:
            return

        self.set_extent(extent)

    def get_point_extent(self, padding: float = 0.03, percentage: float = 98.0):
        if not self.point_coords:
            return None

        coords = np.asarray(self.point_coords, dtype=float)
        coords = _filter_inlier_coords(coords)
        x_vals = coords[:, 0]
        y_vals = coords[:, 1]

        tail_percent = max(0.0, min(100.0, 100.0 - float(percentage))) / 2.0
        lower_percent = tail_percent
        upper_percent = 100.0 - tail_percent

        x_min = float(np.percentile(x_vals, lower_percent))
        x_max = float(np.percentile(x_vals, upper_percent))
        y_min = float(np.percentile(y_vals, lower_percent))
        y_max = float(np.percentile(y_vals, upper_percent))

        x_span = max(x_max - x_min, 1e-3)
        y_span = max(y_max - y_min, 1e-3)

        return (
            (x_min - padding * x_span, x_max + padding * x_span),
            (max(0.0, y_min - padding * y_span), y_max + padding * y_span),
        )

    def set_extent(self, extent) -> None:
        (x_min, x_max), (y_min, y_max) = extent
        self.graph_axes.set_xlim(x_min, x_max)
        self.graph_axes.set_ylim(y_min, y_max)
        self._draw_corr_ticklabels()

    def reset_axes(self):
        ax = self.graph_axes
        ax.cla()
        ax.set_aspect("equal", adjustable="box")
        x_min = -self.smax if self.extend_angle else 0.0
        ax.set_xlim(x_min, self.smax)
        ax.set_ylim(0.0, self.smax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel(self.var_label)
        ax.set_xlabel(self.var_label)
        ax.tick_params(direction="in")
        self._setup_angle_axes()
        self._setup_xaxis()
        self._setup_yaxis()

    def reset_axes_labels(
        self, angle_label: str = "Correlation", var_label: str = "Variance"
    ):
        self.angle_label = angle_label
        self.var_label = var_label
        self.graph_axes.set_xlabel(var_label)
        self.graph_axes.set_ylabel(var_label)
        if getattr(self, "angle_label_artist", None) is not None:
            self.angle_label_artist.set_visible(False)

    def _setup_angle_axes(self):
        self.angle_label_artist = None

    def _setup_xaxis(self):
        self.graph_axes.xaxis.labelpad = 8

    def _setup_yaxis(self):
        self.graph_axes.yaxis.labelpad = 8

    def _clear_corr_ticklabels(self):
        for artist in self.corr_tick_artists:
            try:
                artist.remove()
            except ValueError:
                # Artist may already be removed during interactive redraws.
                pass
        self.corr_tick_artists = []

    def _register_limit_callbacks(self):
        if self._xlim_callback_id is not None:
            self.graph_axes.callbacks.disconnect(self._xlim_callback_id)
        if self._ylim_callback_id is not None:
            self.graph_axes.callbacks.disconnect(self._ylim_callback_id)

        self._xlim_callback_id = self.graph_axes.callbacks.connect(
            "xlim_changed", self._on_limits_changed
        )
        self._ylim_callback_id = self.graph_axes.callbacks.connect(
            "ylim_changed", self._on_limits_changed
        )

    def _on_limits_changed(self, _):
        if self._updating_corr_ticklabels:
            return
        self._draw_corr_ticklabels()

    def _draw_corr_ticklabels(self):
        if self._updating_corr_ticklabels:
            return

        self._updating_corr_ticklabels = True
        self._clear_corr_ticklabels()

        x_min, x_max = self.graph_axes.get_xlim()
        y_min, y_max = self.graph_axes.get_ylim()
        try:
            for corr_tick in self.corr_labels:
                if not self.extend_angle and corr_tick < 0:
                    continue

                direction_x = float(corr_tick)
                direction_y = float(np.sqrt(max(0.0, 1.0 - corr_tick ** 2)))

                upper_bounds = [self.smax]
                lower_bounds = [0.0]

                if abs(direction_x) > 1e-12:
                    upper_bounds.append(x_max / direction_x)
                    lower_bounds.append(x_min / direction_x)
                elif x_min > 0.0 or x_max < 0.0:
                    continue

                if abs(direction_y) > 1e-12:
                    upper_bounds.append(y_max / direction_y)
                    lower_bounds.append(y_min / direction_y)
                elif y_min > 0.0 or y_max < 0.0:
                    continue

                radius_max = min(upper_bounds)
                radius_min = max(lower_bounds)

                if radius_max <= 0.0 or radius_max < radius_min:
                    continue

                label_radius = radius_max + self.corr_label_pad * self.smax
                x, y = self._to_cartesian(label_radius, corr_tick)
                theta = np.arccos(np.clip(corr_tick, -1.0, 1.0))
                artist = self.graph_axes.text(
                    x,
                    y,
                    f"{corr_tick:g}",
                    rotation=np.degrees(theta) - 90,
                    ha="center",
                    va="center",
                    fontsize=self.corr_label_fontsize,
                    clip_on=False,
                )
                self.corr_tick_artists.append(artist)
        finally:
            self._updating_corr_ticklabels = False
