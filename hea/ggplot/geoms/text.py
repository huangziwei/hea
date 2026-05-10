"""``geom_text()`` and ``geom_label()`` — text annotations at (x, y).

Smaller surface than ggplot2's full geom_text: we ship the common
arguments (``hjust``, ``vjust``, ``angle``, ``size``, ``colour``,
``family``, ``fontface``). ``geom_label`` adds a rounded background
box; surface mirrors ggplot2's: ``fill``, ``label_padding``, ``label_r``
(corner radius), ``label_size`` (border line width).

``geom_label_repel`` ports ggrepel's ``geom_label_repel``: a force
simulation pushes overlapping labels apart and away from data points,
with connector segments back to each anchor. The simulation runs in
display coordinates after the figure's first draw (when matplotlib's
renderer can report accurate text bboxes); a one-shot draw_event
callback handles this transparently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..aes import split_layer_kwargs
from .geom import Geom


_PT_PER_MM = 72.27 / 25.4


@dataclass
class GeomText(Geom):
    # Mirrors ggplot2's ``GeomText$default_aes`` (R/geom-text.R). ``size``
    # is in MM (ggplot2's text-size convention) — 11 pt × 25.4 / 72.27 ≈
    # 3.88 mm. ``lineheight`` is the ratio of line spacing to font size.
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "size": 3.88,
        "angle": 0.0,
        "hjust": 0.5,
        "vjust": 0.5,
        "alpha": 1.0,
        "family": "",
        "fontface": "plain",
        "lineheight": 1.2,
    })
    required_aes: tuple = ("x", "y", "label")

    def draw_panel(self, data, ax) -> None:
        from .._util import r_color

        if len(data) == 0:
            return
        x = data["x"].to_numpy()
        y = data["y"].to_numpy()
        labels = data["label"].to_list()

        # Per-row attributes — broadcast-style fallback to default if absent.
        colour = data["colour"].to_list() if "colour" in data.columns else None
        size = data["size"].to_numpy() if "size" in data.columns else None
        angle = data["angle"].to_numpy() if "angle" in data.columns else None
        hjust = data["hjust"].to_numpy() if "hjust" in data.columns else None
        vjust = data["vjust"].to_numpy() if "vjust" in data.columns else None

        x_range, y_range = _axis_ranges_for_inward(ax, x, y, hjust, vjust)

        for i, label in enumerate(labels):
            if label is None:
                continue
            h_raw = hjust[i] if hjust is not None else self.default_aes["hjust"]
            v_raw = vjust[i] if vjust is not None else self.default_aes["vjust"]
            kwargs = {
                "color": r_color(colour[i]) if colour else self.default_aes["colour"],
                "fontsize": float(size[i]) * _PT_PER_MM if size is not None
                            else self.default_aes["size"] * _PT_PER_MM,
                "rotation": float(angle[i]) if angle is not None
                            else self.default_aes["angle"],
                "ha": _hjust_to_ha(_resolve_just(h_raw, x[i], x_range, axis="h")),
                "va": _vjust_to_va(_resolve_just(v_raw, y[i], y_range, axis="v")),
            }
            ax.text(x[i], y[i], str(label), **kwargs)


# ggplot2's compute_just (R/utilities.R) maps these names to numerics.
_HJUST_ALIASES = {
    "left": 0.0, "right": 1.0,
    "center": 0.5, "centre": 0.5, "middle": 0.5,
}
_VJUST_ALIASES = {
    "bottom": 0.0, "top": 1.0,
    "center": 0.5, "centre": 0.5, "middle": 0.5,
}


def _resolve_just(raw, coord, axis_range, *, axis: str) -> float:
    """Resolve a justify value to numeric in [0, 1].

    Accepts numerics (passed through), static strings (``"left"``,
    ``"right"``, ``"top"``, ``"bottom"``, ``"center"``/``"centre"``/
    ``"middle"``), and the position-aware ``"inward"`` / ``"outward"``
    (resolved per ggplot2's ``compute_just``: split on the panel midpoint).
    Unknown strings fall back to 0.5 (matches ggplot2's silent-NA behaviour).
    """
    aliases = _HJUST_ALIASES if axis == "h" else _VJUST_ALIASES
    if isinstance(raw, str):
        if raw in aliases:
            return aliases[raw]
        if raw in ("inward", "outward"):
            lo, hi = axis_range
            if hi <= lo:
                return 0.5
            mid = 0.5 * (lo + hi)
            # ggplot2's just_dir splits at the panel midpoint with a small tol.
            tol = 1e-6 * (hi - lo)
            if coord < mid - tol:
                return 0.0 if raw == "inward" else 1.0
            if coord > mid + tol:
                return 1.0 if raw == "inward" else 0.0
            return 0.5
        return 0.5  # unknown alias → ggplot2 silently uses NA → 0.5
    return float(raw)


def _axis_ranges_for_inward(ax, x, y, hjust, vjust) -> tuple:
    """Pick axis ranges for resolving inward/outward.

    ``draw_panel`` runs before scales call ``apply_to_axis``, so
    ``ax.get_xlim()`` may still be matplotlib's auto view from artists
    drawn earlier in the layer stack. That's the right reference for
    inward/outward — same panel the data lands in. If no other layer
    has drawn yet we fall back to the geom's own data range. If neither
    hjust nor vjust uses ``inward``/``outward`` we skip the work."""
    needs_h = _has_inward_outward(hjust)
    needs_v = _has_inward_outward(vjust)
    if not (needs_h or needs_v):
        return ((0.0, 1.0), (0.0, 1.0))

    def _range(axis_lim, coords):
        lo, hi = axis_lim
        if hi > lo and not (lo == 0.0 and hi == 1.0):
            return (float(lo), float(hi))
        # Fall back to layer data extent.
        if len(coords) == 0:
            return (0.0, 1.0)
        try:
            return (float(coords.min()), float(coords.max()))
        except (TypeError, ValueError):
            return (0.0, 1.0)

    return (_range(ax.get_xlim(), x), _range(ax.get_ylim(), y))


def _has_inward_outward(arr) -> bool:
    if arr is None:
        return False
    try:
        return any(v in ("inward", "outward") for v in arr if isinstance(v, str))
    except TypeError:
        return False


def _hjust_to_ha(h: float) -> str:
    if h <= 0.25:
        return "left"
    if h >= 0.75:
        return "right"
    return "center"


def _vjust_to_va(v: float) -> str:
    if v <= 0.25:
        return "bottom"
    if v >= 0.75:
        return "top"
    return "center"


def geom_text(mapping=None, data=None, *, stat="identity", position="identity",
              na_rm=False, **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat

    aes_params, geom_params = split_layer_kwargs(kwargs)

    return Layer(
        geom=GeomText(),
        stat=resolve_stat(stat),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
        na_rm=na_rm,
    )


@dataclass
class GeomLabel(GeomText):
    """``geom_text`` plus a rounded background box (ggplot2 ``geom_label``).

    Mirrors ggplot2's ``GeomLabel$default_aes`` (R/geom-label.R) — adds
    ``fill = "white"``, ``linewidth = 0.5 * borderwidth = 0.25``,
    ``linetype = "solid"`` to ``GeomText``'s defaults.
    """
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "fill": "white",
        "size": 3.88,
        "angle": 0.0,
        "hjust": 0.5,
        "vjust": 0.5,
        "alpha": 1.0,
        "family": "",
        "fontface": "plain",
        "lineheight": 1.2,
        "linewidth": 0.25,
        "linetype": "solid",
    })
    label_padding: float = 0.25  # ggplot2 default in lines; we treat as box pad
    label_r: float = 0.15        # corner radius
    label_size: float = 0.25     # border line width (mm)

    def draw_panel(self, data, ax) -> None:
        from .._util import r_color

        if len(data) == 0:
            return
        x = data["x"].to_numpy()
        y = data["y"].to_numpy()
        labels = data["label"].to_list()
        n = len(labels)

        colour = data["colour"].to_list() if "colour" in data.columns else [self.default_aes["colour"]] * n
        fill = data["fill"].to_list() if "fill" in data.columns else [self.default_aes["fill"]] * n
        size = data["size"].to_numpy() if "size" in data.columns else [self.default_aes["size"]] * n
        angle = data["angle"].to_numpy() if "angle" in data.columns else [self.default_aes["angle"]] * n
        hjust = data["hjust"].to_numpy() if "hjust" in data.columns else [self.default_aes["hjust"]] * n
        vjust = data["vjust"].to_numpy() if "vjust" in data.columns else [self.default_aes["vjust"]] * n

        # Border width in pt — ggplot2 ``label_size`` is mm.
        border_pt = float(self.label_size) * _PT_PER_MM

        x_range, y_range = _axis_ranges_for_inward(ax, x, y, hjust, vjust)

        for i, label in enumerate(labels):
            if label is None:
                continue
            ax.text(
                x[i], y[i], str(label),
                color=r_color(colour[i]),
                fontsize=float(size[i]) * _PT_PER_MM,
                rotation=float(angle[i]),
                ha=_hjust_to_ha(_resolve_just(hjust[i], x[i], x_range, axis="h")),
                va=_vjust_to_va(_resolve_just(vjust[i], y[i], y_range, axis="v")),
                bbox=dict(
                    boxstyle=f"round,pad={float(self.label_padding)},"
                             f"rounding_size={float(self.label_r)}",
                    facecolor=r_color(fill[i]),
                    edgecolor=r_color(colour[i]),
                    linewidth=border_pt,
                ),
            )


def geom_label(mapping=None, data=None, *, stat="identity", position="identity",
               na_rm=False, label_padding=0.25, label_r=0.15, label_size=0.25,
               **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat

    aes_params, geom_params = split_layer_kwargs(kwargs)

    return Layer(
        geom=GeomLabel(label_padding=label_padding, label_r=label_r,
                       label_size=label_size),
        stat=resolve_stat(stat),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
        na_rm=na_rm,
    )


# ---------------------------------------------------------------------------
# geom_label_repel — port of ggrepel's force-directed label placement
# ---------------------------------------------------------------------------


@dataclass
class GeomLabelRepel(GeomLabel):
    """ggrepel's ``geom_label_repel`` — labels positioned to avoid overlaps
    with each other and with data points; connector lines back to anchors.

    Approach: place labels at ``(x + nudge_x, y + nudge_y)`` initially,
    then defer to a one-shot ``draw_event`` callback. On first draw we
    have the matplotlib renderer (so text bboxes are accurate); we run a
    pixel-space force simulation, move the labels, then add segment
    artists from each label-box edge back to its anchor. A second
    ``draw_idle`` paints the moved state.

    Full ggrepel parameter set is supported. Connectors render via
    :class:`FancyArrowPatch` so curvature, arrows, and line types all
    flow through one path. ``segment_ncp`` is accepted for API parity but
    ignored: matplotlib's ``arc3`` connection style uses a fixed control
    point — pure rad, no n-point control. ``parse`` (R plotmath) isn't
    portable; matplotlib supports its own mathtext with ``$...$``.
    """
    nudge_x: float = 0.0
    nudge_y: float = 0.0
    force: float = 1.0
    force_pull: float = 1.0
    box_padding: float = 0.25         # in lines
    point_padding: float = 0.0        # in lines
    min_segment_length: float = 0.5   # in lines
    max_iter: int = 2000
    max_time: float = 0.5             # seconds; 0 = no time cap
    max_overlaps: int = 10
    seed: int | None = None
    segment_color: str | None = None  # default: same as label colour
    segment_size: float = 0.5         # mm
    segment_alpha: float = 1.0
    segment_linetype: object = 1      # R lty (1=solid) or matplotlib string
    segment_curvature: float = 0.0    # arc3 rad; positive curves one way
    segment_ncp: int = 1              # accepted for parity; ignored
    arrow: bool = False               # arrowhead at the anchor end
    direction: str = "both"           # "both" | "x" | "y"
    xlim: tuple | None = None         # (min, max) in data coords; None = panel
    ylim: tuple | None = None
    verbose: bool = False

    def _make_text_artist(self, ax, x, y, label, *,
                          colour, fill, size, angle, ha, va):
        """Render this label as text inside a rounded box (geom_label style)."""
        from .._util import r_color
        border_pt = float(self.label_size) * _PT_PER_MM
        return ax.text(
            x, y, str(label),
            color=r_color(colour),
            fontsize=float(size) * _PT_PER_MM,
            rotation=float(angle),
            ha=ha, va=va,
            bbox=dict(
                boxstyle=f"round,pad={float(self.label_padding)},"
                         f"rounding_size={float(self.label_r)}",
                facecolor=r_color(fill),
                edgecolor=r_color(colour),
                linewidth=border_pt,
            ),
            zorder=10,
        )

    def draw_panel(self, data, ax) -> None:
        if len(data) == 0:
            return
        x = data["x"].to_numpy().astype(float)
        y = data["y"].to_numpy().astype(float)
        labels = data["label"].to_list()
        n = len(labels)

        colour = data["colour"].to_list() if "colour" in data.columns else [self.default_aes["colour"]] * n
        # ``fill`` only exists on GeomLabel subclasses; fall back to ``colour``
        # so GeomTextRepel can share this code path.
        if "fill" in data.columns:
            fill = data["fill"].to_list()
        elif "fill" in self.default_aes:
            fill = [self.default_aes["fill"]] * n
        else:
            fill = list(colour)
        size = data["size"].to_numpy() if "size" in data.columns else np.full(n, self.default_aes["size"])
        angle = data["angle"].to_numpy() if "angle" in data.columns else np.full(n, self.default_aes["angle"])
        hjust_arr = data["hjust"].to_numpy() if "hjust" in data.columns else None
        vjust_arr = data["vjust"].to_numpy() if "vjust" in data.columns else None

        # Drop None / empty labels — ggrepel's na_rm equivalent.
        keep = [i for i, lbl in enumerate(labels)
                if lbl is not None and str(lbl) != ""]
        if not keep:
            return

        # Initial positions: anchor + nudge.
        x_init = x[keep] + float(self.nudge_x)
        y_init = y[keep] + float(self.nudge_y)

        # Resolve per-label ha/va (string aliases + inward/outward handled).
        x_range, y_range = _axis_ranges_for_inward(
            ax, x[keep], y[keep], hjust_arr, vjust_arr,
        )
        ha_list, va_list = [], []
        for k, i in enumerate(keep):
            h_raw = hjust_arr[i] if hjust_arr is not None else self.default_aes["hjust"]
            v_raw = vjust_arr[i] if vjust_arr is not None else self.default_aes["vjust"]
            ha_list.append(_hjust_to_ha(_resolve_just(h_raw, x[i], x_range, axis="h")))
            va_list.append(_vjust_to_va(_resolve_just(v_raw, y[i], y_range, axis="v")))

        text_artists = []
        for k, i in enumerate(keep):
            ta = self._make_text_artist(
                ax, x_init[k], y_init[k], labels[i],
                colour=colour[i], fill=fill[i],
                size=size[i], angle=angle[i],
                ha=ha_list[k], va=va_list[k],
            )
            # Don't let tight_layout shrink the axes for moved labels.
            ta.set_in_layout(False)
            text_artists.append(ta)

        _schedule_repel(
            ax=ax,
            text_artists=text_artists,
            ha_list=ha_list, va_list=va_list,
            x_anchor=x[keep], y_anchor=y[keep],
            x_init=x_init, y_init=y_init,
            colours=[colour[i] for i in keep],
            sizes=[float(size[i]) for i in keep],
            force=float(self.force),
            force_pull=float(self.force_pull),
            max_iter=int(self.max_iter),
            max_time=float(self.max_time),
            max_overlaps=int(self.max_overlaps),
            box_padding_lines=float(self.box_padding),
            point_padding_lines=float(self.point_padding),
            min_segment_length_lines=float(self.min_segment_length),
            seed=self.seed,
            segment_color=self.segment_color,
            segment_size_mm=float(self.segment_size),
            segment_alpha=float(self.segment_alpha),
            segment_linetype=self.segment_linetype,
            segment_curvature=float(self.segment_curvature),
            arrow=bool(self.arrow),
            direction=str(self.direction),
            xlim=self.xlim,
            ylim=self.ylim,
            verbose=bool(self.verbose),
        )


def geom_label_repel(mapping=None, data=None, *, stat="identity",
                     position="identity", na_rm=False,
                     label_padding=0.25, label_r=0.15, label_size=0.25,
                     nudge_x=0.0, nudge_y=0.0,
                     force=1.0, force_pull=1.0,
                     box_padding=0.25, point_padding=0.0,
                     min_segment_length=0.5,
                     max_iter=2000, max_time=0.5, max_overlaps=10,
                     seed=None,
                     segment_color=None, segment_size=0.5,
                     segment_alpha=1.0, segment_linetype=1,
                     segment_curvature=0.0, segment_ncp=1,
                     arrow=False,
                     direction="both", xlim=None, ylim=None,
                     verbose=False,
                     **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat

    aes_params, geom_params = split_layer_kwargs(kwargs)

    return Layer(
        geom=GeomLabelRepel(
            label_padding=label_padding, label_r=label_r,
            label_size=label_size,
            nudge_x=nudge_x, nudge_y=nudge_y,
            force=force, force_pull=force_pull,
            box_padding=box_padding, point_padding=point_padding,
            min_segment_length=min_segment_length,
            max_iter=max_iter, max_time=max_time,
            max_overlaps=max_overlaps,
            seed=seed,
            segment_color=segment_color, segment_size=segment_size,
            segment_alpha=segment_alpha,
            segment_linetype=segment_linetype,
            segment_curvature=segment_curvature,
            segment_ncp=segment_ncp,
            arrow=arrow,
            direction=direction, xlim=xlim, ylim=ylim,
            verbose=verbose,
        ),
        stat=resolve_stat(stat),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
        na_rm=na_rm,
    )


@dataclass
class GeomTextRepel(GeomText):
    """ggrepel's ``geom_text_repel`` — same force-directed placement and
    connector machinery as :class:`GeomLabelRepel`, but the labels render
    as bare text (no rounded background box). Useful when you want repel
    behaviour without the visual weight of a fill/border."""
    nudge_x: float = 0.0
    nudge_y: float = 0.0
    force: float = 1.0
    force_pull: float = 1.0
    box_padding: float = 0.25
    point_padding: float = 0.0
    min_segment_length: float = 0.5
    max_iter: int = 2000
    max_time: float = 0.5
    max_overlaps: int = 10
    seed: int | None = None
    segment_color: str | None = None
    segment_size: float = 0.5
    segment_alpha: float = 1.0
    segment_linetype: object = 1
    segment_curvature: float = 0.0
    segment_ncp: int = 1
    arrow: bool = False
    direction: str = "both"
    xlim: tuple | None = None
    ylim: tuple | None = None
    verbose: bool = False

    def _make_text_artist(self, ax, x, y, label, *,
                          colour, fill, size, angle, ha, va):
        """Render this label as bare text — no bounding box."""
        from .._util import r_color
        return ax.text(
            x, y, str(label),
            color=r_color(colour),
            fontsize=float(size) * _PT_PER_MM,
            rotation=float(angle),
            ha=ha, va=va,
            zorder=10,
        )

    # Reuse GeomLabelRepel's draw_panel — it already routes through
    # self._make_text_artist for the per-row artist.
    draw_panel = GeomLabelRepel.draw_panel


def geom_text_repel(mapping=None, data=None, *, stat="identity",
                    position="identity", na_rm=False,
                    nudge_x=0.0, nudge_y=0.0,
                    force=1.0, force_pull=1.0,
                    box_padding=0.25, point_padding=0.0,
                    min_segment_length=0.5,
                    max_iter=2000, max_time=0.5, max_overlaps=10,
                    seed=None,
                    segment_color=None, segment_size=0.5,
                    segment_alpha=1.0, segment_linetype=1,
                    segment_curvature=0.0, segment_ncp=1,
                    arrow=False,
                    direction="both", xlim=None, ylim=None,
                    verbose=False,
                    **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat

    aes_params, geom_params = split_layer_kwargs(kwargs)

    return Layer(
        geom=GeomTextRepel(
            nudge_x=nudge_x, nudge_y=nudge_y,
            force=force, force_pull=force_pull,
            box_padding=box_padding, point_padding=point_padding,
            min_segment_length=min_segment_length,
            max_iter=max_iter, max_time=max_time,
            max_overlaps=max_overlaps,
            seed=seed,
            segment_color=segment_color, segment_size=segment_size,
            segment_alpha=segment_alpha,
            segment_linetype=segment_linetype,
            segment_curvature=segment_curvature,
            segment_ncp=segment_ncp,
            arrow=arrow,
            direction=direction, xlim=xlim, ylim=ylim,
            verbose=verbose,
        ),
        stat=resolve_stat(stat),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
        na_rm=na_rm,
    )


def _schedule_repel(**params):
    """Defer the repel sim to the first draw (when bboxes are real).
    Disconnects after running so the second ``draw_idle`` doesn't loop."""
    ax = params["ax"]
    fig = ax.figure
    state = {"cid": None}

    def _on_draw(event):
        # Disconnect FIRST so the redraw we trigger doesn't re-enter us.
        fig.canvas.mpl_disconnect(state["cid"])
        try:
            _do_repel(renderer=event.renderer, **params)
        finally:
            fig.canvas.draw_idle()

    state["cid"] = fig.canvas.mpl_connect("draw_event", _on_draw)


def _do_repel(*, renderer, ax, text_artists, ha_list, va_list,
              x_anchor, y_anchor, x_init, y_init, colours, sizes,
              force, force_pull, max_iter, max_time, max_overlaps,
              box_padding_lines, point_padding_lines, min_segment_length_lines,
              seed, segment_color, segment_size_mm, segment_alpha,
              segment_linetype, segment_curvature, arrow,
              direction, xlim, ylim, verbose):
    import time

    from matplotlib.patches import FancyArrowPatch

    from .._util import r_color
    from ...plot._util import r_lty

    n = len(text_artists)
    if n == 0:
        return

    # Pixel-space conversions. ``transData`` is calibrated by this point
    # — first draw means scale.apply_to_axis has already run.
    anchors = ax.transData.transform(np.column_stack([x_anchor, y_anchor]))

    # Per-label offset between the user-visible text anchor (where ``set_position``
    # plants the text) and the bbox CENTER (what the simulation tracks). Depends
    # on the per-label ha/va. Without this, hjust/vjust != 0.5 would visibly
    # drift after every position update.
    bboxes0 = [ta.get_window_extent(renderer) for ta in text_artists]
    anchor_to_center = np.zeros((n, 2))
    for i, (bb, ha, va) in enumerate(zip(bboxes0, ha_list, va_list)):
        anchor_px = ax.transData.transform([x_init[i], y_init[i]])
        anchor_to_center[i, 0] = (bb.x0 + bb.width / 2) - anchor_px[0]
        anchor_to_center[i, 1] = (bb.y0 + bb.height / 2) - anchor_px[1]

    # ``pos`` tracks bbox CENTERS in display coords throughout the sim.
    pos = ax.transData.transform(np.column_stack([x_init, y_init])) + anchor_to_center
    initial = pos.copy()

    px_per_pt = ax.figure.dpi / 72.0
    avg_fontpx = float(np.mean(sizes)) * _PT_PER_MM * px_per_pt
    line_px = avg_fontpx * 1.2  # 1 line ≈ font height
    box_pad_px = box_padding_lines * line_px
    point_pad_px = point_padding_lines * line_px
    min_seg_px = min_segment_length_lines * line_px

    # Half-extents per label (from real renderer bboxes), inflated by box pad.
    half_w = np.array([bb.width / 2 + box_pad_px for bb in bboxes0])
    half_h = np.array([bb.height / 2 + box_pad_px for bb in bboxes0])

    dir_mask = np.array([
        1.0 if direction in ("both", "x") else 0.0,
        1.0 if direction in ("both", "y") else 0.0,
    ])

    rng = np.random.default_rng(seed)

    # Constraint walls: explicit xlim/ylim override the panel spines.
    if xlim is not None:
        wx = ax.transData.transform([[float(xlim[0]), 0], [float(xlim[1]), 0]])
        wall_xmin, wall_xmax = float(wx[0, 0]), float(wx[1, 0])
    else:
        wxs = ax.transAxes.transform([[0, 0], [1, 0]])
        wall_xmin, wall_xmax = float(wxs[0, 0]), float(wxs[1, 0])
    if ylim is not None:
        wy = ax.transData.transform([[0, float(ylim[0])], [0, float(ylim[1])]])
        wall_ymin, wall_ymax = float(wy[0, 1]), float(wy[1, 1])
    else:
        wys = ax.transAxes.transform([[0, 0], [0, 1]])
        wall_ymin, wall_ymax = float(wys[0, 1]), float(wys[1, 1])

    t_start = time.perf_counter()
    iters_run = 0

    for it in range(max_iter):
        iters_run = it + 1
        forces = np.zeros_like(pos)
        moved = False

        # Pairwise label-label repulsion (rectangle overlap).
        dx = pos[:, 0:1] - pos[:, 0:1].T              # (n, n)
        dy = pos[:, 1:2] - pos[:, 1:2].T
        ox = (half_w[:, None] + half_w[None, :]) - np.abs(dx)
        oy = (half_h[:, None] + half_h[None, :]) - np.abs(dy)
        overlap = (ox > 0) & (oy > 0)
        np.fill_diagonal(overlap, False)
        if overlap.any():
            moved = True
            # Push along smaller-overlap axis. Add tiny jitter so coincident
            # labels don't get stuck (sign(0)=0).
            jitter_x = rng.uniform(-0.5, 0.5, size=dx.shape)
            jitter_y = rng.uniform(-0.5, 0.5, size=dy.shape)
            sx = np.where(dx != 0, np.sign(dx), np.sign(jitter_x))
            sy = np.where(dy != 0, np.sign(dy), np.sign(jitter_y))
            push_x_pair = np.where((ox <= oy) & overlap, ox * sx * 0.5, 0.0)
            push_y_pair = np.where((oy < ox) & overlap, oy * sy * 0.5, 0.0)
            forces[:, 0] += push_x_pair.sum(axis=1) * force
            forces[:, 1] += push_y_pair.sum(axis=1) * force

        # Label-vs-anchor repulsion (push label off any anchor it covers).
        adx = pos[:, 0:1] - anchors[:, 0:1].T
        ady = pos[:, 1:2] - anchors[:, 1:2].T
        ax_inside = (np.abs(adx) < half_w[:, None] + point_pad_px) & \
                    (np.abs(ady) < half_h[:, None] + point_pad_px)
        if ax_inside.any():
            moved = True
            # Push direction = away from each covered anchor; pick the
            # axis with smaller intrusion (cheaper exit).
            ix = (half_w[:, None] + point_pad_px - np.abs(adx))
            iy = (half_h[:, None] + point_pad_px - np.abs(ady))
            sx = np.where(adx != 0, np.sign(adx), 1.0)
            sy = np.where(ady != 0, np.sign(ady), 1.0)
            use_x = ax_inside & (ix <= iy)
            use_y = ax_inside & (iy < ix)
            forces[:, 0] += np.where(use_x, ix * sx, 0.0).sum(axis=1) * force
            forces[:, 1] += np.where(use_y, iy * sy, 0.0).sum(axis=1) * force

        # Wall forces: keep label boxes inside the constraint box.
        push_left = np.maximum(wall_xmin - (pos[:, 0] - half_w), 0.0)
        push_right = np.maximum((pos[:, 0] + half_w) - wall_xmax, 0.0)
        push_bottom = np.maximum(wall_ymin - (pos[:, 1] - half_h), 0.0)
        push_top = np.maximum((pos[:, 1] + half_h) - wall_ymax, 0.0)
        if (push_left.any() or push_right.any()
                or push_bottom.any() or push_top.any()):
            moved = True
            forces[:, 0] += (push_left - push_right) * force
            forces[:, 1] += (push_bottom - push_top) * force

        # Spring back toward initial (anchor + nudge) position.
        forces += (initial - pos) * 0.01 * force_pull

        forces *= dir_mask

        # Damped position update.
        pos += forces * 0.5

        if not moved and np.max(np.abs(forces)) < 0.5:
            break

        if max_time > 0 and (time.perf_counter() - t_start) > max_time:
            if verbose:
                print(f"geom_*_repel: stopped at iter {iters_run} (max_time {max_time}s exceeded)")
            break

    # Final overlap count (for max_overlaps decision).
    dx_f = pos[:, 0:1] - pos[:, 0:1].T
    dy_f = pos[:, 1:2] - pos[:, 1:2].T
    ox_f = (half_w[:, None] + half_w[None, :]) - np.abs(dx_f)
    oy_f = (half_h[:, None] + half_h[None, :]) - np.abs(dy_f)
    overlap_f = (ox_f > 0) & (oy_f > 0)
    np.fill_diagonal(overlap_f, False)
    n_overlapping = int((overlap_f.any(axis=1)).sum())

    if verbose:
        print(f"geom_*_repel: {iters_run} iters, {n_overlapping} labels still overlapping")

    # If too many labels still overlap, hide the excess (matches ggrepel's
    # max.overlaps behaviour: extra labels are silently dropped).
    hide_mask = np.zeros(n, dtype=bool)
    if n_overlapping > max_overlaps:
        # Hide the most-overlapping labels first.
        overlap_count = overlap_f.sum(axis=1)
        n_to_hide = n_overlapping - max_overlaps
        to_hide = np.argsort(-overlap_count)[:n_to_hide]
        hide_mask[to_hide] = True
        for i in np.where(hide_mask)[0]:
            text_artists[i].set_visible(False)
        if verbose:
            print(f"geom_*_repel: hid {n_to_hide} labels (max_overlaps={max_overlaps})")

    # Update text artist positions (data coords). ``set_position`` plants the
    # *text anchor*, not the bbox center, so subtract back the per-label offset.
    text_anchor_px = pos - anchor_to_center
    new_data = ax.transData.inverted().transform(text_anchor_px)
    for i, ta in enumerate(text_artists):
        if hide_mask[i]:
            continue
        ta.set_position((float(new_data[i, 0]), float(new_data[i, 1])))

    # Connector segments. Length is measured from the visible box edge to
    # the anchor (NOT the bbox-center-to-anchor distance) — that's the
    # actual visible line. With default ``min_segment_length=0.5`` lines
    # and ``box_padding=0.25`` lines, an unobstructed repelled label sits
    # with its anchor only ~0.25 lines outside the visible box edge, so
    # no connector is drawn unless the simulation had to push the label
    # further (overlap, wall, etc.). Matches ggrepel.
    linestyle = r_lty(segment_linetype)
    arrowstyle = "-|>" if arrow else "-"
    connstyle = f"arc3,rad={float(segment_curvature)}"
    seg_lw = segment_size_mm * _PT_PER_MM
    for i in range(n):
        if hide_mask[i]:
            continue
        direction = anchors[i] - pos[i]
        d_center = np.hypot(direction[0], direction[1])
        if d_center < 1e-9:
            continue  # anchor coincident with label center
        visible_hw = half_w[i] - box_pad_px
        visible_hh = half_h[i] - box_pad_px
        # ``t`` = parameter along the center→anchor ray that lands on the
        # visible box edge. ``t >= 1`` means the anchor is inside the box
        # (no segment to draw).
        t_x = visible_hw / abs(direction[0]) if direction[0] != 0 else np.inf
        t_y = visible_hh / abs(direction[1]) if direction[1] != 0 else np.inf
        t = min(t_x, t_y)
        if t >= 1.0:
            continue
        seg_length_px = (1.0 - t) * d_center
        if seg_length_px < min_seg_px:
            continue
        edge_px = (pos[i, 0] + t * direction[0], pos[i, 1] + t * direction[1])
        edge_data = ax.transData.inverted().transform(edge_px)
        seg_color = r_color(segment_color) if segment_color else r_color(colours[i])
        # FancyArrowPatch uniformly handles straight + curved + arrow + linestyle.
        # posA = label-box edge, posB = anchor (so the arrow head, if any, lands
        # on the data point — matches ggrepel's arrow=arrow() default).
        patch = FancyArrowPatch(
            posA=(float(edge_data[0]), float(edge_data[1])),
            posB=(float(x_anchor[i]), float(y_anchor[i])),
            connectionstyle=connstyle,
            arrowstyle=arrowstyle,
            color=seg_color,
            linewidth=seg_lw,
            linestyle=linestyle,
            alpha=segment_alpha,
            mutation_scale=10,  # arrow head size in pt
            shrinkA=0, shrinkB=0,
            zorder=9,
        )
        ax.add_patch(patch)


