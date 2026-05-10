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

    Surface mirrors ggrepel's most-used args; not yet supported:
    ``segment_curvature``, ``segment_ncp``, ``xlim``/``ylim``
    (constraint boxes), ``hjust``/``vjust`` per label.
    """
    nudge_x: float = 0.0
    nudge_y: float = 0.0
    force: float = 1.0
    force_pull: float = 1.0
    box_padding: float = 0.25         # in lines
    point_padding: float = 0.0        # in lines
    min_segment_length: float = 0.5   # in lines
    max_iter: int = 2000
    seed: int | None = None
    segment_color: str | None = None  # default: same as label colour
    segment_size: float = 0.5         # mm
    segment_alpha: float = 1.0
    direction: str = "both"           # "both" | "x" | "y"

    def _make_text_artist(self, ax, x, y, label, *,
                          colour, fill, size, angle):
        """Render this label as text inside a rounded box (geom_label style)."""
        from .._util import r_color
        border_pt = float(self.label_size) * _PT_PER_MM
        return ax.text(
            x, y, str(label),
            color=r_color(colour),
            fontsize=float(size) * _PT_PER_MM,
            rotation=float(angle),
            ha="center", va="center",
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

        # Drop None / empty labels — ggrepel's na_rm equivalent.
        keep = [i for i, lbl in enumerate(labels)
                if lbl is not None and str(lbl) != ""]
        if not keep:
            return

        # Initial positions: anchor + nudge. ha/va fixed at center for
        # repel — we move the *whole label*, not the text-within-box anchor.
        x_init = x[keep] + float(self.nudge_x)
        y_init = y[keep] + float(self.nudge_y)

        text_artists = []
        for k, i in enumerate(keep):
            ta = self._make_text_artist(
                ax, x_init[k], y_init[k], labels[i],
                colour=colour[i], fill=fill[i],
                size=size[i], angle=angle[i],
            )
            # Don't let tight_layout shrink the axes for moved labels.
            ta.set_in_layout(False)
            text_artists.append(ta)

        _schedule_repel(
            ax=ax,
            text_artists=text_artists,
            x_anchor=x[keep], y_anchor=y[keep],
            x_init=x_init, y_init=y_init,
            colours=[colour[i] for i in keep],
            sizes=[float(size[i]) for i in keep],
            force=float(self.force),
            force_pull=float(self.force_pull),
            max_iter=int(self.max_iter),
            box_padding_lines=float(self.box_padding),
            point_padding_lines=float(self.point_padding),
            min_segment_length_lines=float(self.min_segment_length),
            seed=self.seed,
            segment_color=self.segment_color,
            segment_size_mm=float(self.segment_size),
            segment_alpha=float(self.segment_alpha),
            direction=str(self.direction),
        )


def geom_label_repel(mapping=None, data=None, *, stat="identity",
                     position="identity", na_rm=False,
                     label_padding=0.25, label_r=0.15, label_size=0.25,
                     nudge_x=0.0, nudge_y=0.0,
                     force=1.0, force_pull=1.0,
                     box_padding=0.25, point_padding=0.0,
                     min_segment_length=0.5,
                     max_iter=2000, seed=None,
                     segment_color=None, segment_size=0.5,
                     segment_alpha=1.0, direction="both",
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
            max_iter=max_iter, seed=seed,
            segment_color=segment_color, segment_size=segment_size,
            segment_alpha=segment_alpha, direction=direction,
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
    seed: int | None = None
    segment_color: str | None = None
    segment_size: float = 0.5
    segment_alpha: float = 1.0
    direction: str = "both"

    def _make_text_artist(self, ax, x, y, label, *,
                          colour, fill, size, angle):
        """Render this label as bare text — no bounding box."""
        from .._util import r_color
        return ax.text(
            x, y, str(label),
            color=r_color(colour),
            fontsize=float(size) * _PT_PER_MM,
            rotation=float(angle),
            ha="center", va="center",
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
                    max_iter=2000, seed=None,
                    segment_color=None, segment_size=0.5,
                    segment_alpha=1.0, direction="both",
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
            max_iter=max_iter, seed=seed,
            segment_color=segment_color, segment_size=segment_size,
            segment_alpha=segment_alpha, direction=direction,
        ),
        stat=resolve_stat(stat),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
        na_rm=na_rm,
    )


def _schedule_repel(*, ax, text_artists, x_anchor, y_anchor, x_init, y_init,
                    colours, sizes, force, force_pull, max_iter,
                    box_padding_lines, point_padding_lines,
                    min_segment_length_lines, seed,
                    segment_color, segment_size_mm, segment_alpha,
                    direction):
    """Defer the repel sim to the first draw (when bboxes are real).
    Disconnects after running so the second ``draw_idle`` doesn't loop."""
    fig = ax.figure
    state = {"cid": None}

    def _on_draw(event):
        # Disconnect FIRST so the redraw we trigger doesn't re-enter us.
        fig.canvas.mpl_disconnect(state["cid"])
        try:
            _do_repel(
                renderer=event.renderer,
                ax=ax, text_artists=text_artists,
                x_anchor=x_anchor, y_anchor=y_anchor,
                x_init=x_init, y_init=y_init,
                colours=colours, sizes=sizes,
                force=force, force_pull=force_pull, max_iter=max_iter,
                box_padding_lines=box_padding_lines,
                point_padding_lines=point_padding_lines,
                min_segment_length_lines=min_segment_length_lines,
                seed=seed,
                segment_color=segment_color,
                segment_size_mm=segment_size_mm,
                segment_alpha=segment_alpha,
                direction=direction,
            )
        finally:
            fig.canvas.draw_idle()

    state["cid"] = fig.canvas.mpl_connect("draw_event", _on_draw)


def _do_repel(*, renderer, ax, text_artists, x_anchor, y_anchor,
              x_init, y_init, colours, sizes, force, force_pull, max_iter,
              box_padding_lines, point_padding_lines, min_segment_length_lines,
              seed, segment_color, segment_size_mm, segment_alpha, direction):
    from .._util import r_color

    n = len(text_artists)
    if n == 0:
        return

    # Pixel-space conversions. ``transData`` is calibrated by this point
    # — first draw means scale.apply_to_axis has already run.
    anchors = ax.transData.transform(np.column_stack([x_anchor, y_anchor]))
    pos = ax.transData.transform(np.column_stack([x_init, y_init]))
    initial = pos.copy()

    px_per_pt = ax.figure.dpi / 72.0
    avg_fontpx = float(np.mean(sizes)) * _PT_PER_MM * px_per_pt
    line_px = avg_fontpx * 1.2  # 1 line ≈ font height
    box_pad_px = box_padding_lines * line_px
    point_pad_px = point_padding_lines * line_px
    min_seg_px = min_segment_length_lines * line_px

    # Half-extents per label (from real renderer bboxes), inflated by box pad.
    bboxes = [ta.get_window_extent(renderer) for ta in text_artists]
    half_w = np.array([bb.width / 2 + box_pad_px for bb in bboxes])
    half_h = np.array([bb.height / 2 + box_pad_px for bb in bboxes])

    dir_mask = np.array([
        1.0 if direction in ("both", "x") else 0.0,
        1.0 if direction in ("both", "y") else 0.0,
    ])

    rng = np.random.default_rng(seed)

    # Panel walls in display coords — keep labels inside the axes spines
    # (ggrepel's default xlim/ylim = panel range). Without this, labels
    # whose anchor sits near an edge can spill past it.
    wall = ax.transAxes.transform([[0, 0], [1, 1]])
    wall_xmin, wall_ymin = wall[0]
    wall_xmax, wall_ymax = wall[1]

    for _ in range(max_iter):
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

        # Wall forces: keep label boxes inside the panel.
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

    # Update text artist positions (data coords).
    new_data = ax.transData.inverted().transform(pos)
    for i, ta in enumerate(text_artists):
        ta.set_position((float(new_data[i, 0]), float(new_data[i, 1])))

    # Connector segments: from box edge nearest the anchor to the anchor.
    for i, ta in enumerate(text_artists):
        seg_dx = pos[i, 0] - anchors[i, 0]
        seg_dy = pos[i, 1] - anchors[i, 1]
        if np.hypot(seg_dx, seg_dy) < min_seg_px:
            continue
        # Ray from label center toward anchor; exit the rectangle there.
        edge_px = _ray_rect_exit(pos[i], anchors[i] - pos[i],
                                 half_w[i] - box_pad_px,  # box edge, not pad
                                 half_h[i] - box_pad_px)
        edge_data = ax.transData.inverted().transform(edge_px)
        seg_color = r_color(segment_color) if segment_color else r_color(colours[i])
        ax.plot(
            [edge_data[0], x_anchor[i]],
            [edge_data[1], y_anchor[i]],
            color=seg_color,
            linewidth=segment_size_mm * _PT_PER_MM,
            alpha=segment_alpha,
            solid_capstyle="round",
            zorder=9,
        )


def _ray_rect_exit(center, direction, half_w, half_h):
    """Where does a ray from rectangle center exit through the side?
    Used to land the connector on the box edge, not the text center."""
    dx, dy = float(direction[0]), float(direction[1])
    if dx == 0.0 and dy == 0.0:
        return (center[0], center[1])
    tx = half_w / abs(dx) if dx != 0 else np.inf
    ty = half_h / abs(dy) if dy != 0 else np.inf
    t = min(tx, ty)
    return (center[0] + t * dx, center[1] + t * dy)
