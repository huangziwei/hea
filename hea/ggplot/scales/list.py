"""``ScalesList`` — registry of scales by aesthetic.

A single :class:`Scale` may cover multiple aesthetics (e.g. one
``scale_color_*`` covers both ``"colour"`` and ``"fill"`` when the same
mapping drives both). The dict is keyed by aesthetic for fast lookup;
identity is preserved so a shared scale isn't double-counted.
"""

from __future__ import annotations

import copy as _copy

from .scale import Scale


class ScalesList:
    def __init__(self):
        self._by_aes: dict[str, Scale] = {}

    def add(self, scale: Scale) -> None:
        for aes in scale.aesthetics:
            self._by_aes[aes] = scale

    def get(self, aesthetic: str) -> Scale | None:
        return self._by_aes.get(aesthetic)

    def has(self, aesthetic: str) -> bool:
        return aesthetic in self._by_aes

    def get_or_default(self, aesthetic: str, data=None) -> Scale | None:
        """Return the registered scale for ``aesthetic``, auto-creating a
        positional scale for ``"x"`` / ``"y"`` if missing.

        With ``data`` supplied, picks :class:`ScaleOrdinal` for discrete
        dtypes (``Utf8``/``Categorical``/``Enum``/``Boolean``) and
        :class:`ScaleContinuous` otherwise — matches ggplot2's
        ``scale_x_discrete`` auto-default for character columns.
        """
        sc = self._by_aes.get(aesthetic)
        if sc is not None:
            return sc
        if aesthetic in ("x", "y"):
            import polars as pl
            is_discrete = (
                data is not None
                and isinstance(data, pl.Series)
                and data.dtype in (pl.Utf8, pl.Categorical, pl.Enum, pl.Boolean)
            )
            if is_discrete:
                from .ordinal import ScaleOrdinal
                sc = ScaleOrdinal(aesthetics=(aesthetic,))
            else:
                from .continuous import ScaleContinuous
                sc = ScaleContinuous(aesthetics=(aesthetic,))
            self._by_aes[aesthetic] = sc
            return sc
        return None

    def get_or_default_for_data(self, aesthetic: str, data) -> Scale | None:
        """Like :meth:`get_or_default` but inspects the data column to pick
        the right default scale for non-positional aesthetics.

        Picks per-aesthetic default palettes that match ggplot2:

        * ``colour`` / ``fill`` — discrete: ``hue_pal``; continuous: ``gradient_pal``
        * ``size`` — discrete: cycle of small steps; continuous: ``rescale_pal((1, 6))``
        * ``alpha`` — discrete: cycle in [0.1, 1]; continuous: ``alpha_pal((0.1, 1))``
        * ``shape`` — discrete only: ``shape_pal``; raises on numeric input
        * ``linetype`` — discrete only: ``linetype_pal``; raises on numeric input
        """
        sc = self._by_aes.get(aesthetic)
        if sc is not None:
            return sc
        if aesthetic in ("x", "y"):
            return self.get_or_default(aesthetic, data=data)

        import polars as pl

        if not isinstance(data, pl.Series):
            return None
        dtype = data.dtype
        is_discrete = dtype in (pl.Utf8, pl.Categorical, pl.Enum, pl.Boolean)
        is_numeric = dtype.is_numeric()

        if aesthetic in ("colour", "fill"):
            if is_discrete:
                from .discrete import ScaleDiscreteColor

                sc = ScaleDiscreteColor(aesthetics=(aesthetic,))
            elif is_numeric:
                from ._palettes import gradient_pal
                from .color_continuous import ScaleContinuousColor

                sc = ScaleContinuousColor(
                    aesthetics=(aesthetic,), palette=gradient_pal(),
                )
            else:
                return None
            self._by_aes[aesthetic] = sc
            return sc

        if aesthetic == "size":
            from ._palettes import rescale_pal
            from .color_continuous import ScaleContinuousColor
            from .discrete import ScaleDiscreteColor

            if is_numeric:
                sc = ScaleContinuousColor(
                    aesthetics=("size",), palette=rescale_pal((1.0, 6.0)),
                )
            elif is_discrete:
                sc = ScaleDiscreteColor(
                    aesthetics=("size",), palette=rescale_pal((1.0, 6.0)),
                )
            else:
                return None
            self._by_aes[aesthetic] = sc
            return sc

        if aesthetic == "alpha":
            from ._palettes import alpha_pal
            from .color_continuous import ScaleContinuousColor
            from .discrete import ScaleDiscreteColor

            if is_numeric:
                sc = ScaleContinuousColor(
                    aesthetics=("alpha",), palette=alpha_pal((0.1, 1.0)),
                )
            elif is_discrete:
                sc = ScaleDiscreteColor(
                    aesthetics=("alpha",), palette=alpha_pal((0.1, 1.0)),
                )
            else:
                return None
            self._by_aes[aesthetic] = sc
            return sc

        if aesthetic == "shape":
            if is_numeric:
                raise ValueError(
                    "A continuous variable cannot be mapped to `shape`. "
                    "Convert via factor() or use scale_shape_manual()."
                )
            if is_discrete:
                from ._palettes import shape_pal
                from .discrete import ScaleDiscreteColor

                sc = ScaleDiscreteColor(aesthetics=("shape",), palette=shape_pal())
                self._by_aes[aesthetic] = sc
                return sc
            return None

        if aesthetic == "linetype":
            if is_numeric:
                raise ValueError(
                    "A continuous variable cannot be mapped to `linetype`. "
                    "Convert via factor() or use scale_linetype_manual()."
                )
            if is_discrete:
                from ._palettes import linetype_pal
                from .discrete import ScaleDiscreteColor

                sc = ScaleDiscreteColor(
                    aesthetics=("linetype",), palette=linetype_pal(),
                )
                self._by_aes[aesthetic] = sc
                return sc
            return None

        return None

    def copy(self) -> "ScalesList":
        """Independent copy — each ``draw()`` builds fresh scales so repeated
        builds don't accumulate state."""
        new = ScalesList()
        # Preserve sharing: same Scale instance bound to multiple aesthetics
        # ends up shared in the copy too.
        seen: dict[int, Scale] = {}
        for aes, sc in self._by_aes.items():
            if id(sc) in seen:
                new._by_aes[aes] = seen[id(sc)]
            else:
                clone = _copy.copy(sc)
                seen[id(sc)] = clone
                new._by_aes[aes] = clone
        return new

    def items(self):
        return self._by_aes.items()
