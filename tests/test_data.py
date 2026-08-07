"""Tests for ``hea.data`` — dataset loader and resolution logic.

The dataset name → package resolution is the main thing to lock down:

* If ``package=`` is omitted, hea searches the merged rdatasets +
  bundled-``datasets/`` index. Unique name → loads it. Ambiguous →
  raises with all candidates so the user can disambiguate. Missing →
  raises with a clear "not found" before any doomed network call.
* If ``package=`` is given but the index already shows the dataset
  lives under different packages, raise immediately rather than
  attempting a 404-bound GitHub download.
"""

from __future__ import annotations

from unittest import mock

import pytest

from hea import data
from hea.io import _dataset_index


def test_data_unique_name_resolves_without_package():
    """Names that exist in exactly one package across rdatasets +
    bundled ``datasets/`` resolve without ``package=``.
    ``PlantGrowth`` is unique (R/datasets only), so this should just
    work."""
    pg = data("PlantGrowth")
    assert pg.height == 30
    assert "weight" in pg.columns and "group" in pg.columns


def test_data_ambiguous_name_raises_with_candidates():
    """When a name shows up in multiple packages, the error must list
    every candidate so the user can pick. ``CO2`` is in both R's
    built-in datasets and Stat2Data."""
    with pytest.raises(ValueError) as exc:
        data("CO2")
    msg = str(exc.value)
    assert "ambiguous" in msg
    # All candidates appear in the error.
    assert "'R'" in msg
    assert "'Stat2Data'" in msg
    # And the suggested fix syntax.
    assert "package=" in msg


def test_data_missing_name_raises_clearly():
    """A name that exists nowhere should raise immediately with a
    helpful message — no network round-trip, no doomed download."""
    with pytest.raises(ValueError, match="not found"):
        data("xyz_definitely_not_a_real_dataset_42")


def test_data_explicit_package_mismatch_raises_before_download():
    """When the user passes a ``package=`` but the local index shows
    the dataset is in *different* package(s), we must raise before
    attempting a GitHub round-trip. Pre-fix this 404'd silently with
    a confusing urllib stack trace.
    """
    # iris is in R (and rdatasets/datasets), not in faraway in our index.
    with pytest.raises(ValueError) as exc:
        data("iris", package="faraway")
    msg = str(exc.value)
    assert "not in package 'faraway'" in msg
    # And the error names the package(s) where it IS available.
    assert "'R'" in msg


def test_data_explicit_package_match_still_works():
    """Sanity: passing ``package=`` for a real (name, package) pair
    still loads the dataset — explicit lookup wasn't broken by the
    auto-resolve change."""
    gala = data("gala", package="faraway")
    assert gala.height == 30
    assert "Species" in gala.columns


def test_dataset_index_merges_rdatasets_and_bundled():
    """The index should include datasets from both sources. Spot-check:
    ``faraway`` is only in our bundled tree (faraway isn't in
    rdatasets), and ``MASS`` items are in rdatasets."""
    idx = _dataset_index()
    # Bundled-only package surfaces.
    assert "gavote" in idx
    assert "faraway" in idx["gavote"]
    # rdatasets package surfaces too.
    assert "Boston" in idx
    assert "MASS" in idx["Boston"]


def test_dataset_index_aliases_datasets_to_R():
    """rdatasets calls R's built-in package ``datasets``; hea calls it
    ``R`` everywhere (matches our ``datasets/R/`` directory). The
    index must use the ``R`` alias, not the raw ``datasets`` label.
    """
    idx = _dataset_index()
    pkgs = idx.get("iris", [])
    assert "R" in pkgs
    assert "datasets" not in pkgs


def test_missing_data_extra_says_what_to_install():
    """``rdatasets``/``pyarrow`` live behind ``hea[data]``, so every failure that
    a missing extra can cause has to name the extra.

    They are 177 MB of the 205 a dependent package would inherit and they serve
    one feature, which is why they moved out of the hard dependencies. The cost
    is that ``data("iris")`` stops working on a bare install — acceptable only
    if the error says so instead of surfacing a bare 404 or polars' pyarrow
    message.
    """
    from hea import io

    assert "hea[data]" in io._DATA_EXTRA_HINT

    # every branch of data() that a missing extra can reach appends the hint
    with mock.patch.object(io, "_have_rdatasets", return_value=False):
        with pytest.raises(ValueError, match=r"hea\[data\]"):
            io.data("definitely-not-a-dataset-name")
        with pytest.raises(ValueError, match=r"hea\[data\]"):
            io.data("iris", package="not-a-package-either")

    # ... and it is silent when the extra is present, so the message does not
    # become noise on a normal install
    with mock.patch.object(io, "_have_rdatasets", return_value=True):
        with pytest.raises(ValueError) as ei:
            io.data("definitely-not-a-dataset-name")
        assert "hea[data]" not in str(ei.value)


def test_from_pandas_translates_the_pyarrow_error():
    """polars' "pyarrow is required" is accurate but says nothing about hea."""
    from hea import io

    class _Boom:
        pass

    with mock.patch.object(
        io.pl, "from_pandas", side_effect=ImportError("pyarrow is required for ...")
    ):
        with pytest.raises(ImportError, match=r"hea\[data\]") as ei:
            io._from_pandas(_Boom())
        assert "pyarrow is required" in str(ei.value)  # upstream's text is kept
