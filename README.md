# hea: R-style Linear Models without *R* (but with Python)

## Install

```bash
pip install hea
```

for core functionalities.

or if you want to load the example datasets:

```bash
pip install "hea[data]"
```

## Usages

### Data transformation and visualization

`hea` supports mostly tidyverse compatible api (can be chained with the classic `+` or with the method-based `.` in the fluent API style) for data manipulation:

```python 
from hea import data

penguins = data("penguins", package="palmerpenguins")
penguins.glimpse()
(
    penguins.ggplot(
        x="flipper_length_mm", y="body_mass_g", color="species", shape="species"
    )
    .geom_point()
    .geom_smooth(method="lm")
)
```

We use Polars DataFrame under the hood. It's basically `tidypoars` + `plotnine`, but in a more integrated manner. More examples can be found in [examples/books/r4ds](./examples/books/r4ds/)

### Linear model(s)

Currently `lm` and `gam` are relatively mature, while `glm` and `gmm` (generalized mixed models, the `lmer` and `glmer` equivalents) are relatively rudimental:

```python
from hea import data
from hea.models import lm

gala = data("gala", package="faraway")
m = lm("Species ~ Area + Adjacent + Elevation + Nearest + Scruz", gala)
m.summary()
```

```python
from hea import data
from hea.models import gam
from hea.family import Gamma

trees = data("trees", "mgcv")

ct1 = gam("Volume ~ s(Height) + s(Girth)", family=Gamma(link="log"), data=trees)
ct1.summary()
ct1.plot_smooth(partial_residuals=True)
```

See more examples in [examples/books/lmwr](./examples/books/lmwr), [examples/books/gam](./examples/books/gam.ipynb) and more.