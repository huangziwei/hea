"""Registry of R → hea name mappings.

Split by namespace so additions stay tidy:

- ``verbs`` — dplyr / tidyr verb dispatch table (filter, mutate, …).
- ``functions`` — base R / stats / forcats / stringr / lubridate scalars.
- ``ggplot`` — ggplot2 geoms / scales / coords / facets / themes.

Each table is a flat dict; the translator imports them lazily.
"""
