# Data contract

The analysis uses monthly drought and climate observations for **Illinois, USA**. The original integrated climate workbook is not redistributed here because its source-by-source licensing and attribution have not yet been consolidated. Place an authorized workbook at:

```text
data/raw/data.xlsx
```

Required columns:

| Group | Columns |
|---|---|
| Time | `Time` (monthly date) |
| Drought targets | `SPI_1M`, `SPI_3M`, `SPI_6M`, `SPI_12M`, `SPEI_1M`, `SPEI_3M`, `SPEI_6M`, `SPEI_12M` |
| Macroclimate | `NAO`, `PDO`, `PNA`, `NINO34`, `ONI`, `SOLAR`, `WP` |
| Local climate | `Precipitation_1M`, `Precipitation_3M`, `Precipitation_6M`, `Precipitation_12M`, `Temperature_1M`, `Temperature_3M`, `Temperature_6M`, `Temperature_12M` |

The four JSON files in `metadata/` define the exact columns used by each accumulation-window task.

For a functional smoke test that does not reproduce scientific results, run:

```bash
python scripts/generate_demo_data.py
```

The generated values are synthetic and must never be interpreted as climate findings.
