# Data: licences, citation, and what may be redistributed

## Source

Open Cities AI Challenge Dataset, published by GFDRR Labs (World Bank) and
hosted openly on Source Cooperative. No account, token or credential is needed.

- Catalogue: https://source.coop/open-cities
- DOI: https://doi.org/10.34911/rdnt.f94cxb

`src/fetch_data.py` walks the dataset's STAC catalogue and records the licence
of every file it downloads into `data/raw/manifest.json`, alongside each file's
size and SHA-256.

## Two licences, not one

This is the detail most easily got wrong, because the collection-level metadata
says only "various":

| Component | Licence | Implication |
| --- | --- | --- |
| Drone imagery (`*.tif`) | **CC BY 4.0** | Redistribution and derivatives permitted with attribution. |
| Building labels (`*.geojson`) | **ODbL-1.0** | Derived from OpenStreetMap. Share-alike applies to derived *databases*. |

Consequences for this repository:

- **No data is committed to git.** `data/` is in `.gitignore`. Anyone
  reproducing the work fetches it from the source, which also means they get
  the current version rather than a stale snapshot.
- **Trained weights and prediction rasters are outputs, not the database.** They
  are not committed either, but that is a size decision, not a licence one.
- **If you publish a derived footprint database** — corrected, extended or
  merged polygons — ODbL share-alike attaches to it. A published *map* or a set
  of accuracy metrics does not trigger it.

## Attribution

Imagery: © contributors, Open Cities AI Challenge, CC BY 4.0.
Labels: © OpenStreetMap contributors, ODbL.

Citation:

> GFDRR Labs (2020). *Open Cities AI Challenge Dataset*, Version 1.0.
> https://doi.org/10.34911/rdnt.f94cxb

## Scenes used

Six tier-1 Dar es Salaam scenes: `a017f9`, `b15fce`, `353093`, `f883a0`,
`42f235`, `0a4c40`. Scene ids, acquisition dates and bounding boxes are read
from the STAC items at download time and written to the manifest, so the record
of what was actually used comes from the data rather than from this file.
