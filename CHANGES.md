# CHANGES — maintained fork

This is a maintained fork of [SensorsINI/ddd20-utils](https://github.com/SensorsINI/ddd20-utils)
(J. Binas, Y. Hu, D. Neil, S.-C. Liu, T. Delbruck), released — like the upstream — under the
**GNU LGPL v3**. Upstream copyright and per-file authorship notices are retained. This file
records the significant changes this fork makes, as required by the LGPL.

## Fork modifications (rogerbooto)

- **Python 3 port.** `queue`/`Queue` compatibility, an `interfaces.oxc` import guard, and
  related fixes so the viewer and exporter run under Python 3.

- **`export_ddd20_hdf.py` rewrite.** Per-frame homogeneous HDF5 export
  (`aps_frame` / `dvs_frame` / OpenXC channels), batched HDF5 writes, an in-memory mode,
  and steering-angle-to-frame nearest-neighbour matching (timestamp synchronization).

- **DDD17 + DDD20 generalization (2026).** A single export path now serves both datasets,
  which were recorded with the same DAVIS346B sensor (260×346), the same caer event packing,
  and the same OpenXC channel names:
  - `--dataset {ddd17,ddd20}` (or an explicit `--sensor`) selects the sensor geometry from a
    small preset table. Geometry is now an explicit, checked parameter rather than a buried
    module global; a mismatch against `interfaces/caer.py:DVS_SHAPE` (which `unpack_frame`
    reshapes to) **fails loudly** instead of silently exporting mis-shaped frames.
  - The exporter is **robust to OpenXC channels absent** from a given recording (skipped with
    a note rather than crashing mid-stream).
  - The output HDF5 is stamped with `dataset` / `sensor` / `dvs_shape` / `binsize_s`
    **provenance** attributes.
  - Backward-compatible: defaults preserve the historical DDD20 behaviour.
  - Verified end-to-end on real DDD17 recordings (run2–5, raw caer HDF5): `aps_frame` and
    `dvs_frame` at (260, 346), steering matched to frames, all OpenXC channels exported.

  **Not covered:** DDD17 `run1_test` is distributed as raw `.aedat`, which requires a separate
  aedat decoder (not part of this exporter) before it can be exported.
