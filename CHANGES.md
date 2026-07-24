# CHANGES — maintained fork

This is a maintained fork of [SensorsINI/ddd20-utils](https://github.com/SensorsINI/ddd20-utils)
(J. Binas, Y. Hu, D. Neil, S.-C. Liu, T. Delbruck), released — like the upstream — under the
**GNU LGPL v3**. Upstream copyright and per-file authorship notices are retained. This file
records the significant changes this fork makes, as required by the LGPL.

## Fork modifications (rogerbooto)

- **Python 3 port.** `queue`/`Queue` compatibility, an `interfaces.oxc` import guard, and
  related fixes so the viewer and exporter run under Python 3.

- **`export_ddd_hdf.py` rewrite.** Per-frame homogeneous HDF5 export
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

- **AEDAT 2.0 input support (2026).** A new, clean-room `aedat.py` decodes raw jAER
  **AEDAT 2.0** recordings (the format DDD17 `run1_test` ships in) and exports them to the
  **same homogeneous HDF5 schema** as the caer path (`aps_frame` (N,260,346) uint8,
  `dvs_frame` (N,260,346) int16, `timestamp` (N,) float).
  - **Sensor:** DAVIS346B (260×346). Decodes APS frames (correlated double sampling,
    `reset − signal`) and DVS polarity events; IMU and external-input marker events are
    excluded. Address bit-layout (`apsOrImu=0x80000000`, `x=(addr&0x003FF000)>>12`,
    `y=(addr&0x7FC00000)>>22`, `pol` bit 11, `adc=addr&0x3FF`, readcycle bits 10–11) was
    reconstructed clean-room from the **jAER `Davis346`/`DavisChip`** (GPL) and
    **AedatTools `ImportAedatDataVersion1or2`** (unlicensed) specifications — **reference
    only, no source copied** — and then **empirically confirmed** against a real DDD17
    `run1_test` file (DVS coords in-range, APS frames reconstruct to real images).
  - **Header parsing** terminates on jAER's `#End Of ASCII Header` sentinel, so an event
    whose address byte happens to be `0x23` (`#`) is not mistaken for a comment line.
  - **Orientation:** frames are emitted **sensor-native** (upside-down, matching
    `export_ddd_hdf` / DDD20 pixel-for-pixel — verified: aedat and caer daytime frames are
    both bottom-brighter). An explicit `orient=` param (`'native'` default, `'upright'`
    rot180 for visualisation only) applies identically to APS and DVS.
  - **Version scope:** AEDAT **2.0 only**. A 3.x/4.x header raises `NotImplementedError`
    (use `dv-processing`).
  - **No steering/OpenXC:** the `run1_test` `.aedat` carries no CAN payload (steering /
    throttle / GPS live in separate `.dat` traces), so **no steering dataset is fabricated**;
    the output is stamped `has_steering=False`.
  - A top-level `export()` dispatcher in `export_ddd_hdf.py` routes by magic bytes
    (`#!AER-DAT`) / `.aedat` extension to the aedat path and `.hdf5` to the existing caer
    path. The verified `export_sequence` body is unchanged.
  - Tested by `test_aedat.py` (synthetic header/DVS/APS/export + orientation, plus
    drive-gated real-file checks).
