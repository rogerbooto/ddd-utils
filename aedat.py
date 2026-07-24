'''Clean-room AEDAT 2.0 (jAER) reader for the DAVIS346B, integrated with the
DDD17/DDD20 homogeneous HDF5 exporter.

DDD17's ``run1_test`` split is distributed as raw jAER ``.aedat`` (AEDAT 2.0)
rather than the caer-packed HDF5 used by the rest of DDD17/DDD20. This module
decodes those files (APS frames + DVS polarity events) and writes the SAME
per-frame HDF5 schema as ``export_ddd_hdf.export_sequence`` (``aps_frame`` /
``dvs_frame`` / ``timestamp``). It does NOT decode OpenXC / CAN steering: the
``run1_test`` ``.aedat`` carries no CAN payload (the steering/throttle/GPS is in
separate ``.dat`` traces), so no steering dataset is fabricated.

Clean-room notice
-----------------
The AEDAT 2.0 DAVIS address bit-layout below was reconstructed from the public
specifications of two reference implementations and then EMPIRICALLY CONFIRMED
against a real DDD17 recording (see the verified constants). No source code was
copied from either reference:

  * jAER ``eu.seebetter.ini.chips.davis.Davis346`` / ``DavisChip`` (GPL) —
    reference for the DVS/APS/IMU address masks and the CDS (reset-signal)
    frame model. GPL: used as a *specification reference only*, not copied.
  * AedatTools ``ImportAedatDataVersion1or2`` (unlicensed) — cross-reference for
    the AEDAT 1/2 header + address decoding. Unlicensed: reference only.

This file is original work released, like the rest of this fork, under the
GNU LESSER GENERAL PUBLIC LICENSE Version 3.

Author: Roger Booto Tokime, 2026.
'''

from __future__ import print_function

import os
import re
import numpy as np
import h5py

# Reuse the exporter's DVS voxel binning + geometry resolution (single source of
# truth for the (on-off) int16 rasterisation and the sensor preset table).
from export_ddd_hdf import raster_evts, resolve_dvs_shape


# ---------------------------------------------------------------------------
# AEDAT 2.0 DAVIS address bit-layout.
#
# Each event record is 8 bytes, BIG-ENDIAN: (uint32 address, uint32 timestamp_us).
# The address decode below matches BOTH jAER (DavisChip) and AedatTools and was
# confirmed on the real file
# ``Davis346B-2016-12-15...-steering-sync-test.aedat``:
#   * DVS x in [0, 345], y in [0, 259], timestamps strictly monotonic;
#   * one APS readout = 89960 signal + 89960 reset samples = 346*260*2, and the
#     reconstructed reset-signal frame is a real image (car interior).
# ---------------------------------------------------------------------------
APS_IMU_MASK = 0x80000000   # bit31 set => APS or IMU sample; clear => DVS event
IMU_TYPE = 0x80000C00       # APS/IMU bit + bits10,11 set => IMU (not a pixel APS)

Y_MASK = 0x7FC00000
Y_SHIFT = 22                # y = (addr & Y_MASK) >> 22   (9 bits, 0..259 valid)
X_MASK = 0x003FF000
X_SHIFT = 12                # x = (addr & X_MASK) >> 12   (10 bits, 0..345 valid)
POL_MASK = 0x00000800
POL_SHIFT = 11              # polarity = (addr & POL_MASK) >> 11  (empirically 0/1)

# APS-only fields
ADC_MASK = 0x000003FF                 # 10-bit ADC value
READCYCLE_MASK = 0x00000C00           # bits 10,11: readout cycle
READCYCLE_SHIFT = 10                  # 0 == reset read, 1 == signal read
EXTERNAL_INPUT_MASK = 0x00000400      # bit10 external-input flag on DVS type

# DAVIS346B geometry (rows y, cols x)
DAVIS346_SHAPE = (260, 346)

# ---------------------------------------------------------------------------
# Frame orientation.
#
# The DAVIS is mounted UPSIDE-DOWN in the DDD17/DDD20 vehicle, so the raw sensor
# frames are stored "native" (upside-down): bright sky lands on the BOTTOM rows.
# ``export_ddd_hdf.export_sequence`` (the caer-HDF5 path) applies NO rotation, and
# the training/eval pipeline consumes these native frames directly (the 180deg
# flip lives only in VISUALISATION code). For pixel-consistency across
# DDD17-aedat / DDD17-hdf5 / DDD20 we must emit frames in the SAME native
# convention.
#
# EMPIRICALLY VERIFIED (2026): the address-bit decode here places pixel (y, x) at
# the same physical location as jAER's ``unpack_frame`` row-major reshape used by
# the caer path. On real files, the top-quartile vs bottom-quartile APS
# brightness sign matches:
#   * caer DDD17 run2 (daytime, strong gradient): bottom brighter (sign -1)
#   * aedat DDD17 run1_test:                       bottom brighter (sign -1)
# Same sign => already aligned; the default ``orient='native'`` is the IDENTITY.
# ``orient='upright'`` (rot180) is offered only for human-facing visualisation and
# must NOT be used for data that feeds the model.
_ORIENTATIONS = ('native', 'upright')


def _orient_frame(frame, orient):
    """Apply the documented orientation transform (default native == identity)."""
    if orient == 'native':
        return frame
    if orient == 'upright':
        return np.rot90(frame, 2)  # 180deg: undo the upside-down mount, view only
    raise ValueError("orient must be one of %s" % (_ORIENTATIONS,))

# jAER writes this exact sentinel as the final ASCII header line; data follows it.
_ASCII_HEADER_END = b'#End Of ASCII Header'
_EVENT_DTYPE = np.dtype('>u4')  # big-endian uint32; records are pairs of these
_CHUNK_EVENTS = 1 << 21         # ~2M events (16 MB) per streamed block


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def read_aedat_header(path_or_file):
    """Parse the ASCII header of an AEDAT 1/2 file.

    Returns a dict with keys:
      * ``version``     — e.g. ``"2.0"`` (from the ``#!AER-DAT<v>`` magic line)
      * ``aechip``      — the ``# AEChip:`` class string, or ``None``
      * ``data_offset`` — byte offset of the first event record

    Accepts a path (str) or an already-opened binary file object (which is left
    positioned at ``data_offset``).

    Robustness note: an event address may legitimately start with the byte
    ``0x23`` (``'#'``), so a naive "first line not starting with #" scan can be
    fooled into consuming binary data as header. We therefore terminate on the
    explicit jAER ``#End Of ASCII Header`` sentinel when present, and only fall
    back to the first non-``#`` line otherwise.
    """
    opened = False
    f = path_or_file
    if isinstance(path_or_file, (str, bytes, os.PathLike)):
        f = open(path_or_file, 'rb')
        opened = True
    try:
        f.seek(0)
        version = None
        aechip = None
        data_offset = None
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                # Reached EOF without a data section.
                data_offset = f.tell()
                break
            if line.startswith(b'#'):
                stripped = line.strip()
                if version is None:
                    m = re.match(rb'#!AER-DAT([0-9]+\.[0-9]+)', stripped)
                    if m:
                        version = m.group(1).decode('ascii')
                if aechip is None and b'AEChip:' in line:
                    aechip = line.split(b'AEChip:', 1)[1].strip().decode(
                        'ascii', 'replace')
                if stripped == _ASCII_HEADER_END:
                    data_offset = f.tell()
                    break
                continue
            # First line that does not start with '#': data begins here.
            data_offset = pos
            break
        if version is None:
            version = '1.0'  # AEDAT 1.0 files omit the magic line
        return {'version': version, 'aechip': aechip,
                'data_offset': int(data_offset)}
    finally:
        if opened:
            f.close()


# ---------------------------------------------------------------------------
# Raw record streaming (memory-bounded, timestamp unwrapping)
# ---------------------------------------------------------------------------
def iter_raw_records(path, data_offset, chunk_events=_CHUNK_EVENTS):
    """Yield ``(addr, ts)`` blocks of big-endian uint32 records from the stream.

    ``addr`` is uint32; ``ts`` is int64 with 32-bit microsecond wrap unwrapped
    across the whole file so timestamps stay monotonic. The file is read in
    bounded chunks (never the whole 200+ MB at once).
    """
    bytes_per = 8
    wrap_offset = 0
    last_ts = None
    with open(path, 'rb') as f:
        f.seek(data_offset)
        while True:
            raw = f.read(bytes_per * chunk_events)
            n = len(raw) // bytes_per
            if n == 0:
                break
            words = np.frombuffer(raw[:n * bytes_per], dtype=_EVENT_DTYPE)
            words = words.reshape(-1, 2)
            addr = words[:, 0].astype(np.uint32)
            ts = words[:, 1].astype(np.int64)
            # Unwrap 32-bit microsecond timestamp wraps.
            if last_ts is not None and ts.size:
                if ts[0] + wrap_offset < last_ts - (1 << 31):
                    wrap_offset += (1 << 32)
            if ts.size:
                # detect internal wraps within the block
                d = np.diff(ts)
                wraps = np.zeros(ts.size, dtype=np.int64)
                neg = np.where(d < -(1 << 31))[0]
                for i in neg:
                    wraps[i + 1:] += (1 << 32)
                ts = ts + wrap_offset + wraps
                wrap_offset += int(wraps[-1]) if wraps.size else 0
                last_ts = int(ts[-1])
            yield addr, ts
            if n < chunk_events:
                break


# ---------------------------------------------------------------------------
# Address decode
# ---------------------------------------------------------------------------
def classify(addr):
    """Return boolean masks ``(dvs, aps, imu)`` for a uint32 address array."""
    aps_or_imu = (addr & APS_IMU_MASK) != 0
    imu = (addr & IMU_TYPE) == IMU_TYPE
    aps = aps_or_imu & ~imu
    dvs = ~aps_or_imu
    return dvs, aps, imu


def decode_dvs(addr, ts):
    """Decode DVS polarity events to an ``Nx4`` float array ``[ts, x, y, pol]``.

    Column order matches ``interfaces.caer.unpack_events`` so the result feeds
    directly into ``raster_evts`` (which reads x=col1, y=col2, pol=col3).
    External-input marker events (bit10) are dropped — they are trigger pulses,
    not pixels.
    """
    dvs, _, _ = classify(addr)
    a = addr[dvs]
    t = ts[dvs]
    keep = (a & EXTERNAL_INPUT_MASK) == 0
    a = a[keep]
    t = t[keep]
    x = (a & X_MASK) >> X_SHIFT
    y = (a & Y_MASK) >> Y_SHIFT
    pol = (a & POL_MASK) >> POL_SHIFT
    return np.stack([t.astype(np.float64), x.astype(np.float64),
                     y.astype(np.float64), pol.astype(np.float64)], axis=1)


def decode_aps_samples(addr, ts):
    """Decode APS samples. Returns ``(x, y, readcycle, adc, ts)`` int arrays."""
    _, aps, _ = classify(addr)
    a = addr[aps]
    t = ts[aps]
    x = ((a & X_MASK) >> X_SHIFT).astype(np.int32)
    y = ((a & Y_MASK) >> Y_SHIFT).astype(np.int32)
    rc = ((a & READCYCLE_MASK) >> READCYCLE_SHIFT).astype(np.int32)
    adc = (a & ADC_MASK).astype(np.int32)
    return x, y, rc, adc, t


class ApsFrameAssembler:
    """Reassemble DAVIS APS frames from a stream of APS samples.

    Each DAVIS readout emits one *signal* pass (readcycle==1) and one *reset*
    pass (readcycle==0), each covering every pixel once. A new frame begins at a
    reset->signal transition (readcycle 0 -> 1). The per-pixel frame value is the
    correlated-double-sampling result ``reset - signal`` (clamped at 0). Feed
    blocks via :meth:`add`; it yields ``(frame_ts_s, uint8_frame)`` tuples for
    every *complete* frame. A partial leading/trailing frame (coverage below
    half the pixels) is discarded.

    The uint8 mapping mirrors ``export_ddd_hdf``'s ``(data // 256)`` convention:
    the 10-bit CDS value is promoted to the 16-bit range used by the DDD20 caer
    frames (``<< 6``) and then ``// 256`` — i.e. ``cds >> 2`` — so aedat exports
    are directly comparable to the caer-HDF5 exports.
    """

    def __init__(self, shape=DAVIS346_SHAPE, t0_us=0):
        self.H, self.W = shape
        self.npix = self.H * self.W
        self.t0_us = t0_us
        # carry-over of an in-progress frame's trailing samples across blocks
        self._cx = np.empty(0, np.int32)
        self._cy = np.empty(0, np.int32)
        self._crc = np.empty(0, np.int32)
        self._cadc = np.empty(0, np.int32)
        self._cts = np.empty(0, np.int64)

    def add(self, x, y, rc, adc, ts):
        # prepend carry-over
        if self._cx.size:
            x = np.concatenate([self._cx, x])
            y = np.concatenate([self._cy, y])
            rc = np.concatenate([self._crc, rc])
            adc = np.concatenate([self._cadc, adc])
            ts = np.concatenate([self._cts, ts])
        # frame boundaries: reset(0) -> signal(1) transitions
        if rc.size >= 2:
            bnd = np.where((rc[:-1] == 0) & (rc[1:] == 1))[0] + 1
        else:
            bnd = np.empty(0, np.int64)
        starts = np.concatenate([[0], bnd])
        out = []
        for i in range(len(starts)):
            s = starts[i]
            e = starts[i + 1] if i + 1 < len(starts) else len(rc)
            is_last = (i == len(starts) - 1)
            if is_last:
                # keep the trailing (possibly incomplete) segment for next block
                self._cx, self._cy = x[s:e], y[s:e]
                self._crc, self._cadc, self._cts = rc[s:e], adc[s:e], ts[s:e]
                break
            frame = self._build(x[s:e], y[s:e], rc[s:e], adc[s:e], ts[s:e])
            if frame is not None:
                out.append(frame)
        return out

    def flush(self):
        """Emit the final buffered frame if it is complete."""
        out = []
        if self._cx.size:
            frame = self._build(self._cx, self._cy, self._crc,
                                self._cadc, self._cts)
            if frame is not None:
                out.append(frame)
        self._cx = np.empty(0, np.int32)
        return out

    def _build(self, x, y, rc, adc, ts):
        if x.size == 0:
            return None
        reset = np.zeros((self.H, self.W), np.int32)
        signal = np.zeros((self.H, self.W), np.int32)
        seen_r = np.zeros((self.H, self.W), bool)
        seen_s = np.zeros((self.H, self.W), bool)
        m0 = rc == 0
        m1 = rc == 1
        inb = (x >= 0) & (x < self.W) & (y >= 0) & (y < self.H)
        r = m0 & inb
        s = m1 & inb
        reset[y[r], x[r]] = adc[r]
        signal[y[s], x[s]] = adc[s]
        seen_r[y[r], x[r]] = True
        seen_s[y[s], x[s]] = True
        # discard partial frames (e.g. the one straddling file start)
        if seen_r.sum() < self.npix // 2 or seen_s.sum() < self.npix // 2:
            return None
        cds = np.clip(reset - signal, 0, None).astype(np.uint16)
        u8 = ((cds << 6) // 256).astype(np.uint8)   # == cds >> 2 (10-bit -> 8-bit)
        frame_ts_s = float((ts.astype(np.float64).mean() - self.t0_us) * 1e-6)
        return frame_ts_s, u8


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_aedat_sequence(input_path, out_path=None, binsize=0.1,
                          dvs_shape=DAVIS346_SHAPE, dataset='ddd17',
                          tstop=None, orient='native'):
    """Decode a DAVIS346B AEDAT 2.0 file to the homogeneous DDD HDF5 schema.

    Output datasets (identical to ``export_ddd_hdf.export_sequence``):
      * ``aps_frame``  — (N, H, W) uint8   CDS frames (reset - signal)
      * ``dvs_frame``  — (N, H, W) int16   (on - off) event voxel per APS frame
      * ``timestamp``  — (N,)      float   seconds relative to recording start

    Provenance attrs stamped on the file: ``dataset``, ``sensor``, ``dvs_shape``,
    ``binsize_s``, ``source_format``, ``aechip``, ``has_steering`` (always
    ``False`` — the ``.aedat`` carries no CAN/OpenXC steering).

    ``orient`` controls frame orientation: ``'native'`` (default) emits frames in
    the sensor-native, upside-down convention that matches ``export_ddd_hdf`` /
    DDD20 exactly (use this for anything that feeds the model); ``'upright'``
    rot180s both APS and DVS for human-facing visualisation only.

    Only AEDAT 2.0 is implemented. A 3.x/4.x header raises ``NotImplementedError``.
    """
    if orient not in _ORIENTATIONS:
        raise ValueError("orient must be one of %s" % (_ORIENTATIONS,))
    shape = tuple(dvs_shape)
    header = read_aedat_header(input_path)
    version = header['version']
    if not version.startswith('2'):
        raise NotImplementedError(
            "AEDAT %s not supported; use dv-processing" % version)

    if out_path is None:
        out_path = input_path + ".exported.hdf5"

    half = binsize / 2.0
    t0_us = None
    end_us = None  # absolute unwrapped-us cutoff for tstop

    assembler = None
    from collections import deque
    dvs_buf = deque()      # recent DVS packets: dict(ts=sec array, data=Nx4)
    active = deque()       # frames pending DVS finalisation

    f_out = h5py.File(out_path, "w")
    f_out.attrs['dataset'] = str(dataset)
    f_out.attrs['sensor'] = 'DAVIS346B'
    f_out.attrs['dvs_shape'] = np.asarray(shape, dtype=np.int32)
    f_out.attrs['binsize_s'] = float(binsize)
    f_out.attrs['source_format'] = 'aedat2.0'
    f_out.attrs['aechip'] = header['aechip'] or 'unknown'
    f_out.attrs['has_steering'] = False
    f_out.attrs['orientation'] = orient  # 'native' == caer/DDD20 convention
    f_out.attrs['exporter'] = 'ddd-utils/aedat.py (AEDAT 2.0 DAVIS346B decoder)'

    ds_aps = f_out.create_dataset('aps_frame', shape=(0,) + shape,
                                  maxshape=(None,) + shape, dtype=np.uint8,
                                  chunks=True, compression="gzip",
                                  compression_opts=1)
    ds_dvs = f_out.create_dataset('dvs_frame', shape=(0,) + shape,
                                  maxshape=(None,) + shape, dtype=np.int16,
                                  chunks=True, compression="gzip",
                                  compression_opts=1)
    ds_ts = f_out.create_dataset('timestamp', shape=(0,), maxshape=(None,),
                                 dtype=float, chunks=True, compression="gzip",
                                 compression_opts=1)

    buf_aps, buf_dvs, buf_ts = [], [], []
    BATCH = 32

    def _flush():
        if not buf_ts:
            return
        n0 = ds_ts.shape[0]
        n1 = n0 + len(buf_ts)
        for ds, buf in ((ds_aps, buf_aps), (ds_dvs, buf_dvs), (ds_ts, buf_ts)):
            ds.resize(n1, axis=0)
            ds[n0:n1] = np.asarray(buf)
        buf_aps.clear()
        buf_dvs.clear()
        buf_ts.clear()

    def _write_frame(frame_ts, aps_u8, dvs_accum):
        # SAME orientation transform on both so APS/DVS stay mutually consistent.
        buf_aps.append(_orient_frame(aps_u8, orient))
        dvs16 = np.clip(dvs_accum, -32768, 32767).astype(np.int16)
        buf_dvs.append(_orient_frame(dvs16, orient))
        buf_ts.append(frame_ts)
        if len(buf_ts) >= BATCH:
            _flush()

    def _finalize_ready(cur_s):
        while active and active[0]['end'] <= cur_s:
            fr = active.popleft()
            _write_frame(fr['ts'], fr['aps'], fr['dvs'])

    def _prune(cur_s):
        while dvs_buf and dvs_buf[0]['ts'][-1] < cur_s - half:
            dvs_buf.popleft()

    for addr, ts in iter_raw_records(input_path, header['data_offset']):
        if ts.size == 0:
            continue
        if t0_us is None:
            t0_us = int(ts[0])
            assembler = ApsFrameAssembler(shape, t0_us=t0_us)
            if tstop is not None:
                end_us = t0_us + int(tstop * 1e6)
        if end_us is not None and ts[0] > end_us:
            break

        cur_s = float((ts[-1] - t0_us) * 1e-6)

        # --- APS: assemble frames, seed DVS window from already-buffered events
        ax, ay, arc, aadc, ats = decode_aps_samples(addr, ts)
        for frame_ts, aps_u8 in assembler.add(ax, ay, arc, aadc, ats):
            fr = {'ts': frame_ts, 'start': frame_ts - half,
                  'end': frame_ts + half, 'aps': aps_u8,
                  'dvs': np.zeros(shape, np.int32)}
            for pkt in dvs_buf:
                tsec = pkt['ts']
                mask = (tsec >= fr['start']) & (tsec <= fr['end'])
                if mask.any():
                    fr['dvs'] += raster_evts(pkt['data'][mask], shape)
            active.append(fr)

        # --- DVS: buffer this block and bin into open frame windows
        dvs = decode_dvs(addr, ts)
        if dvs.shape[0]:
            tsec = (dvs[:, 0] - t0_us) * 1e-6
            pkt = {'ts': tsec, 'data': dvs}
            dvs_buf.append(pkt)
            for fr in active:
                mask = (tsec >= fr['start']) & (tsec <= fr['end'])
                if mask.any():
                    fr['dvs'] += raster_evts(dvs[mask], shape)

        _prune(cur_s)
        _finalize_ready(cur_s)

    # flush trailing frame + all still-open frames
    if assembler is not None:
        for frame_ts, aps_u8 in assembler.flush():
            fr = {'ts': frame_ts, 'start': frame_ts - half,
                  'end': frame_ts + half, 'aps': aps_u8,
                  'dvs': np.zeros(shape, np.int32)}
            for pkt in dvs_buf:
                tsec = pkt['ts']
                mask = (tsec >= fr['start']) & (tsec <= fr['end'])
                if mask.any():
                    fr['dvs'] += raster_evts(pkt['data'][mask], shape)
            active.append(fr)
    while active:
        fr = active.popleft()
        _write_frame(fr['ts'], fr['aps'], fr['dvs'])
    _flush()
    f_out.close()
    return out_path
