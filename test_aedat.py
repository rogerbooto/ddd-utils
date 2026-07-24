'''Tests for the clean-room AEDAT 2.0 DAVIS346B reader (aedat.py).

Run:  python -m pytest test_aedat.py -v

Synthetic tests build AEDAT 2.0 bytes with known events and assert exact decode.
The one real-file test is skipped unless the DDD17 external drive is mounted.
'''

import io
import os
import struct
import numpy as np
import pytest

import aedat
from aedat import (
    read_aedat_header, classify, decode_dvs, decode_aps_samples,
    ApsFrameAssembler, export_aedat_sequence,
    APS_IMU_MASK, X_SHIFT, Y_SHIFT, POL_SHIFT, READCYCLE_SHIFT,
    DAVIS346_SHAPE,
)

REAL_FILE = ("/media/rtokime/Seagate Portable Drive/PHD_Umoncton/datasets/"
             "DDD17/run1_test/"
             "Davis346B-2016-12-15T12-14-57+0100-00INX006-0-steering-sync-test.aedat")


# --------------------------------------------------------------------------
# helpers to build synthetic AEDAT 2.0 bytes
# --------------------------------------------------------------------------
def _hdr(version=b'2.0', aechip=b'eu.seebetter.ini.chips.davis.Davis346B'):
    lines = [
        b'#!AER-DAT' + version,
        b'# This is a raw AE data file - do not edit',
        b'# Data format is int32 address, int32 timestamp (8 bytes total)',
        b'# AEChip: ' + aechip,
        b'#End Of ASCII Header',
    ]
    return b'\r\n'.join(lines) + b'\r\n'


def _dvs_addr(x, y, pol):
    return (x << X_SHIFT) | (y << Y_SHIFT) | (pol << POL_SHIFT)


def _aps_addr(x, y, readcycle, adc):
    return APS_IMU_MASK | (x << X_SHIFT) | (y << Y_SHIFT) \
        | (readcycle << READCYCLE_SHIFT) | (adc & 0x3FF)


def _pack(records):
    """records: list of (addr, ts) -> big-endian 8-byte-per-event bytes."""
    out = bytearray()
    for addr, ts in records:
        out += struct.pack('>II', addr & 0xFFFFFFFF, ts & 0xFFFFFFFF)
    return bytes(out)


# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------
def test_header_parse_version_and_offset():
    hdr = _hdr()
    body = _pack([(_dvs_addr(10, 20, 1), 12345)])
    blob = hdr + body
    h = read_aedat_header(io.BytesIO(blob))
    assert h['version'] == '2.0'
    assert h['aechip'] == 'eu.seebetter.ini.chips.davis.Davis346B'
    assert h['data_offset'] == len(hdr)


def test_header_not_fooled_by_event_starting_with_hash():
    # An event whose address high byte is 0x23 ('#') must NOT be read as header.
    hdr = _hdr()
    # address 0x23xxxxxx: pick a DVS address then force top byte to 0x23
    addr = (_dvs_addr(5, 6, 0) & 0x00FFFFFF) | 0x23000000
    body = _pack([(addr, 999), (_dvs_addr(1, 2, 1), 1000)])
    h = read_aedat_header(io.BytesIO(hdr + body))
    assert h['data_offset'] == len(hdr)


def test_header_version_gate_rejects_v3():
    hdr = _hdr(version=b'3.1')
    with pytest.raises(NotImplementedError):
        # write a tiny file and run the exporter's version gate
        p = _write_tmp(hdr + _pack([(_dvs_addr(0, 0, 0), 1)]))
        try:
            export_aedat_sequence(p, out_path=p + '.out.hdf5')
        finally:
            os.remove(p)


def _write_tmp(blob):
    import tempfile
    fd, p = tempfile.mkstemp(suffix='.aedat')
    with os.fdopen(fd, 'wb') as f:
        f.write(blob)
    return p


# --------------------------------------------------------------------------
# DVS decode (known events)
# --------------------------------------------------------------------------
def test_dvs_decode_exact():
    events = [(10, 20, 1), (0, 0, 0), (345, 259, 1), (100, 50, 0)]
    records = [(_dvs_addr(x, y, p), 1000 + i * 10)
               for i, (x, y, p) in enumerate(events)]
    addr = np.array([r[0] for r in records], dtype=np.uint32)
    ts = np.array([r[1] for r in records], dtype=np.int64)

    dvs, aps, imu = classify(addr)
    assert dvs.all() and not aps.any() and not imu.any()

    out = decode_dvs(addr, ts)  # columns [ts, x, y, pol]
    assert out.shape == (4, 4)
    for i, (x, y, p) in enumerate(events):
        assert out[i, 1] == x
        assert out[i, 2] == y
        assert out[i, 3] == p
        assert out[i, 0] == ts[i]


def test_dvs_feeds_raster_evts_orientation():
    # A single ON event at (x=300, y=100) must land at raster[y, x].
    from export_ddd_hdf import raster_evts
    rec = [(_dvs_addr(300, 100, 1), 5)]
    addr = np.array([rec[0][0]], dtype=np.uint32)
    ts = np.array([rec[0][1]], dtype=np.int64)
    out = decode_dvs(addr, ts)
    img = raster_evts(out, DAVIS346_SHAPE)
    assert img.shape == DAVIS346_SHAPE
    assert img[100, 300] == 1
    assert img.sum() == 1


# --------------------------------------------------------------------------
# APS decode + frame reconstruction (known reset/signal)
# --------------------------------------------------------------------------
def test_aps_sample_decode_exact():
    recs = [(_aps_addr(12, 34, 0, 700), 1), (_aps_addr(12, 34, 1, 200), 2)]
    addr = np.array([r[0] for r in recs], dtype=np.uint32)
    ts = np.array([r[1] for r in recs], dtype=np.int64)
    dvs, aps, imu = classify(addr)
    assert aps.all() and not dvs.any() and not imu.any()
    x, y, rc, adc, tt = decode_aps_samples(addr, ts)
    assert list(x) == [12, 12]
    assert list(y) == [34, 34]
    assert list(rc) == [0, 1]
    assert list(adc) == [700, 200]


def test_aps_frame_reconstruction_shape_dtype_and_values():
    H, W = DAVIS346_SHAPE
    reset_val, signal_val = 800, 300     # CDS = 500 (10-bit) -> uint8 = 500>>2 = 125
    # Build ONE full frame: signal pass (rc=1) then reset pass (rc=0), all pixels.
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    xs = xs.ravel().astype(np.uint32)
    ys = ys.ravel().astype(np.uint32)
    recs = []
    ts = 0
    for x, y in zip(xs, ys):          # signal pass
        recs.append((_aps_addr(int(x), int(y), 1, signal_val), ts)); ts += 1
    for x, y in zip(xs, ys):          # reset pass
        recs.append((_aps_addr(int(x), int(y), 0, reset_val), ts)); ts += 1
    # A trailing reset->signal transition is what closes the frame; append a
    # marker of the next frame's signal pass so the boundary is detected, then
    # rely on flush() for completeness of the built frame.
    addr = np.array([r[0] for r in recs], dtype=np.uint32)
    tarr = np.array([r[1] for r in recs], dtype=np.int64)
    x, y, rc, adc, tt = decode_aps_samples(addr, tarr)

    asm = ApsFrameAssembler(DAVIS346_SHAPE, t0_us=0)
    frames = asm.add(x, y, rc, adc, tt)
    frames += asm.flush()
    assert len(frames) == 1
    fts, frame = frames[0]
    assert frame.shape == (H, W)
    assert frame.dtype == np.uint8
    expected = ((np.uint16(reset_val - signal_val) << 6) // 256)
    assert frame.min() == frame.max() == expected  # uniform frame
    assert expected == 125


def test_aps_partial_frame_discarded():
    # Fewer than half the pixels -> not a real frame.
    recs = [(_aps_addr(i, 0, 0, 700), i) for i in range(10)]
    recs += [(_aps_addr(i, 0, 1, 200), 100 + i) for i in range(10)]
    addr = np.array([r[0] for r in recs], dtype=np.uint32)
    ts = np.array([r[1] for r in recs], dtype=np.int64)
    x, y, rc, adc, tt = decode_aps_samples(addr, ts)
    asm = ApsFrameAssembler(DAVIS346_SHAPE, t0_us=0)
    assert asm.add(x, y, rc, adc, tt) == []
    assert asm.flush() == []


def test_imu_samples_excluded_from_aps_and_dvs():
    from aedat import IMU_TYPE
    imu_addr = np.array([IMU_TYPE | 0x1234], dtype=np.uint32)
    ts = np.array([1], dtype=np.int64)
    dvs, aps, imu = classify(imu_addr)
    assert imu.all() and not aps.any() and not dvs.any()


# --------------------------------------------------------------------------
# end-to-end synthetic export
# --------------------------------------------------------------------------
def test_export_synthetic_end_to_end(tmp_path):
    import h5py
    H, W = DAVIS346_SHAPE
    recs = []
    ts = 0
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    xs = xs.ravel(); ys = ys.ravel()
    # frame 1: signal then reset
    for x, y in zip(xs, ys):
        recs.append((_aps_addr(int(x), int(y), 1, 300), ts)); ts += 1
    for x, y in zip(xs, ys):
        recs.append((_aps_addr(int(x), int(y), 0, 800), ts)); ts += 1
    # a couple of DVS events near the frame time
    recs.append((_dvs_addr(50, 50, 1), ts)); ts += 1
    recs.append((_dvs_addr(60, 60, 0), ts)); ts += 1
    # frame 2 (closes frame 1 via reset->signal boundary), then flush closes it
    for x, y in zip(xs, ys):
        recs.append((_aps_addr(int(x), int(y), 1, 310), ts)); ts += 1
    for x, y in zip(xs, ys):
        recs.append((_aps_addr(int(x), int(y), 0, 790), ts)); ts += 1

    blob = _hdr() + _pack(recs)
    p = tmp_path / "syn.aedat"
    p.write_bytes(blob)
    out = export_aedat_sequence(str(p), out_path=str(tmp_path / "syn.hdf5"),
                                binsize=1.0)
    with h5py.File(out, 'r') as f:
        assert f['aps_frame'].shape[1:] == (H, W)
        assert f['aps_frame'].dtype == np.uint8
        assert f['dvs_frame'].dtype == np.int16
        assert f['timestamp'].shape[0] == f['aps_frame'].shape[0]
        assert f['aps_frame'].shape[0] >= 2
        assert f.attrs['source_format'] == 'aedat2.0'
        assert f.attrs['sensor'] == 'DAVIS346B'
        assert bool(f.attrs['has_steering']) is False


# --------------------------------------------------------------------------
# orientation (native == caer/DDD20 convention)
# --------------------------------------------------------------------------
def test_orient_native_is_identity_and_upright_is_rot180():
    from aedat import _orient_frame
    img = np.arange(260 * 346, dtype=np.uint8).reshape(260, 346)
    assert np.array_equal(_orient_frame(img, 'native'), img)         # identity
    assert np.array_equal(_orient_frame(img, 'upright'), np.rot90(img, 2))
    with pytest.raises(ValueError):
        _orient_frame(img, 'sideways')


def test_orient_applies_equally_to_aps_and_dvs(tmp_path):
    # Build a frame + one DVS event, export native vs upright, and assert the
    # DVS event moves EXACTLY the same way the APS frame does (mutual consistency).
    import h5py
    H, W = DAVIS346_SHAPE
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    xs = xs.ravel(); ys = ys.ravel()
    recs = []
    ts = 0
    for x, y in zip(xs, ys):
        recs.append((_aps_addr(int(x), int(y), 1, 300), ts)); ts += 1
    for x, y in zip(xs, ys):
        recs.append((_aps_addr(int(x), int(y), 0, 800), ts)); ts += 1
    recs.append((_dvs_addr(40, 30, 1), ts)); ts += 1          # ON event at (x=40,y=30)
    for x, y in zip(xs, ys):
        recs.append((_aps_addr(int(x), int(y), 1, 300), ts)); ts += 1
    for x, y in zip(xs, ys):
        recs.append((_aps_addr(int(x), int(y), 0, 800), ts)); ts += 1
    blob = _hdr() + _pack(recs)
    p = tmp_path / "o.aedat"; p.write_bytes(blob)

    outs = {}
    for orient in ('native', 'upright'):
        o = export_aedat_sequence(str(p), out_path=str(tmp_path / (orient + ".hdf5")),
                                  binsize=1.0, orient=orient)
        with h5py.File(o, 'r') as f:
            outs[orient] = (f['aps_frame'][0].copy(), f['dvs_frame'][0].copy(),
                            str(f.attrs['orientation']))
    aps_n, dvs_n, _ = outs['native']
    aps_u, dvs_u, tag = outs['upright']
    assert tag == 'upright'
    assert np.array_equal(aps_u, np.rot90(aps_n, 2))
    assert np.array_equal(dvs_u, np.rot90(dvs_n, 2))
    # native DVS event sits at [y=30, x=40]; upright at rot180 location
    assert dvs_n[30, 40] == 1
    assert dvs_u[H - 1 - 30, W - 1 - 40] == 1


@pytest.mark.skipif(not os.path.exists(REAL_FILE),
                    reason="DDD17 run1_test .aedat drive not mounted")
def test_real_file_orientation_matches_caer_native_sign():
    # Native aedat APS must have the SAME top-vs-bottom brightness sign as the
    # caer/DDD20 convention (bottom brighter, i.e. sign(top-bottom) < 0), because
    # the DAVIS is mounted upside-down and frames are stored sensor-native.
    h = read_aedat_header(REAL_FILE)
    with open(REAL_FILE, 'rb') as f:
        f.seek(h['data_offset'])
        buf = f.read(8 * 1_500_000)
    arr = np.frombuffer(buf, dtype='>u4').reshape(-1, 2)
    addr = arr[:, 0].astype(np.uint32); ts = arr[:, 1].astype(np.int64)
    x, y, rc, adc, tt = decode_aps_samples(addr, ts)
    asm = ApsFrameAssembler(DAVIS346_SHAPE, t0_us=int(ts[0]))
    frames = asm.add(x, y, rc, adc, tt)
    _, frame = frames[0]  # native
    q = frame.shape[0] // 4
    top = float(frame[:q].mean()); bot = float(frame[-q:].mean())
    assert bot > top, "aedat native APS should be bottom-brighter (caer convention)"


# --------------------------------------------------------------------------
# real file (skipped if drive absent)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(REAL_FILE),
                    reason="DDD17 run1_test .aedat drive not mounted")
def test_real_file_header_and_first_frame():
    h = read_aedat_header(REAL_FILE)
    assert h['version'] == '2.0'
    assert 'Davis346B' in (h['aechip'] or '')
    # decode first ~1.5M events, assert DVS coords in-range + APS frame plausible
    with open(REAL_FILE, 'rb') as f:
        f.seek(h['data_offset'])
        buf = f.read(8 * 1_500_000)
    arr = np.frombuffer(buf, dtype='>u4').reshape(-1, 2)
    addr = arr[:, 0].astype(np.uint32)
    ts = arr[:, 1].astype(np.int64)
    assert (np.diff(ts) >= 0).mean() > 0.99
    out = decode_dvs(addr, ts)
    assert out[:, 1].max() <= 345 and out[:, 2].max() <= 259
    x, y, rc, adc, tt = decode_aps_samples(addr, ts)
    asm = ApsFrameAssembler(DAVIS346_SHAPE, t0_us=int(ts[0]))
    frames = asm.add(x, y, rc, adc, tt)
    assert len(frames) >= 1
    _, frame = frames[0]
    assert frame.shape == DAVIS346_SHAPE
    assert frame.dtype == np.uint8
    assert frame.max() > frame.min()  # not a blank image
