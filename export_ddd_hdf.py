'''Export a DDD17 / DDD20 recording to homogeneous format.

The raw DDD HDF recordings save data in a custom (caer) format that is not
friendly to batch processing. This script exports the data into a newly created
HDF5 with a nicer, per-frame layout (aps_frame / dvs_frame / OpenXC channels).

DDD17 (Binas et al. 2017) and DDD20 (Hu et al. 2020) were recorded with the same
DAVIS346B sensor (260x346), use the same caer event packing, and expose the same
OpenXC channel names, so a single export path serves both datasets. The sensor
geometry and the OpenXC field set are selected with ``--dataset {ddd17,ddd20}``
(or an explicit ``--sensor``); the exporter is robust to OpenXC channels that are
absent from a given recording and stamps dataset/sensor provenance on the output.

NOTE: ``export_sequence`` reads only the raw caer HDF5 container. DDD17's
``run1_test`` is distributed as raw ``.aedat`` (AEDAT 2.0) and is decoded by the
sibling ``aedat.py``; the top-level ``export()`` dispatcher here routes ``.aedat``
inputs there and ``.hdf5`` inputs to ``export_sequence``.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch

Experimental viewer for DAVIS + OpenXC data
Author: J. Binas <jbinas@gmail.com>, 2017

DDD17/DDD20 generalization (dataset/sensor parameterization, missing-channel
robustness, provenance) contributed by Roger Booto Tokime, 2026.

This software is released under the
GNU LESSER GENERAL PUBLIC LICENSE Version 3.
'''

from __future__ import print_function
import os
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import h5py
import cv2
import time
import queue as Queue
import multiprocessing as mp
from collections import deque
from interfaces.caer import DVS_SHAPE, unpack_header, unpack_data

CHUNK_SIZE = 128

DISPLAY = False # whether to turn on display. setting DISPLAY=False makes our lives easier in headless servers

# --- DDD17 + DDD20 generalization -------------------------------------------
# Both datasets were recorded with the DAVIS346B (260x346). The presets below
# make the sensor geometry an explicit, checked parameter rather than a buried
# module global; a future DAVIS240C recording is one table entry away (and would
# additionally require interfaces/caer.py:DVS_SHAPE to match — see the loud check
# in export_sequence, since unpack_frame reshapes to that constant).
SENSOR_PRESETS = {
    'davis346b': (260, 346),   # DDD17 + DDD20
    'davis240c': (180, 240),   # reserved for future DAVIS240C recordings
}
DATASET_SENSOR = {
    'ddd17': 'davis346b',
    'ddd20': 'davis346b',
}


def resolve_dvs_shape(dataset=None, sensor=None):
    """Resolve the (H, W) DVS/APS geometry from a dataset name or explicit sensor.

    Precedence: explicit ``sensor`` > ``dataset`` preset > interfaces.caer.DVS_SHAPE.
    Raises ``ValueError`` on an unknown dataset/sensor so a typo fails loudly
    instead of silently exporting mis-shaped frames.
    """
    if sensor is not None:
        key = str(sensor).lower()
        if key not in SENSOR_PRESETS:
            raise ValueError(f"Unknown sensor {sensor!r}; known: {sorted(SENSOR_PRESETS)}")
        return SENSOR_PRESETS[key]
    if dataset is not None:
        key = str(dataset).lower()
        if key not in DATASET_SENSOR:
            raise ValueError(f"Unknown dataset {dataset!r}; known: {sorted(DATASET_SENSOR)}")
        return SENSOR_PRESETS[DATASET_SENSOR[key]]
    return DVS_SHAPE


def _present_tables(input_path, wanted):
    """Intersect the wanted OpenXC/dvs tables with those actually in the recording.

    DDD17 and DDD20 expose the same channel names, but individual recordings may
    omit a channel. Skipping absent channels (with a note) keeps the exporter from
    crashing on a KeyError deep inside the streaming process. 'dvs' is mandatory.
    """
    with h5py.File(input_path, 'r') as _f:
        keys = set(_f.keys())
    present = {k for k in wanted if k in keys}
    if 'dvs' not in present:
        raise ValueError(f"{input_path}: no 'dvs' group found — not a raw DDD caer recording")
    missing = set(wanted) - present
    if missing:
        print(f"[export] note: OpenXC channels absent from recording, skipped: {sorted(missing)}")
    return present

#  exported_h5_path = os.path.join(
#      os.environ["HOME"], "data", "DDD19", "exported.h5")
#  exported_data = h5py.File(exported_h5_path, "w")
#  frame_data = exported_data.create_dataset(
#      name="frame",
#      shape=(0, 260, 346),
#      maxshape=(None, 260, 346),
#      dtype="uint8")
#  frame_time = exported_data.create_dataset(
#      name="frame_ts",
#      shape=(0, 1),
#      maxshape=(None, 1),
#      dtype="float32")
#  event_data = exported_data.create_dataset(
#      name="event",
#      shape=(0, 4),
#      maxshape=(None, 4),
#      dtype="uint32")


VIEW_DATA = {
        'dvs',
        'steering_wheel_angle',
        'engine_speed',
        'accelerator_pedal_position',
        'brake_pedal_status',
        'vehicle_speed',
        }


# this changed in version 3
CV_AA = cv2.LINE_AA if int(cv2.__version__[0]) > 2 else cv2.CV_AA

def raster_evts(data, dvs_shape=None):
    """
    Bin polarity events into a 2D voxel grid:
      data[:,0] = timestamps (µs)
      data[:,1] = y coordinates
      data[:,2] = x coordinates
      data[:,3] = polarity (0 or 1)
    Returns an int16 array of shape ``dvs_shape`` (defaults to DVS_SHAPE) with
    (on − off) counts.
    """
    shape = tuple(dvs_shape) if dvs_shape is not None else DVS_SHAPE
    _histrange = [(0, v) for v in shape]
    pol_on  = data[:,3] == 1
    pol_off = ~pol_on
    img_on, _, _  = np.histogram2d(
        data[pol_on, 2], data[pol_on, 1],
        bins=shape, range=_histrange
    )
    img_off, _, _ = np.histogram2d(
        data[pol_off,2], data[pol_off,1],
        bins=shape, range=_histrange
    )
    return (img_on - img_off).astype(np.int16)


import os
import numpy as np
import h5py
import queue as Queue
from interfaces.caer import DVS_SHAPE, unpack_data
# … other imports …

def export_sequence(
    input_path: str,
    out_path: str = None,
    binsize: float = 0.1,
    export_aps: bool = True,
    export_dvs: bool = True,
    display: bool = False,
    in_memory: bool = False,
    tstart: float = 0.0,
    tstop: float = None,
    dataset: str = None,
    dvs_shape=None,
):
    """
    Three modes:
      * display=True        → live OpenCV viewer, no on-disk/output
      * in_memory=True      → collect into RAM and return arrays
      * neither (default)   → write an HDF5 to out_path and return its path

    ``dataset`` ({'ddd17','ddd20'}) selects the sensor geometry preset; pass an
    explicit ``dvs_shape=(H, W)`` to override. Defaults preserve the historical
    DDD20 behaviour (DAVIS346B = (260, 346)).
    """
    global DISPLAY
    DISPLAY = bool(display)

    # Resolve sensor geometry. DDD17 and DDD20 are both DAVIS346B = (260, 346).
    shape = tuple(dvs_shape) if dvs_shape is not None else tuple(resolve_dvs_shape(dataset))
    # APS frames are decoded by interfaces/caer.py:unpack_frame, which reshapes to
    # the module-level DVS_SHAPE. If the requested geometry disagrees, exported APS
    # frames would be silently mis-shaped (or crash on reshape) — fail loudly.
    if shape != tuple(DVS_SHAPE):
        raise ValueError(
            f"Requested DVS shape {shape} != interfaces.caer.DVS_SHAPE {tuple(DVS_SHAPE)}. "
            "APS frames are decoded via interfaces/caer.py:unpack_frame, which reshapes to "
            "that module-level constant; set DVS_SHAPE/SENSOR there to match before exporting "
            "a different sensor. (DDD17 and DDD20 are both DAVIS346B = (260, 346).)"
        )

    if out_path is None:
        out_path = input_path + ".exported.hdf5"

    # open streams — skip OpenXC channels absent from this particular recording
    present = _present_tables(input_path, VIEW_DATA)
    f_in = HDF5Stream(input_path, present)
    m    = MergedStream(f_in)

    # Store recording start time for making timestamps relative
    recording_start_us = m.tmin  # Recording start in microseconds

    # optional seek
    if tstart > 0:
        m.search(int(m.tmin + tstart * 1e6))

    # compute absolute microsecond cutoff for tstop
    end_us = None
    if tstop is not None:
        end_us = int(m.tmin + tstop * 1e6)

    # viewer if requested
    if display:
        viewer = Viewer(tmin=m.tmin*1e-6, tmax=m.tmax*1e-6, zoom=1.0, rotate180=True)

    # prepare on‐disk HDF5
    if not display and not in_memory:
        dtypes = {k: float for k in present.union({'timestamp'})}
        if export_aps: dtypes['aps_frame'] = (np.uint8, shape)
        if export_dvs: dtypes['dvs_frame'] = (np.int16, shape)

        f_out = h5py.File(out_path, "w")
        # provenance: which dataset/sensor/geometry produced this export
        f_out.attrs['dataset'] = str(dataset) if dataset else 'ddd20'
        f_out.attrs['sensor'] = 'DAVIS346B'
        f_out.attrs['dvs_shape'] = np.asarray(shape, dtype=np.int32)
        f_out.attrs['binsize_s'] = float(binsize)
        f_out.attrs['exporter'] = 'ddd20-utils/export_ddd20_hdf.py (DDD17+DDD20 generalized)'
        for name, dt in dtypes.items():
            if isinstance(dt, tuple):
                f_out.create_dataset(name, shape=(0,) + dt[1],
                                     maxshape=(None,) + dt[1],
                                     dtype=dt[0], chunks=True,
                                     compression="gzip", compression_opts=1)
            else:
                f_out.create_dataset(name, shape=(0,),
                                     maxshape=(None,),
                                     dtype=dt, chunks=True,
                                     compression="gzip", compression_opts=1)

        # Batch writing buffers for performance
        BATCH_SIZE = 100
        buffers = {name: [] for name in dtypes.keys()}

        # Temporary buffers for matching steering to frames
        temp_steering_timestamps = []
        temp_steering_angles = []

        def _flush_buffers():
            """Flush all buffers to disk"""
            for name, buf in buffers.items():
                if len(buf) > 0:
                    ds = f_out[name]
                    old_size = ds.shape[0]
                    new_size = old_size + len(buf)
                    ds.resize(new_size, axis=0)
                    ds[old_size:new_size] = np.array(buf)
                    buf.clear()

        def _append(name, data):
            buffers[name].append(data)
            if len(buffers[name]) >= BATCH_SIZE:
                # Flush this specific buffer
                buf = buffers[name]
                ds = f_out[name]
                old_size = ds.shape[0]
                new_size = old_size + len(buf)
                ds.resize(new_size, axis=0)
                ds[old_size:new_size] = np.array(buf)
                buf.clear()

    # prepare in‐memory buffers
    if in_memory:
        timestamps = []
        aps_frames = []
        dvs_frames = []
        steering_timestamps = []
        steering_angles = []

    half_window = binsize / 2.0
    event_buffer = deque()
    active_frames = deque()

    def _prune_event_buffer(current_time):
        min_time = current_time - half_window
        while event_buffer and event_buffer[0]['ts'][-1] < min_time:
            event_buffer.popleft()

    def _finalize_ready_frames(current_time):
        while active_frames and active_frames[0]['end'] <= current_time:
            frame = active_frames.popleft()
            if export_dvs:
                dvs_frame = np.clip(frame['dvs'], -32768, 32767).astype(np.int16)
            if in_memory:
                timestamps.append(frame['ts'])
                aps_frames.append(frame['aps'])
                if export_dvs:
                    dvs_frames.append(dvs_frame)
            else:
                _append('timestamp', frame['ts'])
                _append('aps_frame', frame['aps'])
                if export_dvs:
                    _append('dvs_frame', dvs_frame)

    # main loop
    while m.has_data:
        try:
            sys_ts, d = m.get()
        except Queue.Empty:
            continue
        if not d:
            continue

        # break when we exceed tstop
        if end_us is not None and sys_ts > end_us:
            break

        # 1) display‐only
        if display:
            viewer.show(d, sys_ts)
            continue

        # 2) in‐memory
        if in_memory:
            etype = d.get('etype', d.get('name'))
            if etype == 'special_event':
                unpack_data(d)
            elif etype == 'steering_wheel_angle':
                # Convert to relative timestamp (seconds from recording start)
                # sys_ts is already in seconds, recording_start_us is in microseconds
                relative_ts = sys_ts - (recording_start_us * 1e-6)
                steering_timestamps.append(relative_ts)
                steering_angles.append(d.get('value', d.get('data')))
            elif etype == 'frame_event' and export_aps:
                _finalize_ready_frames(sys_ts - (recording_start_us * 1e-6))
                unpack_data(d)  # Unpacks frame data
                aps = (d['data'] // 256).astype(np.uint8)
                # Use sys_ts (relative to recording start) for consistency with steering
                relative_ts = sys_ts - (recording_start_us * 1e-6)
                frame = {
                    'ts': relative_ts,
                    'start': relative_ts - half_window,
                    'end': relative_ts + half_window,
                    'aps': aps,
                    'dvs': np.zeros(shape, dtype=np.int32),
                }
                if export_dvs:
                    for pkt in event_buffer:
                        ts = pkt['ts']
                        if ts[0] > relative_ts:
                            break
                        mask = (ts >= frame['start']) & (ts <= relative_ts)
                        if mask.any():
                            frame['dvs'] += raster_evts(pkt['data'][mask], shape)
                active_frames.append(frame)
            elif etype == 'polarity_event' and export_dvs:
                unpack_data(d)
                packet_offset = sys_ts - d['timestamp']
                event_ts = d['data'][:, 0] * 1e-6 + packet_offset - (recording_start_us * 1e-6)
                event_buffer.append({'ts': event_ts, 'data': d['data']})
                _prune_event_buffer(event_ts[-1])
                for frame in active_frames:
                    mask = (event_ts >= frame['start']) & (event_ts <= frame['end'])
                    if mask.any():
                        frame['dvs'] += raster_evts(d['data'][mask], shape)
                _finalize_ready_frames(event_ts[-1])
            continue

        # 3) on‐disk
        etype = d.get('etype', d.get('name'))
        if etype == 'special_event':
            unpack_data(d)
        elif etype == 'steering_wheel_angle':
            # Collect steering data for later matching
            # Convert to relative timestamp (seconds from recording start)
            # sys_ts is already in seconds, recording_start_us is in microseconds
            relative_ts = sys_ts - (recording_start_us * 1e-6)
            temp_steering_timestamps.append(relative_ts)
            temp_steering_angles.append(d.get('value', d.get('data')))
        elif etype in present and etype != 'steering_wheel_angle':
            _append(etype, d.get('value', d.get('data')))
        elif etype == 'frame_event' and export_aps:
            _finalize_ready_frames(sys_ts - (recording_start_us * 1e-6))
            unpack_data(d)  # Unpacks frame data
            aps = (d['data'] // 256).astype(np.uint8)
            # Use sys_ts (relative to recording start) for consistency with steering
            relative_ts = sys_ts - (recording_start_us * 1e-6)
            frame = {
                'ts': relative_ts,
                'start': relative_ts - half_window,
                'end': relative_ts + half_window,
                'aps': aps,
                'dvs': np.zeros(shape, dtype=np.int32),
            }
            if export_dvs:
                for pkt in event_buffer:
                    ts = pkt['ts']
                    if ts[0] > relative_ts:
                        break
                    mask = (ts >= frame['start']) & (ts <= relative_ts)
                    if mask.any():
                        frame['dvs'] += raster_evts(pkt['data'][mask], shape)
            active_frames.append(frame)
        elif etype == 'polarity_event' and export_dvs:
            unpack_data(d)
            packet_offset = sys_ts - d['timestamp']
            event_ts = d['data'][:, 0] * 1e-6 + packet_offset - (recording_start_us * 1e-6)
            event_buffer.append({'ts': event_ts, 'data': d['data']})
            _prune_event_buffer(event_ts[-1])
            for frame in active_frames:
                mask = (event_ts >= frame['start']) & (event_ts <= frame['end'])
                if mask.any():
                    frame['dvs'] += raster_evts(d['data'][mask], shape)
            _finalize_ready_frames(event_ts[-1])

    # teardown & returns
    if display:
        viewer.close()
        return out_path

    if in_memory:
        _finalize_ready_frames(float('inf'))
        # Match steering angles to frame timestamps using nearest-neighbor
        frame_timestamps = np.array(timestamps)
        steering_ts_arr = np.array(steering_timestamps)
        steering_ang_arr = np.array(steering_angles)

        matched_steering = np.zeros(len(frame_timestamps))
        for i, frame_ts in enumerate(frame_timestamps):
            # Find nearest steering sample
            idx = np.argmin(np.abs(steering_ts_arr - frame_ts))
            matched_steering[i] = steering_ang_arr[idx]

        return (
            frame_timestamps,                        # shape [N], sensor timestamps
            np.stack(aps_frames, axis=0),           # shape [N,H,W]
            np.stack(dvs_frames, axis=0),           # shape [N,H,W]
            matched_steering                         # shape [N], matched to frames
        )

    # default on‐disk - flush remaining buffers before closing
    _finalize_ready_frames(float('inf'))
    _flush_buffers()

    # Match steering angles to frame timestamps and write to file
    if len(temp_steering_timestamps) > 0 and 'timestamp' in f_out:
        frame_timestamps = f_out['timestamp'][:]
        steering_ts_arr = np.array(temp_steering_timestamps)
        steering_ang_arr = np.array(temp_steering_angles)

        matched_steering = np.zeros(len(frame_timestamps))
        for i, frame_ts in enumerate(frame_timestamps):
            # Find nearest steering sample
            idx = np.argmin(np.abs(steering_ts_arr - frame_ts))
            matched_steering[i] = steering_ang_arr[idx]

        # Write matched steering angles to file
        if 'steering_wheel_angle' in f_out:
            del f_out['steering_wheel_angle']
        f_out.create_dataset('steering_wheel_angle', data=matched_steering,
                           dtype=float, compression="gzip", compression_opts=1)

    f_out.close()
    return out_path


def export(input_path, **kwargs):
    """Top-level dispatcher: route an input file to the right export path.

    Routing is by magic bytes first, extension second:
      * first bytes ``#!AER-DAT`` or a ``.aedat`` extension -> AEDAT 2.0 decoder
        (:func:`aedat.export_aedat_sequence`); APS + DVS only, no OpenXC steering.
      * ``.hdf5`` / ``.h5`` (or anything else) -> the caer-HDF5
        :func:`export_sequence` (the verified DDD17/DDD20 path, unchanged).

    ``kwargs`` are forwarded to the chosen exporter; pass only the args that
    exporter accepts (e.g. ``binsize``, ``out_path``, ``dataset``, ``tstop``).
    The AEDAT path is imported lazily to avoid an import cycle (aedat.py imports
    ``raster_evts`` from this module).
    """
    magic = b''
    try:
        with open(input_path, 'rb') as _f:
            magic = _f.read(9)
    except OSError:
        pass
    is_aedat = magic.startswith(b'#!AER-DAT') or str(input_path).lower().endswith('.aedat')
    if is_aedat:
        from aedat import export_aedat_sequence
        return export_aedat_sequence(input_path, **kwargs)
    return export_sequence(input_path, **kwargs)


def _flush_q(q):
    """Flush a multiprocessing.Queue."""
    while True:
        try:
            q.get(timeout=1e-3)
        except Queue.Empty:
            if q.empty():
                break


class HDF5Stream(mp.Process):
    def __init__(self, filename, tables, bufsize=64):
        super(HDF5Stream, self).__init__()
        self.f = h5py.File(filename, 'r')
        self.tables = tables
        self.q = {k: mp.Queue(bufsize) for k in self.tables}
        self.run_search = mp.Event()
        self.exit = mp.Event()
        self.done = mp.Event()
        self.skip_to = mp.Value('L', 0)
        self._init_count()
        self._init_time()
        self.daemon = True
        self.start()

    def run(self):
        while self.blocks_rem and not self.exit.is_set():
            blocks_read = 0
            for k in list(self.blocks_rem.keys()):
                if self.q[k].full():
                    time.sleep(1e-6)
                    continue
                i = self.block_offset[k]
                self.q[k].put(self.f[k]['data'][i*CHUNK_SIZE:(i+1)*CHUNK_SIZE])
                self.block_offset[k] += 1
                if self.blocks_rem[k].value:
                    self.blocks_rem[k].value -= 1
                else:
                    self.blocks_rem.pop(k)
                blocks_read += 1
            if not blocks_read:
                time.sleep(1e-6)
            if self.run_search.is_set():
                self._search()
        self.f.close()
        print('closed input file')
        while not self.exit.is_set():
            time.sleep(1e-3)
        # print('[DEBUG] flushing stream queues')
        for k in self.q:
            # print('[DEBUG] flushing', k)
            _flush_q(self.q[k])
            self.q[k].close()
            self.q[k].join_thread()
        # print('[DEBUG] flushed all stream queues')
        self.done.set()
        print('stream done')

    def get(self, k, block=True, timeout=None):
        return self.q[k].get(block, timeout)

    def _init_count(self, offset={}):
        """
        Initialize block offsets, sizes, and remaining blocks using integer division
        so that multiprocessing.Value('L', v) always receives integers.
        """
        # how many complete chunks were skipped per stream
        self.block_offset = {
            k: offset.get(k, 0) // CHUNK_SIZE
            for k in self.tables
        }
        # total number of samples remaining per stream
        self.size = {
            k: len(self.f[k]['data']) - off * CHUNK_SIZE
            for k, off in self.block_offset.items()
        }
        # how many full CHUNK_SIZE blocks remain
        self.blocks = {
            k: sz // CHUNK_SIZE
            for k, sz in self.size.items()
        }
        # wrap each block count in a multiprocessing.Value('L', int)
        self.blocks_rem = {
            k: mp.Value('L', v)
            for k, v in self.blocks.items() if v
        }


    def _init_time(self):
        self.ts_start = {}
        self.ts_stop = {}
        self.ind_stop = {}
        for k in self.tables:
            ts_start = self.f[k]['timestamp'][self.block_offset[k]*CHUNK_SIZE]
            self.ts_start[k] = mp.Value('L', ts_start)
            b = self.block_offset[k] + self.blocks_rem[k].value - 1
            while b > self.block_offset[k] and \
                    self.f[k]['timestamp'][b*CHUNK_SIZE] == 0:
                b -= 1
            print(k, 'final block:', b)
            self.ts_stop[k] = mp.Value(
                'L', self.f[k]['timestamp'][(b + 1) * CHUNK_SIZE - 1])
            self.ind_stop[k] = b

    def init_search(self, t):
        ''' start streaming from given time point '''
        if self.run_search.is_set():
            return
        self.skip_to.value = np.uint64(t)
        self.run_search.set()

    def _search(self):
        t = self.skip_to.value
        offset = {k: self._bsearch_by_timestamp(k, t) for k in self.tables}
        for k in self.tables:
            _flush_q(self.q[k])
        self._init_count(offset)
        # self._init_time()
        self.run_search.clear()

    def _bsearch_by_timestamp(self, k, t):
        '''performs binary search on timestamp, returns closest block index'''
        l, r = 0, self.ind_stop[k]
        print('searching', k, t)
        while True:
            if r - l < 2:
                print('selecting block', l)
                return l * CHUNK_SIZE
            if self.f[k]['timestamp'][(l + (r - l) / 2) * CHUNK_SIZE] > t:
                r = l + (r - l) / 2
            else:
                l += (r - l) / 2


class MergedStream(mp.Process):
    ''' Unpacks and merges data from HDF5 stream '''
    def __init__(self, fbuf, bufsize=256):
        super(MergedStream, self).__init__()
        self.fbuf = fbuf
        self.ts_start = self.fbuf.ts_start
        self.ts_stop = self.fbuf.ts_stop
        self.q = mp.Queue(bufsize)
        self.run_search = mp.Event()
        self.skip_to = mp.Value('L', 0)
        self._init_state()
        self.done = mp.Event()
        self.fetched_all = mp.Event()
        self.exit = mp.Event()
        self.daemon = True
        self.start()

    def run(self):
        while self.blocks_rem and not self.exit.is_set():
            # find next event
            if self.q.full():
                time.sleep(1e-4)
                continue
            next_k = min(self.current_ts, key=self.current_ts.get)
            self.q.put((self.current_ts[next_k], self.current_dat[next_k]))
            self._inc_current(next_k)
            # get new blocks if necessary
            for k in {k for k in self.blocks_rem if self.i[k] == CHUNK_SIZE}:
                self.current_blk[k] = self.fbuf.get(k)
                self.i[k] = 0
                if self.blocks_rem[k]:
                    self.blocks_rem[k] -= 1
                else:
                    self.blocks_rem.pop(k)
                    self.current_ts.pop(k)
            if self.run_search.is_set():
                self._search()
        self.fetched_all.set()
        self.fbuf.exit.set()
        while not self.fbuf.done.is_set():
            time.sleep(1)
            # print('[DEBUG] waiting for stream process')
        while not self.exit.is_set():
            time.sleep(1)
            # print('[DEBUG] waiting for merger process')
        _flush_q(self.q)
        # print('[DEBUG] flushed merger q ->', self.q.qsize())
        self.q.close()
        self.q.join_thread()
        # print('[DEBUG] joined merger q')
        self.done.set()

    def close(self):
        self.exit.set()

    def _init_state(self):
        keys = self.fbuf.blocks_rem.keys()
        self.blocks_rem = {k: self.fbuf.blocks_rem[k].value for k in keys}
        self.current_blk = {k: self.fbuf.get(k) for k in keys}
        self.i = {k: 0 for k in keys}
        self.current_dat = {}
        self.current_ts = {}
        for k in keys:
            self._inc_current(k)

    def _inc_current(self, k):
        ''' get next event of given type and increment row pointer '''
        row = self.current_blk[k][self.i[k]]
        if k == 'dvs':
            ts, d = caer_event_from_row(row)
        else:  # vi event
            ts = row[0] * 1e-6
            d = {'etype': k, 'timestamp': row[0], 'data': row[1]}
        if not ts and k in self.current_ts:
            self.current_ts.pop(k)
            self.blocks_rem.pop(k)
            return False
        self.current_ts[k], self.current_dat[k] = ts, d
        self.i[k] += 1

    def get(self, block=False):
        return self.q.get(block)

    @property
    def has_data(self):
        return not (self.fetched_all.is_set() and self.q.empty())

    @property
    def tmin(self):
        return self.ts_start['dvs'].value

    @property
    def tmax(self):
        return self.ts_stop['dvs'].value

    def search(self, t, block=True):
        if self.run_search.is_set():
            return
        self.skip_to.value = np.uint64(t)
        self.run_search.set()

    def _search(self):
        self.fbuf.init_search(self.skip_to.value)
        while self.fbuf.run_search.is_set():
            time.sleep(1e-6)
        _flush_q(self.q)
        self._init_state()
        self.q.put((0, {'etype': 'timestamp_reset'}))
        self.run_search.clear()


class Interface(object):
    def __init__(self,
                 tmin=0, tmax=0,
                 search_callback=None,
                 update_callback=None,
                 create_callback=None,
                 destroy_callback=None):
        self.tmin, self.tmax = tmin, tmax
        self.search_callback = search_callback
        self.update_callback = update_callback
        self.create_callback = create_callback
        self.destroy_callback = destroy_callback

    def _set_t(self, t):
        self.t_now = int(t - self.tmin)
        if self.update_callback is not None:
            self.update_callback(t)

    def close(self):
        if self.close_callback is not None:
            self.close_callback


class Viewer(Interface):
    ''' Simple visualizer for events '''
    def __init__(self, max_fps=40, zoom=1, rotate180=False, **kwargs):
        super(Viewer, self).__init__(**kwargs)
        self.zoom = zoom
        if DISPLAY:
            cv2.namedWindow('frame')
            # tobi added from https://stackoverflow.com/questions/21810452/
            # cv2-imshow-command-doesnt-work-properly-in-opencv-python/
            # 24172409#24172409
            cv2.startWindowThread()
            cv2.namedWindow('polarity')
            ox = 0
            oy = 0
            cv2.moveWindow('frame', ox, oy)
            cv2.moveWindow('polarity', ox + int(448*self.zoom), oy)
        self.set_fps(max_fps)
        self.pol_img = 0.5 * np.ones(DVS_SHAPE)
        self.t_now = 0
        self.t_pre = {}
        self.count = {}
        self.cache = {}
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.display_info = True
        self.display_color = 0
        self.playback_speed = 1.  # seems to do nothing
        self.rotate180 = rotate180
        # sets contrast for full scale event count for white/black
        self.dvs_contrast = 2
        self.paused = False

    def set_fps(self, max_fps):
        self.min_dt = 1. / max_fps

    def show(self, d, t=None):
        if not DISPLAY:
            return

        # figure out event type
        etype = d.get('etype', d.get('name'))

        # cache steering‐wheel angles as they arrive
        if etype == 'steering_wheel_angle':
            # d['data'] or d['value'] holds the scalar angle in degrees
            self.cache['steering_wheel_angle'] = float(d.get('value', d.get('data')))
            return

        if etype == 'frame_event':
            # unpack the uint16 image
            unpack_data(d)
            frame = (d['data'] // 256).astype(np.uint8)

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # rotate the image 180°
            frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)

            # overlay steering‐wheel angle if we have one
            self._plot_steering_wheel(frame_bgr)

            cv2.imshow('frame', frame_bgr)

        elif etype == 'polarity_event':
            # unpack polarity events into d['data']
            unpack_data(d)
            vox = raster_evts(d['data'])  # int16

            # normalize to 0–255 for display
            lo, hi = vox.min(), vox.max()
            if hi > lo:
                disp = ((vox - lo) * (255.0 / (hi - lo))).astype(np.uint8)
            else:
                disp = np.zeros_like(vox, dtype=np.uint8)

            cv2.imshow('polarity', disp)

        # let OpenCV process its window events
        cv2.waitKey(1)



    def _plot_steering_wheel(self, img):
        if 'steering_wheel_angle' not in self.cache:
            return

        # center and radius
        cx, cy = (173, 130)
        r = 65

        # current angle in degrees
        a = self.cache['steering_wheel_angle']
        a_rad = a/180.0 * np.pi + np.pi/2
        if self.rotate180:
            a_rad = np.pi - a_rad

        # end point of the spoke
        end_x = int(cx + np.cos(a_rad) * r)
        end_y = int(cy - np.sin(a_rad) * r)

        # draw a white circle and a green spoke
        cv2.circle(img, (cx, cy), r, (255,255,255), 1, CV_AA)
        cv2.line( img, (cx, cy), (end_x, end_y), (0,255,0), 2, CV_AA)

        # put the text just below the wheel
        cv2.putText(
            img,
            f"{a:.1f} deg",
            (cx - 30, cy + r + 20),
            self.font, 0.5, (0,0,255), 1, CV_AA
        )



    def _print(self, img, pos, name, unit, autohide=False):
        if name not in self.cache:
            return
        v = self.cache[name]
        if autohide and v == 0:
            return
        cv2.putText(
            img, '%d %s' % (v, unit),
            (pos[0]-40, pos[1]+20), self.font, 0.4,
            self.display_color, 1, CV_AA)

    def _print_string(self, img, pos, string):
         cv2.putText(
            img, '%s' % string,
            (pos[0], pos[1]), self.font, 0.4, self.display_color, 1, CV_AA)

    def _plot_timeline(self, img):
        pos = (50, 10)
        p = int(346 * self.t_now / (self.tmax - self.tmin))
        cv2.line(img, (0, 2), (p, 2), 255, 1, CV_AA)
        cv2.putText(
            img, '%d s' % self.t_now,
            (pos[0]-40, pos[1]+20), self.font, 0.4, self.display_color, 1, CV_AA)

    def close(self):
        cv2.destroyAllWindows()


class Controller(Interface):
    def __init__(self, filename, **kwargs):
        super(Controller, self).__init__(**kwargs)
        global DISPLAY
        self.display = DISPLAY
        if self.display:
            cv2.namedWindow('control')
            cv2.moveWindow('control', 400, 698)
        self.f = h5py.File(filename, 'r')
        self.tmin, self.tmax = self._get_ts()
        self.len = int(self.tmax - self.tmin) + 1
        img = np.zeros((100, self.len))
        self.plot_pixels(img, 'headlamp_status', 0, 10)
        self.plot_line(img, 'steering_wheel_angle', 20, 30)
        self.plot_line(img, 'vehicle_speed', 69, 30)
        self.width = 978
        self.img = cv2.resize(
            img, (self.width, 100), interpolation=cv2.INTER_NEAREST)
        if self.display:
            cv2.setMouseCallback('control', self._set_search)
        self.t_pre = 0
        self.update(0)
        self.f.close()

    def update(self, t):
        self._set_t(t)
        t = int(float(self.width) / self.len * (t - self.tmin))
        if t == self.t_pre:
            return
        self.t_pre = t
        img = self.img.copy()
        img[:, :t+1] = img[:, :t+1] * 0.5 + 0.5
        if self.display:
            cv2.imshow('control', img)
            cv2.waitKey(1)

    def plot_line(self, img, name, offset, height):
        x, y = self.get_xy(name)
        if x is None:
            return
        y -= y.min()
        y = y / y.max() * height
        x = x.clip(0, self.len - 1)
        img[offset+height-y.astype(int), x] = 1

    def plot_pixels(self, img, name, offset=0, height=1):
        x, y = self.get_xy(name)
        if x is None:
            return
        img[offset:offset+height, x] = y

    def _set_search(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        t = self.len * 1e6 * x / float(self.width) + self.tmin * 1e6
        self._search_callback(t)

    def _get_ts(self):
        ts = self.f['dvs']['timestamp']
        tmin = ts[0]
        i = -1
        while ts[i] == 0:
            i -= 1
        tmax = ts[i]
        print('tmin/tmax', tmin, tmax)
        return int(tmin * 1e-6), int(tmax * 1e-6)

    def get_xy(self, name):
        d = self.f[name]['data']
        print('name', name)
        gtz_ids = d[:, 0] > 0
        if not gtz_ids.any():
            return None, 0
        gtz = d[gtz_ids, :]
        return (gtz[:, 0] * 1e-6 - self.tmin).astype(int), gtz[:, 1]


def caer_event_from_row(row):
    '''
    Takes binary dvs data as input,
    returns unpacked event data or False if event type does not exist.
    '''
    sys_ts, head, body = (v.tobytes() for v in row)
    if not sys_ts:
        # rows with 0 timestamp do not contain any data
        return 0, False
    d = unpack_header(head)
    d['dvs_data'] = body
    return int(sys_ts) * 1e-6, unpack_data(d)


def main_cli():
    parser = argparse.ArgumentParser(
        description="Export a raw DDD17/DDD20 caer HDF5 recording to homogeneous HDF5",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument('filename', help="Raw DDD17/DDD20 recording (.hdf5, caer format)")
    parser.add_argument('--dataset', choices=sorted(DATASET_SENSOR), default='ddd20',
                        help="Source dataset; selects the sensor geometry preset "
                             "(ddd17 and ddd20 are both DAVIS346B = 260x346). Default: ddd20.")
    parser.add_argument('--sensor', choices=sorted(SENSOR_PRESETS), default=None,
                        help="Explicit sensor override (takes precedence over --dataset).")
    parser.add_argument('--binsize',    type=float, default=0.1,
                        help="Time bin size in seconds (neg for fixed-event count)")
    parser.add_argument('--export_aps', type=int,   choices=[0,1], default=1,
                        help="1 to include APS frames")
    parser.add_argument('--export_dvs', type=int,   choices=[0,1], default=1,
                        help="1 to include DVS voxel frames")
    parser.add_argument('--out_file',   default=None,
                        help="Path to write exported HDF5 (defaults to filename + '.exported.hdf5')")
    parser.add_argument('--tstart',     type=float, default=0.0,
                        help="Start time in seconds relative to recording tmin")
    parser.add_argument('--tstop',      type=float, default=None,
                        help="Stop time in seconds relative to recording tmin")
    parser.add_argument('--display',    type=int,   choices=[0,1], default=0,
                        help="1 to show OpenCV windows during export (not recommended for headless)")
    args = parser.parse_args()

    # explicit --sensor wins; otherwise the --dataset preset resolves the geometry
    shape = resolve_dvs_shape(dataset=args.dataset, sensor=args.sensor)

    out = export_sequence(
        args.filename,
        out_path=args.out_file,
        binsize=args.binsize,
        export_aps=bool(args.export_aps),
        export_dvs=bool(args.export_dvs),
        display=bool(args.display),
        tstart=args.tstart,
        tstop=args.tstop,
        dataset=args.dataset,
        dvs_shape=shape,
    )
    print(f"[DONE] Exported to {out}")


if __name__ == '__main__':
    main_cli()
