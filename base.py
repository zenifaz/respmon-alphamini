import os
import cv2
import time
import copy
import csv
import logging
import peakutils
import numpy as np
import pyqtgraph as pg
from tqdm import tqdm
from collections import deque
from pyqtgraph.Qt import QtWidgets
from scipy.signal import butter, lfilter
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe import Image, ImageFormat
from tools import reduce_bounding_box, Benchmarker
from transforms import uint8_to_float, float_to_uint8, eulerian_magnification_bandpass, butter_lowpass_filter


# ── Breath Logger ──────────────────────────────────────────────────────────────
class RelativeBreathLogger:
    def __init__(self, filename="respiratory_session_data.csv", parent=None):
        self.filename           = filename
        self.parent             = parent
        self.start_time         = time.time()
        self.last_logged_inhale = -1.0

        with open(self.filename, mode='w', newline='') as f:
            csv.writer(f).writerow(["Elapsed Time (s)", "Phase", "Duration (s)"])

    def log_from_peaks(self, peak_times, filtered_data, peak_indices):
        if len(peak_times) < 2:
            return
        i        = len(peak_times) - 2
        t_inhale = round(float(peak_times[i]), 3)
        t_next   = round(float(peak_times[i + 1]), 3)

        if t_inhale <= self.last_logged_inhale:
            return

        t_exhale        = round((t_inhale + t_next) / 2, 3)
        inhale_duration = round(t_exhale - t_inhale, 3)
        exhale_duration = round(t_next - t_exhale, 3)

        self._write("Inhale", t_inhale, inhale_duration)
        self._write("Exhale", t_exhale, exhale_duration)
        self.last_logged_inhale = t_inhale

    def _write(self, phase, elapsed, duration):
        with open(self.filename, mode='a', newline='') as f:
            csv.writer(f).writerow([elapsed, phase, duration])
        logging.info(f"[BREATH] {phase:7s}  elapsed={elapsed:.3f}s  duration={duration:.3f}s")

    def process_frame_value(self, current_value):
        pass


# ── Post-session Analyzer ──────────────────────────────────────────────────────
class RespiratoryAnalyzer:
    def __init__(self, filename="respiratory_session_data.csv"):
        self.filename = filename
        self.data     = self._load()

    def _load(self):
        data = {"inhale_durations": [], "exhale_durations": [],
                "timestamps": [], "phases": []}
        try:
            with open(self.filename, newline='') as f:
                reader = csv.DictReader(f)
                prev   = None
                for row in reader:
                    phase = row["Phase"]
                    if phase == prev:
                        continue
                    prev = phase
                    data["timestamps"].append(float(row["Elapsed Time (s)"]))
                    data["phases"].append(phase)
                    dur = float(row["Duration (s)"])
                    if phase == "Inhale":
                        data["inhale_durations"].append(dur)
                    else:
                        data["exhale_durations"].append(dur)
        except FileNotFoundError:
            logging.warning(f"CSV not found: {self.filename}")
        return data

    def _stats(self, durations):
        vals = [d for d in durations if d > 0]
        if not vals:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        return {"count": len(vals),
                "mean":  round(float(np.mean(vals)), 3),
                "min":   round(float(np.min(vals)),  3),
                "max":   round(float(np.max(vals)),  3),
                "std":   round(float(np.std(vals)),  3)}

    def get_respiratory_rate(self):
        if len(self.data["timestamps"]) < 2:
            return 0.0
        total  = self.data["timestamps"][-1] - self.data["timestamps"][0]
        cycles = len(self.data["phases"]) // 2
        return round((cycles / total) * 60, 2) if total > 0 else 0.0

    def _ie_ratio(self):
        i = self._stats(self.data["inhale_durations"])
        e = self._stats(self.data["exhale_durations"])
        if i["mean"] == 0 or e["mean"] == 0:
            return 0.0
        return round(i["mean"] / e["mean"], 3)

    def print_report(self):
        ts = self.data["timestamps"]
        s  = {
            "respiratory_rate_bpm":   self.get_respiratory_rate(),
            "total_breaths":          len(self.data["phases"]) // 2,
            "total_duration_seconds": round(ts[-1] - ts[0], 2) if len(ts) >= 2 else 0,
            "inhale":  self._stats(self.data["inhale_durations"]),
            "exhale":  self._stats(self.data["exhale_durations"]),
            "ie_ratio": self._ie_ratio(),
        }
        print("\n" + "=" * 60)
        print("RESPIRATORY ANALYSIS REPORT")
        print("=" * 60)
        print(f"Respiratory Rate:        {s['respiratory_rate_bpm']} BPM")
        print(f"Total Breaths:           {s['total_breaths']}")
        print(f"Session Duration:        {s['total_duration_seconds']} seconds")
        print(f"Inhale-to-Exhale Ratio:  {s['ie_ratio']}:1")
        print()
        print("INHALE STATISTICS (seconds):")
        for k, v in s['inhale'].items():
            print(f"  {k.capitalize():8s}: {v}")
        print()
        print("EXHALE STATISTICS (seconds):")
        for k, v in s['exhale'].items():
            print(f"  {k.capitalize():8s}: {v}")
        print("=" * 60 + "\n")


# ── Main Monitor ───────────────────────────────────────────────────────────────
class RespiratoryMonitor:

    def __init__(self, capture_target=0, save_calibration_image=False,
                 visualize='pyqtgraph', fig_size=None, fps_limit=10,
                 error_reset_delay=10.0, save_all_data=True,
                 motion_extraction_method='average'):

        assert isinstance(fps_limit, (int, float)) and fps_limit > 0
        assert isinstance(save_calibration_image, bool)
        assert visualize in ('pyqtgraph', None)
        assert motion_extraction_method in ('average', 'flow')

        self.benchmarker              = Benchmarker()
        self.error_reset_delay        = error_reset_delay
        self.save_all_data            = save_all_data
        self.fig_size                 = fig_size
        self.save_calibration_image   = save_calibration_image
        self.capture_target           = capture_target
        self.visualize                = visualize
        self.motion_extraction_method = motion_extraction_method
        self.logger                   = RelativeBreathLogger(parent=self)
        self._running                 = True

        # ── MediaPipe PoseLandmarker ──────────────────────────────
        try:
            import mediapipe as mp
            options = PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path='pose_landmarker.task'),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            self.pose = PoseLandmarker.create_from_options(options)
            print("[+] MediaPipe PoseLandmarker loaded.")
        except Exception as e:
            print(f"[-] MediaPipe failed: {e}")
            self.pose = None

        # ── Camera ───────────────────────────────────────────────
        self.cap = cv2.VideoCapture(capture_target, cv2.CAP_DSHOW)
        time.sleep(1.5)
        self.fps    = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.maximum_bounding_box_area        = np.inf
        self.calibration_buffer_target_length = 128
        self.freq_min             = 0.1
        self.freq_max             = 1.0
        self.temporal_threshold   = 0.7
        self.threshold            = 0.08
        self.measure_buffer_length         = 128
        self.confidence_interval           = 0.95
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3,
                                   minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS |
                                        cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.gaussian_cutoff               = 10.0
        self.filter_order                  = 3
        self.peak_minimum_sample_distance  = 0
        self.measure_initialization_length = 12

        if self.fps == 0:
            self.fps = np.nan
        self.fps_limit = fps_limit

        self.x = self.y = self.w = self.h = None
        self.disable_error_detection = False
        self.calibration_buffer_idx  = 0

        self.calibration_buffer = np.zeros(
            (self.calibration_buffer_target_length, self.width, self.height),
            dtype=np.float32)

        self.all_data       = []
        self.data           = deque()
        self.t              = deque()
        self.freq           = deque()
        self.confidence     = deque()
        self.num_peaks      = deque()
        self.num_peaks_mean = deque()
        self.motion_data    = deque()
        self.filtered_data  = []
        self.peak_indices   = []
        self.peak_times     = []

        self.current_frame          = None
        self.raw_bgr                = None
        self.raw_upright_gray       = None
        self.cropped_image          = None
        self.previous_cropped_image = None
        self.display_frame          = None
        self.motion_key_points      = None
        self.video_out              = None
        self.error_message          = None

        self.buffers = [self.data, self.confidence, self.t, self.freq,
                        self.num_peaks, self.num_peaks_mean, self.motion_data]

        if visualize == 'pyqtgraph':
            self.ui = self._init_ui()

        self.state                    = 'initialize'
        self.calibration_start_time   = np.nan
        self.loop_start_time          = np.nan
        self.reset_start_time         = np.nan
        self.calibration_progress_bar = tqdm(total=self.calibration_buffer_target_length)

    # ── Camera ────────────────────────────────────────────────────
    def next_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False
        self.raw_bgr          = frame.copy()
        self.raw_upright_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        gray    = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        return uint8_to_float(gray)

    # ── MediaPipe chest ROI ───────────────────────────────────────
    def _get_chest_roi_mediapipe(self):
        if self.raw_bgr is None or self.pose is None:
            return None
        try:
            rgb    = cv2.cvtColor(self.raw_bgr, cv2.COLOR_BGR2RGB)
            uh, uw = rgb.shape[:2]   # upright: uh=480, uw=640
            result = self.pose.detect(Image(image_format=ImageFormat.SRGB, data=rgb))

            if not result.pose_landmarks:
                return None

            lm = result.pose_landmarks[0]
            ls = lm[11];  rs = lm[12]   # shoulders
            lh = lm[23];  rh = lm[24]   # hips

            # Upright pixel coords
            # Chest box: horizontally between shoulders, vertically from just below shoulders to hips
            left   = int(min(ls.x, rs.x) * uw)
            right  = int(max(ls.x, rs.x) * uw)
            top    = int(((ls.y + rs.y) / 2) * uh)          # shoulder line
            bottom = int(((lh.y + rh.y) / 2) * uh)          # hip line

            # Push top down 20% of torso height to exclude neck/collarbone
            torso_h = bottom - top
            top     = top + int(torso_h * 0.20)

            # Add horizontal margin
            margin = int((right - left) * 0.1)
            left   = max(0, left - margin)
            right  = min(uw, right + margin)

            if right - left < 10 or bottom - top < 10:
                return None

            # Translate upright (left, top, right, bottom) → rotated 90° CW
            # In rotated frame: rot_w = uh, rot_h = uw
            # A point (x, y) in upright → (uh - y, x) in rotated
            # So the box:
            #   rot_x = uh - bottom   (right edge of upright box becomes top of rotated)
            #   rot_y = left
            #   rot_w = bottom - top  (upright vertical → rotated horizontal)
            #   rot_h = right - left  (upright horizontal → rotated vertical)
            rot_x = max(0, uh - bottom)
            rot_y = max(0, left)
            rot_w = max(10, bottom - top)
            rot_h = max(10, right - left)

            logging.info(f"MediaPipe chest ROI (rotated): x={rot_x} y={rot_y} w={rot_w} h={rot_h}")
            return rot_x, rot_y, rot_w, rot_h

        except Exception as e:
            logging.debug(f"MediaPipe ROI error: {e}")
            return None

    # ── UI ────────────────────────────────────────────────────────
    def _init_ui(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = pg.GraphicsLayoutWidget(show=True, title="Respiratory Monitor")

        def on_key(event):
            if event.key() in (pg.QtCore.Qt.Key.Key_Escape, pg.QtCore.Qt.Key.Key_Q):
                print("\n[EXIT] Closing…")
                self._running = False
                QtWidgets.QApplication.quit()
        win.keyPressEvent = on_key

        win.resize(*(self.fig_size or (1500, 900)))
        layout = pg.GraphicsLayout()
        win.addItem(layout)
        pg.setConfigOptions(antialias=True)

        lp = layout.addPlot(title="Raw Signal")
        lp.showGrid(x=True, y=True)
        lp.enableAutoRange('xy', False)
        raw_signal  = lp.plot(pen='y')
        peak_plot   = lp.plot(pen=None, symbolBrush=(255, 0, 0), symbolPen=None)
        top_ci      = lp.plot(pen='w')
        bot_ci      = lp.plot(pen='w')
        fill_ci     = pg.FillBetweenItem(top_ci, bot_ci, (255, 0, 0, 100))
        lp.addItem(fill_ci)
        fitted_plot = lp.plot(pen='g')

        img_view      = layout.addViewBox()
        img_view.setAspectLocked(True)
        capture_image = pg.ImageItem(border='w')
        img_view.addItem(capture_image)

        rp             = layout.addPlot(title="Frequency Plot (bpm)")
        rp.showGrid(x=True, y=True)
        rp.enableAutoRange('xy', False)
        frequency_plot = rp.plot()

        bpm_text = pg.TextItem(text='??? BPM', anchor=(-0.1, 1.2),
                               color=(255, 255, 255, 255),
                               border=(0, 0, 0, 255), fill=(0, 0, 0, 127))
        font = pg.QtGui.QFont()
        font.setBold(True)
        font.setPointSize(24)
        bpm_text.setFont(font)
        img_view.addItem(bpm_text)
        bpm_text.setPos(0, 0)

        return {"raw_signal": raw_signal, "capture_image": capture_image,
                "frequency_plot": frequency_plot, "peak_plot": peak_plot,
                "bpm_text": bpm_text,
                "top_confidence_interval": top_ci,
                "bottom_confidence_interval": bot_ci,
                "fill_confidence_interval": fill_ci,
                "fitted_plot": fitted_plot,
                "window": win, "plots": [lp, rp], "application": app}

    def set_window_title(self, title):
        self.ui["window"].setWindowTitle(title)

    def set_image(self, img):
        self.ui["capture_image"].setImage(img)

    def set_plot_autoscale(self, enabled, axes='xy'):
        for p in self.ui["plots"]:
            p.enableAutoRange(axes, enabled)

    def set_plot_x_range(self, lo, hi):
        if np.isnan(lo) or np.isnan(hi) or lo == hi:
            return
        for p in self.ui["plots"]:
            p.setXRange(lo, hi, padding=0)

    def update_ui(self):
        if self.visualize != 'pyqtgraph':
            return
        if self.state == "calibration":
            self.set_window_title(
                f'Calibrating… {self.calibration_buffer_idx}'
                f'/{self.calibration_buffer_target_length}')
            if self.current_frame is not None:
                self.set_image(float_to_uint8(self.current_frame))
        elif self.state == "measure":
            if self.cropped_image is None:
                self.set_plot_autoscale(True)
                return
            self.display_frame = float_to_uint8(self.cropped_image)
            if self.motion_extraction_method == 'flow' and self.motion_key_points is not None:
                mask = np.zeros_like(self.display_frame)
                for pt in self.motion_key_points:
                    a, b = pt.ravel()
                    mask = cv2.circle(mask, (int(a), int(b)), 2, 255, -1)
                self.display_frame = cv2.add(self.display_frame, mask)
            if len(self.peak_times) > 0:
                self.ui["peak_plot"].setData(
                    self.peak_times,
                    np.take(self.filtered_data, self.peak_indices))
            self.set_window_title('Measuring' + '.' * (len(self.filtered_data) % 4))
            if len(self.filtered_data) >= 2 and len(self.t) >= 2:
                self.set_plot_x_range(min(self.t), max(self.t))
                self.ui["raw_signal"].setData(list(self.t), self.filtered_data)
            self.set_image(self.display_frame)
            if len(self.freq) >= 2 and len(self.t) >= 2:
                self.ui["frequency_plot"].setData(
                    list(np.array(self.t)[-len(self.freq):]), list(self.freq))
                self.ui["bpm_text"].setText(f'{self.freq[-1]:.1f} BPM')
        elif self.state == "error":
            self.ui["bpm_text"].setText('??? BPM')
            self.set_window_title(
                f'Error – recalibrating in '
                f'{self.error_reset_delay - (time.time() - self.reset_start_time):.1f}s')
        QtWidgets.QApplication.processEvents()

    # ── Signal processing (original Respmon) ──────────────────────
    def find_peaks(self):
        width   = self.peak_minimum_sample_distance
        indices = peakutils.indexes(self.filtered_data, min_dist=width)
        final_idxs = []
        fits       = []
        for idx in indices:
            w = width
            if idx - width < 0:
                w = idx
            if idx + w > len(self.t):
                w = len(self.t) - idx
            ti    = np.array(self.t)[idx - w: idx + w]
            datai = np.array(self.filtered_data)[idx - w: idx + w]
            try:
                params = peakutils.gaussian_fit(ti, datai, center_only=False)
                y      = [peakutils.gaussian(x, *params) for x in ti]
                ssr    = np.sum(np.power(np.subtract(y, datai), 2.0))
                sst    = np.sum(np.power(np.subtract(y, datai), 2.0))
                fits.append(1 - (ssr / sst))
                if params[2] < self.gaussian_cutoff:
                    final_idxs.append(idx)
            except RuntimeError:
                pass
        return final_idxs, fits

    def measure(self):
        self.filtered_data = np.array(
            butter_lowpass_filter(self.data, self.freq_max * 0.5,
                                  self.fps, self.filter_order))
        self.peak_indices, fits = self.find_peaks()
        self.peak_times = np.take(self.t, self.peak_indices)
        diffs = [b - a for a, b in zip(self.peak_times, self.peak_times[1:])]
        if len(diffs) > 0:
            self.freq.append(60.0 / np.mean(diffs))
        self.logger.log_from_peaks(self.peak_times, self.filtered_data, self.peak_indices)

    def extract_motion(self):
        if self.motion_extraction_method == "average":
            return np.average(self.cropped_image)

        if self.previous_cropped_image is None:
            self.previous_cropped_image = float_to_uint8(self.cropped_image.copy())
            self.motion_key_points = cv2.goodFeaturesToTrack(
                self.previous_cropped_image, mask=None, **self.feature_params)
            if self.motion_key_points is None or len(self.motion_key_points) < 1:
                self.trigger_error("No motion key points found.")
            return 0.0

        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_cropped_image,
            float_to_uint8(self.cropped_image),
            self.motion_key_points, None, **self.lk_params)
        if p1 is None:
            return np.nan

        good_new = p1[st == 1]
        good_old = self.motion_key_points[st == 1]
        self.previous_cropped_image = float_to_uint8(self.cropped_image.copy())
        self.motion_key_points = good_new.reshape(-1, 1, 2)

        if len(good_new) == 0 or len(good_old) == 0:
            return np.nan

        raw = list(np.mean(good_old - good_new, axis=0))
        self.motion_data.append(raw)

        if len(self.motion_data) >= 2:
            x, y     = np.transpose(self.motion_data)
            coords   = np.vstack([x, y])
            cov_mat  = np.cov(coords)
            eig_vals, eig_vecs = np.linalg.eig(cov_mat)
            sort_idx = np.argsort(eig_vals)[::-1]
            evec1, _ = eig_vecs[:, sort_idx]
            return np.array(self.motion_data).dot(evec1)[-1]
        return 0.0

    # ── Calibration ───────────────────────────────────────────────
    def initialize(self):
        self.calibration_start_time = time.time()
        self.calibration_buffer_idx = 0

    def detect_fps(self):
        if self.fps == 0 or self.fps is np.nan:
            self.fps = (self.calibration_buffer_target_length /
                        (time.time() - self.calibration_start_time))
        if self.fps > self.fps_limit:
            self.fps = self.fps_limit
        logging.info(f"FPS locked to {self.fps:.1f}")

    def find_chest_roi(self):
        """
        Try MediaPipe first (reliable, height-invariant).
        Fall back to Eulerian only if MediaPipe fails.
        """
        roi = self._get_chest_roi_mediapipe()
        if roi:
            logging.info(f"MediaPipe chest ROI: x={roi[0]} y={roi[1]} "
                         f"w={roi[2]} h={roi[3]}")
            return roi

        logging.info("MediaPipe failed — falling back to Eulerian.")
        location = self.locate(
            self.calibration_buffer[:self.calibration_buffer_idx],
            self.fps,
            freq_min=self.freq_min, freq_max=self.freq_max,
            amplification=1000,
            temporal_threshold=self.temporal_threshold,
            threshold=int(np.round(self.threshold * 255)),
            save_calibration_image=self.save_calibration_image)

        if location:
            x, y, w, h = location
            # Force into chest zone (30-75% of rotated frame height)
            frame_h   = self.calibration_buffer.shape[1]
            chest_min = int(frame_h * 0.30)
            chest_max = int(frame_h * 0.75)
            if y < chest_min or y > chest_max:
                y = chest_min
                h = chest_max - chest_min
            return x, y, w, h

        return None

    def trigger_error(self, msg=""):
        self.state            = 'error'
        self.error_message    = msg
        self.reset_start_time = time.time()
        logging.warning(f"Error: {msg}")

    def detect_errors(self):
        return self.data[-1] is np.nan

    def reset(self):
        self.state = 'initialize'
        for b in self.buffers:
            b.clear()
        self.ui["raw_signal"].clear()
        self.ui["frequency_plot"].clear()
        self.ui["peak_plot"].clear()
        self.ui["bpm_text"].setText("??? BPM")
        self.ui["top_confidence_interval"].clear()
        self.ui["bottom_confidence_interval"].clear()
        self.ui["fitted_plot"].clear()
        self.filtered_data          = []
        self.peak_indices           = []
        self.peak_times             = []
        self.calibration_buffer_idx = 0
        self.previous_cropped_image = None
        self.motion_key_points      = None
        if self.video_out is not None:
            self.video_out.release()
            self.video_out = None

    def sync_to_fps(self):
        fps   = self.fps if not np.isnan(self.fps) else self.fps_limit
        sleep = (1.0 / fps) - (time.time() - self.loop_start_time)
        if sleep > 0:
            time.sleep(sleep)

    def skip_calibration(self, x, y, w, h):
        self.x = x; self.y = y; self.w = w; self.h = h
        self.peak_minimum_sample_distance = int(np.floor(self.fps / self.freq_max))
        self.state = 'measure'

    # ── Main loop ─────────────────────────────────────────────────
    def run(self):
        self.benchmarker.add_tag('Measurement Loop')
        self.benchmarker.add_tag('Frame Capture')
        self.benchmarker.add_tag('Calibration Measurement')

        print("[*] Warming up camera…")
        for attempt in range(200):
            frame = self.next_frame()
            if frame is not False and frame is not None:
                self.current_frame = frame
                print(f"[+] Camera ready (attempt {attempt + 1})")
                break
            time.sleep(0.025)
        else:
            logging.error("Camera failed to open. Aborting.")
            return

        self._running = True

        while self.cap.isOpened() and self._running:
            self.loop_start_time = time.time()

            self.benchmarker.tick_start('Frame Capture')
            frame_data = self.next_frame()
            self.benchmarker.tick_end('Frame Capture')

            if frame_data is False or frame_data is None:
                break
            self.current_frame = frame_data

            if self.state == 'initialize':
                self.initialize()
                self.state = 'calibration'

            elif self.state == 'calibration':
                if self.calibration_buffer_idx < self.calibration_buffer_target_length:
                    self.calibration_buffer[self.calibration_buffer_idx][:] = self.current_frame
                    self.calibration_buffer_idx += 1
                    self.calibration_progress_bar.update(1)
                else:
                    logging.info("Calibration complete. Finding chest ROI…")
                    self.detect_fps()
                    self.peak_minimum_sample_distance = int(np.floor(self.fps / self.freq_max))

                    self.benchmarker.tick_start('Calibration Measurement')
                    location = self.find_chest_roi()
                    self.benchmarker.tick_end('Calibration Measurement')

                    if location is None:
                        logging.info("ROI not found – retrying…")
                        self.calibration_buffer_idx = 0
                        self.calibration_progress_bar.reset()
                        continue

                    self.x, self.y, self.w, self.h = location
                    self.x, self.y, self.w, self.h = reduce_bounding_box(
                        self.x, self.y, self.w, self.h,
                        self.maximum_bounding_box_area)
                    logging.info(f"ROI locked: x={self.x} y={self.y} "
                                 f"w={self.w} h={self.h}")
                    self.calibration_progress_bar.close()
                    self.state = 'measure'

            elif self.state == 'measure':
                if self.save_all_data and self.video_out is None:
                    self.video_out = cv2.VideoWriter(
                        str(self.capture_target) + '.mp4',
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        self.fps, (self.w, self.h))

                self.benchmarker.tick_start('Measurement Loop')
                self.cropped_image = self.current_frame[
                    self.y: self.y + self.h, self.x: self.x + self.w]

                for b in self.buffers:
                    if len(b) >= self.measure_buffer_length:
                        b.popleft()

                motion_val = self.extract_motion()
                self.data.append(motion_val)
                self.logger.process_frame_value(motion_val)
                self.t.append(0.0 if len(self.t) == 0 else self.t[-1] + 1.0 / self.fps)

                if self.save_all_data and self.video_out:
                    gray_bgr = cv2.cvtColor(float_to_uint8(self.cropped_image),
                                            cv2.COLOR_GRAY2BGR)
                    self.video_out.write(gray_bgr)
                    self.all_data.append((self.t[-1], motion_val))

                if len(self.data) > self.measure_initialization_length:
                    self.measure()
                    if not self.disable_error_detection and self.detect_errors():
                        self.trigger_error("poor signal")

                self.benchmarker.tick_end('Measurement Loop')

            elif self.state == 'error':
                if time.time() - self.reset_start_time >= self.error_reset_delay:
                    self.reset()
                    self.state = 'calibration'

            self.update_ui()
            self.sync_to_fps()

        logging.info("Capture closed.")
        self.cap.release()
        if self.save_all_data and self.video_out:
            self.video_out.release()
            np.save(str(self.capture_target) + '.npy', self.all_data)
        cv2.destroyAllWindows()

    # ── Eulerian fallback ─────────────────────────────────────────
    @staticmethod
    def locate(calibration_video_data, fps,
               freq_min=0.1, freq_max=1.0, amplification=500,
               pyramid_levels=9, skip_levels_at_top=4,
               temporal_threshold=0.7, threshold=20,
               threshold_type=cv2.THRESH_BINARY,
               verbose=False, save_calibration_image=False):

        logging.info("Eulerian magnification ROI search…")
        op, raw = eulerian_magnification_bandpass(
            calibration_video_data, fps, freq_min, freq_max, amplification,
            skip_levels_at_top=skip_levels_at_top,
            pyramid_levels=pyramid_levels,
            threshold=temporal_threshold, verbose=verbose)

        avg_frame = np.average(op, axis=0)
        avg_norm  = ((avg_frame - avg_frame.min()) /
                     (avg_frame.max() - avg_frame.min()))
        avg = float_to_uint8(avg_norm)

        _, thresh   = cv2.threshold(avg, threshold, 255, threshold_type)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        if save_calibration_image:
            total_avg = float_to_uint8(np.average(calibration_video_data, axis=0))
            drawn     = cv2.rectangle(total_avg + copy.deepcopy(avg),
                                      (x, y), (x + w, y + h), 255, 2)
            i = 0
            while os.path.exists(f"calibration{i}.png"):
                i += 1
            cv2.imwrite(f"calibration{i}.png", drawn)

        return x, y, w, h
    