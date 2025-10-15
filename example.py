# inpaint_dashboard.py
# Panel dashboard for diffusion inpainting results over 1024 images per .npy file.

import os
import numpy as np
import panel as pn
from repaint import setup, main
from pathlib import Path
import math
import queue
import threading

pn.extension()  # no global sizing

import numpy as np

def _coerce_flat_5x5(x):
    """
    Accepts: list, list-of-lists, or np.array for a single 5x5 sample
             or a batch (list/array of many samples). Returns (5,5) float32.
    Rules:
      - If x looks like a batch (ndim >= 2 and size multiple of 25), we take the LAST sample.
      - If x is 1D of length 25, we reshape (5,5).
      - If x is 2D 5x5, we use it directly.
    """
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        if a.size != 25:
            raise ValueError(f"Expected 25 values, got {a.size} in 1D array")
        return a.reshape(5, 5)
    if a.ndim == 2:
        # Either 5x5 (single) or Nx25 (batch) or 25xN (unlikely)
        if a.shape == (5, 5):
            return a
        if a.shape[1] == 25:          # Nx25 batch
            return a[-1].reshape(5, 5)
        if a.shape[0] == 25:          # 25xN -> take last column-set if N==1 else error
            if a.shape[1] == 1:
                return a[:, 0].reshape(5, 5)
    if a.ndim >= 3:
        # Treat as batch of 5x5: (...,5,5) -> take last item
        if a.shape[-2:] == (5, 5):
            return a.reshape(-1, 5, 5)[-1]
        # Or batch of 25 -> take last and reshape
        if a.shape[-1] == 25:
            return a.reshape(-1, 25)[-1].reshape(5, 5)
    # Last resort: flatten and expect 25
    a = a.ravel()
    if a.size != 25:
        raise ValueError(f"Could not coerce to 5x5; got flattened size {a.size}")
    return a.reshape(5, 5)

# =========================
# ---- Helper functions ----
# =========================

def _ensure_chw(img: np.ndarray) -> np.ndarray:
    """Return (H, W, C) with C in {1,3}. Accepts (H,W), (H,W,1), (H,W,3)."""
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] not in (1, 3):
        raise ValueError(f"Expected last dim to be 1 or 3; got {img.shape}")
    return img

def _to_uint8_with_ref(arr, ref_min, ref_max):
    a = _ensure_chw(arr).astype(np.float32)

    # Compute denom as Python float
    denom = float(ref_max) - float(ref_min)
    if denom == 0.0:
        out = np.zeros(a.shape, dtype=np.uint8)
    else:
        norm = (a - float(ref_min)) / denom
        out = (norm * 255.0).clip(0, 255).astype(np.uint8)

    # 1-channel -> RGB
    if out.shape[-1] == 1:
        out = np.repeat(out, 3, axis=1 if out.ndim == 3 and out.shape[0] in (1,3) else 2)
    return np.ascontiguousarray(out)

def _as_svg_from_uint8(rgb_u8: np.ndarray, scale: int) -> str:
    """
    rgb_u8: (H,W,3) uint8 already in display space (do NOT normalize here).
    scale: integer pixel size for each source pixel.
    """
    if rgb_u8.dtype != np.uint8 or rgb_u8.ndim != 3 or rgb_u8.shape[-1] != 3:
        raise ValueError("Expected (H,W,3) uint8")
    H, W = rgb_u8.shape[:2]
    S = int(scale)
    rects = []
    # Build SVG with crisp edges and no smoothing
    for y in range(H):
        row = rgb_u8[y]
        for x in range(W):
            r, g, b = map(int, row[x])
            rects.append(
                f'<rect x="{x*S}" y="{y*S}" width="{S}" height="{S}" fill="rgb({r},{g},{b})" />'
            )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W*S}" height="{H*S}" '
        f'shape-rendering="crispEdges">{"".join(rects)}</svg>'
    )

def _nearest_upscale_uint8(img_u8, scale):
    s = int(scale)  # <- ensure int
    x = np.ascontiguousarray(img_u8)
    if s <= 1:
        return x
    x = np.repeat(x, s, axis=0)
    x = np.repeat(x, s, axis=1)
    return np.ascontiguousarray(x)

def _to_uint8_for_display(arr: np.ndarray) -> np.ndarray:
    """Map array to uint8 [0,255] for safe display. Handles bool, int, float."""
    a = arr
    if a.dtype == np.bool_:
        return (a.astype(np.uint8) * 255)
    if np.issubdtype(a.dtype, np.integer):
        # normalize per image min/max if dynamic range is odd
        amin, amax = a.min(), a.max()
        if amin == amax:
            return np.zeros_like(a, dtype=np.uint8)
        a = (a - amin) / float(amax - amin)
        return (a * 255.0).clip(0, 255).astype(np.uint8)
    # float
    # try to infer if already in [0,1] or [-1,1]
    a = a.astype(np.float32)
    if a.min() >= 0.0 and a.max() <= 1.0:
        pass
    elif a.min() >= -1.0 and a.max() <= 1.0:
        a = (a + 1.0) * 0.5
    else:
        amin, amax = a.min(), a.max()
        if amin == amax:
            return np.zeros_like(a, dtype=np.uint8)
        a = (a - amin) / float(amax - amin)
    return (a * 255.0).clip(0, 255).astype(np.uint8)

def _tile_for_display(img: np.ndarray) -> np.ndarray:
    img = _ensure_chw(img)
    img = _to_uint8_for_display(img)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return np.ascontiguousarray(img)  # important


def _compute_residual(orig: np.ndarray, inpainted: np.ndarray) -> np.ndarray:
    """
    Residual = abs(orig - inpainted), normalized per-image for display.
    Keeps channels consistent.
    """
    o = orig.astype(np.float32)
    i = inpainted.astype(np.float32)
    r = np.abs(o - i)
    # Normalize per-image
    if r.max() > 0:
        r = r / r.max()
    return r

# =========================
# --------- UI ------------
# =========================

class DiffusionInpaintUI:
    """
    Panel UI to load a file 'X.npy' (X from 1..N), run mask + inpaint for 1024 images,
    and browse results. Plug in user-provided make_mask_fn and inpaint_fn.
    """

    def __init__(
        self,
        data_dir: str,
        expected_count: int = 1024,
        file_min: int = 1,
        file_max: int = 1000,
    ):
        self.data_dir = data_dir
        self.expected_count = expected_count

        self.full_images = None

        # State
        self.current_file_index = file_min
        self.cache = {}  # file_index -> dict with 'orig','mask','inpaint','resid'
        self._busy = False

        # Widgets
        self.status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.source_id = pn.pane.Markdown("", sizing_mode="stretch_width")

        self.start_btn = pn.widgets.Button(name="Start Inpainting", button_type="primary")
        self.progress = pn.indicators.Progress(name="Progress", value=0, max=100, width=300, height=20, bar_color="primary")

        self.dir_input = pn.widgets.TextInput(name="Data directory", value=self.data_dir)
        npy_list = Path(self.dir_input.value).glob('*.npy')
        npy_list = [str(x)[len(self.dir_input.value):-4] for x in npy_list]
        
        self.file_index = pn.widgets.DiscreteSlider(name="File index (X.npy)", options=["-"], value="-")
        self.file_index.param.watch(self._load_images_from_npy, "value")

        if len(npy_list):
            self._log(f"{len(npy_list)} .npy files found in {self.dir_input.value}")
            self.file_index.options = npy_list
            self.file_index.value = npy_list[0]
            self.file_index.disabled = False
            self.start_btn.disabled = False


        else:
            self._log(f"No .npy files found in {self.dir_input.value}")
            self.file_index.options = ["-"]
            self.file_index.value = "-"
            self.file_index.disabled = True
            self.start_btn.disabled = True

        self.dir_input.param.watch(self._update_dir_and_files, "value")

        self.T_select = pn.widgets.IntInput(name="Number of iterations", value=60, start=1, end=1000, step=1)
        self.js_select = pn.widgets.IntInput(name="Resample Amount", value=3, start=1, end=30, step=1)
        self.jl_select = pn.widgets.IntInput(name="Jump Length Resample", value=1, start=1, end=30, step=1)
        self.inpaint_size = pn.widgets.IntInput(name="Inpaint Size", value=5, start=1, end=15, step=2)
        powers_of_two = [2**n for n in range(0, 11)]
        self.batch_size = pn.widgets.DiscreteSlider(name="Batch Size (Reduce if out of memory error)", options=powers_of_two, value=16)

        self.idx_slider = pn.widgets.IntSlider(name="Image index", start=0, end=self.expected_count-1, value=0, step=1)
        self.prev_btn = pn.widgets.Button(name="◀ Prev", button_type="default")
        self.next_btn = pn.widgets.Button(name="Next ▶", button_type="default")
        self.jump_input = pn.widgets.IntInput(name="Jump to index", start=0, end=self.expected_count-1, value=0)
        self.jump_btn = pn.widgets.Button(name="Go", button_type="default")

        pn.extension(
            raw_css=[
                """
                .pixelated img {
                    image-rendering: pixelated;
                    image-rendering: crisp-edges;
                }
                """
            ]
        )

        self.original_pane  = pn.pane.SVG(sizing_mode='fixed')
        self.residual_pane  = pn.pane.SVG(sizing_mode='fixed')
        self.inpainted_pane = pn.pane.SVG(sizing_mode='fixed')
        self.mask_pane = pn.pane.SVG(sizing_mode='fixed')


        # Wire events
        self.start_btn.on_click(self._on_start)
        self.prev_btn.on_click(lambda e: self._bump_index(-1))
        self.next_btn.on_click(lambda e: self._bump_index(+1))
        self.idx_slider.param.watch(self._on_slider, "value")
        self.jump_btn.on_click(self._on_jump)

        # Layout
        self.view = pn.Column(
            pn.pane.Markdown("### Diffusion Inpainting Dashboard"),
            pn.Row(self.T_select, self.js_select, self.jl_select, sizing_mode='fixed'),
            pn.Row(self.inpaint_size, self.batch_size,sizing_mode='fixed'),
            pn.Spacer(height=10, sizing_mode='fixed'),
            pn.Row(self.dir_input, self.file_index, sizing_mode='fixed'),
            pn.Spacer(height=10, sizing_mode='fixed'),
            pn.Row(self.start_btn, self.progress, sizing_mode='fixed'),
            self.status,
            self.source_id,
            pn.Spacer(height=10, sizing_mode='fixed'),
            pn.Row(
                pn.Column("Original",  self.original_pane,  sizing_mode='fixed'),
                pn.Column("Mask",      self.mask_pane,      sizing_mode='fixed'),
                pn.Column("Residuals", self.residual_pane,  sizing_mode='fixed'),
                pn.Column("Inpainted", self.inpainted_pane, sizing_mode='fixed'),
                sizing_mode='fixed'
            ),
            pn.Spacer(height=10, sizing_mode='fixed'),
            pn.Row(self.prev_btn, self.idx_slider, self.next_btn, self.jump_input, self.jump_btn, sizing_mode='fixed'),
            sizing_mode='fixed',
        )

        # Initialize empty view
        self._update_display(None)

    # -------- actions --------

    def _accept_snapshot(self, snapshot: dict):
        rec = self.cache[self.current_file_index]

        # How many we already integrated (before this snapshot)
        prev_seen = getattr(self, "_seen", 0)

        L = min(
            len(snapshot.get("ids", [])),
            len(snapshot.get("o_pixels", [])),
            len(snapshot.get("c_pixels_norm", [])),
            len(snapshot.get("centre_pixels", [])),

        )

        if L <= prev_seen:
            return  # nothing new

        # --- build new chunk (same as you had) ---
        ids_chunk = snapshot["ids"][prev_seen:L]
        o_chunk = np.asarray(snapshot["o_pixels"][prev_seen:L], dtype=np.float32)
        cn_chunk = np.asarray(snapshot["c_pixels_norm"][prev_seen:L], dtype=np.float32)
        mask_centre_chunk = np.asarray(snapshot["centre_pixels"][prev_seen:L])

        P = int(o_chunk.shape[-1])
        side = int(round(P ** 0.5))
        if side * side != P:
            raise ValueError(f"Expected mask_size^2 length, got {P}")

        o_chunk = o_chunk.reshape(-1, side, side)
        cn_chunk = cn_chunk.reshape(-1, side, side)
        resid_chunk = o_chunk - cn_chunk
        n_new = o_chunk.shape[0]

        # Init/append
        if not isinstance(rec.get("orig"), np.ndarray) or rec["orig"].size == 0:
            rec["orig"] = o_chunk
            rec["inpaint"] = cn_chunk
            rec["resid"] = resid_chunk
            rec["ids"] = list(ids_chunk)
            rec["mask"] = [None] * n_new
            rec["centre_pixels"] = mask_centre_chunk
        else:
            rec["orig"] = np.concatenate([rec["orig"], o_chunk], axis=0)
            rec["inpaint"] = np.concatenate([rec["inpaint"], cn_chunk], axis=0)
            rec["resid"] = np.concatenate([rec["resid"], resid_chunk], axis=0)
            rec["ids"].extend(ids_chunk)
            rec["mask"].extend([None] * n_new)
            rec["centre_pixels"] = np.concatenate([rec["centre_pixels"], mask_centre_chunk], axis=0)

        # Update counters
        self._seen = L

        # --- slider bounds ---
        new_end = rec["orig"].shape[0] - 1
        self.idx_slider.end = new_end

        # --- FORCE FIRST RENDER ---
        # If this is the very first data we’ve integrated, show image 0 now.
        if prev_seen == 0:
            # (value is already 0, watcher won't fire; call explicitly)
            self.idx_slider.value = 0
            self._update_display(0)

        self._log(f"Integrated +{n_new} items (total {self._seen}).")

    def _load_images_from_npy(self,event):
        if self.file_index.value != "-":
            filename = f"{self.dir_input.value}{self.file_index.value}.npy"
            self.full_images = np.load(f"{self.dir_input.value}{self.file_index.value}.npy")[:,32:-32,32:-32]


    def _update_dir_and_files(self, event):
        npy_list = Path(self.dir_input.value).glob('*.npy')
        npy_list = [str(x)[len(self.dir_input.value):-4] for x in npy_list]
        if len(npy_list):

            self._log(f"{len(npy_list)} .npy files found in {self.dir_input.value}")
            self.file_index.options = npy_list
            self.file_index.value = npy_list[0]
            self.file_index.disabled = False
            self.start_btn.disabled = False


        else:

            self._log(f"No .npy files found in {self.dir_input.value}")
            self.file_index.options = ["-"]
            self.file_index.value = "-"
            self.file_index.disabled = True
            self.start_btn.disabled = True

    def _start_poller(self, period_ms=50):
        # Runs on UI thread; periodically drains queue
        if hasattr(self, "_poller") and self._poller is not None:
            try:
                self._poller.stop()
            except Exception:
                pass
        self._poller = pn.state.add_periodic_callback(self._drain_queue, period=period_ms)

    def _stop_poller(self):
        if hasattr(self, "_poller") and self._poller is not None:
            try:
                self._poller.stop()
            except Exception:
                pass
            self._poller = None

    def _log(self, msg: str):
        self.status.object = f"**Status:** {msg}"

    def _drain_queue(self):
        processed = 0
        try:
            while not self._delta_q.empty():
                kind, payload, _ = self._delta_q.get_nowait()
                processed += 1
                if kind == "snapshot":
                    self._accept_snapshot(payload)
                elif kind == "done":
                    self._finish_ok(getattr(self, "_seen", 0))
                    self._stop_poller()
                    self._busy = False
                    self.start_btn.disabled = False

                elif kind == "error":
                    self._handle_error(payload)
                    self._stop_poller()
                    self._busy = False
                    self.start_btn.disabled = False

        except Exception as e:
            self._log(f"Error while draining queue: {e}")
            self._stop_poller()
            self._busy = False
            self.start_btn.disabled = False

        return processed


    def _on_start(self, _):
        if self._busy:
            return
        self._busy = True
        self.start_btn.disabled = True
        self.progress.value = 0
        self.progress.active = True
        self._log("Starting streaming inpainting…")

        # Reset run state
        self.current_file_index = self.file_index.value
        self.cache[self.current_file_index] = dict(orig=[], mask=[], inpaint=[], resid=[], ids=[], centre_pixels=[])
        self._seen = 0  # how many rows integrated so far

        self.idx_slider.start = 0
        self.idx_slider.end = 0
        self.idx_slider.value = 0
        self._update_display(None)


        # Thread-safe queue to pass deltas back to UI
        self._delta_q = queue.Queue()

        # Start a UI-side poller to drain queue
        self._start_poller(period_ms=50)

        conf_arg = setup(
            self.T_select.value,
            self.js_select.value,
            self.jl_select.value,
            self.inpaint_size.value,
        )

        def worker():
            try:
                for snapshot in main(conf_arg,self.progress, self.batch_size.value, self.file_index.value, data_dir=self.dir_input.value):
                    self._delta_q.put(("snapshot", snapshot, None))
                # on completion, pass total we saw (optional)
                self._delta_q.put(("done", None, None))
            except Exception as e:
                self._delta_q.put(("error", e, None))


        threading.Thread(target=worker, daemon=True).start()


    def _accept_delta_and_update(self, delta: dict, n: int):
        """
        Runs on UI thread (called by _drain_queue). Accepts either:
        - per-step item: delta['o_pixels'] -> 25 values
        - growing snapshot: delta['o_pixels'] -> list of many items -> use last
        """
        rec = self.cache[self.current_file_index]

        print("537 rec: ",rec)

        # Support both naming styles (ids optional)
        # If you're streaming full snapshots, these will be list-of-lists; helper handles it.
        try:
            o_5x5 = _coerce_flat_5x5(delta["o_pixels"])
        except KeyError:
            # Some pipelines name it differently
            o_5x5 = _coerce_flat_5x5(delta["o"])  # adjust if needed

        try:
            c_5x5 = _coerce_flat_5x5(delta["c_pixels_norm"])
        except KeyError:
            c_5x5 = _coerce_flat_5x5(delta["c"])  # adjust if needed

        # Accumulate
        rec.setdefault("orig", [])
        rec.setdefault("inpaint", [])
        rec.setdefault("resid", [])
        rec.setdefault("mask", [])
        rec.setdefault("centre_pixels", [])

        rec["orig"].append(o_5x5)
        rec["inpaint"].append(c_5x5)
        rec["resid"].append(o_5x5 - c_5x5)
        rec["mask"].append(None)

        rec["centre_pixels"].append(delta["centre_pixels"])


        # Keep arrays for fast slicing
        for k in ("orig", "inpaint", "resid"):
            rec[k] = np.asarray(rec[k], dtype=np.float32)
        rec["centre_pixels"] = np.asarray(rec["centre_pixels"],dtype=np.int)
        # Update slider bounds; choose whether to auto-jump to the latest
        new_end = len(rec["orig"]) - 1
        self.idx_slider.end = new_end
        if new_end == 0:
            self.idx_slider.value = 0
            self._update_display(0)

        self._log(f"Received item {n} (cache size={new_end+1})")


    def _finish_ok(self, count: int):
        self.progress.active = False
        self._log(f"Done. Received {count} items.")

    def _handle_error(self, e: Exception):
        self._log(f"Error: {e}")

    def _mark_not_busy(self):
        self._busy = False
        self.start_btn.disabled = False


    def _bump_index(self, delta: int):
        new_idx = int(self.idx_slider.value) + delta
        new_idx = max(self.idx_slider.start, min(self.idx_slider.end, new_idx))
        self.idx_slider.value = new_idx

    def _on_slider(self, event):
        if event is None or event.new is None:
            return
        self._update_display(event.new)


    def _on_jump(self, _):
        val = int(self.jump_input.value)
        val = max(self.idx_slider.start, min(self.idx_slider.end, val))
        self.idx_slider.value = val

    def _update_display(self, idx: int | None):
        SCALE = 1

        fidx = self.current_file_index

        record = self.cache.get(fidx)
        if record is None or idx is None:
            # Clear panes
            self.original_pane.object = None
            self.mask_pane.object = _as_svg_from_uint8(np.ones((64,64,3), dtype=np.uint8)*255, 1)
            self.residual_pane.object = None
            self.inpainted_pane.object = None
            return

        orig   = record["orig"][idx]
        resid  = record["resid"][idx]
        inpaint= record["inpaint"][idx]
        centre_pixels_ = record["centre_pixels"][idx]

        self.source_id.object = f"**Source ID:** {record['ids'][idx]}"

        s = self.inpaint_size.value
        c_offset = int(np.ceil(s/2)-1)

        orig_infilled = self.full_images[idx].copy()
        inpaint_infilled = self.full_images[idx].copy()

        record["orig"][idx] = orig_infilled[centre_pixels_[0]:centre_pixels_[1],centre_pixels_[2]:centre_pixels_[3]]
        inpaint_infilled[centre_pixels_[0]:centre_pixels_[1],centre_pixels_[2]:centre_pixels_[3]] = record["inpaint"][idx]
        mask = np.ones((64,64,3), dtype=np.uint8)*200
        mask[centre_pixels_[0]:centre_pixels_[1],centre_pixels_[2]:centre_pixels_[3], :] = 0

        orig = orig_infilled
        inpaint = inpaint_infilled

        ref_min = float(min(orig.min(),    inpaint.min()))
        ref_max = float(max(orig.max(),    inpaint.max()))

        orig_u8    = _to_uint8_with_ref(orig,    ref_min, ref_max)
        inpaint_u8 = _to_uint8_with_ref(inpaint, ref_min, ref_max)

        # Compute signed residuals
        resid_signed = inpaint - orig

        # Normalize to [-1, 1] range
        r_range = float(ref_max) - float(ref_min)
        resid_norm = resid_signed / (r_range if r_range > 0.0 else 1.0)

        # Map to RGB (blue-white-red)
        #   -1 → blue (0, 0, 255)
        #    0 → white (255, 255, 255)
        #   +1 → red  (255, 0, 0)
        resid_rgb = np.zeros(resid_norm.shape + (3,), dtype=np.uint8)
        resid_rgb[..., 0] = np.clip(255 * (resid_norm > 0) * resid_norm + 255 * (resid_norm <= 0), 0, 255)  # Red channel
        resid_rgb[..., 1] = np.clip(255 * (1 - np.abs(resid_norm)), 0, 255)                                # Green channel
        resid_rgb[..., 2] = np.clip(255 * (-resid_norm > 0) * (-resid_norm) + 255 * (resid_norm >= 0), 0, 255)  # Blue channel
        resid_u8 = resid_rgb

        S_NEAREST = 5
        o = _nearest_upscale_uint8(orig_u8,    S_NEAREST)
        r = _nearest_upscale_uint8(resid_u8,   S_NEAREST)
        c = _nearest_upscale_uint8(inpaint_u8, S_NEAREST)
        m = _nearest_upscale_uint8(mask,    S_NEAREST)

        self.original_pane.object  = _as_svg_from_uint8(o, 1)
        self.residual_pane.object  = _as_svg_from_uint8(r, 1)
        self.inpainted_pane.object = _as_svg_from_uint8(c, 1)
        self.mask_pane.object = _as_svg_from_uint8(m, 1)



    # Expose a Panel component
    def panel(self):
        return self.view


# Edit these paths/funcs before serving:
DEFAULT_DATA_DIR = "data/"   # directory containing 1.npy, 2.npy, ...

ui = DiffusionInpaintUI(
    data_dir=DEFAULT_DATA_DIR
)

# When run as `panel serve inpaint_dashboard.py --show`
# the object `pn.serve(ui.panel())` is not needed; Panel discovers the `servable()`.
app = ui.panel()
app.servable()
