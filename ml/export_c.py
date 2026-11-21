"""
ml/export_c.py — export trained model and feature extractor as Arduino C headers.

Run:  python -m ml.export_c

Generates (never edit these by hand):
  arduino/signal_controller/siren_model.h    — decision tree as inline if/else
  arduino/signal_controller/siren_features.h — streaming feature extractor
  arduino/signal_controller/config.h         — constants from ml/config.py
  arduino/siren_probe/siren_probe.ino        — calibration+measurement sketch
"""

import json, math, pathlib, subprocess, sys, time

from ml.config import (
    SR, WIN, WINS_PER_CHUNK, HYST, DC_SHIFT, LEVEL_GATE,
    CLASSES, TREE_JSON, METRICS_JSON, ARDUINO_DIR, PROBE_DIR
)

FNAMES = ["pmin", "pmax", "pmean", "sweep"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tree() -> dict:
    if not pathlib.Path(TREE_JSON).exists():
        print(f"ERROR: {TREE_JSON} not found. Run python -m ml.train first.")
        sys.exit(1)
    return json.loads(pathlib.Path(TREE_JSON).read_text(encoding="utf-8"))


def _load_metrics() -> dict:
    return json.loads(pathlib.Path(METRICS_JSON).read_text(encoding="utf-8")) \
        if pathlib.Path(METRICS_JSON).exists() else {}


def _emit_node(tree: dict, node: int, indent: int) -> list[str]:
    """Recursively emit if/else C code for one tree node."""
    pad = "  " * indent
    left  = tree["children_left"][node]
    right = tree["children_right"][node]

    if left == -1:   # leaf
        values = tree["value"][node]
        # values shape: [1, n_classes]
        cls = int(max(range(len(values[0])), key=lambda i: values[0][i]))
        return [f"{pad}return {cls}; /* {CLASSES[cls]} */\n"]

    feat    = tree["feature"][node]
    thresh  = tree["threshold"][node]
    # Integer split: float threshold → floor (safe for int16 feature values)
    ithresh = int(math.floor(thresh))
    fname   = FNAMES[feat]

    lines = [f"{pad}if ({fname} <= {ithresh}) {{\n"]
    lines += _emit_node(tree, left, indent + 1)
    lines += [f"{pad}}} else {{\n"]
    lines += _emit_node(tree, right, indent + 1)
    lines += [f"{pad}}}\n"]
    return lines


def gen_model_h(tree: dict, metrics: dict) -> str:
    acc   = metrics.get("held_out_accuracy", "?")
    f1    = metrics.get("held_out_macro_f1", "?")
    ts    = metrics.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))

    body_lines = _emit_node(tree, 0, 1)
    body = "".join(body_lines)

    return f"""\
/* GENERATED {ts} — DO NOT EDIT
   Training: held-out accuracy={acc}, macro-F1={f1}
   Tree params: {tree["params"]}
*/
#pragma once
#include <stdint.h>

/* Class constants */
#define SIREN_NOISE     0
#define SIREN_AMBULANCE 1
#define SIREN_FIRE      2

static inline uint8_t siren_classify(int16_t pmin, int16_t pmax,
                                      int16_t pmean, int16_t sweep) {{
{body}\
}}
"""


def gen_features_h() -> str:
    return f"""\
/* GENERATED — DO NOT EDIT
   Implements siren-feature-parity spec (streaming, integer-only, <40 B static RAM).
   Call siren_feed() once per ADC sample; it returns 1 when a chunk is ready.
   Read results from the SirenFeatures global struct.
*/
#pragma once
#include <stdint.h>

#define SR_HZ            {SR}
#define WIN_SIZE         {WIN}
#define WINS_PER_CHUNK   {WINS_PER_CHUNK}
#define HYST_COUNTS      {HYST}
#define DC_SHIFT_BITS    {DC_SHIFT}
#define LEVEL_GATE_VALUE {LEVEL_GATE}

typedef struct {{
  int16_t pmin, pmax, pmean, sweep;
  int16_t lvl;
}} SirenChunk;

typedef struct {{
  int32_t  dc16;         /* Q4 fixed-point DC */
  int8_t   state;        /* -1=unknown, 0=low, 1=high */
  int16_t  cross;
  int32_t  level_sum;
  uint8_t  win_cnt;
  uint8_t  chunk_wins;
  int16_t  pitch_buf[WINS_PER_CHUNK];
  int16_t  level_buf[WINS_PER_CHUNK];
  SirenChunk out;        /* filled when chunk completes */
}} SirenState;

/* static instance — lives in .bss, zero-initialised at boot */
static SirenState _ss = {{ .dc16 = 512 << DC_SHIFT_BITS, .state = -1 }};

/* Feed one ADC sample.  Returns 1 when a chunk is ready in _ss.out. */
static inline uint8_t siren_feed(int16_t x) {{
  /* EMA DC removal */
  _ss.dc16 += (((int32_t)x << DC_SHIFT_BITS) - _ss.dc16) >> DC_SHIFT_BITS;
  int16_t dc = (int16_t)(_ss.dc16 >> DC_SHIFT_BITS);

  _ss.level_sum += (x > dc) ? (x - dc) : (dc - x);

  /* Hysteresis zero-crossing */
  if (x > dc + HYST_COUNTS) {{
    if (_ss.state == 0) _ss.cross++;
    _ss.state = 1;
  }} else if (x < dc - HYST_COUNTS) {{
    if (_ss.state == 1) _ss.cross++;
    _ss.state = 0;
  }}

  _ss.win_cnt++;
  if (_ss.win_cnt < WIN_SIZE) return 0;

  /* Window end */
  int16_t pitch = (int16_t)(((int32_t)_ss.cross * 75L) >> 2);
  int16_t level = (int16_t)(_ss.level_sum >> 8);
  _ss.pitch_buf[_ss.chunk_wins] = pitch;
  _ss.level_buf[_ss.chunk_wins] = level;

  _ss.cross = 0;
  _ss.level_sum = 0;
  _ss.win_cnt = 0;
  _ss.chunk_wins++;

  if (_ss.chunk_wins < WINS_PER_CHUNK) return 0;

  /* Chunk end */
  int16_t pmin = 32767, pmax = -32768;
  int32_t psum = 0, lsum = 0, sweep = 0;
  for (uint8_t i = 0; i < WINS_PER_CHUNK; i++) {{
    int16_t p = _ss.pitch_buf[i];
    if (p < pmin) pmin = p;
    if (p > pmax) pmax = p;
    psum += p;
    lsum += _ss.level_buf[i];
    if (i > 0) {{
      int16_t d = p - _ss.pitch_buf[i-1];
      sweep += (d < 0) ? -d : d;
    }}
  }}
  _ss.out.pmin  = pmin;
  _ss.out.pmax  = pmax;
  _ss.out.pmean = (int16_t)(psum / WINS_PER_CHUNK);
  _ss.out.sweep = (int16_t)(sweep / (WINS_PER_CHUNK - 1));
  _ss.out.lvl   = (int16_t)(lsum / WINS_PER_CHUNK);

  _ss.chunk_wins = 0;
  return 1;
}}
"""


def gen_config_h() -> str:
    return f"""\
/* GENERATED from ml/config.py — DO NOT EDIT */
#pragma once
#define CFG_SR             {SR}
#define CFG_WIN            {WIN}
#define CFG_WINS_PER_CHUNK {WINS_PER_CHUNK}
#define CFG_HYST           {HYST}
#define CFG_DC_SHIFT       {DC_SHIFT}
#define CFG_LEVEL_GATE     {LEVEL_GATE}
"""


def gen_probe_ino() -> str:
    return f"""\
/*
 * siren_probe.ino — calibration and sampling-rate measurement sketch.
 * GENERATED — DO NOT EDIT CONSTANTS; they come from config.h.
 *
 * Baud rate : 115200
 * Serial output per chunk: pmin,pmax,pmean,sweep,lvl,class,sr_est
 *
 * Calibration mode: send 'c' over Serial (or hold pin 7 LOW at boot).
 *   Prints lvl continuously so you can tune LEVEL_GATE against
 *   quiet room noise and phone-speaker sirens at 0.5 m and 2 m.
 */

#include "config.h"
#include "siren_features.h"
#include "siren_model.h"

#define CAL_PIN   7      /* pull LOW to enter calibration mode */

static bool cal_mode = false;
static unsigned long chunk_start_us = 0;
static uint32_t sample_count = 0;

void setup() {{
  Serial.begin(115200);
  pinMode(CAL_PIN, INPUT_PULLUP);
  cal_mode = (digitalRead(CAL_PIN) == LOW);
  if (cal_mode) {{
    Serial.println(F("# CALIBRATION MODE — lvl only"));
  }} else {{
    Serial.println(F("# pmin,pmax,pmean,sweep,lvl,class,sr_est"));
  }}
  chunk_start_us = micros();
}}

void loop() {{
  /* Check for serial 'c' to toggle calibration */
  if (Serial.available()) {{
    char c = Serial.read();
    if (c == 'c') cal_mode = !cal_mode;
  }}

  int16_t adc = (int16_t)analogRead(A0);
  sample_count++;

  if (cal_mode) {{
    /* Feed feature extractor just for lvl */
    if (siren_feed(adc)) {{
      Serial.print(F("lvl="));
      Serial.println(_ss.out.lvl);
    }}
    return;
  }}

  if (siren_feed(adc)) {{
    /* Measure SR from elapsed time */
    unsigned long elapsed_us = micros() - chunk_start_us;
    uint16_t sr_est = (uint16_t)((uint32_t)sample_count * 1000000UL / elapsed_us);
    chunk_start_us = micros();
    sample_count = 0;

    uint8_t cls = SIREN_NOISE;
    if (_ss.out.lvl >= LEVEL_GATE_VALUE) {{
      cls = siren_classify(_ss.out.pmin, _ss.out.pmax,
                            _ss.out.pmean, _ss.out.sweep);
    }}

    Serial.print(_ss.out.pmin);  Serial.print(',');
    Serial.print(_ss.out.pmax);  Serial.print(',');
    Serial.print(_ss.out.pmean); Serial.print(',');
    Serial.print(_ss.out.sweep); Serial.print(',');
    Serial.print(_ss.out.lvl);   Serial.print(',');
    Serial.print(cls);           Serial.print(',');
    Serial.println(sr_est);
  }}
}}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tree    = _load_tree()
    metrics = _load_metrics()

    # Create directories
    arduino_dir = pathlib.Path(ARDUINO_DIR)
    probe_dir   = pathlib.Path(PROBE_DIR)
    arduino_dir.mkdir(parents=True, exist_ok=True)
    probe_dir.mkdir(parents=True, exist_ok=True)

    # Write headers
    (arduino_dir / "siren_model.h").write_text(gen_model_h(tree, metrics), encoding="utf-8")
    print(f"Written: {arduino_dir/'siren_model.h'}")

    (arduino_dir / "siren_features.h").write_text(gen_features_h(), encoding="utf-8")
    print(f"Written: {arduino_dir/'siren_features.h'}")

    (arduino_dir / "config.h").write_text(gen_config_h(), encoding="utf-8")
    print(f"Written: {arduino_dir/'config.h'}")

    (probe_dir / "siren_probe.ino").write_text(gen_probe_ino(), encoding="utf-8")
    print(f"Written: {probe_dir/'siren_probe.ino'}")

    # Try arduino-cli
    try:
        result = subprocess.run(
            ["arduino-cli", "compile", "--fqbn", "arduino:avr:uno",
             str(probe_dir)],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            # Parse flash/SRAM from output
            out = result.stdout + result.stderr
            print("\narduino-cli output:")
            print(out)
        else:
            print(f"\narduino-cli failed (exit {result.returncode}):")
            print(result.stderr[:500])
    except FileNotFoundError:
        print("\narduino-cli not found — not compiled. "
              "Run arduino-cli on real hardware to check flash/SRAM.")
    except subprocess.TimeoutExpired:
        print("\narduino-cli timed out.")

    print("\nExport complete.")


if __name__ == "__main__":
    main()
