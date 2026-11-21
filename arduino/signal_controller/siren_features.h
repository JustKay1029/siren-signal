/* GENERATED — DO NOT EDIT
   Implements siren-feature-parity spec (streaming, integer-only, <40 B static RAM).
   Call siren_feed() once per ADC sample; it returns 1 when a chunk is ready.
   Read results from the SirenFeatures global struct.
*/
#pragma once
#include <stdint.h>

#define SR_HZ            9600
#define WIN_SIZE         256
#define WINS_PER_CHUNK   37
#define HYST_COUNTS      6
#define DC_SHIFT_BITS    6
#define LEVEL_GATE_VALUE 40

typedef struct {
  int16_t pmin, pmax, pmean, sweep;
  int16_t lvl;
} SirenChunk;

typedef struct {
  int32_t  dc16;         /* Q4 fixed-point DC */
  int8_t   state;        /* -1=unknown, 0=low, 1=high */
  int16_t  cross;
  int32_t  level_sum;
  uint8_t  win_cnt;
  uint8_t  chunk_wins;
  int16_t  pitch_buf[WINS_PER_CHUNK];
  int16_t  level_buf[WINS_PER_CHUNK];
  SirenChunk out;        /* filled when chunk completes */
} SirenState;

/* static instance — lives in .bss, zero-initialised at boot */
static SirenState _ss = { .dc16 = 512 << DC_SHIFT_BITS, .state = -1 };

/* Feed one ADC sample.  Returns 1 when a chunk is ready in _ss.out. */
static inline uint8_t siren_feed(int16_t x) {
  /* EMA DC removal */
  _ss.dc16 += (((int32_t)x << DC_SHIFT_BITS) - _ss.dc16) >> DC_SHIFT_BITS;
  int16_t dc = (int16_t)(_ss.dc16 >> DC_SHIFT_BITS);

  _ss.level_sum += (x > dc) ? (x - dc) : (dc - x);

  /* Hysteresis zero-crossing */
  if (x > dc + HYST_COUNTS) {
    if (_ss.state == 0) _ss.cross++;
    _ss.state = 1;
  } else if (x < dc - HYST_COUNTS) {
    if (_ss.state == 1) _ss.cross++;
    _ss.state = 0;
  }

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
  for (uint8_t i = 0; i < WINS_PER_CHUNK; i++) {
    int16_t p = _ss.pitch_buf[i];
    if (p < pmin) pmin = p;
    if (p > pmax) pmax = p;
    psum += p;
    lsum += _ss.level_buf[i];
    if (i > 0) {
      int16_t d = p - _ss.pitch_buf[i-1];
      sweep += (d < 0) ? -d : d;
    }
  }
  _ss.out.pmin  = pmin;
  _ss.out.pmax  = pmax;
  _ss.out.pmean = (int16_t)(psum / WINS_PER_CHUNK);
  _ss.out.sweep = (int16_t)(sweep / (WINS_PER_CHUNK - 1));
  _ss.out.lvl   = (int16_t)(lsum / WINS_PER_CHUNK);

  _ss.chunk_wins = 0;
  return 1;
}
