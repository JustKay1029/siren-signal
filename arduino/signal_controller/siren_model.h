/* GENERATED 2026-11-19T23:36:57 — DO NOT EDIT
   Training: held-out accuracy=0.8032, macro-F1=0.7957
   Tree params: {'max_depth': 3, 'min_samples_leaf': 2}
*/
#pragma once
#include <stdint.h>

/* Class constants */
#define SIREN_NOISE     0
#define SIREN_AMBULANCE 1
#define SIREN_FIRE      2

static inline uint8_t siren_classify(int16_t pmin, int16_t pmax,
                                      int16_t pmean, int16_t sweep) {
  if (pmax <= 646) {
    if (pmin <= 9) {
      if (pmax <= 177) {
        return 0; /* traffic */
      } else {
        return 1; /* ambulance */
      }
    } else {
      if (pmax <= 552) {
        return 0; /* traffic */
      } else {
        return 0; /* traffic */
      }
    }
  } else {
    if (sweep <= 60) {
      if (pmax <= 1452) {
        return 1; /* ambulance */
      } else {
        return 1; /* ambulance */
      }
    } else {
      if (sweep <= 193) {
        return 2; /* firetruck */
      } else {
        return 1; /* ambulance */
      }
    }
  }
}
