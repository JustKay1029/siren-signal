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

void setup() {
  Serial.begin(115200);
  pinMode(CAL_PIN, INPUT_PULLUP);
  cal_mode = (digitalRead(CAL_PIN) == LOW);
  if (cal_mode) {
    Serial.println(F("# CALIBRATION MODE — lvl only"));
  } else {
    Serial.println(F("# pmin,pmax,pmean,sweep,lvl,class,sr_est"));
  }
  chunk_start_us = micros();
}

void loop() {
  /* Check for serial 'c' to toggle calibration */
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'c') cal_mode = !cal_mode;
  }

  int16_t adc = (int16_t)analogRead(A0);
  sample_count++;

  if (cal_mode) {
    /* Feed feature extractor just for lvl */
    if (siren_feed(adc)) {
      Serial.print(F("lvl="));
      Serial.println(_ss.out.lvl);
    }
    return;
  }

  if (siren_feed(adc)) {
    /* Measure SR from elapsed time */
    unsigned long elapsed_us = micros() - chunk_start_us;
    uint16_t sr_est = (uint16_t)((uint32_t)sample_count * 1000000UL / elapsed_us);
    chunk_start_us = micros();
    sample_count = 0;

    uint8_t cls = SIREN_NOISE;
    if (_ss.out.lvl >= LEVEL_GATE_VALUE) {
      cls = siren_classify(_ss.out.pmin, _ss.out.pmax,
                            _ss.out.pmean, _ss.out.sweep);
    }

    Serial.print(_ss.out.pmin);  Serial.print(',');
    Serial.print(_ss.out.pmax);  Serial.print(',');
    Serial.print(_ss.out.pmean); Serial.print(',');
    Serial.print(_ss.out.sweep); Serial.print(',');
    Serial.print(_ss.out.lvl);   Serial.print(',');
    Serial.print(cls);           Serial.print(',');
    Serial.println(sr_est);
  }
}
