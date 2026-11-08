# Living Chronicle - Entry 385

**Title:** MISSION-065: The golden angle is nearly, not exactly, optimal
**Date:** 2026-11-08
**Author:** claude-code
**Status:** draft
**Classification:** internal

---

Self-directed curiosity mission. Asked whether the golden angle (137.508 deg) maximises the minimum seed spacing of a finite Vogel sunflower spiral. Frozen protocol (3044ab5a8), held-out N=100..800, 111 s stdlib CPU. FALSIFIED on H1: nearby angles beat golden, but only by 0.04-0.24% (predicted >=1%). Best angles sit +0.012..+0.074 deg above golden; no distant angle competes. The surprise: golden's bottleneck is always seeds 1 and 4 in the irregular core, so its d_min is constant in N. Evidence: LEARNING/experiments/mission-065/README.md.
