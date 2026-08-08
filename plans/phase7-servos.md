# Phase 7 — Servos: what's left

Execution plan for the servo/control stage. Design and reasoning live in
[`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) §6; this file only tracks what still has to happen.

**Hardware decided:** Arduino Uno + 2× DS3218 (**180°** variant) on PWM, PC talking to the board over
the §6 ASCII line protocol.

---

## Already done (code)

| Piece | Where |
|---|---|
| Firmware: protocol + 4 safety rules | `firmware/pdp_servo/pdp_servo.ino` |
| Serial backend: protocol, `READY` wait, ping | `pdp/control/serial_servo.py` |
| Safety layer + keep-alive | `pdp/control/loop.py` |
| Closed-loop aiming | `pdp/logic/policy.py` |
| Runtime parameters | `configs/runtime/live.yaml` |

None of it has ever run against hardware. 48 unit tests pass, all with fakes.

---

## Open decision — asynchronous `ERR 5`

The firmware prints `ERR 5` on its own when the watchdog trips, but `serial_servo.py` reads exactly one
line per command sent. That unsolicited line **desynchronises the ack stream permanently**: every
later `apply()` reads the previous command's reply. Nothing crashes; the warnings just stop matching
the channel they name.

Pick one:

- [ ] **A — Drop the line.** Firmware tracks the state internally and says nothing. Simplest.
- [ ] **B — Tag async messages.** Prefix them (`! WATCHDOG`) and have Python skip those lines when
      reading acks. Keeps the diagnostic, costs a little parsing on both sides.

Worth fixing before hardware testing: the watchdog *will* trip during step 3 below, on purpose.

---

## 1. Buy the rest of the hardware

- [ ] Arduino Uno
- [ ] External **6 V / 3 A** supply — **never** the Arduino's 5 V pin. Two DS3218 can pull several amps
      stalled; browning out the board looks exactly like a software bug and isn't one
- [ ] ~1000 µF capacitor across the servo supply, near the servos
- [ ] 3-pin servo leads, and the 2-axis rig itself

Common ground between the supply and the Arduino is mandatory — without it nothing works.

## 2. Compile and upload

- [ ] Compile `firmware/pdp_servo/pdp_servo.ino` — **never been compiled**, no Arduino toolchain on this
      machine. First place a trivial error would show up
- [ ] Confirm `SERVO_SWEEP_DEG = 180.0` matches the servos actually bought
- [ ] Upload and confirm the `READY pdp-servo v1 ch=2` banner appears on the serial monitor

## 3. Test the firmware alone, no Python

Serial monitor at 115200. This is the whole reason the protocol is plain text — each half must be
debuggable on its own.

- [ ] `S 0 20` → replies `OK 0 20.00`, servo moves
- [ ] `S 0 999` → replies `ERR 3`, servo does **not** move (limits reject, never clip)
- [ ] `S 5 0` → replies `ERR 2` (bad channel)
- [ ] `P` → replies `OK`
- [ ] Stop typing → rig returns to centre on its own after 500 ms
- [ ] Send a large step and watch it ramp rather than snap (slew limit)

## 4. Connect Python

- [ ] Find the real COM port (`live.yaml` says `COM3` as a placeholder)
- [ ] Set `control.backend: serial` in `configs/runtime/live.yaml`
- [ ] Run `pdp live` and confirm the `READY` handshake is logged
- [ ] Confirm keep-alive works: hold aim on a still target and check the rig does **not** recentre

## 5. Calibrate against the built rig

- [ ] **`invert_pan` / `invert_tilt`** — flip if the rig drives away from the target. Cannot be known
      until the rig is assembled
- [ ] **Travel limits** — currently ±45° pan, ±30° tilt. Pan uses only half of a 180° servo's range;
      widen toward ±90° if the mechanics and the iPhone cable allow
- [ ] Keep the firmware limits **at least as tight** as `control.limits_deg`
- [ ] **`gain`** — 0.5 to start. Raise if it converges too slowly, lower if it oscillates
- [ ] **Field of view** — `fov_h_deg: 69.0` / `fov_v_deg: 42.0` are derived from the iPhone 15 main
      camera optics, *not measured*. Camo may crop; the ultra-wide lens is a completely different
      number (~120°)

## 6. Verify the safety rules

The §8 "done" criterion for this phase. All of it with the rig assembled and moving:

- [ ] **Watchdog** — pull the USB mid-run; the rig must return to neutral on its own
- [ ] **Clamp** — command past the limit; must be rejected, not clipped
- [ ] **Slew limit** — a large jump must ramp, not snap
- [ ] **Deadband** — a stationary target must not make the servos buzz

---

## Assumptions to check with the board in hand

Three choices made without hardware to verify them against:

- **Pins 9 and 10** for pan and tilt
- **500–2500 µs** pulse range for the DS3218 — confirm against the seller's datasheet
- **±45° / ±30°** limits fitting the mechanics you actually build
