/*
 * pdp servo firmware — Arduino Uno + 2x DS3218 (pan / tilt).
 *
 * Speaks the ASCII line protocol from docs/ARCHITECTURE.md §6:
 *
 *     ->  S <ch> <deg> [<speed>]\n   set channel to an absolute angle
 *     ->  P\n                        ping
 *     <-  OK <ch> <deg>\n
 *     <-  ERR <code>\n
 *     <-  READY pdp-servo v1 ch=2\n  (once, on boot)
 *
 * Angles are SIGNED degrees, the same convention the PC speaks: 0 is neutral
 * (rig centred), negative is left/down. The mapping to servo microseconds
 * happens here and nowhere else.
 *
 * Every safety rule below is ALSO enforced in pdp/control/loop.py. That is
 * deliberate, not redundant: the PC can crash, hang, or have its USB pulled,
 * and when it does the rig has to be safe with no help from it. This firmware
 * is the half that keeps running when the other half is gone.
 *
 * Wiring:
 *   - Servo signal wires to PINS below.
 *   - Servo power from an EXTERNAL 6 V supply, never the Arduino 5 V pin: two
 *     DS3218 can pull several amps in a stall and will brown-out the board,
 *     which looks exactly like a software bug and is not one.
 *   - Supply ground MUST be tied to Arduino ground, or nothing works.
 *   - ~1000 uF across the servo supply, close to the servos.
 */

#include <Servo.h>

// ---- configuration -------------------------------------------------------

// The DS3218 ships in a 180-degree and a 270-degree variant with the same
// pulse range; only the sweep differs. Set this to the one you actually
// bought — getting it wrong scales every angle silently.
const float SERVO_SWEEP_DEG = 180.0;

const int SERVO_MIN_US = 500;    // full one way
const int SERVO_MID_US = 1500;   // mechanical centre == 0 deg logical
const int SERVO_MAX_US = 2500;   // full the other way

const uint8_t CH_COUNT = 2;
const uint8_t PINS[CH_COUNT] = {9, 10};   // 0 = pan, 1 = tilt

// Per-channel travel limits. Keep these at least as tight as
// control.limits_deg in configs/runtime/live.yaml. The PC clamps as well;
// this is the backstop for when the PC is the thing that is wrong.
const float LIMIT_LO[CH_COUNT] = {-45.0, -30.0};
const float LIMIT_HI[CH_COUNT] = { 45.0,  30.0};

const float NEUTRAL_DEG  = 0.0;
const float SLEW_DPS     = 180.0;  // deg/s ceiling: a detector glitch must not
                                   // snap a servo hard enough to strip a gear
const float DEADBAND_DEG = 1.0;    // ignore sub-threshold requests, or the
                                   // servo buzzes forever on detector jitter
const unsigned long WATCHDOG_MS = 500;
const unsigned long TICK_MS     = 20;      // 50 Hz motion update
const long          BAUD        = 115200;  // must match ControlConfig.baud

// ---- state ---------------------------------------------------------------

Servo servos[CH_COUNT];
float target[CH_COUNT];   // where we have been told to go
float current[CH_COUNT];  // where we have actually slewed to

unsigned long lastCommandMs = 0;
unsigned long lastTickMs = 0;
bool watchdogTripped = false;

char buf[48];
uint8_t bufLen = 0;

// ---- helpers -------------------------------------------------------------

float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

// Signed logical degrees -> servo pulse width.
void writeServo(uint8_t ch, float deg) {
  float us = SERVO_MID_US + deg * (SERVO_MAX_US - SERVO_MIN_US) / SERVO_SWEEP_DEG;
  servos[ch].writeMicroseconds((int) clampf(us, SERVO_MIN_US, SERVO_MAX_US));
}

// ---- command handling ----------------------------------------------------

void handleSet(char *args) {
  char *chTok  = strtok(args, " ");
  char *degTok = strtok(NULL, " ");
  // The speed field is in the §6 protocol but pdp/control/loop.py does the
  // rate limiting itself and never sends one. Accept and ignore it so both
  // forms work.
  if (chTok == NULL || degTok == NULL) {
    Serial.println(F("ERR 1"));   // malformed
    return;
  }

  int ch = atoi(chTok);
  if (ch < 0 || ch >= CH_COUNT) {
    Serial.println(F("ERR 2"));   // bad channel
    return;
  }

  float deg = atof(degTok);
  // §6: out-of-range commands are REJECTED, not silently clipped. Clipping
  // would hide a policy bug that is worth knowing about.
  if (deg < LIMIT_LO[ch] || deg > LIMIT_HI[ch]) {
    Serial.println(F("ERR 3"));   // out of range
    return;
  }

  // Deadband applies to the REQUEST against the last accepted target, not to
  // the output. Applying it to the output would also block the final step of
  // a legitimate slew — same reasoning as the comment in control/loop.py.
  if (fabs(deg - target[ch]) >= DEADBAND_DEG) {
    target[ch] = deg;
  }

  lastCommandMs = millis();
  if (watchdogTripped) {
    watchdogTripped = false;
  }

  Serial.print(F("OK "));
  Serial.print(ch);
  Serial.print(' ');
  Serial.println(target[ch], 2);
}

void handleLine(char *line) {
  switch (line[0]) {
    case 'S':
      handleSet(line + 1);
      break;
    case 'P':
      lastCommandMs = millis();   // a ping is proof the PC is alive
      Serial.println(F("OK"));
      break;
    default:
      Serial.println(F("ERR 1"));
  }
}

// Non-blocking: never stall the motion update waiting for a full line.
void readSerial() {
  while (Serial.available() > 0) {
    char c = (char) Serial.read();
    if (c == '\n' || c == '\r') {
      if (bufLen > 0) {
        buf[bufLen] = '\0';
        handleLine(buf);
        bufLen = 0;
      }
    } else if (bufLen < sizeof(buf) - 1) {
      buf[bufLen++] = c;
    } else {
      bufLen = 0;                 // overlong garbage; resync on next newline
      Serial.println(F("ERR 4"));
    }
  }
}

// ---- motion --------------------------------------------------------------

void tick() {
  // Watchdog: no valid command for WATCHDOG_MS -> return to neutral and hold.
  // Slew-limited on the way, so a lost connection doesn't produce a snap.
  if (millis() - lastCommandMs > WATCHDOG_MS) {
    if (!watchdogTripped) {
      watchdogTripped = true;
      Serial.println(F("ERR 5"));   // reported once, so the PC log shows it
    }
    for (uint8_t ch = 0; ch < CH_COUNT; ch++) {
      target[ch] = NEUTRAL_DEG;
    }
  }

  const float maxStep = SLEW_DPS * (TICK_MS / 1000.0);
  for (uint8_t ch = 0; ch < CH_COUNT; ch++) {
    float delta = target[ch] - current[ch];
    if (fabs(delta) > maxStep) {
      current[ch] += (delta > 0) ? maxStep : -maxStep;
    } else {
      current[ch] = target[ch];
    }
    current[ch] = clampf(current[ch], LIMIT_LO[ch], LIMIT_HI[ch]);
    writeServo(ch, current[ch]);
  }
}

// ---- lifecycle -----------------------------------------------------------

void setup() {
  Serial.begin(BAUD);

  for (uint8_t ch = 0; ch < CH_COUNT; ch++) {
    target[ch] = NEUTRAL_DEG;
    current[ch] = NEUTRAL_DEG;
    servos[ch].attach(PINS[ch], SERVO_MIN_US, SERVO_MAX_US);
    writeServo(ch, NEUTRAL_DEG);
  }

  lastCommandMs = millis();
  lastTickMs = millis();

  // Opening the port resets the board. The PC waits for this banner rather
  // than guessing at a fixed delay — see SerialServoBackend._wait_ready.
  Serial.print(F("READY pdp-servo v1 ch="));
  Serial.println(CH_COUNT);
}

void loop() {
  readSerial();

  unsigned long now = millis();
  if (now - lastTickMs >= TICK_MS) {
    lastTickMs = now;
    tick();
  }
}
