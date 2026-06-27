// ESP32 FireBeetle AGV firmware for Qt A-star navigation
// Based on the uploaded working firmware. Requires imu_manager.h/.cpp in the same Arduino sketch folder.

#include <Arduino.h>
#include "soc/gpio_struct.h"
#include "imu_manager.h"

// ====================================================
// MOTOR PINS
// ====================================================

const int stepPin1 = 16;
const int dirPin1 = 26;
const int enPin1 = 25;

const int stepPin2 = 4;
const int dirPin2 = 13;
const int enPin2 = 14;


// ====================================================
// MOTOR LIMITS
// ====================================================

const uint32_t MAX_PPS = 20000;


// ====================================================
// TRAPEZOID / RAMP PARAMETERS
// ====================================================

float ACCEL_PPS_PER_SEC = 7000.0f;
float DECEL_PPS_PER_SEC = 9000.0f;

const uint32_t RAMP_INTERVAL_MS = 20;


// ====================================================
// IMU STRAIGHT HOLD PARAMETERS
// ====================================================

int BASE_PPS = 6500;

// Adjustable from serial:
// SET_IMU_KP 70
// SET_IMU_MAX 250
float KP_IMU_PPS_PER_DEG = 70.0f;
int MAX_IMU_CORR_PPS = 350;

const uint32_t CONTROL_INTERVAL_MS = 50;


// ====================================================
// IMU TURN PARAMETERS
// ====================================================

const int TURN_PPS_FAST = 1200;
const int TURN_PPS_SLOW = 700;

const float TURN_SLOW_ZONE_DEG = 12.0f;
const float TURN_DONE_DEG = 2.0f;

// If TURN_REL 90 turns wrong direction, change to -1
const int TURN_SIGN = 1;


// ====================================================
// TIMER OBJECTS
// ====================================================

hw_timer_t* leftTimer = NULL;
hw_timer_t* rightTimer = NULL;

portMUX_TYPE leftMux = portMUX_INITIALIZER_UNLOCKED;
portMUX_TYPE rightMux = portMUX_INITIALIZER_UNLOCKED;


// ====================================================
// IMU OBJECT
// ====================================================

ImuManager imu;


// ====================================================
// ROBOT MODE
// ====================================================

enum RobotMode {
  MODE_STOP = 0,
  MODE_DIRECT_VEL = 1,
  MODE_IMU_STRAIGHT = 2,
  MODE_IMU_TURN = 3
};

RobotMode robotMode = MODE_STOP;


// ====================================================
// STATE VARIABLES
// ====================================================

volatile uint32_t leftPPS = 0;
volatile uint32_t rightPPS = 0;

volatile bool leftStepState = false;
volatile bool rightStepState = false;

int currentLeftPps = 0;
int currentRightPps = 0;

int targetLeftPps = 0;
int targetRightPps = 0;

float targetHeadingDeg = 0.0f;
float headingErrorDeg = 0.0f;

unsigned long lastControlMs = 0;
unsigned long lastRampMs = 0;
unsigned long lastPrintMs = 0;

String command;


// ====================================================
// ANGLE HELPERS
// ====================================================

float normalizeAngleDeg(float a) {
  while (a > 180.0f) {
    a -= 360.0f;
  }

  while (a < -180.0f) {
    a += 360.0f;
  }

  return a;
}


float imuHeadingDeg() {
  ImuOrientation o = imu.getOrientation();
  return o.relative_heading_deg;
}


// ====================================================
// TIMER ISR
// ====================================================

void IRAM_ATTR onLeftTimer() {
  portENTER_CRITICAL_ISR(&leftMux);

  leftStepState = !leftStepState;

  if (leftStepState) {
    GPIO.out_w1ts = (1UL << stepPin1);
  } else {
    GPIO.out_w1tc = (1UL << stepPin1);
  }

  portEXIT_CRITICAL_ISR(&leftMux);
}


void IRAM_ATTR onRightTimer() {
  portENTER_CRITICAL_ISR(&rightMux);

  rightStepState = !rightStepState;

  if (rightStepState) {
    GPIO.out_w1ts = (1UL << stepPin2);
  } else {
    GPIO.out_w1tc = (1UL << stepPin2);
  }

  portEXIT_CRITICAL_ISR(&rightMux);
}


// ====================================================
// MOTOR PPS APPLY
// ====================================================

void setLeftPPS(uint32_t pps) {
  pps = constrain(pps, 0U, MAX_PPS);

  portENTER_CRITICAL(&leftMux);

  leftPPS = pps;

  if (pps == 0) {
    timerStop(leftTimer);
    leftStepState = false;
    GPIO.out_w1tc = (1UL << stepPin1);
  } else {
    uint32_t period_us = 500000UL / pps;
    timerAlarm(leftTimer, period_us, true, 0);
    timerStart(leftTimer);
  }

  portEXIT_CRITICAL(&leftMux);
}


void setRightPPS(uint32_t pps) {
  pps = constrain(pps, 0U, MAX_PPS);

  portENTER_CRITICAL(&rightMux);

  rightPPS = pps;

  if (pps == 0) {
    timerStop(rightTimer);
    rightStepState = false;
    GPIO.out_w1tc = (1UL << stepPin2);
  } else {
    uint32_t period_us = 500000UL / pps;
    timerAlarm(rightTimer, period_us, true, 0);
    timerStart(rightTimer);
  }

  portEXIT_CRITICAL(&rightMux);
}


void applyLeftSignedPPS(int pps) {
  currentLeftPps = pps;

  if (pps == 0) {
    setLeftPPS(0);
    return;
  }

  digitalWrite(dirPin1, pps >= 0 ? HIGH : LOW);
  setLeftPPS(abs(pps));
}


void applyRightSignedPPS(int pps) {
  currentRightPps = pps;

  if (pps == 0) {
    setRightPPS(0);
    return;
  }

  digitalWrite(dirPin2, pps >= 0 ? HIGH : LOW);
  setRightPPS(abs(pps));
}


int rampToward(int current, int target, int maxDelta) {
  if (current < target) {
    current += maxDelta;

    if (current > target) {
      current = target;
    }
  }

  else if (current > target) {
    current -= maxDelta;

    if (current < target) {
      current = target;
    }
  }

  return current;
}


void updateVelocityRamp() {
  unsigned long now = millis();

  if (now - lastRampMs < RAMP_INTERVAL_MS) {
    return;
  }

  float dt = (now - lastRampMs) / 1000.0f;
  lastRampMs = now;

  if (dt <= 0.0f) {
    return;
  }

  int leftDelta;
  int rightDelta;

  if (abs(targetLeftPps) < abs(currentLeftPps)) {
    leftDelta = (int)(DECEL_PPS_PER_SEC * dt);
  } else {
    leftDelta = (int)(ACCEL_PPS_PER_SEC * dt);
  }

  if (abs(targetRightPps) < abs(currentRightPps)) {
    rightDelta = (int)(DECEL_PPS_PER_SEC * dt);
  } else {
    rightDelta = (int)(ACCEL_PPS_PER_SEC * dt);
  }

  if (leftDelta < 1) {
    leftDelta = 1;
  }

  if (rightDelta < 1) {
    rightDelta = 1;
  }

  int newLeft = rampToward(currentLeftPps, targetLeftPps, leftDelta);
  int newRight = rampToward(currentRightPps, targetRightPps, rightDelta);

  if (newLeft != currentLeftPps) {
    applyLeftSignedPPS(newLeft);
  }

  if (newRight != currentRightPps) {
    applyRightSignedPPS(newRight);
  }
}


void setWheelVelocity(int left, int right) {
  left = constrain(left, -(int)MAX_PPS, (int)MAX_PPS);
  right = constrain(right, -(int)MAX_PPS, (int)MAX_PPS);

  targetLeftPps = left;
  targetRightPps = right;
}


void stopRobot() {
  robotMode = MODE_STOP;

  targetLeftPps = 0;
  targetRightPps = 0;

  currentLeftPps = 0;
  currentRightPps = 0;

  setLeftPPS(0);
  setRightPPS(0);
}


// ====================================================
// IMU STRAIGHT HOLD
// ====================================================

void updateImuStraight() {
  if (!imu.isReady()) {
    setWheelVelocity(BASE_PPS, BASE_PPS);
    return;
  }

  float imuHeading = imuHeadingDeg();

  headingErrorDeg = normalizeAngleDeg(
    targetHeadingDeg - imuHeading
  );

  float correction = -KP_IMU_PPS_PER_DEG * headingErrorDeg;

  correction = constrain(
    correction,
    -MAX_IMU_CORR_PPS,
    MAX_IMU_CORR_PPS
  );

  int left = BASE_PPS - (int)correction;
  int right = BASE_PPS + (int)correction;

  setWheelVelocity(left, right);
}


// ====================================================
// IMU TURN
// ====================================================

void updateImuTurn() {
  if (!imu.isReady()) {
    stopRobot();
    Serial.print("ERR TURN IMU_NOT_READY ");
    Serial.println(imu.getStateText());
    return;
  }

  float imuHeading = imuHeadingDeg();

  headingErrorDeg = normalizeAngleDeg(
    targetHeadingDeg - imuHeading
  );

  if (abs(headingErrorDeg) <= TURN_DONE_DEG) {

    stopRobot();

    Serial.print("OK TURN_DONE target=");
    Serial.print(targetHeadingDeg, 2);

    Serial.print(" imu=");
    Serial.print(imuHeading, 2);

    Serial.print(" err=");
    Serial.println(headingErrorDeg, 2);

    return;
  }

  int turnPps = TURN_PPS_FAST;

  if (abs(headingErrorDeg) <= TURN_SLOW_ZONE_DEG) {
    turnPps = TURN_PPS_SLOW;
  }

  int left;
  int right;

  if (headingErrorDeg > 0.0f) {
    left = -turnPps * TURN_SIGN;
    right = turnPps * TURN_SIGN;
  } else {
    left = turnPps * TURN_SIGN;
    right = -turnPps * TURN_SIGN;
  }

  setWheelVelocity(left, right);
}


// ====================================================
// ROBOT CONTROL UPDATE
// ====================================================

void updateRobotControl() {
  unsigned long now = millis();

  if (now - lastControlMs >= CONTROL_INTERVAL_MS) {
    lastControlMs = now;

    if (robotMode == MODE_IMU_STRAIGHT) {
      updateImuStraight();
    }

    else if (robotMode == MODE_IMU_TURN) {
      updateImuTurn();
    }
  }

  if (now - lastPrintMs >= 300) {
    lastPrintMs = now;

    Serial.print("RUN mode=");
    Serial.print((int)robotMode);

    Serial.print(" imuState=");
    Serial.print(imu.getStateText());

    Serial.print(" imu=");
    Serial.print(imuHeadingDeg(), 2);

    Serial.print(" target=");
    Serial.print(targetHeadingDeg, 2);

    Serial.print(" err=");
    Serial.print(headingErrorDeg, 2);

    Serial.print(" L=");
    Serial.print(currentLeftPps);
    Serial.print("/");
    Serial.print(targetLeftPps);

    Serial.print(" R=");
    Serial.print(currentRightPps);
    Serial.print("/");
    Serial.print(targetRightPps);

    Serial.print(" kp=");
    Serial.print(KP_IMU_PPS_PER_DEG, 1);

    Serial.print(" maxCorr=");
    Serial.println(MAX_IMU_CORR_PPS);
  }
}


// ====================================================
// SERIAL COMMANDS
// ====================================================

void processCommand(const String& cmd) {
  int left;
  int right;
  float value1;
  float value2;

  if (sscanf(cmd.c_str(), "VEL %d %d", &left, &right) == 2) {

    if (left == 0 && right == 0) {
      robotMode = MODE_STOP;
      setWheelVelocity(0, 0);
    } else {
      robotMode = MODE_DIRECT_VEL;
      setWheelVelocity(left, right);
    }

    Serial.print("OK VEL ");
    Serial.print(left);
    Serial.print(" ");
    Serial.println(right);
  }

  else if (cmd == "LOCK_HEADING_GO") {

    if (!imu.isReady()) {
      Serial.print("ERR IMU_NOT_READY ");
      Serial.println(imu.getStateText());
      return;
    }

    targetHeadingDeg = imuHeadingDeg();
    headingErrorDeg = 0.0f;

    robotMode = MODE_IMU_STRAIGHT;

    Serial.print("OK LOCK_HEADING_GO target=");
    Serial.println(targetHeadingDeg, 2);
  }

  else if (sscanf(cmd.c_str(), "TURN_REL %f", &value1) == 1) {

    if (!imu.isReady()) {
      Serial.print("ERR TURN IMU_NOT_READY ");
      Serial.println(imu.getStateText());
      return;
    }

    float currentHeading = imuHeadingDeg();

    targetHeadingDeg = normalizeAngleDeg(
      currentHeading + value1
    );

    headingErrorDeg = normalizeAngleDeg(
      targetHeadingDeg - currentHeading
    );

    robotMode = MODE_IMU_TURN;

    Serial.print("OK TURN_START deg=");
    Serial.print(value1, 2);

    Serial.print(" current=");
    Serial.print(currentHeading, 2);

    Serial.print(" target=");
    Serial.print(targetHeadingDeg, 2);

    Serial.print(" err=");
    Serial.println(headingErrorDeg, 2);
  }

  else if (cmd == "STOP") {

    stopRobot();

    Serial.println("OK STOP");
  }

  else if (cmd == "STATUS") {

    Serial.print("STATUS mode=");
    Serial.print((int)robotMode);

    Serial.print(" imuState=");
    Serial.print(imu.getStateText());

    Serial.print(" imu=");
    Serial.print(imuHeadingDeg(), 2);

    Serial.print(" target=");
    Serial.print(targetHeadingDeg, 2);

    Serial.print(" err=");
    Serial.print(headingErrorDeg, 2);

    Serial.print(" L=");
    Serial.print(currentLeftPps);
    Serial.print("/");
    Serial.print(targetLeftPps);

    Serial.print(" R=");
    Serial.print(currentRightPps);
    Serial.print("/");
    Serial.print(targetRightPps);

    Serial.print(" base=");
    Serial.print(BASE_PPS);

    Serial.print(" kp=");
    Serial.print(KP_IMU_PPS_PER_DEG, 1);

    Serial.print(" maxCorr=");
    Serial.println(MAX_IMU_CORR_PPS);
  }

  else if (cmd == "IMU RECAL") {

    stopRobot();

    Serial.print("IMU current state before recal: ");
    Serial.println(imu.getStateText());

    if (imu.getState() == IMU_ERROR || imu.getState() == IMU_BOOT) {

      Serial.println("IMU is not recoverable by recalibrate(). Trying imu.begin() again...");

      bool beginOk = imu.begin();

      if (!beginOk) {
        Serial.println("ERR IMU BEGIN FAILED");
        return;
      }

      Serial.print("IMU state after begin: ");
      Serial.println(imu.getStateText());
    }

    Serial.println("IMU recalibrating, keep robot still...");

    bool ok = imu.recalibrate();

    if (ok) {
      Serial.println("OK IMU RECAL");
    } else {
      Serial.print("ERR IMU RECAL FAILED state=");
      Serial.println(imu.getStateText());
    }
  }

  else if (sscanf(cmd.c_str(), "SET_IMU_KP %f", &value1) == 1) {

    KP_IMU_PPS_PER_DEG = value1;

    Serial.print("OK SET_IMU_KP ");
    Serial.println(KP_IMU_PPS_PER_DEG, 2);
  }

  else if (sscanf(cmd.c_str(), "SET_IMU_MAX %f", &value1) == 1) {

    MAX_IMU_CORR_PPS = (int)value1;

    Serial.print("OK SET_IMU_MAX ");
    Serial.println(MAX_IMU_CORR_PPS);
  }

  else if (sscanf(cmd.c_str(), "SET_BASE %f", &value1) == 1) {

    BASE_PPS = (int)value1;
    BASE_PPS = constrain(BASE_PPS, 0, (int)MAX_PPS);

    Serial.print("OK SET_BASE ");
    Serial.println(BASE_PPS);
  }

  else if (sscanf(cmd.c_str(), "SET_RAMP %f %f", &value1, &value2) == 2) {

    ACCEL_PPS_PER_SEC = value1;
    DECEL_PPS_PER_SEC = value2;

    Serial.print("OK SET_RAMP accel=");
    Serial.print(ACCEL_PPS_PER_SEC, 1);

    Serial.print(" decel=");
    Serial.println(DECEL_PPS_PER_SEC, 1);
  }

  else {

    Serial.print("ERR ");
    Serial.println(cmd);
  }
}


// ====================================================
// SETUP
// ====================================================

void setup() {
  Serial.begin(115200);

  pinMode(stepPin1, OUTPUT);
  pinMode(dirPin1, OUTPUT);
  pinMode(enPin1, OUTPUT);

  pinMode(stepPin2, OUTPUT);
  pinMode(dirPin2, OUTPUT);
  pinMode(enPin2, OUTPUT);

  digitalWrite(stepPin1, LOW);
  digitalWrite(stepPin2, LOW);

  // T60 enable is active LOW
  digitalWrite(enPin1, LOW);
  digitalWrite(enPin2, LOW);

  leftTimer = timerBegin(1000000);
  timerAttachInterrupt(leftTimer, &onLeftTimer);
  timerAlarm(leftTimer, 1000, true, 0);
  timerStop(leftTimer);

  rightTimer = timerBegin(1000000);
  timerAttachInterrupt(rightTimer, &onRightTimer);
  timerAlarm(rightTimer, 1000, true, 0);
  timerStop(rightTimer);

  Serial.println("Starting IMU. Keep robot still...");

  bool imuOk = imu.begin();

  if (imuOk) {
    Serial.println("IMU begin OK. Waiting until IMU_READY...");
  } else {
    Serial.println("IMU begin FAILED");
  }

  stopRobot();

  Serial.println("ESP32 Robot Ready");
  Serial.println("Protocol:");
  Serial.println("  VEL left_pps right_pps");
  Serial.println("  LOCK_HEADING_GO");
  Serial.println("  TURN_REL deg");
  Serial.println("  IMU RECAL");
  Serial.println("  STOP");
  Serial.println("  STATUS");
  Serial.println("  SET_IMU_KP value");
  Serial.println("  SET_IMU_MAX value");
  Serial.println("  SET_BASE value");
  Serial.println("  SET_RAMP accel decel");
}


// ====================================================
// LOOP
// ====================================================

void loop() {
  imu.update();

  while (Serial.available()) {

    char c = Serial.read();

    if (c == '\n' || c == '\r') {

      if (command.length() > 0) {
        processCommand(command);
        command = "";
      }

    } else {

      command += c;

      if (command.length() > 80) {
        command = "";
      }
    }
  }

  updateRobotControl();

  updateVelocityRamp();
}