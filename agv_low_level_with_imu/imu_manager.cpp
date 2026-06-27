#include "imu_manager.h"


ImuManager::ImuManager() {
  _mpu_addr = MPU6050_ADDR_LOW;
  _state = IMU_BOOT;
  _last_update_us = 0;
  _calibration_start_ms = 0;
  _last_print_ms = 0;

  _gyro_offset_x = 0.0f;
  _gyro_offset_y = 0.0f;
  _gyro_offset_z = 0.0f;
  _reference_heading_deg = 0.0f;
  _relative_heading_deg = 0.0f;
  _heading_candidate_deg = 0.0f;
  _heading_stable_count = 0;
  _heading_candidate_valid = false;

  _roll_deg = 0.0f;
  _pitch_deg = 0.0f;
  _yaw_deg = 0.0f;

  _q0 = 1.0f;
  _q1 = 0.0f;
  _q2 = 0.0f;
  _q3 = 0.0f;
}


bool ImuManager::begin() {
  _state = IMU_BOOT;

  Wire.begin();
  Wire.setClock(I2C_CLOCK_HZ);
  delay(250);

  if (!initMpu6050()) {
    _state = IMU_ERROR;
    return false;
  }

  _state = IMU_CALIBRATING;
  calibrateGyro();

  if (!initializeOrientation()) {
    _state = IMU_ERROR;
    return false;
  }

  _last_update_us = micros();
  _calibration_start_ms = millis();
  _heading_candidate_valid = false;
  _heading_stable_count = 0;
  _reference_heading_deg = 0.0f;
  _relative_heading_deg = 0.0f;

  return true;
}


bool ImuManager::recalibrate() {
  if (_state == IMU_ERROR || _state == IMU_BOOT) {
    return false;
  }

  _state = IMU_CALIBRATING;
  _relative_heading_deg = 0.0f;
  _heading_candidate_deg = 0.0f;
  _heading_candidate_valid = false;
  _heading_stable_count = 0;

  calibrateGyro();

  if (!initializeOrientation()) {
    _state = IMU_ERROR;
    return false;
  }

  getEulerDegrees();
  _reference_heading_deg = _yaw_deg;
  _relative_heading_deg = 0.0f;
  _last_update_us = micros();
  _calibration_start_ms = millis();
  _state = IMU_READY;

  return true;
}


void ImuManager::update() {
  if (_state == IMU_ERROR || _state == IMU_BOOT) {
    return;
  }

  uint32_t now_us = micros();
  if ((uint32_t)(now_us - _last_update_us) < SAMPLE_INTERVAL_US) {
    return;
  }

  float dt = (now_us - _last_update_us) / 1000000.0f;
  _last_update_us = now_us;

  MpuRaw raw = {0, 0, 0, 0, 0, 0, 0};
  if (!readMpuRaw(raw)) {
    return;
  }

  float ax = raw.ax / ACCEL_SCALE;
  float ay = raw.ay / ACCEL_SCALE;
  float az = raw.az / ACCEL_SCALE;

  float gx_deg_per_sec = (raw.gx - _gyro_offset_x) / GYRO_SCALE;
  float gy_deg_per_sec = (raw.gy - _gyro_offset_y) / GYRO_SCALE;
  float gz_deg_per_sec = (raw.gz - _gyro_offset_z) / GYRO_SCALE;

  float gx = gx_deg_per_sec * DEG_TO_RAD_F;
  float gy = gy_deg_per_sec * DEG_TO_RAD_F;
  float gz = gz_deg_per_sec * DEG_TO_RAD_F;

  updateMadgwickImu(gx, gy, gz, ax, ay, az, dt);
  getEulerDegrees();

  if (_state == IMU_CALIBRATING) {
    updateStartupReference(_yaw_deg);
    return;
  }

  _relative_heading_deg += gz_deg_per_sec * dt * GYRO_HEADING_SIGN;
  _relative_heading_deg = wrapAngleDeg(_relative_heading_deg);
}


bool ImuManager::isReady() const {
  return _state == IMU_READY;
}


ImuState ImuManager::getState() const {
  return _state;
}


const char *ImuManager::getStateText() const {
  switch (_state) {
    case IMU_BOOT:
      return "BOOT";
    case IMU_CALIBRATING:
      return "IMU_CALIBRATING";
    case IMU_READY:
      return "IMU_READY";
    case IMU_ERROR:
      return "IMU_ERROR";
    default:
      return "UNKNOWN";
  }
}


ImuOrientation ImuManager::getOrientation() const {
  ImuOrientation orientation = {
    _roll_deg,
    _pitch_deg,
    _yaw_deg,
    _relative_heading_deg
  };
  return orientation;
}


void ImuManager::printSerial(Stream &serial) const {
  if (_state != IMU_READY) {
    serial.println(getStateText());
    return;
  }

  serial.println("IMU_READY");
  serial.print("ROLL:");
  serial.println(_roll_deg, 2);
  serial.print("PITCH:");
  serial.println(_pitch_deg, 2);
  serial.print("YAW:");
  serial.println(_yaw_deg, 2);
  serial.print("HEADING:");
  serial.println(_relative_heading_deg, 2);
}


void ImuManager::printHeadingSerial(Stream &serial) const {
  if (_state != IMU_READY) {
    serial.println(getStateText());
    return;
  }

  serial.print("HEADING:");
  serial.println(_relative_heading_deg, 2);
}


bool ImuManager::findMpu6050() {
  if (devicePresent(MPU6050_ADDR_LOW)) {
    _mpu_addr = MPU6050_ADDR_LOW;
    return true;
  }

  if (devicePresent(MPU6050_ADDR_HIGH)) {
    _mpu_addr = MPU6050_ADDR_HIGH;
    return true;
  }

  return false;
}


bool ImuManager::initMpu6050() {
  if (!findMpu6050()) {
    return false;
  }

  if (!writeByte(_mpu_addr, 0x6B, 0x00)) {
    return false;
  }
  delay(100);

  writeByte(_mpu_addr, 0x6B, 0x01);
  writeByte(_mpu_addr, 0x6A, 0x00);
  writeByte(_mpu_addr, 0x37, 0x02);
  delay(10);

  writeByte(_mpu_addr, 0x19, 0x09);
  writeByte(_mpu_addr, 0x1A, 0x03);
  writeByte(_mpu_addr, 0x1B, 0x00);
  writeByte(_mpu_addr, 0x1C, 0x00);

  return true;
}


bool ImuManager::readMpuRaw(MpuRaw &raw) {
  uint8_t data[14];

  if (!readBytes(_mpu_addr, 0x3B, data, sizeof(data))) {
    return false;
  }

  raw.ax = readS16BE(&data[0]);
  raw.ay = readS16BE(&data[2]);
  raw.az = readS16BE(&data[4]);
  raw.temp = readS16BE(&data[6]);
  raw.gx = readS16BE(&data[8]);
  raw.gy = readS16BE(&data[10]);
  raw.gz = readS16BE(&data[12]);

  return true;
}


void ImuManager::calibrateGyro() {
  const uint16_t warmup_samples = 200;
  const uint16_t samples = 1000;
  int32_t sum_x = 0;
  int32_t sum_y = 0;
  int32_t sum_z = 0;
  uint16_t good_samples = 0;

  for (uint16_t i = 0; i < warmup_samples; i++) {
    MpuRaw raw = {0, 0, 0, 0, 0, 0, 0};
    readMpuRaw(raw);
    delay(3);
  }

  for (uint16_t i = 0; i < samples; i++) {
    MpuRaw raw = {0, 0, 0, 0, 0, 0, 0};
    if (readMpuRaw(raw)) {
      sum_x += raw.gx;
      sum_y += raw.gy;
      sum_z += raw.gz;
      good_samples++;
    }
    delay(3);
  }

  if (good_samples > 0) {
    _gyro_offset_x = (float)sum_x / good_samples;
    _gyro_offset_y = (float)sum_y / good_samples;
    _gyro_offset_z = (float)sum_z / good_samples;
  }
}


bool ImuManager::initializeOrientation() {
  MpuRaw raw = {0, 0, 0, 0, 0, 0, 0};
  if (!readMpuRaw(raw)) {
    return false;
  }

  float ax = raw.ax / ACCEL_SCALE;
  float ay = raw.ay / ACCEL_SCALE;
  float az = raw.az / ACCEL_SCALE;

  float roll = atan2(ay, az);
  float pitch = atan2(-ax, sqrt((ay * ay) + (az * az)));
  float yaw = 0.0f;

  setQuaternionFromEuler(roll, pitch, yaw);
  return true;
}


void ImuManager::updateStartupReference(float yaw_deg) {
  if (millis() - _calibration_start_ms < STARTUP_STABILIZATION_MS) {
    return;
  }

  if (!_heading_candidate_valid) {
    _heading_candidate_deg = yaw_deg;
    _heading_candidate_valid = true;
    _heading_stable_count = 0;
    return;
  }

  if (fabs(angleDiffDeg(yaw_deg, _heading_candidate_deg)) <= HEADING_STABLE_DEG) {
    _heading_stable_count++;
  } else {
    _heading_candidate_deg = yaw_deg;
    _heading_stable_count = 0;
  }

  if (_heading_stable_count < HEADING_STABLE_SAMPLES) {
    return;
  }

  _reference_heading_deg = yaw_deg;
  _relative_heading_deg = 0.0f;
  _state = IMU_READY;
}


void ImuManager::updateMadgwickImu(
  float gx,
  float gy,
  float gz,
  float ax,
  float ay,
  float az,
  float dt
) {
  if (ax == 0.0f && ay == 0.0f && az == 0.0f) {
    return;
  }

  float recip_norm = invSqrt((ax * ax) + (ay * ay) + (az * az));
  if (recip_norm == 0.0f) {
    return;
  }

  ax *= recip_norm;
  ay *= recip_norm;
  az *= recip_norm;

  float q0 = _q0;
  float q1 = _q1;
  float q2 = _q2;
  float q3 = _q3;

  float f1 = 2.0f * (q1 * q3 - q0 * q2) - ax;
  float f2 = 2.0f * (q0 * q1 + q2 * q3) - ay;
  float f3 = 2.0f * (0.5f - q1 * q1 - q2 * q2) - az;

  float j_11_or_24 = 2.0f * q2;
  float j_12_or_23 = 2.0f * q3;
  float j_13_or_22 = 2.0f * q0;
  float j_14_or_21 = 2.0f * q1;
  float j_32 = 4.0f * q1;
  float j_33 = 4.0f * q2;

  float s0 = j_14_or_21 * f2 - j_11_or_24 * f1;
  float s1 = j_12_or_23 * f1 + j_13_or_22 * f2 - j_32 * f3;
  float s2 = j_12_or_23 * f2 - j_33 * f3 - j_13_or_22 * f1;
  float s3 = j_14_or_21 * f1 + j_11_or_24 * f2;

  recip_norm = invSqrt((s0 * s0) + (s1 * s1) + (s2 * s2) + (s3 * s3));
  if (recip_norm == 0.0f) {
    return;
  }

  s0 *= recip_norm;
  s1 *= recip_norm;
  s2 *= recip_norm;
  s3 *= recip_norm;

  float q_dot_0 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz) - MADGWICK_BETA * s0;
  float q_dot_1 = 0.5f * (q0 * gx + q2 * gz - q3 * gy) - MADGWICK_BETA * s1;
  float q_dot_2 = 0.5f * (q0 * gy - q1 * gz + q3 * gx) - MADGWICK_BETA * s2;
  float q_dot_3 = 0.5f * (q0 * gz + q1 * gy - q2 * gx) - MADGWICK_BETA * s3;

  q0 += q_dot_0 * dt;
  q1 += q_dot_1 * dt;
  q2 += q_dot_2 * dt;
  q3 += q_dot_3 * dt;

  recip_norm = invSqrt((q0 * q0) + (q1 * q1) + (q2 * q2) + (q3 * q3));
  if (recip_norm == 0.0f) {
    return;
  }

  _q0 = q0 * recip_norm;
  _q1 = q1 * recip_norm;
  _q2 = q2 * recip_norm;
  _q3 = q3 * recip_norm;
}


void ImuManager::getEulerDegrees() {
  _roll_deg = atan2(
    2.0f * (_q0 * _q1 + _q2 * _q3),
    1.0f - 2.0f * (_q1 * _q1 + _q2 * _q2)
  ) * RAD_TO_DEG_F;

  float sin_pitch = 2.0f * (_q0 * _q2 - _q3 * _q1);
  if (sin_pitch > 1.0f) {
    sin_pitch = 1.0f;
  }
  if (sin_pitch < -1.0f) {
    sin_pitch = -1.0f;
  }

  _pitch_deg = asin(sin_pitch) * RAD_TO_DEG_F;

  _yaw_deg = atan2(
    2.0f * (_q0 * _q3 + _q1 * _q2),
    1.0f - 2.0f * (_q2 * _q2 + _q3 * _q3)
  ) * RAD_TO_DEG_F;

  if (_yaw_deg < 0.0f) {
    _yaw_deg += 360.0f;
  }
}


void ImuManager::setQuaternionFromEuler(float roll, float pitch, float yaw) {
  float cr = cos(roll * 0.5f);
  float sr = sin(roll * 0.5f);
  float cp = cos(pitch * 0.5f);
  float sp = sin(pitch * 0.5f);
  float cy = cos(yaw * 0.5f);
  float sy = sin(yaw * 0.5f);

  _q0 = cr * cp * cy + sr * sp * sy;
  _q1 = sr * cp * cy - cr * sp * sy;
  _q2 = cr * sp * cy + sr * cp * sy;
  _q3 = cr * cp * sy - sr * sp * cy;

  float recip_norm = invSqrt((_q0 * _q0) + (_q1 * _q1) + (_q2 * _q2) + (_q3 * _q3));
  if (recip_norm == 0.0f) {
    _q0 = 1.0f;
    _q1 = 0.0f;
    _q2 = 0.0f;
    _q3 = 0.0f;
    return;
  }

  _q0 *= recip_norm;
  _q1 *= recip_norm;
  _q2 *= recip_norm;
  _q3 *= recip_norm;
}


bool ImuManager::writeByte(uint8_t addr, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}


bool ImuManager::readBytes(uint8_t addr, uint8_t reg, uint8_t *data, uint8_t len) {
  Wire.beginTransmission(addr);
  Wire.write(reg);

  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  uint8_t received = Wire.requestFrom((int)addr, (int)len);
  if (received != len) {
    return false;
  }

  for (uint8_t i = 0; i < len; i++) {
    data[i] = Wire.read();
  }

  return true;
}


bool ImuManager::devicePresent(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}


int16_t ImuManager::readS16BE(const uint8_t *data) const {
  return (int16_t)(((uint16_t)data[0] << 8) | data[1]);
}


float ImuManager::invSqrt(float value) const {
  if (value <= 0.0f) {
    return 0.0f;
  }

  return 1.0f / sqrt(value);
}


float ImuManager::angleDiffDeg(float current, float reference) const {
  float diff = current - reference;
  return wrapAngleDeg(diff);
}


float ImuManager::wrapAngleDeg(float angle) const {
  while (angle > 180.0f) {
    angle -= 360.0f;
  }

  while (angle < -180.0f) {
    angle += 360.0f;
  }

  return angle;
}
