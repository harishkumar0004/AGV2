#ifndef ESP32_IMU_MANAGER_H
#define ESP32_IMU_MANAGER_H

#include <Arduino.h>
#include <Wire.h>


enum ImuState {
  IMU_BOOT,
  IMU_CALIBRATING,
  IMU_READY,
  IMU_ERROR
};


struct ImuOrientation {
  float roll_deg;
  float pitch_deg;
  float yaw_deg;
  float relative_heading_deg;
};


class ImuManager {
public:
  ImuManager();

  bool begin();
  void update();
  bool recalibrate();
  bool isReady() const;
  ImuState getState() const;
  const char *getStateText() const;
  ImuOrientation getOrientation() const;
  void printSerial(Stream &serial) const;
  void printHeadingSerial(Stream &serial) const;

private:
  struct MpuRaw {
    int16_t ax;
    int16_t ay;
    int16_t az;
    int16_t temp;
    int16_t gx;
    int16_t gy;
    int16_t gz;
  };

  static const uint32_t I2C_CLOCK_HZ = 400000UL;
  static const uint32_t SAMPLE_INTERVAL_US = 20000UL;
  static const uint32_t STARTUP_STABILIZATION_MS = 5000UL;
  static const uint16_t HEADING_STABLE_SAMPLES = 50;
  static constexpr float HEADING_STABLE_DEG = 0.75f;
  static constexpr float ACCEL_SCALE = 16384.0f;
  static constexpr float GYRO_SCALE = 131.0f;
  static constexpr float DEG_TO_RAD_F = PI / 180.0f;
  static constexpr float RAD_TO_DEG_F = 180.0f / PI;
  static constexpr float MADGWICK_BETA = 0.12f;
  static constexpr float GYRO_HEADING_SIGN = 1.0f;

  static const uint8_t MPU6050_ADDR_LOW = 0x68;
  static const uint8_t MPU6050_ADDR_HIGH = 0x69;

  uint8_t _mpu_addr;
  ImuState _state;
  uint32_t _last_update_us;
  uint32_t _calibration_start_ms;
  uint32_t _last_print_ms;

  float _gyro_offset_x;
  float _gyro_offset_y;
  float _gyro_offset_z;
  float _reference_heading_deg;
  float _relative_heading_deg;
  float _heading_candidate_deg;
  uint16_t _heading_stable_count;
  bool _heading_candidate_valid;

  float _roll_deg;
  float _pitch_deg;
  float _yaw_deg;

  float _q0;
  float _q1;
  float _q2;
  float _q3;

  bool findMpu6050();
  bool initMpu6050();
  bool readMpuRaw(MpuRaw &raw);
  void calibrateGyro();
  bool initializeOrientation();
  void updateStartupReference(float yaw_deg);
  void updateMadgwickImu(
    float gx,
    float gy,
    float gz,
    float ax,
    float ay,
    float az,
    float dt
  );
  void getEulerDegrees();
  void setQuaternionFromEuler(float roll, float pitch, float yaw);

  bool writeByte(uint8_t addr, uint8_t reg, uint8_t value);
  bool readBytes(uint8_t addr, uint8_t reg, uint8_t *data, uint8_t len);
  bool devicePresent(uint8_t addr);
  int16_t readS16BE(const uint8_t *data) const;
  float invSqrt(float value) const;
  float angleDiffDeg(float current, float reference) const;
  float wrapAngleDeg(float angle) const;
};

#endif
