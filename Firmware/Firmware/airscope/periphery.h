#ifndef PERIPHERY_H
#define PERIPHERY_H

#include <Wire.h>
#include "esp_camera.h"
#include "Arduino.h"
#include "soc/soc.h" 
#include <WiFi.h>

// IMU address
const int LSM6DSL_ADDR=0x6A;
const int OUTX_L_XL = 0x28; // AccelX. AccelY would be 0x2A. AccelZ would be 0x2C
const int OUTX_L_G = 0x22; // RawGyroX. RawGyroY would be 0x24. RawGyroZ would be 0x26
const int WHO_AM_I_REG = 0x0F;
const int CTRL1_XL = 0x10;
const int CTRL2_G = 0x11;

// MAG address
const int QMC6310_ADDR = 0x1C;  // QMC6310U I2C address
const int QMC6310_REG_OUT_X_L = 0x00;
const int QMC6310_REG_OUT_X_H =0x01;
const int QMC6310_REG_OUT_Y_L =0x02;
const int QMC6310_REG_OUT_Y_H =0x03;
const int QMC6310_REG_OUT_Z_L =0x04;
const int QMC6310_REG_OUT_Z_H =0x05;
const int QMC6310_REG_STATUS =0x06;
const int QMC6310_REG_TEMP_L =0x07;
const int QMC6310_REG_TEMP_H =0x08;
const int QMC6310_REG_CONTROL1 =0x09;
const int QMC6310_REG_CONTROL2 = 0x0A;
const int QMC6310_REG_RESET = 0x0B;
const int QMC6310_REG_AXIS_SIGN = 0x29;


// Temperature sensor
const int tempSensorAddress = 0x48;
const int ledAddress = 0x64;
const int sensorPin = 33;
const int i2cSDA = 22;
const int i2cSCL = 19;

extern bool led_on;
extern float batt_value;
extern float temp_value;

// IMU value
extern float Accel_X;
extern float Accel_Y;
extern float Accel_Z;
extern float Gyro_X;
extern float Gyro_Y;
extern float Gyro_Z;

// MAG value
extern int16_t magX;
extern int16_t magY;
extern int16_t magZ;

// I2C
void initI2C();
// Temp
void configTempSensor(byte msb, byte lsb);
float readTempValue();
void checkTempSensorConfig();
//IMU
void configIMU();
void writeRegister_IMU(uint8_t reg, uint8_t value);
void readRegister_IMU(uint8_t reg, uint8_t* buffer, uint8_t len);

int16_t readAxis(uint8_t reg);
int16_t readRawAccelX();
int16_t readRawAccelY();
int16_t readRawAccelZ();
int16_t readRawGyroX();
int16_t readRawGyroY();
int16_t readRawGyroZ();
float readFloatAccelX();
float readFloatAccelY();
float readFloatAccelZ();
float readFloatGyroX();
float readFloatGyroY();
float readFloatGyroZ();
float convertAccel(int16_t axisValue);
float convertGyro(int16_t axisValue);

//MAG

void configMAG();
float readMagData_X();
float readMagData_Y();
float readMagData_Z();
void writeRegister_MAG(uint8_t reg, uint8_t value);
uint8_t readRegister_MAG(uint8_t reg);
int16_t readRegister16(uint8_t regL, uint8_t regH);
bool waitForDataReady(uint16_t timeout);
bool checkStatus();

// LED
void enableLED();
void setLedValue(int val);
// Batt
float readBattValue();
void get_initial_batt_value();
// Camera
void init_camera();
void test_capture();
// Wifi
void init_wifi(const char* ssid, const char* password);
// Loop
void update_tb();
void update_IMU_MAG();
// Pins define
// #define CAMERA_MODEL_AI_THINKER
#define CAMERA_MODEL_PICO_11

#if defined(CAMERA_MODEL_AI_THINKER)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Pico v1.1
#elif defined(CAMERA_MODEL_PICO_11)
#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    25
#define SIOD_GPIO_NUM    7
#define SIOC_GPIO_NUM    8
#define VSYNC_GPIO_NUM   20
#define HREF_GPIO_NUM    27 
#define PCLK_GPIO_NUM    0 

#define Y9_GPIO_NUM      32
#define Y8_GPIO_NUM      35
#define Y7_GPIO_NUM      26
#define Y6_GPIO_NUM      34
#define Y5_GPIO_NUM      38
#define Y4_GPIO_NUM      39
#define Y3_GPIO_NUM      37 
#define Y2_GPIO_NUM      36 
#endif

#endif
