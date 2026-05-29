#include "Arduino.h"
#include "periphery.h"

bool led_on = false;
float batt_value = 0;
float temp_value = 0;

// Updated in configIMU() / configMAG() based on whether the chip ACKs.
bool has_imu = false;
bool has_mag = false;

float Accel_X = 0;
float Accel_Y = 0;
float Accel_Z = 0;
float Gyro_X = 0;
float Gyro_Y = 0;
float Gyro_Z = 0;

int16_t magX = 0;
int16_t magY = 0;
int16_t magZ = 0;


// ------------------- Server ---------------------- // 
void update_tb(){
  batt_value = readBattValue();  // get the battery value
  temp_value = readTempValue();  // get the temperature value
}

void update_IMU_MAG(){
  // If the chip isn't on the bus, leave the globals at their previous
  // value (init 0). Skipping the reads avoids the per-frame I2C traffic
  // and the "Error: No data available" spam.
  if (has_imu) {
    Accel_X = readFloatAccelX();
    Accel_Y = readFloatAccelY();
    Accel_Z = readFloatAccelZ();
    Gyro_X  = readFloatGyroX();
    Gyro_Y  = readFloatGyroY();
    Gyro_Z  = readFloatGyroZ();
  }
  if (has_mag) {
    magX = readMagData_X();
    magY = readMagData_Y();
    magZ = readMagData_Z();
  }
}

void get_initial_batt_value(){
  batt_value = readBattValue();
  Serial.printf("Initial battery value: %.2f", batt_value);
}

// -------------------- I2C ------------------------- //
void initI2C() {
  Wire.begin(i2cSDA, i2cSCL); // Initialize I2C bus with SDA pin 22 and SCL pin 19


  // configTempSensor

  // config IMU
  configIMU();

  // config MAG
  configMAG();
}

// -------------------- Temp ------------------------- //
void configTempSensor(byte msb, byte lsb) {
  Wire.beginTransmission(tempSensorAddress);
  Wire.write(0x01); // Configuration register address
  Wire.write(0x00); // MSB
  Wire.write(lsb); // LSB
  Wire.endTransmission();
}

void checkTempSensorConfig() {
  Wire.beginTransmission(tempSensorAddress);
  Wire.write(0x01); // 配置寄存器地址
  Wire.endTransmission();

  // 从配置寄存器读取2字节数据
  Wire.requestFrom(tempSensorAddress, 2);
  if (Wire.available() == 2) {
    byte msb = Wire.read(); // 读取配置寄存器的高位字节
    byte lsb = Wire.read(); // 读取配置寄存器的低位字节

    // 打印配置寄存器的内容
    Serial.print("Configuration Register MSB: ");
    Serial.print(msb, BIN); // 以二进制形式打印
    Serial.print(" LSB: ");
    Serial.println(lsb, BIN); // 以二进制形式打印
  }
}

float readTempValue() {
  Wire.beginTransmission(tempSensorAddress);
  Wire.write(0x00); // Temperature register address
  Wire.endTransmission();

  Wire.requestFrom(tempSensorAddress, 2);
  if (Wire.available() == 2) {
    byte msb = Wire.read();
    byte lsb = Wire.read();
    // TMP102: 12-bit two's-complement value, left-justified in a 16-bit
    // register (top 12 bits = T, bottom 4 = 0). The previous code did
    // `(raw << 1) >> 5` which discards the sign bit and then shoves
    // bit 14 *into* the sign slot — so any temp >=64C (bit 14 set)
    // came out as a large negative number.
    int16_t raw = (int16_t)((msb << 8) | lsb);
    int16_t t12 = raw >> 4;   // arithmetic shift sign-extends 12->16 bit
    return t12 * 0.0625f;
  }
  return 0.0; // Return 0 if temperature reading fails
}

// -------------------- IMU ------------------------- //
void configIMU() {
    // Probe the bus first. Wire.endTransmission() returns 0 only when the
    // slave actually ACKs; on this PICO_11 board IMU/MAG are optional, so
    // a missing chip should leave the firmware otherwise functional.
    Wire.beginTransmission(LSM6DSL_ADDR);
    if (Wire.endTransmission() != 0) {
        has_imu = false;
        Serial.println("[imu] LSM6DSL @ 0x6A not detected -- accel/gyro will be 0");
        return;
    }
    has_imu = true;
    writeRegister_IMU(CTRL1_XL, 0x60); // ODR = 1.66 kHz, 2g full scale, 400 Hz filter
    writeRegister_IMU(CTRL2_G,  0x60); // ODR = 1.66 kHz, 2000 dps
    Serial.println("[imu] LSM6DSL detected");
}

void writeRegister_IMU(uint8_t reg, uint8_t value) {
    Wire.beginTransmission(LSM6DSL_ADDR);
    Wire.write(reg);
    Wire.write(value);
    Wire.endTransmission();
}

void readRegister_IMU(uint8_t reg, uint8_t* buffer, uint8_t len) {
    Wire.beginTransmission(LSM6DSL_ADDR);
    Wire.write(reg);
    Wire.endTransmission();
    Wire.requestFrom(LSM6DSL_ADDR, len);
    for (uint8_t i = 0; i < len; i++) {
        buffer[i] = Wire.read();
    }
}

int16_t readAxis(uint8_t reg) {
    int16_t value;
    readRegister_IMU(reg, (uint8_t*)&value, 2);
    return value;
}

int16_t readRawAccelX() {
    return readAxis(OUTX_L_XL);
}

int16_t readRawAccelY() {
    return readAxis(OUTX_L_XL + 2);
}

int16_t readRawAccelZ() {
    return readAxis(OUTX_L_XL + 4);
}

int16_t readRawGyroX() {
    return readAxis(OUTX_L_G);
}

int16_t readRawGyroY() {
    return readAxis(OUTX_L_G + 2);
}

int16_t readRawGyroZ() {
    return readAxis(OUTX_L_G + 4);
}

float readFloatAccelX() {
    return convertAccel(readRawAccelX());
}

float readFloatAccelY() {
    return convertAccel(readRawAccelY());
}

float readFloatAccelZ() {
    return convertAccel(readRawAccelZ());
}

float readFloatGyroX() {
    return convertGyro(readRawGyroX());
}

float readFloatGyroY() {
    return convertGyro(readRawGyroY());
}

float readFloatGyroZ() {
    return convertGyro(readRawGyroZ());
}

float convertAccel(int16_t axisValue) {
    // Assuming full scale = ±2g
    return axisValue * 0.000061; // 0.061 mg/LSB
}

float convertGyro(int16_t axisValue) {
    // Assuming full scale = ±2000 dps
    return axisValue * 0.07; // 70 mdps/LSB
}

// -------------------- MAG ------------------------- //

void configMAG() {
    // Same address-probe pattern as configIMU.
    Wire.beginTransmission(QMC6310_ADDR);
    if (Wire.endTransmission() != 0) {
        has_mag = false;
        Serial.println("[mag] QMC6310 @ 0x1C not detected -- mag will be 0");
        return;
    }
    has_mag = true;
    writeRegister_MAG(QMC6310_REG_AXIS_SIGN, 0x06);
    writeRegister_MAG(QMC6310_REG_RESET,     0x08);   // 8 Gauss FS
    writeRegister_MAG(QMC6310_REG_CONTROL2,  0xCD);   // normal, ODR=200 Hz

    uint8_t control1 = readRegister_MAG(QMC6310_REG_CONTROL1);
    uint8_t control2 = readRegister_MAG(QMC6310_REG_CONTROL2);
    uint8_t axisSign = readRegister_MAG(QMC6310_REG_AXIS_SIGN);
    Serial.printf("[mag] QMC6310 detected: ctrl1=0x%02X ctrl2=0x%02X axis=0x%02X\n",
                  control1, control2, axisSign);
}

float readMagData_X() {
        return readRegister16(QMC6310_REG_OUT_X_L, QMC6310_REG_OUT_X_H);  
}

float readMagData_Y() {
        return readRegister16(QMC6310_REG_OUT_Y_L, QMC6310_REG_OUT_Y_H);
}

float readMagData_Z() {
        return readRegister16(QMC6310_REG_OUT_Z_L, QMC6310_REG_OUT_Z_H);
}

void writeRegister_MAG(uint8_t reg, uint8_t value) {
    Wire.beginTransmission(QMC6310_ADDR);
    Wire.write(reg);
    Wire.write(value);
    Wire.endTransmission();
}

uint8_t readRegister_MAG(uint8_t reg) {
    // Bypass the bus entirely when the device isn't present, so the
    // recording loop's per-frame mag reads don't print 6 lines of
    // "Error: No data available" per frame.
    if (!has_mag) return 0;

    Wire.beginTransmission(QMC6310_ADDR);
    Wire.write(reg);
    Wire.endTransmission();

    Wire.requestFrom(QMC6310_ADDR, 1);
    if (Wire.available()) {
        return Wire.read();
    } else {
        Serial.println("Error: No data available");
        return 0;
    }
}

int16_t readRegister16(uint8_t regL, uint8_t regH) {
    uint8_t low = readRegister_MAG(regL);
    uint8_t high = readRegister_MAG(regH);
    return (high << 8) | low;
}
bool waitForDataReady(uint16_t timeout) {
    uint32_t start = millis();
    while (millis() - start < timeout) {
        if (checkStatus()) {
            return true;
        }
        delay(1); // Adjust delay as needed for responsiveness
    }
    return false;
}

bool checkStatus() {
    uint8_t status = readRegister_MAG(QMC6310_REG_STATUS);
    return status & 0x01;
}

// -------------------- LED ------------------------- //
void enableLED(){
    Wire.beginTransmission(ledAddress); // device address  TPL04001-A #46 (0x2E) ,  TPL04001-B #46 (0x2E)
    Wire.write(byte(0x01));     // sends instruction.  0x00 = Write
    Wire.write(byte(0x22));            // sends value
    Wire.endTransmission();     // end transmission
    led_on = true;
}

void setLedValue(int val) {
  Wire.beginTransmission(ledAddress);
  Wire.write(0x04); // Register address for setting torch value
  Wire.write(val);  // Value to set
  Wire.endTransmission();
}

// -------------------- Batt ------------------------- //
float readBattValue() {
  float sensorValue = analogRead(sensorPin);
  return sensorValue * 4.0 / 2250.0;
}


// ------------------- WIFI --------------------- //
void init_wifi(const char* ssid, const char* password) {
    // Start WiFi connection
    WiFi.begin(ssid, password);
    WiFi.setSleep(false);

    // Wait until the device is connected to the WiFi network
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    // Connection successful
    Serial.println("");
    Serial.println("WiFi connected");
}


// -------------------- Sensor --------------------- //

void init_camera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;  
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;

    config.xclk_freq_hz = 16000000;
    config.frame_size = FRAMESIZE_UXGA; // only defined here
    config.pixel_format = PIXFORMAT_JPEG; // for streaming
    config.jpeg_quality = 6;

    config.grab_mode = CAMERA_GRAB_LATEST;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.fb_count = 2;
    
    // camera init
    Serial.println("Before Camera Init");
    Serial.printf("Internal Total heap %d, internal Free Heap %d\n", ESP.getHeapSize(), ESP.getFreeHeap());
    Serial.printf("SPIRam Total heap   %d, SPIRam Free Heap   %d\n", ESP.getPsramSize(), ESP.getFreePsram());

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x", err);
        return;
    }
    Serial.println("Camera init success!");
    Serial.println("After Camera Init");
    Serial.printf("Internal Total heap %d, internal Free Heap %d\n", ESP.getHeapSize(), ESP.getFreeHeap());
    Serial.printf("SPIRam Total heap   %d, SPIRam Free Heap   %d\n", ESP.getPsramSize(), ESP.getFreePsram());


    // sensor prop init
    sensor_t * s = esp_camera_sensor_get();
   s->set_vflip(s, 1); // flip it back
   s->set_brightness(s, 0);                  // -2 to 2
   s->set_contrast(s, 0);                    // -2 to 2
   s->set_saturation(s, 0);                  // -2 to 2
   s->set_special_effect(s, 2);              // Special effect.  0 - None, 1 - Negative, 2 - Grayscale, 3 - Red Tint, 4 - Green Tint, 5 - Blue Tint, 6 - Sepia
   
   s->set_whitebal(s, 0);                    // 0 = disable , 1 = enable
   s->set_awb_gain(s, 0);                    // 0 = disable , 1 = enable
   s->set_wb_mode(s, 0);                     // 0 to 4 - if awb_gain enabled (0 - Auto, 1 - Sunny, 2 - Cloudy, 3 - Office, 4 - Home)

   s->set_bpc(s, 1);                         // 0 = disable , 1 = enable
   s->set_wpc(s, 1);                         // 0 = disable , 1 = enable
   s->set_raw_gma(s, 0);                     // 0 = disable , 1 = enable
   s->set_lenc(s, 0);                        // 0 = disable , 1 = enable
   s->set_hmirror(s, 0);                     // Horizontal mirror.  0 = disable , 1 = enable
   s->set_dcw(s, 0);                         // 0 = disable , 1 = enable
  
    s->set_exposure_ctrl(s, 0);               // 0 = disable , 1 = enable
    s->set_aec2(s, 0);                        // 0 = disable , 1 = enable
    s->set_ae_level(s, 0);                    // -2 to 2
    s->set_aec_value(s, 1200);                 // 0 to 1200

   s->set_gain_ctrl(s, 0);                   // 0 = disable , 1 = enable
   s->set_agc_gain(s, 1);                    // 0 to 30
   s->set_gainceiling(s, (gainceiling_t)2);  // 0 to 6

}

void set_camera_quality(int quality){
    sensor_t * s = esp_camera_sensor_get();
    s->set_quality(s, quality);
    delay(500);
    Serial.printf("Camera quality set to %d. \n", quality);
  }

void test_capture(){
  
  camera_fb_t * fb = NULL;
  fb = esp_camera_fb_get();  
  if(!fb) {
    Serial.println("Camera capture failed");
    return;
  }
  Serial.println("Fb get");
  Serial.printf("Internal Total heap %d, internal Free Heap %d\n", ESP.getHeapSize(), ESP.getFreeHeap());
  Serial.printf("SPIRam Total heap   %d, SPIRam Free Heap   %d\n", ESP.getPsramSize(), ESP.getFreePsram());
  Serial.println(fb->len);
  Serial.println(fb->width);
  Serial.println(fb->height);
  esp_camera_fb_return(fb); 
  Serial.println("Fb return");
  Serial.printf("Internal Total heap %d, internal Free Heap %d\n", ESP.getHeapSize(), ESP.getFreeHeap());
  Serial.printf("SPIRam Total heap   %d, SPIRam Free Heap   %d\n", ESP.getPsramSize(), ESP.getFreePsram());
  
}