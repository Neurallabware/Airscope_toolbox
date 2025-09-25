/*********
  Rui Santos
  Complete project details at https://RandomNerdTutorials.com/esp32-cam-take-photo-save-microsd-card
  
  IMPORTANT!!! 
   - Select Board "AI Thinker ESP32-CAM"
   - GPIO 0 must be connected to GND to upload a sketch
   - After connecting GPIO 0 to GND, press the ESP32-CAM on-board RESET button to put your board in flashing mode
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files.
  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.
*********/

#include "capture.h"
#include "periphery.h"

#include "soc/soc.h"           // Disable brownour problems
#include "soc/rtc_cntl_reg.h"  // Disable brownour problems
#include "driver/rtc_io.h"
#include <WiFi.h>

const char* ssid = "Redmi_1B3E";
const char* password = "ptbnxaxg";

void startCameraServer();
int capture_time = 5;
bool start_capture = false;


void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); //disable brownout detector
  Serial.begin(115200);
  
  init_writer(); // sd card writter
  init_camera();
  initI2C();
  get_initial_batt_value();
  init_wifi(ssid, password);
  start_capture_service(); // here we start the capture service, which is based on RTOS
  startCameraServer(); // we also start with web service
  Serial.print("Preview Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");  

}

void loop() {
  //TODO implement a shutdown feature
}
