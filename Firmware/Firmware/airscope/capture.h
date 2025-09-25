#ifndef CAPTURE_H
#define CAPTURE_H

#include "SD_MMC.h"            // SD Card ESP32
#include "Arduino.h"
#include "periphery.h"
#include "time.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <EEPROM.h> 
#include <WiFiUdp.h>

// eeprom
#define EEPROM_SIZE 1
#define NTP_TIMEOUT 1000
extern int current_EEP; 
void eep_init(); // init eep function
int eep_next(); // get next eep value
void eep_update(); // update eep value (++)

// file writing related functions
void init_writer();
bool create_nested_directories(const char *path);
void write_frame_to_sd(File &file, camera_fb_t *fb);
void write_frame_to_jpeg(String path, camera_fb_t * fb);

// Time related functions
extern tm initial_time;
extern unsigned long initial_mills;
extern File datafile;
extern File logfile;
constexpr const char* ntpServer = "106.55.184.199"; // deprecated
extern char udpServerIp[64]; // now we set udpServer

void udp_sync();
void ntp_sync(); // init initial_time and initial mills
void ntp_sync_dummy(); // pretend to initial time (to a fixed time 2000.1.1 12:00, not through WIFI)
void write_timestamp(); // log timestamp to logfile
void create_record_files(); // create the folder, datafile and logfile of current recording according to the initial time

// core record function, wo RTOS
void record_frames(int seconds); // capture for n seconds
void test_single_frame_write();

// RTOS record function
extern bool recording_on;
extern float total_frames;
extern TaskHandle_t the_camera_loop_task;
extern TaskHandle_t the_sd_loop_task;
void sd_loop(void *pvParameters); 
void camera_loop(void *pvParameters);
void record_another_frame();
void start_capture_service();
unsigned long time_past();

#endif
