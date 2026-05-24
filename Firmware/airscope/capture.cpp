#include "capture.h"
#include "periphery.h"
#include "config.h"
#include <WiFi.h>


int current_EEP = -1;
tm initial_time;
unsigned long initial_millis;
unsigned long capture_start_time;
float total_size;
float total_frames = -1;
File datafile; // global recording of the file data
File logfile; // global recording of the frame time
File IMUfile; // glocal recording of the IMU data

#define fbs 8 // was 64 -- how many kb of static ram for psram -> sram buffer for sd write
const int BUFFER_SIZE = 1024 * fbs + 20; // Adjust the buffer size as needed
const int BLOCK_SIZE = 1024 * fbs;
uint8_t framebuffer_static[BUFFER_SIZE];

const int IMU_BLOCK_SIZE = 1024;
static char IMU_buffer[IMU_BLOCK_SIZE]; // Buffer to accumulate data
static size_t buffer_index = 0; // Current position in the buffer

// --------------------------- EEPROM ---------------------------- //

void eep_init(){
    EEPROM.begin(EEPROM_SIZE);
    current_EEP = EEPROM.read(0);
    Serial.println("EEPROM initialized successfully");
  }
  
int eep_next(){
    return current_EEP + 1;
  }

void eep_update(){
    current_EEP++;
    EEPROM.write(0, current_EEP);
    EEPROM.commit();
  }
  
// ---------------------------- Init ----------------------------- //

void init_writer(){
    // Add Internal PU
    pinMode(13, INPUT_PULLUP);
    pinMode(2, INPUT_PULLUP);
    //SD card setup
    Serial.println("Starting SD Card");
    Serial.print("SD Card type: ");
    Serial.println(SD_MMC.cardType());
    
    if (!SD_MMC.begin()) {
        Serial.println("SD Card Mount Failed");
        return;
    }

    uint8_t cardType = SD_MMC.cardType();
    if (cardType == CARD_NONE) {
        Serial.println("No SD Card attached");
        return;
    }

    Serial.println("SD Card initialized successfully");

    eep_init();
}

void write_boot_log() {
    File f = SD_MMC.open("/log.txt", FILE_WRITE);
    if (!f) {
        Serial.println("Failed to open /log.txt for boot log");
        return;
    }

    uint8_t mac[6];
    WiFi.macAddress(mac);
    char mac_str[18];
    snprintf(mac_str, sizeof(mac_str), "%02X:%02X:%02X:%02X:%02X:%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

    f.printf("MAC: %s\n", mac_str);
    f.printf("DeviceName: %s\n", g_config.device_name.c_str());
    f.printf("WiFiSSID: %s\n", g_config.wifi_ssid.c_str());
    f.printf("ChipRev: %d\n", ESP.getChipRevision());
    f.printf("ChipCores: %d\n", ESP.getChipCores());
    f.printf("SDKVersion: %s\n", ESP.getSdkVersion());
    f.printf("FlashSize: %u\n", ESP.getFlashChipSize());
    f.printf("PsramSize: %u\n", ESP.getPsramSize());
    f.printf("SketchMD5: %s\n", ESP.getSketchMD5().c_str());
    f.printf("BootMillis: %lu\n", millis());
    f.flush();
    f.close();

    Serial.printf("Boot log written to /log.txt (MAC %s)\n", mac_str);
}

// -------------------------- Frame writing : to file, with buffer  --------------------------- /
// Function to write frame to SD card using buffer
 void write_frame_to_sd(File &file, camera_fb_t *fb) {
     int frame_len = fb->len;
     int remaining = frame_len;
     uint8_t *buffer = fb->buf;
     unsigned long iterStartTime; 
     while (remaining > 0) {

         iterStartTime = millis();
         
         int to_write = remaining > BLOCK_SIZE ? BLOCK_SIZE : remaining;
         memcpy(framebuffer_static, buffer, to_write);
         if (to_write < BLOCK_SIZE) {
            memset(framebuffer_static + to_write, 0, BLOCK_SIZE - to_write);
         }
         size_t written = file.write(framebuffer_static, BLOCK_SIZE);
         
//         if (written != to_write) {
//             Serial.println("Failed to write frame to file");
//             break;
//         }
         remaining -= to_write;
         buffer += to_write;
     }
     file.flush(); // Ensure data is written to the file
 }

//void write_frame_to_sd(File &file, camera_fb_t *fb) {
//
//
//    long startTime, iterStartTime;
//    startTime = millis();
//    for (int i = 0; i <10; i++){
//         iterStartTime = millis();
//        size_t error = file.write(framebuffer_static, 8 * 1024);    
//        Serial.printf("Iteration %d: Time taken = %ld ms\n", i + 1, millis() - iterStartTime);
//      }
//    file.flush(); // Ensure data is written to the file
//}

//void write_frame_to_sd(File &file, camera_fb_t *fb) {
//    long startTime, iterStartTime;
//    startTime = millis();
//
//    for (int i = 0; i < 10; i++) {
//
//        iterStartTime = millis();
//        size_t written = file.write(framebuffer_static, 8 * 1024);
//        long iterEndTime = millis();
//
//        Serial.printf("Iteration %d: Time taken = %ld ms\n", i + 1, iterEndTime - iterStartTime);
//        
//        if (written != 8 * 1024) {
//            Serial.println("Failed to write frame to file");
//            break;
//        }
//    }
//
//    iterStartTime = millis();
//    size_t written = file.write(framebuffer_static, 4 * 1024 + 35);
//    long iterEndTime = millis();
//
//    Serial.printf("Iteration %d: Time taken = %ld ms\n", 10, iterEndTime - iterStartTime);
//        
//
//    // Measure the flush time
//    long flushStartTime = millis();
//    file.flush(); // Ensure data is written to the file
//    long flushEndTime = millis();
//
//    Serial.printf("Flushing Time: %ld ms\n", flushEndTime - flushStartTime);
//}




// -------------------------- Frame writing: single frame to jpeg ---------------------- //
void write_frame_to_jpeg(String path, camera_fb_t * fb) {
  
  File file = SD_MMC.open(path.c_str(), FILE_WRITE);
  if(!file){
        Serial.println("Failed to open file in writing mode");
    } else {
        file.write(fb->buf, fb->len); // payload (image), payload length
        Serial.printf("Saved file to path: %s\n", path.c_str());
    }
   file.close();
  }

void test_single_frame_write() {
    camera_fb_t * fb = esp_camera_fb_get();
    if(!fb) {
        Serial.println("Camera capture failed");
        return;
    }
   Serial.printf("Size %d \n", fb->len);
   int pictureNumber = eep_next();
   String path = "/picture" + String(pictureNumber) + ".jpg";
   Serial.printf("Picture file name: %s\n", path.c_str());
  
   write_frame_to_jpeg(path, fb);
   eep_update();
   esp_camera_fb_return(fb);
   
  }

// -------------------------- Time and file handler --------------------------- //

void ntp_sync() {
    Serial.println("Beginning NTP synchronization...");
    configTime(8 * 3600, 0, ntpServer);
    if (!getLocalTime(&initial_time)) {
        Serial.println("Failed to obtain time");
        Serial.println("Will use dummy");
        ntp_sync_dummy();
        return;
    }
    initial_millis = millis();

    char timeStr[64];
    strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &initial_time);
    Serial.print("NTP synchronization completed. Initial time: ");
    Serial.println(timeStr);
    
    return;
}


void ntp_sync_dummy() {
    // Set initial_time to 2000-01-01 12:00:00
    initial_time.tm_year = 100; // years since 1900
    initial_time.tm_mon = 0;    // January
    initial_time.tm_mday = 1;
    initial_time.tm_hour = 12;
    initial_time.tm_min = 0;
    initial_time.tm_sec = 0;
    initial_time.tm_isdst = -1; // Not set by localtime
    // Reset initial_millis
    initial_millis = millis();
}

unsigned long time_past() {
    unsigned long current_millis = millis();
    unsigned long relative_millis = current_millis - initial_millis;
    return relative_millis;
  }

void write_timestamp() {
    unsigned long relative_millis = time_past();

    char time_str[32];
    snprintf(time_str, sizeof(time_str), "%lu\n", relative_millis);

    if (logfile.write((uint8_t*)time_str, strlen(time_str)) != strlen(time_str)) {
        Serial.println("Failed to write timestamp to log file");
        return;
    }
    logfile.flush(); // Ensure data is written to the file
}

// old version
// void write_IMU_value(File &file) {
//     unsigned long relative_millis = time_past();

//     // get those of values

//     update_IMU_MAG(); // 1. Pls check the time spent for executing this fucntion

//     unsigned long relative_millis2 = time_past();
//     char time_str[32];
//     snprintf(time_str, sizeof(time_str), "update_IMU_MAG takes %lu\n", relative_millis2 - relative_millis);
//     Serial.print(time_str);
//     // 

//     static char response[128]; // smaller size
//     // Format the response string as raw text. This is for all of the variables
//     snprintf(response, sizeof(response), "AccX: %.6f,AccY: %.6f,AccZ: %.6f,GyroX: %.6f,GyroY: %.6f,GyroZ: %.6f, MagX: %d, MagY %d, MagZ: %d,\n", \
//                                           Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z, magX, magY, magZ);
//     // snprintf(response, sizeof(response), "AccX: %.6f,AccY: %.6f,AccZ: %.6f, MagX: %d, MagY %d, MagZ: %d,\n", \
//     //                                       Accel_X, Accel_Y, Accel_Z, magX, magY, magZ); // 2. Add Gryo?
//     if (file.write((uint8_t*)response, strlen(response)) != strlen(response)) {
//         Serial.println("Failed to write IMU to log file");
//         return;
//     }

//     // Serial.println(response);
    
//     file.flush(); // Ensure data is written to the file # 
// }


// new version, block technique
void write_IMU_value(File &file) {
    unsigned long relative_millis = time_past();

    // Get IMU values (update_IMU_MAG remains the same)
    update_IMU_MAG();

    // Format data and add it to the buffer
    int bytes_written = snprintf(IMU_buffer + buffer_index, 
                                 IMU_BLOCK_SIZE - buffer_index, 
                                 "AccX: %.6f,AccY: %.6f,AccZ: %.6f,GyroX: %.6f,GyroY: %.6f,GyroZ: %.6f, MagX: %d, MagY %d, MagZ: %d,\n", 
                                 Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z, magX, magY, magZ);

    if (bytes_written < 0) {
        Serial.println("Error formatting IMU data");
        return;
    }

    buffer_index += bytes_written;

    // Check if buffer is almost full, then pad and write
    if (buffer_index >= IMU_BLOCK_SIZE - 128) { 
        memset(IMU_buffer + buffer_index, 0, IMU_BLOCK_SIZE - buffer_index); 

        if (file.write((uint8_t*)IMU_buffer, IMU_BLOCK_SIZE) != IMU_BLOCK_SIZE) {
            Serial.println("Failed to write padded IMU block to log file");
        } else {
            file.flush(); 
        }

        buffer_index = 0; 
    } 
}


char udpPacketBuffer[128];
unsigned long udpEchoTime;
WiFiUDP udp;
char udpServerIp[64] = "192.168.31.208"; // idle default; overwritten by GET /udpip from the host
char record_parent_name[64] = "rec";     // overwritten by GET /recordname?name=...

void udp_sync() {
  // Clear UDP packet buffer
  memset(udpPacketBuffer, 0, sizeof(udpPacketBuffer));

  Serial.println("UDP Sync");
  udp.flush();
  udp.begin(12345); // Ensure UDP is started on the desired port
  unsigned long started = millis();
  udp.beginPacket(udpServerIp, 12345);  // Replace with your server's IP and port number
  udp.print("TIME");
  udp.endPacket();
  
  // Wait for response
  while (!udp.parsePacket()) {
    delay(1);
    if (millis() - started > NTP_TIMEOUT) {
      udp.stop();  
      Serial.println("Timeout: Failed to get UDP time");
      strcpy(udpPacketBuffer, "/01-Jan-2000/00-00-00-000");
      unsigned long current_millis = millis();
      udpEchoTime = current_millis - started;
      initial_millis = millis();
      return;
    }
  }

  // Read response
  int len = udp.read(udpPacketBuffer, sizeof(udpPacketBuffer) - 1);
  initial_millis = millis();
  if (len > 0) {
    udpPacketBuffer[len] = '\0';
    Serial.print("Received time from server: ");
    Serial.println(udpPacketBuffer);
  } else {
    Serial.println("Timeout: Failed to get UDP time");
    strcpy(udpPacketBuffer, "/01-Jan-2000/00-00-00-000");
    unsigned long current_millis = millis();
    udpEchoTime = current_millis - started;
  }

  // Calculate synchronization time
  unsigned long current_millis = millis();
  udpEchoTime = current_millis - started;
}


void create_record_files() {
    // Creates the recording's folder + the data.dat / log.txt / IMU.txt files
    // inside it. Folder path is /<record_parent_name>/<udp_timestamp>/.
    udp_sync();

    char folder_path[192];
    snprintf(folder_path, sizeof(folder_path), "/%s%s",
             record_parent_name, udpPacketBuffer);

    if (!create_nested_directories(folder_path)) {
        Serial.print("Failed to create nested directories: ");
        Serial.println(folder_path);
        return;
    }

    // Create data file
    char datafile_path[256];
    snprintf(datafile_path, sizeof(datafile_path), "%s/data.dat", folder_path);
    datafile = SD_MMC.open(datafile_path, FILE_WRITE);
    if (!datafile) {
        Serial.println("Failed to open data file for writing");
        return;
    }

    // Create log file
    char logfile_path[256];
    snprintf(logfile_path, sizeof(logfile_path), "%s/log.txt", folder_path);
    logfile = SD_MMC.open(logfile_path, FILE_WRITE);
    if (!logfile) {
        Serial.println("Failed to open log file for writing");
        return;
    }

    // Create IMU file
    char IMUfile_path[256];
    snprintf(IMUfile_path, sizeof(IMUfile_path), "%s/IMU.txt", folder_path);
    IMUfile = SD_MMC.open(IMUfile_path, FILE_WRITE);
    if (!IMUfile) {
        Serial.println("Failed to open IMU file for writing");
        return;
    }

    // Write initial time
    logfile.printf("InitialTime: %s\n", udpPacketBuffer);
    logfile.printf("UDPEchoTime: %u\n", udpEchoTime);
    logfile.printf("InitialMillis: %u\n", initial_millis);
    logfile.flush();

    // Write sensor config
    sensor_t *s = esp_camera_sensor_get();
    if (s != NULL) {
        logfile.printf("xclk: %u\n", s->xclk_freq_hz);
        logfile.printf("pixformat: %u\n", s->pixformat);
        logfile.printf("special_effect: %u\n", s->status.special_effect);
        logfile.printf("framesize: %u\n", s->status.framesize);
        logfile.printf("quality: %u\n", s->status.quality);
        logfile.printf("dcw: %u\n", s->status.dcw);
        
        logfile.printf("brightness: %d\n", s->status.brightness);
        logfile.printf("contrast: %d\n", s->status.contrast);
        logfile.printf("saturation: %d\n", s->status.saturation);
        logfile.printf("sharpness: %d\n", s->status.sharpness);
        
        logfile.printf("wb_mode: %u\n", s->status.wb_mode);
        logfile.printf("awb: %u\n", s->status.awb);
        logfile.printf("awb_gain: %u\n", s->status.awb_gain);
        
        logfile.printf("aec: %u\n", s->status.aec);
        logfile.printf("aec2: %u\n", s->status.aec2);
        logfile.printf("ae_level: %d\n", s->status.ae_level);
        logfile.printf("aec_value: %u\n", s->status.aec_value);
        
        logfile.printf("agc: %u\n", s->status.agc);
        logfile.printf("agc_gain: %u\n", s->status.agc_gain);
        logfile.printf("gainceiling: %u\n", s->status.gainceiling);
        
        logfile.printf("bpc: %u\n", s->status.bpc);
        logfile.printf("wpc: %u\n", s->status.wpc);
        logfile.printf("raw_gma: %u\n", s->status.raw_gma);
        logfile.printf("lenc: %u\n", s->status.lenc);
        logfile.printf("hmirror: %u\n", s->status.hmirror);
        logfile.printf("colorbar: %u\n", s->status.colorbar);
        logfile.printf("led_intensity: %d\n", 0); // Assuming led_intensity is 0
    }

    logfile.flush();
    
}

// void create_record_files() {

//     char folder_path[64];
//     snprintf(folder_path, sizeof(folder_path), "/%04d-%02d-%02d_%02d-%02d-%02d", 
//             initial_time.tm_year + 1900, initial_time.tm_mon + 1, initial_time.tm_mday, 
//             initial_time.tm_hour, initial_time.tm_min, initial_time.tm_sec);

//     if (!create_nested_directories(folder_path)) {
//         Serial.print("Failed to create nested directories: ");
//         Serial.println(folder_path);
//         return;
//     }

//     // Create data file
//     char datafile_path[128];
//     snprintf(datafile_path, sizeof(datafile_path), "%s/data.dat", folder_path);
//     datafile = SD_MMC.open(datafile_path, FILE_WRITE);
//     if (!datafile) {
//         Serial.println("Failed to open data file for writing");
//         return;
//     }

//     // Create log file
//     char logfile_path[128];
//     snprintf(logfile_path, sizeof(logfile_path), "%s/log.txt", folder_path);
//     logfile = SD_MMC.open(logfile_path, FILE_WRITE);
//     if (!logfile) {
//         Serial.println("Failed to open log file for writing");
//         return;
//     }

//     // Write initial time
//     char timeStr[64];
//     strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &initial_time);
//     logfile.printf("InitialTime: %s\n", timeStr);
//     logfile.printf("InitialMillis: %u\n", initial_millis);
//     logfile.flush();

//     // Write sensor config
//     sensor_t *s = esp_camera_sensor_get();
//     if (s != NULL) {
//         logfile.printf("xclk: %u\n", s->xclk_freq_hz);
//         logfile.printf("pixformat: %u\n", s->pixformat);
//         logfile.printf("special_effect: %u\n", s->status.special_effect);
//         logfile.printf("framesize: %u\n", s->status.framesize);
//         logfile.printf("quality: %u\n", s->status.quality);
//         logfile.printf("dcw: %u\n", s->status.dcw);
        
//         logfile.printf("brightness: %d\n", s->status.brightness);
//         logfile.printf("contrast: %d\n", s->status.contrast);
//         logfile.printf("saturation: %d\n", s->status.saturation);
//         logfile.printf("sharpness: %d\n", s->status.sharpness);
        
//         logfile.printf("wb_mode: %u\n", s->status.wb_mode);
//         logfile.printf("awb: %u\n", s->status.awb);
//         logfile.printf("awb_gain: %u\n", s->status.awb_gain);
        
//         logfile.printf("aec: %u\n", s->status.aec);
//         logfile.printf("aec2: %u\n", s->status.aec2);
//         logfile.printf("ae_level: %d\n", s->status.ae_level);
//         logfile.printf("aec_value: %u\n", s->status.aec_value);
        
//         logfile.printf("agc: %u\n", s->status.agc);
//         logfile.printf("agc_gain: %u\n", s->status.agc_gain);
//         logfile.printf("gainceiling: %u\n", s->status.gainceiling);
        
//         logfile.printf("bpc: %u\n", s->status.bpc);
//         logfile.printf("wpc: %u\n", s->status.wpc);
//         logfile.printf("raw_gma: %u\n", s->status.raw_gma);
//         logfile.printf("lenc: %u\n", s->status.lenc);
//         logfile.printf("hmirror: %u\n", s->status.hmirror);
//         logfile.printf("colorbar: %u\n", s->status.colorbar);
//         logfile.printf("led_intensity: %d\n", 0); // Assuming led_intensity is 0
//     }

//     logfile.flush();
    
// }

bool create_nested_directories(const char *path) {
    char temp[256];
    char *pos = temp;
    // Copy the path to a temporary variable
    strncpy(temp,  path, sizeof(temp));
    temp[sizeof(temp) - 1] = '\0';
    // Iterate through the path and create each level of directory
    while (*pos) {
        if (*pos == '/') {
            *pos = '\0';
            if (!SD_MMC.exists(temp)) {
                if (!SD_MMC.mkdir(temp)) {
                    Serial.print("Failed to create folder: ");
                    Serial.println(temp);
                    return false;
                }
            }
            *pos = '/';
        }
        pos++;
    }
    // Create the final directory
    if (!SD_MMC.exists(temp)) {
        if (!SD_MMC.mkdir(temp)) {
            Serial.print("Failed to create folder: ");
            Serial.println(temp);
            return false;
        }
    }
    return true;
}

void record_init() {
     // sync time
     Serial.println("initial time");
     // create record files
     Serial.println("create record files");
     create_record_files();

     Serial.println("init global variables");
     // init global variables
     total_size = 0;
     total_frames = 0;
     capture_start_time = millis();
  }

void record_end(){
    // modify this with correspondance of create_record_files
    // TODO add IMU and MAG file
    unsigned long total_capture_time = millis() - capture_start_time;
    float fps = total_frames / (total_capture_time / 1000.0);
    float data_rate = total_size / (total_capture_time / 1000.0); // KB per second
 
    char end_log_str[256];
    snprintf(end_log_str, sizeof(end_log_str), "Total capture time: %lu ms, Total frames: %.2f, Total size: %.2f KB, FPS: %.2f, Data rate: %.2f KB/s\n", 
             total_capture_time, total_frames, total_size, fps, data_rate);
    Serial.println(end_log_str);
    logfile.write((const uint8_t*)end_log_str, strlen(end_log_str));
    logfile.flush();

    datafile.close();
    logfile.close();
    IMUfile.close();
    total_frames = -1;
  }

// ----------------------------- record without RTOS ------------------------- //

void record_single_frame() {
    unsigned long frame_start_time = millis(); // Start time for capturing the frame
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        return;
    }
    unsigned long frame_captured_time = millis(); // Calculate capture time
    
    write_frame_to_sd(datafile, fb);
    unsigned long frame_written_time = millis(); // Calculate SD time

    // Calculate times
    unsigned long capture_time = frame_captured_time - frame_start_time;
    unsigned long sd_write_time = frame_written_time - frame_captured_time;
    unsigned long total_time = frame_written_time - frame_start_time;
    unsigned long relative_mills = frame_captured_time - initial_millis;
    float frame_size_kb = fb->len / 1024.0;

    total_size += frame_size_kb;
    total_frames += 1;
    
    // log
    char log_str[256];
    snprintf(log_str, sizeof(log_str), "Size: %.2f KB, Capture time: %lu ms, SD write time: %lu ms, Total time: %lu ms, Time: %lu\n", 
             frame_size_kb, capture_time, sd_write_time, total_time, relative_mills);
    Serial.print(log_str);
    logfile.write((const uint8_t*)log_str, strlen(log_str));
    logfile.flush();


    // IMU
    write_IMU_value(IMUfile);

    esp_camera_fb_return(fb);    
  }

void record_frames(int seconds) {

    record_init();
    unsigned long start_time = millis();
    unsigned long end_time = start_time + seconds * 1000;
    
    while (millis() < end_time) {
        record_single_frame(); 
    }
    record_end();
}

// ------------------------- RTOS capture service ----------------------------- //
static SemaphoreHandle_t capture_ready;
static SemaphoreHandle_t frame_ready;

TaskHandle_t the_camera_loop_task;
TaskHandle_t the_sd_loop_task;

camera_fb_t * fb_curr = NULL;
bool recording_on = false;
// control the acqusition state based on recording_on
void camera_loop(void *pvParameters) {
    Serial.print("the camera loop, core ");  Serial.print(xPortGetCoreID());
    Serial.print(", priority = "); Serial.println(uxTaskPriorityGet(NULL));
    while(1) {
//      Serial.printf("Inside camera loop");
//      Serial.printf("Total frames %d\n", total_frames);
//      if (recording_on) {
//        Serial.println("recording on");
//        }
      if (total_frames < 0 && recording_on == false){
        // IDLE
        delay(100);
//        Serial.println("IDLE...");
        }
      else if (total_frames < 0 && recording_on == true){
        // START
//        Serial.println("start!");
        record_init();
        Serial.println("recording started!");
        xSemaphoreGive(capture_ready); 
        }
      else if (total_frames >= 0 && recording_on == true){
        // RECORDING
      //  Serial.println("another_frame");
        record_another_frame();
        }
      else if (total_frames >= 0 && recording_on == false){
        // END
        xSemaphoreTake(capture_ready, portMAX_DELAY);
        record_end();
        Serial.println("recording stopped!");
        }
      }
  }

void record_another_frame(){
    xSemaphoreTake( capture_ready, portMAX_DELAY );
    fb_curr = esp_camera_fb_get();  
    if (!fb_curr) {
        return;
    }
    xSemaphoreGive( frame_ready);
  }

void sd_loop(void *pvParameters){
    Serial.print("the_sd_loop, core ");  Serial.print(xPortGetCoreID());
    Serial.print(", priority = "); Serial.println(uxTaskPriorityGet(NULL));
    while(1) {

        xSemaphoreTake(frame_ready , portMAX_DELAY );            // a new frame was captured in camera loop

        // write to SD
        long start = millis();
        write_frame_to_sd(datafile, fb_curr);                       // do the actual sd wrte

        // logs
        write_timestamp();
        write_IMU_value(IMUfile);
        total_frames += 1;
        total_size += fb_curr->len/1024;
        long end_t = millis();
        Serial.printf("Start: %ld ms, End: %ld ms, Exec: %ld ms, Frame size: %d KB\n", start, end_t, end_t - start, fb_curr->len / 1024);

        // return fb
        esp_camera_fb_return(fb_curr);
        
        // tell the camera loop to prepare a new frame
        xSemaphoreGive(capture_ready);                     // tell camera loop we are done
      }
  }

void start_capture_service() {
    
    frame_ready = xSemaphoreCreateBinary(); 
    capture_ready = xSemaphoreCreateBinary();

    recording_on = false;
    total_frames = -1;
    // prio 6 - higher than the camera loop(), and the streaming
    xTaskCreatePinnedToCore(camera_loop, "the_camera_loop", 10000, NULL, 6, &the_camera_loop_task, 0); // prio 3, core 0 //v56 core 1 as http dominating 0 ... back to 0, raise prio
    // used a deep stack to prevent stackoverflow. TODO: Check how much is used in each step
    delay(100);

    // prio 4 - higher than the cam_loop(), and the streaming
    xTaskCreatePinnedToCore( sd_loop, "the_sd_loop", 10000, NULL, 4, &the_sd_loop_task, 1);  // prio 4, core 1
    

}
