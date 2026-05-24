/*********
  Airscope v2 firmware - NO-BLE variant.

  Same as airscope_v2 except BLE is stripped. Used to isolate whether the
  MJPEG stream instability seen on Princeton's Wi-Fi comes from BLE
  (compile-time linkage of the BT controller and/or its deinit leaving the
  radio in a degraded coexistence state) or from the ESP32 core 2.0.17
  itself / network. Keep the same FQBN + core so BLE-presence is the only
  variable.

  Board: AI Thinker ESP32-CAM (PICO_11 pin map in periphery.h).
  Dependencies: ArduinoJson (Library Manager).
*********/

#include "capture.h"
#include "periphery.h"
#include "config.h"

#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "driver/rtc_io.h"
#include <WiFi.h>

void startCameraServer();
int  capture_time = 5;          // overridden from config in setup()
bool start_capture = false;

static void apply_camera_defaults_from_config();

void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    Serial.begin(115200);

    // SD must come up first so config_load_from_sd() has somewhere to read.
    init_writer();

    // Load defaults, then layer SD config on top (if available).
    config_load_defaults();
    config_load_from_sd();
    capture_time = g_config.default_capture_time;

    // Write /log.txt at SD root so the MAC address is easy to find for pairing.
    write_boot_log();

    init_camera();
    apply_camera_defaults_from_config();

    initI2C();
    get_initial_batt_value();

    init_wifi(g_config.wifi_ssid.c_str(), g_config.wifi_password.c_str());

    start_capture_service();   // RTOS capture loop
    startCameraServer();       // HTTP + MJPEG

    Serial.print("Preview Ready! Use 'http://");
    Serial.print(WiFi.localIP());
    Serial.print("' to connect. Device name: ");
    Serial.println(g_config.device_name);
}

void loop() {
    delay(100);
}

// Push the config's camera defaults into the running sensor.
static void apply_camera_defaults_from_config() {
    sensor_t *s = esp_camera_sensor_get();
    if (!s) return;

    s->set_framesize(s,      (framesize_t)g_config.framesize);
    s->set_quality(s,        g_config.quality);
    s->set_brightness(s,     g_config.brightness);
    s->set_contrast(s,       g_config.contrast);
    s->set_saturation(s,     g_config.saturation);
    s->set_special_effect(s, g_config.special_effect);
    s->set_wb_mode(s,        g_config.wb_mode);
    s->set_whitebal(s,       g_config.awb);
    s->set_awb_gain(s,       g_config.awb_gain);
    s->set_exposure_ctrl(s,  g_config.aec);
    s->set_aec2(s,           g_config.aec2);
    s->set_ae_level(s,       g_config.ae_level);
    s->set_aec_value(s,      g_config.aec_value);
    s->set_gain_ctrl(s,      g_config.agc);
    s->set_agc_gain(s,       g_config.agc_gain);
    s->set_gainceiling(s,    (gainceiling_t)g_config.gainceiling);
    s->set_bpc(s,            g_config.bpc);
    s->set_wpc(s,            g_config.wpc);
    s->set_raw_gma(s,        g_config.raw_gma);
    s->set_lenc(s,           g_config.lenc);
    s->set_hmirror(s,        g_config.hmirror);
    s->set_vflip(s,          g_config.vflip);
    s->set_dcw(s,            g_config.dcw);

    if (g_config.led_intensity > 0) {
        enableLED();
        setLedValue(g_config.led_intensity);
    }
}
