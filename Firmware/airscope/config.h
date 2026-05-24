#ifndef CONFIG_H
#define CONFIG_H

#include "Arduino.h"

// On-SD config layout: /config.json on the SD card root. Read once at boot,
// rewritten when the host calls POST /config (Save Config in the UI).
//
// Anything missing from the file falls back to the hardcoded DEFAULT_*
// constants in config.cpp, so a boot with no SD card or a corrupt file still
// comes up. Host IP for UDP sync is NOT stored here — the host pushes its
// current IP via /udpip after /whoami on every connect.

struct AirscopeConfig {
    String wifi_ssid;
    String wifi_password;
    String device_name;       // e.g. "scope-A", or "def-XXXX" if unconfigured
    int    default_capture_time;  // seconds

    // Camera defaults
    int framesize;
    int quality;
    int brightness;
    int contrast;
    int saturation;
    int special_effect;
    int wb_mode;
    int awb;
    int awb_gain;
    int aec;
    int aec2;
    int ae_level;
    int aec_value;
    int agc;
    int agc_gain;
    int gainceiling;
    int bpc;
    int wpc;
    int raw_gma;
    int lenc;
    int hmirror;
    int vflip;
    int dcw;
    int led_intensity;
};

extern AirscopeConfig g_config;

// Fills g_config with hardcoded defaults (always safe to call).
void config_load_defaults();

// Attempts /config.json from SD; merges over current g_config. Returns true
// if at least one field was loaded from disk.
bool config_load_from_sd();

// Writes current g_config back to /config.json. Returns true on success.
bool config_save_to_sd();

// Returns "def-XXXX" where XXXX = last 2 bytes of MAC (uppercase hex).
// Stable across reboots.
String config_default_device_name();

#endif
