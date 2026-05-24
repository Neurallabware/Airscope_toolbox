#include "config.h"
#include "FS.h"
#include "SD_MMC.h"
#include <ArduinoJson.h>   // install via Library Manager: "ArduinoJson"
#include <WiFi.h>

AirscopeConfig g_config;

static const char* CONFIG_PATH = "/config.json";

String config_default_device_name() {
    uint8_t mac[6];
    WiFi.macAddress(mac);
    char buf[12];
    snprintf(buf, sizeof(buf), "def-%02X%02X", mac[4], mac[5]);
    return String(buf);
}

void config_load_defaults() {
    // Hardcoded fallbacks. These are what runs if SD is missing or
    // config.json is unreadable. Default to the open institutional IoT
    // SSID so a freshly flashed board with no SD card still associates
    // on the lab network.
    //
    // Only seven fields are persisted to /config.json (see
    // config_load_from_sd / config_save_to_sd below): the network creds,
    // the device name, and the four most-tweaked camera knobs (quality,
    // gain, exposure, LED). Everything else here is hardcoded firmware
    // baseline that the host can still override at runtime via /control;
    // those tweaks just don't survive a reboot.
    g_config.wifi_ssid           = "servicenet";
    g_config.wifi_password       = "";
    g_config.device_name         = config_default_device_name();
    g_config.default_capture_time = 5;

    g_config.framesize        = 13;    // UXGA (1600x1200)
    g_config.quality          = 6;
    g_config.brightness       = 0;
    g_config.contrast         = 0;
    g_config.saturation       = 0;
    g_config.special_effect   = 2;     // grayscale
    g_config.wb_mode          = 0;
    g_config.awb              = 0;
    g_config.awb_gain         = 0;
    g_config.aec              = 0;
    g_config.aec2             = 0;
    g_config.ae_level         = 0;
    g_config.aec_value        = 1200;
    g_config.agc              = 0;
    g_config.agc_gain         = 1;
    g_config.gainceiling      = 2;
    g_config.bpc              = 1;
    g_config.wpc              = 1;
    g_config.raw_gma          = 0;
    g_config.lenc             = 0;
    g_config.hmirror          = 0;
    g_config.vflip            = 1;
    g_config.dcw              = 0;
    g_config.led_intensity    = 0;
}

bool config_load_from_sd() {
    if (!SD_MMC.exists(CONFIG_PATH)) {
        Serial.printf("[config] %s not present, using defaults\n", CONFIG_PATH);
        return false;
    }
    File f = SD_MMC.open(CONFIG_PATH, FILE_READ);
    if (!f) {
        Serial.printf("[config] failed to open %s\n", CONFIG_PATH);
        return false;
    }
    StaticJsonDocument<2048> doc;
    DeserializationError err = deserializeJson(doc, f);
    f.close();
    if (err) {
        Serial.printf("[config] parse error: %s, using defaults\n", err.c_str());
        return false;
    }

    bool got_any = false;
#define LOAD_STR(name) \
    if (doc.containsKey(#name)) { g_config.name = doc[#name].as<const char*>(); got_any = true; }
#define LOAD_INT(name) \
    if (doc.containsKey(#name)) { g_config.name = doc[#name].as<int>(); got_any = true; }

    // Only seven keys are recognized. Anything else in the file is ignored.
    LOAD_STR(wifi_ssid);
    LOAD_STR(wifi_password);
    LOAD_STR(device_name);
    LOAD_INT(quality);
    LOAD_INT(agc_gain);     // "gain" in user-facing docs
    LOAD_INT(aec_value);    // "exposure" in user-facing docs
    LOAD_INT(led_intensity);

#undef LOAD_STR
#undef LOAD_INT

    Serial.printf("[config] loaded from SD (ssid=%s, device=%s)\n",
                  g_config.wifi_ssid.c_str(), g_config.device_name.c_str());
    return got_any;
}

bool config_save_to_sd() {
    StaticJsonDocument<512> doc;
    // Only seven persisted keys. The rest live in firmware defaults.
    doc["wifi_ssid"]     = g_config.wifi_ssid;
    doc["wifi_password"] = g_config.wifi_password;
    doc["device_name"]   = g_config.device_name;
    doc["quality"]       = g_config.quality;
    doc["agc_gain"]      = g_config.agc_gain;       // "gain"
    doc["aec_value"]     = g_config.aec_value;      // "exposure"
    doc["led_intensity"] = g_config.led_intensity;

    File f = SD_MMC.open(CONFIG_PATH, FILE_WRITE);
    if (!f) {
        Serial.printf("[config] save: failed to open %s for write\n", CONFIG_PATH);
        return false;
    }
    if (serializeJsonPretty(doc, f) == 0) {
        Serial.println("[config] save: serializeJsonPretty wrote 0 bytes");
        f.close();
        return false;
    }
    f.close();
    Serial.println("[config] saved to SD");
    return true;
}
