#include <Wire.h>
#include <Adafruit_ADS1X15.h>

Adafruit_ADS1015 ads;

// ---------------- configuration ----------------
const uint32_t  SERIAL_BAUD = 460800;   // raised: 115200 can't carry ~1000 text lines/s
const adsGain_t ADS_GAIN    = GAIN_SIXTEEN;   // ±0.256V (0.125 mV/bit)
const uint32_t  I2C_HZ      = 400000;   // 400 kHz I2C (4x faster transactions)
const unsigned long SAMPLE_INTERVAL_US = 1000;  // 1000 us -> ~1000 Hz

// 0 = print millivolts (default, Main.py-compatible: time_ms,voltage_mV)
// 1 = print raw signed ADC counts instead (changes format -> Main.py needs adjusting)
#define OUTPUT_RAW_COUNTS 0

unsigned long lastSample = 0;

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial) { delay(1); }

  Serial.println("Geophone - Direct Differential (A0-A1), continuous ~1kHz");
  Wire.begin(4, 5);          // SDA=GPIO4(D2), SCL=GPIO5(D1)
  Wire.setClock(I2C_HZ);     // faster I2C

  if (!ads.begin(0x48)) {
    Serial.println("ERROR: ADS1015 not found!");
    while (1) { delay(100); }
  }

  ads.setGain(ADS_GAIN);
  ads.setDataRate(RATE_ADS1015_3300SPS);   // run the chip fast (3300 SPS)

  // Start CONTINUOUS differential conversions on A0(+)-A1(-).
  // After this, each sample is one fast read with no per-sample reconfigure.
  ads.startADCReading(ADS1X15_REG_CONFIG_MUX_DIFF_0_1, /*continuous=*/true);
  delay(2);  // let the first conversion complete

#if OUTPUT_RAW_COUNTS
  Serial.println("time_ms,raw_counts");
#else
  Serial.println("time_ms,voltage_mV");
#endif
}

void loop() {
  unsigned long now = micros();
  if (now - lastSample >= SAMPLE_INTERVAL_US) {
    lastSample = now;

    int16_t raw = ads.getLastConversionResults();   // fast read (continuous mode)

    Serial.print(millis());        // ms timestamp (kept for Main.py compatibility)
    Serial.print(",");
#if OUTPUT_RAW_COUNTS
    Serial.println(raw);
#else
    Serial.println(raw * 0.125, 3);   // millivolts
#endif
  }
}
