/* pdm_payload.h -- the wire format, C side.
 *
 * Byte-for-byte twin of pdmcore.io.payload. Any change here needs the same
 * change there and a bump of PDM_PAYLOAD_SCHEMA_VERSION in both -- the ingest
 * service rejects versions it does not know rather than guessing a layout, so a
 * mismatch takes the fleet dark rather than making it silently wrong. That is
 * the correct failure direction and still an outage.
 */

#ifndef PDM_PAYLOAD_H
#define PDM_PAYLOAD_H

#include <stddef.h>
#include <stdint.h>

#include "pdm_dsp.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PDM_PAYLOAD_SCHEMA_VERSION 3u
#define PDM_PAYLOAD_HEADER_SIZE    44u

typedef struct {
    uint64_t device_id;
    uint64_t timestamp_ms;
    uint32_t uptime_s;
    uint16_t battery_mv;
    int16_t  rssi_dbm;
    float    shaft_hz;
    float    temperature_c;
    uint16_t flags;          /* PDM_FRAME_* from pdm_pipeline.h */
    uint16_t sequence;       /* wraps at 16 bits; used for gap detection */
} pdm_payload_meta;

/* Encoded size for n features: 44 + 4n + 4. */
size_t pdm_payload_size(uint16_t n_features);

/* Pack a frame. Returns PDM_ERR_SIZE if `cap` is too small. */
pdm_status pdm_payload_encode(const pdm_payload_meta *meta,
                              const float *features, uint16_t n_features,
                              uint8_t *out, size_t cap, size_t *written);

/* CRC-32 (IEEE 802.3), identical to zlib.crc32. */
uint32_t pdm_crc32(const uint8_t *data, size_t len);

#ifdef __cplusplus
}
#endif
#endif /* PDM_PAYLOAD_H */
