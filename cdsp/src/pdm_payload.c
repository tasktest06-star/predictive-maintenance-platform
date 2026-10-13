/* pdm_payload.c -- the C half of the wire format.
 *
 * Byte-for-byte twin of pdmcore.io.payload. 44-byte header, n float32 features,
 * CRC-32. Little-endian on the wire, which every target here already is, but
 * the encoder writes bytes explicitly rather than memcpy-ing a struct: struct
 * padding and endianness are exactly the two things that make a wire format
 * work on the developer's machine and fail on the device.
 *
 * The CRC is computed even though the link is usually TLS. It catches corruption
 * that happens *before* transmission -- a brown-out during the DMA copy out of
 * the accelerometer FIFO, or a flash-wear bit flip in the store-and-forward
 * queue. TLS protects the link, not the device.
 */

#include "pdm_payload.h"

#include <string.h>

/* CRC-32 (IEEE 802.3), the same polynomial and bit order as zlib.crc32, which
 * is what the Python side uses. Bitwise rather than table-driven: 1 kB of table
 * is 0.4 % of a 256 kB part's RAM, and at 300 bytes per frame once an hour the
 * ~8x speed difference is 30 microseconds nobody will ever measure. */
static uint32_t crc32_update(uint32_t crc, const uint8_t *data, size_t len)
{
    crc = ~crc;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int b = 0; b < 8; ++b) {
            crc = (crc >> 1) ^ (0xEDB88320u & (uint32_t)(-(int32_t)(crc & 1u)));
        }
    }
    return ~crc;
}

uint32_t pdm_crc32(const uint8_t *data, size_t len)
{
    return crc32_update(0u, data, len);
}

static void put_u8(uint8_t *p, size_t *o, uint8_t v)
{
    p[(*o)++] = v;
}

static void put_u16(uint8_t *p, size_t *o, uint16_t v)
{
    p[(*o)++] = (uint8_t)(v & 0xFFu);
    p[(*o)++] = (uint8_t)((v >> 8) & 0xFFu);
}

static void put_u32(uint8_t *p, size_t *o, uint32_t v)
{
    for (int i = 0; i < 4; ++i) {
        p[(*o)++] = (uint8_t)((v >> (8 * i)) & 0xFFu);
    }
}

static void put_u64(uint8_t *p, size_t *o, uint64_t v)
{
    for (int i = 0; i < 8; ++i) {
        p[(*o)++] = (uint8_t)((v >> (8 * i)) & 0xFFu);
    }
}

static void put_f32(uint8_t *p, size_t *o, float v)
{
    /* Type-punning through a union rather than a pointer cast: a cast from
     * float* to uint32_t* is a strict-aliasing violation that GCC is entitled
     * to miscompile at -O2, and does. */
    union { float f; uint32_t u; } conv;
    conv.f = v;
    put_u32(p, o, conv.u);
}

size_t pdm_payload_size(uint16_t n_features)
{
    return PDM_PAYLOAD_HEADER_SIZE + 4u * (size_t)n_features + 4u;
}

pdm_status pdm_payload_encode(const pdm_payload_meta *meta,
                              const float *features, uint16_t n_features,
                              uint8_t *out, size_t cap, size_t *written)
{
    if (meta == NULL || features == NULL || out == NULL || written == NULL) {
        return PDM_ERR_ARG;
    }
    const size_t need = pdm_payload_size(n_features);
    if (cap < need) {
        return PDM_ERR_SIZE;
    }

    size_t o = 0;
    out[o++] = 'P';
    out[o++] = 'D';
    out[o++] = 'M';
    out[o++] = '3';
    put_u8(out, &o, PDM_PAYLOAD_SCHEMA_VERSION);
    put_u8(out, &o, (uint8_t)PDM_FEATURE_SPEC_VERSION);
    put_u16(out, &o, n_features);
    put_u64(out, &o, meta->device_id);
    put_u64(out, &o, meta->timestamp_ms);
    put_u32(out, &o, meta->uptime_s);
    put_u16(out, &o, meta->battery_mv);
    put_u16(out, &o, (uint16_t)(int16_t)meta->rssi_dbm);
    put_f32(out, &o, meta->shaft_hz);
    put_f32(out, &o, meta->temperature_c);
    put_u16(out, &o, meta->flags);
    put_u16(out, &o, meta->sequence);
    if (o != PDM_PAYLOAD_HEADER_SIZE) {
        /* The header layout and the declared size have diverged. A runtime
         * check rather than a static assert because C99 has no static_assert;
         * it costs one comparison per frame and prevents a whole fleet from
         * emitting undecodable data. */
        return PDM_ERR_SIZE;
    }

    for (uint16_t i = 0; i < n_features; ++i) {
        put_f32(out, &o, features[i]);
    }
    const uint32_t crc = pdm_crc32(out, o);
    put_u32(out, &o, crc);

    *written = o;
    return PDM_OK;
}
