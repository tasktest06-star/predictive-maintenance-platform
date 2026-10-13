# ESP-IDF component build for the ESP32-S3 node.
#
#   . $IDF_PATH/export.sh
#   idf.py set-target esp32s3
#   idf.py build flash monitor
#
# Default sensor profile is the ADXL355. For the wideband profile:
#   idf.py build -DPDM_SENSOR=IIS3DWB
# and read hardware/esp32s3/README.md first -- the two parts are not
# substitutes and the choice changes what this node can diagnose.

set(CDSP ${CMAKE_CURRENT_LIST_DIR}/../../../cdsp)

idf_component_register(
    SRCS
        main.c
        board_esp32s3.c
        mqtt_transport.c
        nvs_queue.c
        ${CDSP}/src/pdm_app.c
        ${CDSP}/src/pdm_pipeline.c
        ${CDSP}/src/pdm_payload.c
        ${CDSP}/src/pdm_window.c
        ${CDSP}/src/pdm_fft.c
        ${CDSP}/src/pdm_spectrum.c
        ${CDSP}/src/pdm_features.c
        ${CDSP}/src/pdm_envelope.c
        ${CDSP}/src/pdm_bearing.c
        ${CDSP}/src/pdm_model.c
    INCLUDE_DIRS
        "."
        "model"
        "${CDSP}/include"
    REQUIRES
        driver esp_adc esp_timer esp_wifi esp_event nvs_flash mqtt esp-tls
        esp_hw_support freertos
)

if(PDM_SENSOR STREQUAL "IIS3DWB")
    target_compile_definitions(${COMPONENT_LIB} PRIVATE PDM_SENSOR_IIS3DWB=1)
else()
    target_compile_definitions(${COMPONENT_LIB} PRIVATE PDM_SENSOR_ADXL355=1)
endif()

target_compile_definitions(${COMPONENT_LIB} PRIVATE PDM_WANT_FEATURE_NAMES)

target_compile_options(${COMPONENT_LIB} PRIVATE
    -Os -Wall -Wextra -Wdouble-promotion -Wconversion -Wshadow
    # The LX7 has a single-precision FPU. -Wdouble-promotion matters for the
    # same reason it does on the Cortex-M kits: an accidental promotion drops
    # into software emulation. ESP-IDF's default flags do not include it.
    -ffunction-sections -fdata-sections
)
