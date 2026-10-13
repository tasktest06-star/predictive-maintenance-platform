# nRF Connect SDK application.
#
#   west build -b nrf9160dk_nrf9160_ns hardware/nrf9160/firmware
#   west flash
#
# The _ns (non-secure) target is required: the modem library runs in the secure
# domain behind TF-M, and a build against the secure target will link but will
# not be able to reach the modem.

cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(pdm_nrf9160)

set(CDSP ${CMAKE_CURRENT_SOURCE_DIR}/../../../cdsp)

target_sources(app PRIVATE
    main.c
    board_nrf9160.c
    lte_transport.c
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
)

target_include_directories(app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/model
    ${CDSP}/include
)

target_compile_definitions(app PRIVATE
    # 46 kB of window/twiddle tables move from RAM to flash. On 256 kB that is
    # the difference between the pipeline fitting comfortably and not fitting.
    # Regenerate the tables with:
    #   make -C cdsp tables N_FFT=4096 ENV_N_FFT=1024 \
    #       TABLES=hardware/nrf9160/firmware/model/pdm_tables.h
    PDM_TABLES_IN_FLASH=1
    PDM_WANT_FEATURE_NAMES
)

target_compile_options(app PRIVATE
    -Os -Wall -Wextra -Wdouble-promotion -Wconversion -Wshadow
    # -Wdouble-promotion matters more here than anywhere: the M33 runs at 64 MHz,
    # the slowest core in the lineup, so a 4096-point FFT is already ~8 ms and
    # software double emulation would put it past 300 ms -- longer than the
    # acquisition it is processing.
)
