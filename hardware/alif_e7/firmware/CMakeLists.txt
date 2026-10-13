# Alif Ensemble E7, HE core.
#
#   cmake -B build -DCMAKE_TOOLCHAIN_FILE=$ALIF_SDK/cmake/toolchain-arm-none-eabi.cmake \
#         -DALIF_SDK=$ALIF_SDK -DALIF_CORE=HE
#   cmake --build build -j
#   $ALIF_SDK/app-release-exec/app-write-mram -p
#
# Enable the experimental waveform model with -DAPP_WAVEFORM_MODEL=1. Read the
# README first: it is an experiment, and leaving it off is what keeps this kit's
# frames comparable with the other seven.

cmake_minimum_required(VERSION 3.20)
project(pdm_alif_e7 C ASM)
set(CMAKE_C_STANDARD 99)

if(NOT DEFINED ALIF_SDK)
    message(FATAL_ERROR "Set -DALIF_SDK=/path/to/alif_ensemble-camera-app or the "
                        "Ensemble SDK root (not vendored; registration required)")
endif()

set(CDSP ${CMAKE_CURRENT_SOURCE_DIR}/../../../cdsp)
option(APP_WAVEFORM_MODEL "Enable the Ethos-U55 waveform model" OFF)

add_executable(pdm_alif_e7
    main.c
    board_alif_e7.c
    ble_transport.c
    mram_queue.c
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
if(APP_WAVEFORM_MODEL)
    target_sources(pdm_alif_e7 PRIVATE waveform_model.c model/waveform_vela.c)
    target_compile_definitions(pdm_alif_e7 PRIVATE APP_WAVEFORM_MODEL=1)
endif()

target_include_directories(pdm_alif_e7 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/model
    ${CDSP}/include
    ${ALIF_SDK}/Device/common/include ${ALIF_SDK}/Device/core/include
    ${ALIF_SDK}/drivers/include ${ALIF_SDK}/se_services/include
    ${ALIF_SDK}/THIRD_PARTY/CMSIS/CMSIS/Core/Include
    ${ALIF_SDK}/THIRD_PARTY/CMSIS-NN/Include
)

target_compile_definitions(pdm_alif_e7 PRIVATE
    PDM_TABLES_IN_FLASH=1 PDM_WANT_FEATURE_NAMES
    M55_HE ARM_MATH_MVEF   # Helium: CMSIS-DSP uses the MVE paths
)

target_compile_options(pdm_alif_e7 PRIVATE
    -mcpu=cortex-m55 -mthumb -mfloat-abi=hard -mfpu=auto
    # -mfpu=auto, not fpv5-sp-d16: it is what enables the MVE (Helium) vector
    # unit. Without it the compiler emits scalar code, CMSIS-DSP silently falls
    # back to its generic paths, and the FFT takes twice as long -- a 2x
    # regression that produces identical numbers and is therefore invisible to
    # every test in this repository.
    -Os -ffunction-sections -fdata-sections
    -Wall -Wextra -Wdouble-promotion -Wconversion -Wshadow
)

target_link_options(pdm_alif_e7 PRIVATE
    -mcpu=cortex-m55 -mthumb -mfloat-abi=hard -mfpu=auto
    -T${CMAKE_CURRENT_SOURCE_DIR}/he_mram.ld
    -Wl,--gc-sections -Wl,-Map=pdm_alif_e7.map
    -specs=nano.specs -specs=nosys.specs
)
target_link_libraries(pdm_alif_e7 PRIVATE m)
