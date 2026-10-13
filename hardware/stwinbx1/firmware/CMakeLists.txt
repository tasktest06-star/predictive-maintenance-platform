# STEVAL-STWINBX1 firmware.
#
#   cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/arm-none-eabi.cmake \
#         -DSTM32CUBE_U5=/path/to/STM32CubeU5
#   cmake --build build -j
#   make -C .. flash
#
# The STM32CubeU5 package is NOT vendored: it is ~1.5 GB and its licence does
# not permit redistribution. The path is a required cache variable so the build
# fails with a clear message rather than a hundred missing headers.

cmake_minimum_required(VERSION 3.20)
project(pdm_stwinbx1 C ASM)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT DEFINED STM32CUBE_U5)
    message(FATAL_ERROR
        "Set -DSTM32CUBE_U5=/path/to/STM32CubeU5 (download from st.com; not "
        "vendored here, ~1.5 GB and not redistributable)")
endif()

set(CDSP ${CMAKE_CURRENT_SOURCE_DIR}/../../../cdsp)

add_executable(pdm_stwinbx1
    main.c
    board_stwinbx1.c
    board_init.c
    ble_transport.c
    flash_queue.c
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

target_include_directories(pdm_stwinbx1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/model
    ${CDSP}/include
    ${STM32CUBE_U5}/Drivers/STM32U5xx_HAL_Driver/Inc
    ${STM32CUBE_U5}/Drivers/CMSIS/Device/ST/STM32U5xx/Include
    ${STM32CUBE_U5}/Drivers/CMSIS/Include
)

target_compile_definitions(pdm_stwinbx1 PRIVATE
    STM32U585AI USE_HAL_DRIVER
    # Pull the feature names into flash so the boot-time cross-check against
    # the exported model can report WHICH feature disagrees, not just that the
    # count is wrong. 1.4 kB of .rodata for a diagnosable failure.
    PDM_WANT_FEATURE_NAMES
)

target_compile_options(pdm_stwinbx1 PRIVATE
    -mcpu=cortex-m33 -mthumb -mfpu=fpv5-sp-d16 -mfloat-abi=hard
    -Os -ffunction-sections -fdata-sections -fno-common
    -Wall -Wextra -Wdouble-promotion -Wconversion -Wshadow
    # -Wdouble-promotion is not optional on this part: the M33 FPU is
    # single-precision only, so an accidental promotion drops into software
    # emulation and a 4096-point FFT goes from ~1.1 ms to ~40 ms.
    $<$<CONFIG:Debug>:-g3>
)

target_link_options(pdm_stwinbx1 PRIVATE
    -mcpu=cortex-m33 -mthumb -mfpu=fpv5-sp-d16 -mfloat-abi=hard
    -T${CMAKE_CURRENT_SOURCE_DIR}/STM32U585AIIXQ_FLASH.ld
    -Wl,--gc-sections -Wl,-Map=pdm_stwinbx1.map
    -specs=nano.specs -specs=nosys.specs
    -u _printf_float
)

target_link_libraries(pdm_stwinbx1 PRIVATE m)

# Report flash and RAM use at the end of every build. The numbers in
# hardware/stwinbx1/README.md come from here, and a build that silently grows
# past the SRAM1 budget is a build that fails on the bench, not at compile time.
add_custom_command(TARGET pdm_stwinbx1 POST_BUILD
    COMMAND ${CMAKE_SIZE} -A -d $<TARGET_FILE:pdm_stwinbx1>
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:pdm_stwinbx1> pdm_stwinbx1.bin
    COMMAND ${CMAKE_OBJCOPY} -O ihex   $<TARGET_FILE:pdm_stwinbx1> pdm_stwinbx1.hex
)
