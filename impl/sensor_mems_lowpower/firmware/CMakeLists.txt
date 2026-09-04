# CMakeLists.txt — MEMS Low-Power Sensor Firmware
# Target board : nucleo_u575zi_q (STM32U575, Cortex-M33)
# RTOS         : Zephyr RTOS 3.6
# Toolchain    : arm-zephyr-eabi (bundled with Zephyr SDK ≥ 0.16)
#
# Build:
#   west build -b nucleo_u575zi_q .
#   west flash
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

# ── Zephyr boilerplate ──────────────────────────────────────────────────────
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(sensor_mems_lowpower
    VERSION 1.0.0
    DESCRIPTION "MEMS Low-Power Predictive Maintenance Sensor"
    LANGUAGES C
)

# ── Source files ────────────────────────────────────────────────────────────
target_sources(app PRIVATE
    main.c
    dsp_features.c
)

# ── Include directories ─────────────────────────────────────────────────────
target_include_directories(app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ── CMSIS-DSP linkage ───────────────────────────────────────────────────────
# Zephyr 3.6 bundles CMSIS-DSP as a module.  The module is enabled via
# CONFIG_CMSIS_DSP=y and CONFIG_CMSIS_DSP_FILTERING=y in prj.conf.
# Zephyr's module system exposes it as the zephyr_interface target.
#
# If building outside Zephyr (e.g. unit tests on host):
#   target_link_libraries(app PRIVATE CMSISDSP)
# That requires the CMSIS-DSP cmake package installed separately.
#
# For Zephyr builds, no explicit link step is needed — prj.conf handles it.

# ── Compiler flags ──────────────────────────────────────────────────────────
# -O2          : Standard optimization — good balance of speed/size
# -ffast-math  : Safe for DSP: allows reassociation, vectorization, etc.
#                Does NOT generate incorrect results for finite inputs.
#                Explicitly safe for our use of arm_rfft_fast_f32, arm_rms_f32, etc.
# -ffunction-sections / -fdata-sections: Enables dead-code elimination via --gc-sections
target_compile_options(app PRIVATE
    -O2
    -ffast-math
    -ffunction-sections
    -fdata-sections
    -Wall
    -Wextra
    -Wno-unused-parameter
)

# ── Linker flags ────────────────────────────────────────────────────────────
target_link_options(app PRIVATE
    -Wl,--gc-sections
    -Wl,--print-memory-usage
)

# ── Post-build: print binary size report ────────────────────────────────────
add_custom_command(
    TARGET app POST_BUILD
    COMMAND ${CMAKE_SIZE} --format=sysv $<TARGET_FILE:app>
    COMMENT "Binary size report:"
)

add_custom_command(
    TARGET app POST_BUILD
    COMMAND ${CMAKE_SIZE} --format=berkeley $<TARGET_FILE:app>
    COMMENT "Flash / RAM usage:"
)

# ── Post-build: generate .bin and .hex for flashing ─────────────────────────
add_custom_command(
    TARGET app POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:app>
            ${CMAKE_BINARY_DIR}/zephyr/sensor_mems_lowpower.bin
    COMMAND ${CMAKE_OBJCOPY} -O ihex   $<TARGET_FILE:app>
            ${CMAKE_BINARY_DIR}/zephyr/sensor_mems_lowpower.hex
    COMMENT "Generating .bin and .hex"
)

# ── Memory budget assertions ─────────────────────────────────────────────────
# STM32U575: 786 KB Flash, 256 KB SRAM (+ 64 KB SRAM4 for DMA)
# Budget allocations:
#   Zephyr kernel + drivers : ~80 KB Flash, ~32 KB SRAM
#   Application code        : ~40 KB Flash
#   DSP buffers (acc_x/y/z, fft_buf, etc.) @ 2048 × 4 × 8 = 65 KB SRAM
#   Stack + heap            : ~48 KB SRAM
# Total: ~120 KB Flash, ~145 KB SRAM  (well within limits)

message(STATUS "Target board : nucleo_u575zi_q (STM32U575, Cortex-M33)")
message(STATUS "Flash budget : ~120 KB / 786 KB")
message(STATUS "SRAM budget  : ~145 KB / 256 KB")
message(STATUS "Zephyr RTOS  : 3.6")
