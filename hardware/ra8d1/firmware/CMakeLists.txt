# Renesas EK-RA8D1 node.
#
# The FSP build is normally driven from e2 studio, which generates ra_cfg/,
# ra_gen/ and the linker script from the project's .xml configuration. This
# CMake file builds the application sources against an FSP tree that already
# exists, which is what a CI runner can do; it does not replace the configurator.
#
#   cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/arm-none-eabi.cmake \
#         -DFSP_ROOT=$HOME/ra8d1_project
#   cmake --build build -j

cmake_minimum_required(VERSION 3.20)
project(pdm_ra8d1 C ASM)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(FSP_ROOT "" CACHE PATH "e2 studio project root containing ra/, ra_gen/, ra_cfg/")
if(NOT FSP_ROOT)
    message(FATAL_ERROR
        "FSP_ROOT is not set. The Renesas FSP is ~400 MB and is not vendored here; "
        "generate a project in e2 studio for the EK-RA8D1 and point FSP_ROOT at it. "
        "To check this kit's sources without the FSP, run tools/syntax_check.sh, "
        "which compiles them against a stub.")
endif()

set(PDM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

# -------------------------------------------------------------------- flags --
# -mcpu=cortex-m85 selects Helium (MVE) automatically; -mfpu=auto lets the
# compiler use the whole single-precision MVE unit rather than the FPv5 subset.
#
# Repeating the Alif branch's finding because it applies identically here:
# -mfpu=fpv5-sp-d16 compiles, links, produces bit-identical results and passes
# every test in this repository while running the FFT about twice as slow,
# because it hides the vector unit. There is no test that can catch it. It is a
# review item, which is why it is written down here.
set(ARCH_FLAGS
    -mcpu=cortex-m85
    -mfpu=auto
    -mfloat-abi=hard
    -mthumb)

# -Wdouble-promotion is not optional: the M85's FPU and Helium are
# single-precision. One accidental promotion drops into software double
# arithmetic -- the numbers stay right, only the energy budget breaks, so
# nothing fails and nobody notices.
set(WARN_FLAGS
    -Wall -Wextra -Werror -Wpedantic -Wshadow -Wconversion
    -Wdouble-promotion -Wstrict-prototypes -Wmissing-prototypes
    -Wpointer-arith -Wundef)

# ------------------------------------------------------------------ sources --
add_executable(pdm_ra8d1
    main.c
    board_ra8d1.c
    pdm_order.c
    ${PDM_ROOT}/cdsp/src/pdm_dsp.c
    ${PDM_ROOT}/cdsp/src/pdm_filter.c
    ${PDM_ROOT}/cdsp/src/pdm_features.c
    ${PDM_ROOT}/cdsp/src/pdm_model.c
    ${PDM_ROOT}/cdsp/src/pdm_payload.c
    ${PDM_ROOT}/cdsp/src/pdm_pipeline.c
    ${PDM_ROOT}/cdsp/src/pdm_app.c
    # Board-specific and NOT written: see the README's honest-status table.
    # Listed so the gap shows up as a link error rather than hiding behind a
    # stub that returns success.
    ble_transport.c
    flash_queue.c
    ra8_platform.c)

target_include_directories(pdm_ra8d1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/model
    ${PDM_ROOT}/cdsp/include
    ${FSP_ROOT}/ra_gen
    ${FSP_ROOT}/ra_cfg/fsp_cfg
    ${FSP_ROOT}/ra/fsp/inc
    ${FSP_ROOT}/ra/fsp/inc/api
    ${FSP_ROOT}/ra/fsp/inc/instances)

target_compile_options(pdm_ra8d1 PRIVATE ${ARCH_FLAGS} ${WARN_FLAGS} -Os -ffunction-sections -fdata-sections)
target_link_options(pdm_ra8d1 PRIVATE
    ${ARCH_FLAGS}
    -T${FSP_ROOT}/script/fsp.ld
    -Wl,--gc-sections
    -Wl,-Map=pdm_ra8d1.map
    --specs=nosys.specs)
target_link_libraries(pdm_ra8d1 PRIVATE m)

# The config checker runs before the build, not after: a config error produces
# a firmware that works and is wrong, which is the expensive kind.
add_custom_target(check_config ALL
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PDM_ROOT}/core:${PDM_ROOT}/tools
            python3 ${CMAKE_CURRENT_SOURCE_DIR}/../tools/check_config.py
    COMMENT "cross-checking app_config.h against the Python reference")
add_dependencies(pdm_ra8d1 check_config)

add_custom_command(TARGET pdm_ra8d1 POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:pdm_ra8d1> pdm_ra8d1.bin
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:pdm_ra8d1>)
