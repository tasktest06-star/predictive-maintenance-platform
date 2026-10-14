/* hal_data.h -- a STUB of the Renesas FSP surface this kit's driver uses.
 *
 * NOT the vendor header. e2 studio generates the real hal_data.h from the
 * project configuration, and it is specific to the pin and peripheral choices
 * made there. This file exists so board_ra8d1.c can be syntax-checked on a
 * machine with no Renesas toolchain -- which is this machine, and probably the
 * CI runner too.
 *
 * What it proves: the driver's types, signatures, struct members and control
 * flow are self-consistent, and it compiles clean under the same
 * -Werror -Wconversion -Wdouble-promotion set as the rest of the firmware.
 *
 * What it does NOT prove: that any register value, clock setting, DMA
 * configuration or pin assignment is correct. Only hardware proves that. See
 * `tools/syntax_check.sh` and the kit README's honest-status table.
 *
 * Every declaration below is transcribed from the FSP 5.x API documentation.
 * If a signature here disagrees with the real header the stub build will pass
 * and the real build will fail, which is the right way round.
 */

#ifndef HAL_DATA_STUB_H
#define HAL_DATA_STUB_H

#include <stdint.h>

typedef enum {
    FSP_SUCCESS = 0,
    FSP_ERR_ASSERTION,
    FSP_ERR_TIMEOUT,
    FSP_ERR_ABORTED,
    FSP_ERR_NOT_OPEN
} fsp_err_t;

/* --- BSP ---------------------------------------------------------------- */
typedef enum { BSP_DELAY_UNITS_SECONDS, BSP_DELAY_UNITS_MILLISECONDS,
               BSP_DELAY_UNITS_MICROSECONDS } bsp_delay_units_t;
typedef enum { BSP_WARM_START_RESET, BSP_WARM_START_POST_CLOCK,
               BSP_WARM_START_POST_C } bsp_warm_start_event_t;
void R_BSP_SoftwareDelay(uint32_t delay, bsp_delay_units_t units);
void R_BSP_WarmStart(bsp_warm_start_event_t event);

/* --- IOPORT ------------------------------------------------------------- */
typedef enum { BSP_IO_LEVEL_LOW = 0, BSP_IO_LEVEL_HIGH = 1 } bsp_io_level_t;
typedef enum {
    BSP_IO_PORT_04_PIN_06 = 0x0406,
    BSP_IO_PORT_06_PIN_00 = 0x0600
} bsp_io_port_pin_t;
typedef struct { void *p_ctrl; } ioport_instance_ctrl_t;
extern ioport_instance_ctrl_t g_ioport_ctrl;
fsp_err_t R_IOPORT_PinWrite(ioport_instance_ctrl_t *p_ctrl,
                            bsp_io_port_pin_t pin, bsp_io_level_t level);

/* --- SPI (SCI_B in simple SPI mode) ------------------------------------- */
typedef enum { SPI_BIT_WIDTH_8_BITS = 0, SPI_BIT_WIDTH_16_BITS = 1 } spi_bit_width_t;
typedef enum {
    SPI_EVENT_TRANSFER_COMPLETE = 1,
    SPI_EVENT_TRANSFER_ABORTED  = 2
} spi_event_t;
typedef struct { void *p_reg; } sci_b_spi_instance_ctrl_t;
typedef struct { uint32_t bitrate; } spi_cfg_t;
extern sci_b_spi_instance_ctrl_t g_spi0_ctrl;
extern const spi_cfg_t           g_spi0_cfg;
fsp_err_t R_SCI_B_SPI_Open(sci_b_spi_instance_ctrl_t *p_ctrl, const spi_cfg_t *p_cfg);
fsp_err_t R_SCI_B_SPI_WriteRead(sci_b_spi_instance_ctrl_t *p_ctrl,
                                const void *p_src, void *p_dest,
                                uint32_t length, spi_bit_width_t bit_width);

/* --- GPT (tachometer input capture) ------------------------------------- */
typedef enum { TIMER_EVENT_CYCLE_END, TIMER_EVENT_CAPTURE_A,
               TIMER_EVENT_CAPTURE_B } timer_event_t;
typedef struct {
    timer_event_t event;
    uint32_t      capture;
    void         *p_context;
} timer_callback_args_t;
typedef struct { void *p_reg; } gpt_instance_ctrl_t;
typedef struct { uint32_t period_counts; } timer_cfg_t;
extern gpt_instance_ctrl_t g_timer_tacho_ctrl;
extern const timer_cfg_t   g_timer_tacho_cfg;
fsp_err_t R_GPT_Open(gpt_instance_ctrl_t *p_ctrl, const timer_cfg_t *p_cfg);
fsp_err_t R_GPT_Start(gpt_instance_ctrl_t *p_ctrl);

/* --- ADC (battery sense) ------------------------------------------------ */
typedef enum { ADC_STATE_IDLE = 0, ADC_STATE_SCAN_IN_PROGRESS = 1 } adc_state_t;
typedef struct { adc_state_t state; } adc_status_t;
typedef enum { ADC_CHANNEL_0 = 0, ADC_CHANNEL_1 = 1 } adc_channel_t;
typedef struct { void *p_reg; } adc_instance_ctrl_t;
extern adc_instance_ctrl_t g_adc0_ctrl;
fsp_err_t R_ADC_ScanStart(adc_instance_ctrl_t *p_ctrl);
fsp_err_t R_ADC_StatusGet(adc_instance_ctrl_t *p_ctrl, adc_status_t *p_status);
fsp_err_t R_ADC_Read(adc_instance_ctrl_t *p_ctrl, adc_channel_t reg_id, uint16_t *p_data);

/* --- WDT ---------------------------------------------------------------- */
typedef struct { void *p_reg; } wdt_instance_ctrl_t;
extern wdt_instance_ctrl_t g_wdt0_ctrl;
fsp_err_t R_WDT_Refresh(wdt_instance_ctrl_t *p_ctrl);

/* --- RTC ---------------------------------------------------------------- */
typedef struct {
    int tm_sec, tm_min, tm_hour, tm_mday, tm_mon, tm_year, tm_wday, tm_yday, tm_isdst;
} rtc_time_t;
typedef struct { void *p_reg; } rtc_instance_ctrl_t;
extern rtc_instance_ctrl_t g_rtc_ctrl;
fsp_err_t R_RTC_CalendarTimeGet(rtc_instance_ctrl_t *p_ctrl, rtc_time_t *p_time);

#endif /* HAL_DATA_STUB_H */
