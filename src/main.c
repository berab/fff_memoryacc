#include "am_mcu_apollo.h"
#include "am_hal_global.h"
#include "am_bsp.h"
#include "am_util.h"

#include "fff.h"


// UART configuration
void *phUART;

const am_hal_uart_config_t g_sUartConfig =
{
    .ui32BaudRate = 9600,
    .eDataBits = AM_HAL_UART_DATA_BITS_8,
    .eParity = AM_HAL_UART_PARITY_NONE,
    .eStopBits = AM_HAL_UART_ONE_STOP_BIT,
    .eFlowControl = AM_HAL_UART_FLOW_CTRL_NONE,
    .eTXFifoLevel = AM_HAL_UART_FIFO_LEVEL_16,
    .eRXFifoLevel = AM_HAL_UART_FIFO_LEVEL_16,
};

#if AM_BSP_UART_PRINT_INST == 0
void am_uart_isr(void)
#elif AM_BSP_UART_PRINT_INST == 1
void am_uart1_isr(void)
#elif AM_BSP_UART_PRINT_INST == 2
void am_uart2_isr(void)
#elif AM_BSP_UART_PRINT_INST == 3
void am_uart3_isr(void)
#endif
{
    uint32_t ui32Status;
    am_hal_uart_interrupt_status_get(phUART, &ui32Status, true);
    am_hal_uart_interrupt_clear(phUART, ui32Status);
    am_hal_uart_interrupt_service(phUART, ui32Status);
}

void uart_print(char *pcStr)
{
    uint32_t ui32StrLen = 0;
    uint32_t ui32BytesWritten = 0;

    while (pcStr[ui32StrLen] != 0)
    {
        ui32StrLen++;
    }

    const am_hal_uart_transfer_t sUartWrite =
    {
        .eType = AM_HAL_UART_BLOCKING_WRITE,
        .pui8Data = (uint8_t *) pcStr,
        .ui32NumBytes = ui32StrLen,
        .pui32BytesTransferred = &ui32BytesWritten,
        .ui32TimeoutMs = 100,
        .pfnCallback = NULL,
        .pvContext = NULL,
        .ui32ErrorStatus = 0
    };

    am_hal_uart_transfer(phUART, &sUartWrite);
}

#define TIMER_NUM 5
volatile uint32_t g_TimerCount = 0;
uint32_t timer_init(uint32_t ui32TimerNum) {
    am_hal_timer_config_t       TimerConfig;
    uint32_t ui32Status         = AM_HAL_STATUS_SUCCESS;

    // Set the timer configuration, the default timer configuration is HFRC_DIV16, EDGE, compares=0, no trig.
    am_hal_timer_default_config_set(&TimerConfig);
    ui32Status = am_hal_timer_config(ui32TimerNum, &TimerConfig);
    if ( ui32Status != AM_HAL_STATUS_SUCCESS )
    {
        am_util_stdio_printf("Failed to configure TIMER%d, return value=%d\n", ui32TimerNum, ui32Status);
        return ui32Status;
    }

    // Stop and clear the timer.
    am_hal_timer_clear(ui32TimerNum);
    am_hal_timer_stop(ui32TimerNum);

    // Timer interrupt not needed for this purpose.
    return ui32Status;
} // timer_init()

// Main
int main(void) {
    timer_init(TIMER_NUM);
    // Set the default cache configuration
    am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
    am_hal_cachectrl_enable();

    //
    // Configure the board for low power operation.
    //
    am_bsp_low_power_init();

#ifndef HIGH_PERF
    am_hal_pwrctrl_mcu_mode_select(AM_HAL_PWRCTRL_MCU_MODE_LOW_POWER);
#else
    am_hal_pwrctrl_mcu_mode_select(AM_HAL_PWRCTRL_MCU_MODE_HIGH_PERFORMANCE);
#endif

    // Initialize UART for printing
    am_hal_uart_initialize(AM_BSP_UART_PRINT_INST, &phUART);
    am_hal_uart_power_control(phUART, AM_HAL_SYSCTRL_WAKE, false);
    am_hal_uart_configure(phUART, &g_sUartConfig);
    am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_TX, g_AM_BSP_GPIO_COM_UART_TX);
    am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_RX, g_AM_BSP_GPIO_COM_UART_RX);
    NVIC_SetPriority((IRQn_Type)(UART0_IRQn + AM_BSP_UART_PRINT_INST), AM_IRQ_PRIORITY_DEFAULT);
    NVIC_EnableIRQ((IRQn_Type)(UART0_IRQn + AM_BSP_UART_PRINT_INST));
    am_hal_interrupt_master_enable();
    am_util_stdio_printf_init(uart_print);

    // Initialize LED0 for status indication
    am_hal_gpio_pinconfig(AM_BSP_GPIO_LED0, am_hal_gpio_pincfg_output);
    am_hal_gpio_state_write(AM_BSP_GPIO_LED0, AM_HAL_GPIO_OUTPUT_CLEAR);
    while (1)
    {
        am_hal_timer_clear(TIMER_NUM);   // The clear function also starts the timer
        for (int i = 0; i < N_SAMPLES; i++) {
            fff();
        }

        am_hal_timer_stop(TIMER_NUM);
        g_TimerCount = am_hal_timer_read(TIMER_NUM);
        am_util_stdio_printf("elapsedcount %d\n\r", (g_TimerCount));
        am_util_stdio_printf("elapsed_ms=%d.%03d\n\r", (int)(g_TimerCount / 6000), (int)((g_TimerCount % 6000) / 6));
        // uint32_t ui32Freq;
        //
        // am_hal_clkgen_status_t sClkGenStatus;
        // am_hal_clkgen_status_get(&sClkGenStatus);
        // am_util_stdio_printf("Current HFRC adjustment = %u\n\r",sClkGenStatus.ui32SysclkFreq);
        // am_hal_gpio_state_write(AM_BSP_GPIO_LED0, AM_HAL_GPIO_OUTPUT_TOGGLE);

        // Go to Deep Sleep.
        am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
    }
}
