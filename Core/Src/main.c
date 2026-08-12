#include "main.h"
#include "fff.h"
#include <stdio.h>
#include <string.h>

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
void SystemClock_Config_HalfSpeed(void);
static void MX_USART1_UART_Init(void);
int __io_putchar(int ch);
int _write(int file, char *ptr, int len);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

UART_HandleTypeDef huart1;

/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
volatile uint32_t g_elapsed_ms = 0;
int main(void)
{

    /* USER CODE BEGIN 1 */

    /* USER CODE END 1 */

    /* MCU Configuration--------------------------------------------------------*/

    /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
    HAL_Init();

    /* USER CODE BEGIN Init */
    /* Enable GPIO clock and initialize LED2 */
#ifdef LED
    __HAL_RCC_GPIOB_CLK_ENABLE();
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = LED2_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(LED2_GPIO_Port, &GPIO_InitStruct);
#endif
#ifdef LOWPWR
    HAL_PWREx_EnableLowPowerRunMode();
#else
    HAL_PWREx_DisableLowPowerRunMode();
#endif

    /* USER CODE END Init */

    /* Configure the system clock */
    SystemClock_Config();

    /* USER CODE BEGIN SysInit */

    /* USER CODE END SysInit */

    /* Initialize all configured peripherals */
    /* USER CODE BEGIN 2 */
    MX_USART1_UART_Init();

    /* USER CODE END 2 */

    /* Infinite loop */
    /* USER CODE BEGIN WHILE */
    /* USER CODE END WHILE */
#ifdef LED
    HAL_GPIO_TogglePin(LED2_GPIO_Port, LED2_Pin);
#endif
    /* Measure time (ms) to run MODEL() 100 times (globals for flash read) */
    uint32_t start_ms = HAL_GetTick();
    for (int i = 0; i < N_SAMPLES; i++) {
        fff();
    }
    uint32_t end_ms = HAL_GetTick();
    uint32_t elapsed_ms = end_ms - start_ms;
    HAL_GPIO_TogglePin(LED2_GPIO_Port, LED2_Pin);
    printf("\n\relapsed_ms=%ld\n\r", elapsed_ms);
    /* USER CODE BEGIN 3 */
    /* USER CODE END 3 */
    }

    /**
     * @brief System Clock Configuration
     * @retval None
     */
    void SystemClock_Config(void)
    {

        RCC_OscInitTypeDef RCC_OscInitStruct = {0};
        RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

        /** Configure the main internal regulator output voltage
        */
        if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
        {
            Error_Handler();
        }

        /** Initializes the RCC Oscillators according to the specified parameters
         * in the RCC_OscInitTypeDef structure.
         */
        RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_MSI;
        RCC_OscInitStruct.MSIState = RCC_MSI_ON;
        RCC_OscInitStruct.MSICalibrationValue = 0;
        RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_6;
        RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
        RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_MSI;
        RCC_OscInitStruct.PLL.PLLM = 1;
        RCC_OscInitStruct.PLL.PLLN = 40;
        RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV7;
        RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
        RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
        if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
        {
            Error_Handler();
        }

        /** Initializes the CPU, AHB and APB buses clocks
        */
        RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
            |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
        RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
        RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
        RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
        RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

        if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
        {
            Error_Handler();
        }

    }

    /* USER CODE BEGIN 4 */

    static void MX_USART1_UART_Init(void)
    {
        huart1.Instance = USART1;
        huart1.Init.BaudRate = 9600;
        huart1.Init.WordLength = UART_WORDLENGTH_8B;
        huart1.Init.StopBits = UART_STOPBITS_1;
        huart1.Init.Parity = UART_PARITY_NONE;
        huart1.Init.Mode = UART_MODE_TX_RX;
        huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
        huart1.Init.OverSampling = UART_OVERSAMPLING_16;
        huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
        huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
        if (HAL_UART_Init(&huart1) != HAL_OK) {
            Error_Handler();
        }
    }

    int __io_putchar(int ch)
    {
        HAL_UART_Transmit(&huart1, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
        return ch;
    }

    int _write(int file, char *ptr, int len)
    {
        (void)file;
        int DataIdx;
        for (DataIdx = 0; DataIdx < len; DataIdx++) {
            __io_putchar(*ptr++);
        }
        return len;
    }

    /* USER CODE END 4 */

    /**
     * @brief  This function is executed in case of error occurrence.
     * @retval None
     */
    void Error_Handler(void)
    {
        /* USER CODE BEGIN Error_Handler_Debug */
        /* User can add his own implementation to report the HAL error return state */
        __disable_irq();
        while (1)
        {
        }
        /* USER CODE END Error_Handler_Debug */
    }
#ifdef USE_FULL_ASSERT
    /**
     * @brief  Reports the name of the source file and the source line number
     *         where the assert_param error has occurred.
     * @param  file: pointer to the source file name
     * @param  line: assert_param error line source number
     * @retval None
     */
    void assert_failed(uint8_t *file, uint32_t line)
    {
        /* USER CODE BEGIN 6 */
        /* User can add his own implementation to report the file name and line number,
           ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
        /* USER CODE END 6 */
    }
#endif /* USE_FULL_ASSERT */
