# Acceleration on conditional NNs via memory caching: APOLLO4 Lite Blue EVB benchmarking

## Load HAL libraries:
```bash
git submodule update --init --recursive
```

## Requirements:
- A STM32L475 Board (here, b-l475e-iot01a used, you might need to change the Makefile for other boards)
- ARM GCC toolchain 15.2.1
- OpenOCD
- CMSIS-Core (to Driver/CMSIS) from https://github.com/ARM-software/CMSIS_5
- CMSIS-L4 (to Driver/CMSIS/Device/ST/) from https://github.com/STMicroelectronics/cmsis-device-l4.git

## Compile FFF with memory acceleration
```bash
make clean all SORTED=1
```

## Run debuggin 
```bash
make clean all debug SORTED=1
```

## Run latency benchmarking 
```bash
make clean all flash SORTED=1 TIMING=1
make tcount
```

## Makefile 
Please check other features in Makefile

## Get MEM info 
```bash
arm-none-eabi-size ./build/main.elf
```
