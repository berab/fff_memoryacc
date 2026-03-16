# Acceleration on conditional NNs via memory caching: APOLLO4 Lite Blue EVB benchmarking

## Requirements:
- Apollo4 Blue Lite EVB
- ARM GCC toolchain 15.2.1
- JLink GDB Server

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
arm-none-eabi-size ./bin/main.axf
```
