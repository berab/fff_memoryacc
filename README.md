# Acceleration on conditional NNs via memory caching: APOLLO4 Lite Blue EVB benchmarking

## Requirements:
- Apollo4 Blue Lite EVB
- ARM GCC toolchain 15.2.1

## Make:
```bash
make clean all
```

## RUN JLink GDB Server:
```bash
JLinkGDBServerCLExe -singlerun -nogui -port 61234 -device AMAP42KL-KBR
```

## Connect GDB to the Server (.gdbinit is autocreated):
```bash
gdb-multiarch
```

## Run FFF 
```bash
make clean all
```

## Get MEM info 
```bash
arm-none-eabi-size ./bin/main.axf
```
