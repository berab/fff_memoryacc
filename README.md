## Run FFF
```bash
make clean all
./fff
```

## Memory optimized FFF via leaf statistics
```bash
make clean all SORTED=1
./fff
```
## Get memory footprint
```bash
arm-none-eabi-size fff
```

## Get memory characteristics
```bash
valgrind --tool=cachegrind ./fff
cg_annotate cachegrind.out.<pid>
```
