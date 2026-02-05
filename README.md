## Get MEM cahracteristics
```bash
valgrind --tool=cachegrind ./fff
cg_annotate cachegrind.out.<pid>
valgrind --tool=cachegrind ./vfff
cg_annotate cachegrind.out.<pid>
```
