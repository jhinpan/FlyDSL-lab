# ATT stall attribution

Turns a `rocprofv3 --att` thread trace into per-instruction stall cycles, which is
the only way to see *where* a kernel's cycles go once the SQ counters have been
exhausted. `rocprofv3` itself only emits the raw `.att` blob plus the code
objects; the decode step below is what produces numbers you can read.

ROCm < 7.13 does not ship the decoder. Fetch it once:

```bash
gh release download 0.1.6 --repo ROCm/rocprof-trace-decoder \
    --pattern "*ubuntu-22.04*.deb"
dpkg-deb -x rocprof-trace-decoder-*.deb /tmp/dec
# -> /tmp/dec/opt/rocm/lib/librocprof-trace-decoder.so
```

## Capture

`--att-target-cu` traces a single CU, so the kernel must be long enough for that
CU to see real work. `--kernel-include-regex` keeps the trace to the kernel under
study; without it the blob is dominated by framework kernels.

```bash
rocprofv3 --att --att-library-path /tmp/dec/opt/rocm/lib \
    --att-target-cu 0 --att-simd-select 0xF --att-activity 8 \
    --att-buffer-size 536870912 --att-serialize-all \
    --kernel-include-regex "my_kernel" -d out -o run -- ./app
```

## Decode

```bash
g++ -O2 -std=c++17 -I/opt/rocm/include att_stall_decode.cpp \
    -o att_stall_decode -L/opt/rocm/lib -lrocprofiler-sdk -Wl,-rpath,/opt/rocm/lib
```

The decoder needs each code object's load base and size, which `rocprofv3` records
in the run's sqlite database:

```bash
ARGS=$(python3 - <<'PY'
import sqlite3, glob, os
c = sqlite3.connect(glob.glob('out/*.db')[0])
t = [n for (n,) in c.execute("select name from sqlite_master where type='table'")
     if 'info_code_object' in n][0]
print(' '.join(f'{i}:{b}:{s}:out/run_gfx950_code_object_id_{i}.out'
               for i, b, s in c.execute(f'select id,load_base,load_size from "{t}"')
               if os.path.exists(f'out/run_gfx950_code_object_id_{i}.out')))
PY
)
ATT_DECODER_PATH=/tmp/dec/opt/rocm/lib ./att_stall_decode out/*.att $ARGS > stalls.csv
```

Stderr carries the wave-state totals (EXEC / WAIT / STALL), which say how much of
the wave lifetime is actually retiring instructions.

## Report

Disassemble the code object holding the kernel, then join:

```bash
llvm-objdump -d --mcpu=gfx950 out/run_gfx950_code_object_id_8.out > kernel.s
./att_stall_report.py stalls.csv kernel.s 8 25
```

The report ranks stall cycles by instruction category, by mnemonic, and by
individual PC. Divide a mnemonic's stall by (traced waves x loop trips) to get
cycles per loop iteration, which is directly comparable across kernels.
