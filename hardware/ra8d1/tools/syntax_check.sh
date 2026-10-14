#!/bin/sh
# syntax_check.sh -- compile and link this kit's firmware on the host against a
# stub FSP, with the same warning set the real build uses.
#
# This is not a substitute for a Renesas build. It catches what it can catch
# without e2 studio installed: type errors, wrong signatures, bad struct
# members, implicit conversions, double promotion (which on a Cortex-M85 means
# dropping into software emulation and losing the Helium advantage entirely),
# and any function the driver calls that nobody defines.
#
# It does NOT check a single register value, clock setting or pin assignment.
#
#   sh tools/syntax_check.sh
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$HERE/../.." && pwd)
CC=${CC:-cc}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

# -Wdouble-promotion is not optional on this part: the M85's FPU and Helium are
# single-precision, so one accidental promotion turns a 0.6 ms FFT into tens of
# milliseconds of software double arithmetic -- and the numbers stay correct, so
# nothing fails except the energy budget.
WARN="-Wall -Wextra -Werror -Wpedantic -Wshadow -Wconversion -Wdouble-promotion \
-Wstrict-prototypes -Wmissing-prototypes -Wpointer-arith -Wundef"

$CC -std=c99 -O2 $WARN \
    -I"$REPO/cdsp/include" -I"$HERE/firmware" -I"$HERE/firmware/model" \
    -I"$HERE/tests/fsp_stub" \
    -c "$HERE/firmware/board_ra8d1.c" -o "$OUT/board.o"
echo "  board_ra8d1.c   compiles clean"

$CC -std=c99 -O2 $WARN \
    -I"$REPO/cdsp/include" -I"$HERE/firmware" -I"$HERE/firmware/model" \
    -I"$HERE/tests/fsp_stub" \
    -c "$HERE/firmware/pdm_order.c" -o "$OUT/order.o"
echo "  pdm_order.c     compiles clean"

$CC -std=c99 -O2 $WARN \
    -I"$REPO/cdsp/include" -I"$HERE/firmware" -I"$HERE/firmware/model" \
    -I"$HERE/tests/fsp_stub" \
    -c "$HERE/firmware/main.c" -o "$OUT/main.o"
echo "  main.c          compiles clean"

# The shared sources, built with THIS kit's app_config.h -- which is the point:
# it is the three-axis configuration that selects the multi-channel code path.
#
# -Wdouble-promotion is dropped for these two: pdm_pipeline.c uses double
# accumulators deliberately in the places where single precision loses
# significance (spectral centroid over 2049 bins, cumulative energy), and the
# shared Makefile makes the same exception. The rule that matters is that this
# KIT's own sources are strictly single-precision, and they are checked above
# with the flag on.
SHAREDWARN=$(echo "$WARN" | sed 's/-Wdouble-promotion//')
for src in pdm_app.c pdm_pipeline.c; do
    $CC -std=c99 -O2 $SHAREDWARN \
        -I"$REPO/cdsp/include" -I"$HERE/firmware" -I"$HERE/firmware/model" \
        -I"$HERE/tests/fsp_stub" \
        -c "$REPO/cdsp/src/$src" -o "$OUT/${src%.c}.o"
    echo "  $src   compiles clean against this kit's app_config.h"
done

$CC -std=c99 -O2 $WARN \
    -I"$REPO/cdsp/include" -I"$HERE/tests/fsp_stub" \
    -c "$HERE/tests/fsp_stub/fsp_stub.c" -o "$OUT/stub.o"

# Link. Unresolved symbols here are files the build references and nobody wrote.
if $CC "$OUT"/*.o "$REPO/cdsp/libpdmdsp.a" -lm -o "$OUT/firmware" 2>"$OUT/link.log"; then
    echo "  links clean"
else
    echo "  link reports unresolved symbols (expected: the transport and flash queue):"
    grep -o "undefined reference to \`[a-z_0-9]*'" "$OUT/link.log" | sort -u | sed 's/^/    /'
fi
echo "syntax check complete -- this proves nothing about register values"
