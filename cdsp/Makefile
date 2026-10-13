# Portable C99 DSP library + native cross-check against the Python reference.
#
# `make test` compiles for the build host and runs the golden-vector comparison,
# so the firmware maths is verified on every commit with no hardware attached.
# `make lib` produces the static library the firmware builds link against.
#
# Strict by default: -Werror with the pedantic set, because this code runs
# unattended for years on a battery and a warning here is a field failure later.

CC      ?= cc
PY      ?= python3
CFLAGS  ?= -std=c99 -O2 -g
WARN     = -Wall -Wextra -Werror -Wpedantic -Wshadow -Wconversion \
           -Wdouble-promotion -Wstrict-prototypes -Wmissing-prototypes \
           -Wpointer-arith -Wcast-align -Wundef
# The test harness deliberately mixes double accumulators with float data (the
# reference is float64, the library is float32), so -Wdouble-promotion is noise
# there. It stays on for the library itself, where an accidental promotion means
# software float emulation on the target.
TESTWARN = $(filter-out -Wdouble-promotion,$(WARN))
INCLUDE  = -Iinclude
LDLIBS   = -lm

# pdm_pipeline.c and pdm_app.c are excluded from the library on purpose: they
# need the kit's app_config.h and exported pdm_detector.h on the include path,
# so they are compiled per kit (and, for the host tests, against
# tests/ref_config/).
SRC      = $(filter-out src/pdm_pipeline.c src/pdm_app.c,$(wildcard src/*.c))
OBJ      = $(SRC:.c=.o)
LIB      = libpdmdsp.a
GOLDEN   = tests/golden.h
LAYOUT   = include/pdm_features_layout.h
TESTBIN  = tests/test_cdsp
PIPEBIN  = tests/test_pipeline
VDEVBIN  = tests/virtual_device

.PHONY: all lib test test-pipeline virtual-device golden layout clean size
all: lib

lib: $(LIB)

$(LIB): $(OBJ)
	$(AR) rcs $@ $^

src/%.o: src/%.c include/pdm_dsp.h
	$(CC) $(CFLAGS) $(WARN) $(INCLUDE) -c $< -o $@

# Golden vectors come from the Python reference. Regenerate whenever pdmcore.dsp
# changes -- a diff here means the two implementations have diverged.
golden:
	$(PY) tools/gen_golden.py --out $(GOLDEN)

# The feature layout is generated from pdmcore.features so the C and Python
# sides of the contract cannot drift.
layout:
	$(PY) tools/gen_feature_layout.py --out $(LAYOUT)

$(GOLDEN):
	$(PY) tools/gen_golden.py --out $(GOLDEN)

$(TESTBIN): tests/test_cdsp.c $(GOLDEN) $(LIB)
	$(CC) $(CFLAGS) $(TESTWARN) $(INCLUDE) -Itests tests/test_cdsp.c $(LIB) $(LDLIBS) -o $@

# The shared pipeline, compiled against the host reference config and the
# exported reference model. Proves edge/cloud feature agreement.
$(PIPEBIN): tests/test_pipeline.c src/pdm_pipeline.c $(GOLDEN) $(LIB) \
            tests/ref_config/app_config.h tests/ref_config/pdm_detector.h
	$(CC) $(CFLAGS) $(TESTWARN) $(INCLUDE) -Itests -Itests/ref_config \
	    tests/test_pipeline.c src/pdm_pipeline.c $(LIB) $(LDLIBS) -o $@

# The real firmware driven from a file of samples: same pdm_app.c, pipeline and
# wire encoder a microcontroller runs, emitting real frames on stdout. Used by
# core/tests/test_virtual_device.py to close the loop from C to cloud.
$(VDEVBIN): tests/virtual_device.c tests/board_host.c src/pdm_app.c src/pdm_pipeline.c \
            $(LIB) tests/ref_config/app_config.h tests/ref_config/pdm_detector.h
	$(CC) $(CFLAGS) $(TESTWARN) $(INCLUDE) -Itests -Itests/ref_config \
	    tests/virtual_device.c tests/board_host.c src/pdm_app.c src/pdm_pipeline.c \
	    $(LIB) $(LDLIBS) -o $@

virtual-device: $(VDEVBIN)

test: $(TESTBIN) $(PIPEBIN) $(VDEVBIN)
	./$(TESTBIN)
	@echo
	./$(PIPEBIN)

test-pipeline: $(PIPEBIN)
	./$(PIPEBIN)

# Code size per object, the number that decides whether this fits in flash
# alongside the vendor SDK.
size: $(OBJ)
	@$(SIZE) -t $(OBJ) 2>/dev/null || size -t $(OBJ)

clean:
	rm -f $(OBJ) $(LIB) $(TESTBIN) $(PIPEBIN) $(VDEVBIN)

distclean: clean
	rm -f $(GOLDEN)
