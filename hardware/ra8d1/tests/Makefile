# Host tests for the RA8D1 kit.
#
#   make -C hardware/ra8d1/tests        # build and run everything
#
# These run on the host, not on the board. test_order measures a real,
# quantitative claim (how much peak amplitude angular resampling recovers);
# the syntax check compiles the driver against a stub FSP.

REPO   := ../../..
CC     ?= cc
WARN   := -Wall -Wextra -Werror -Wpedantic -Wshadow -Wconversion -Wdouble-promotion \
          -Wstrict-prototypes -Wmissing-prototypes -Wpointer-arith -Wundef
CFLAGS := -std=c99 -O2 -g $(WARN)
INC    := -I$(REPO)/cdsp/include -I../firmware
LIB    := $(REPO)/cdsp/libpdmdsp.a

.PHONY: all test test-order syntax clean

all: test

test: test-order syntax
	@echo
	@echo "kit host tests complete"

$(LIB):
	$(MAKE) -C $(REPO)/cdsp lib

test_order: test_order.c ../firmware/pdm_order.c ../firmware/pdm_order.h $(LIB)
	$(CC) $(CFLAGS) $(INC) test_order.c ../firmware/pdm_order.c $(LIB) -lm -o $@

test-order: test_order
	./test_order

syntax:
	sh ../tools/syntax_check.sh

clean:
	rm -f test_order
