CC = gcc
CFLAGS = -Wall
DEBUGFLAGS = -g
OPT = -O0
CLAFGS += $(OPT)
TARGET = main

SORTED ?= 0
ifeq ($(SORTED), 1)
	CFLAGS += -DSORTED
endif
TEST_BIN = test_bin
MAIN_SOURCES = $(TARGET).c
MAIN_SOURCES += fff.c
TEST_SOURCES = test.c

.PHONY: all run test clean debug time

all: $(TARGET)

$(TARGET): $(MAIN_SOURCES) $(TENSORS)
	$(CC) $(CFLAGS) $(MAIN_SOURCES) -o $(TARGET)

$(TEST_BIN): $(TEST_SOURCES) $(TENSORS)
	$(CC) $(CFLAGS) $(TEST_SOURCES) -o $(TEST_BIN)

run: $(TARGET)
	./$(TARGET)

test: $(TEST_BIN)
	@echo "Running tests..."
	@./$(TEST_BIN) && echo "✅ All tests passed" || (echo "❌ Tests failed" && exit 1)

debug: CFLAGS += $(DEBUGFLAGS)
debug: clean $(TEST_BIN)
	gdb $(TEST_BIN)

# Build and time the program with clean timing (stdout -> /dev/null)
time: $(TARGET)
	@echo "Timing $(TARGET)..."
	@time -p sh -c './$(TARGET) > /dev/null'

clean:
	rm -f $(TARGET) $(TEST_BIN) $(TENSORS)
