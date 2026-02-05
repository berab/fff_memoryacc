CC = gcc
CFLAGS = -Wall -O0
DEBUGFLAGS = -g -O0

TARGET = main
TEST_BIN = test_bin
MAIN_SOURCES = main.c
TEST_SOURCES = test.c

.PHONY: all run test clean debug

all: $(TARGET) $(TEST_BIN)

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

clean:
	rm -f $(TARGET) $(TEST_BIN) $(TENSORS)
