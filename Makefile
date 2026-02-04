CC = gcc
CFLAGS = -Wall -O0
PYTHON = python3

TARGET = main
SOURCES = main.c
TENSORS = tensor1.bin tensor2.bin

.PHONY: all run clean

all: $(TARGET)

$(TENSORS): convert_array.py
	$(PYTHON) convert_array.py

$(TARGET): $(SOURCES) $(TENSORS)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(TENSORS)
