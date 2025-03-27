#ifndef CONFIG_H
#define CONFIG_H

// Constants
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9

// Uncomment to enable verbose mode
// #define VERBOSE

// Macro for verbose mode print statements
#ifdef VERBOSE
    #define VERBOSE_PRINT(fmt, ...) printf(fmt, __VA_ARGS__)
#else
    #define VERBOSE_PRINT(fmt, ...)  // Do nothing when VERBOSE is not defined
#endif

#endif // CONFIG_H
