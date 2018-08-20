# netron

A simple neural network library in C.

# Usage

To use the library include 'netron.h' in the main file

```C
#include "netron.c"
```
Here is a short demonstration of library's functionalities.

```C
float learning_rate = 0.2f;
int iterations = 1000;
int hidden_layers = 2;
int hidden_units = 4;

n_netron letters;
 
// Initialize network
netron_init(&letters, learning_rate, iterations,
            hidden_layers, hidden_units, emit);

/*
Letters.
Imagine these # and . represent black and white pixels.
*/

float *a = character(
  ".#####."
  "#.....#"
  "#.....#"
  "#######"
  "#.....#"
  "#.....#"
  "#.....#");
  
float *b = character(
  "######."
  "#.....#"
  "#.....#"
  "######."
  "#.....#"
  "#.....#"
  "######.");

float *c = character(
  "#######"
  "#......"
  "#......"
  "#......"
  "#......"
  "#......"
  "#######");

float *abc[3] = { a, b, c };
// Put all values in single array.
float *input = format(abc, 3, 49);
// Map letter to a number.
float output[3] = { map('a'), map('b'), map('c') };

/*
Learn the letters A through C.
*/

learn(&letters, input, 3, 49,
                output, 3, 1);

/*
Predict the letter B, even with a few pixels off.
*/

float *b_modified = character(
  "######."
  "##....#"
  "#.....#"
  "######."
  "#....##"
  "#.....#"
  "######.");

Matrix result = predict(&letters, b_modified, 1, 49);
print_elements(result); // { 0.37 }
```



