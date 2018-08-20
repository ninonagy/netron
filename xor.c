#include "netron.h"

void emit(char *str, int iteration, Matrix errors, Matrix *results)
{
    printf("Iteration #%i \n", iteration);
    print_elements(results[3]);
}

int main()
{
    float learning_rate = 0.2f;
    int iterations = 10000;
    int hidden_layers = 1;
    int hidden_units = 3;

    n_netron xor;
 
    // Initialize network
    netron_init(&xor, learning_rate, iterations,
                hidden_layers, hidden_units, emit);

    float input[4][2] = {{0, 0},
                         {0, 1},
                         {1, 0},
                         {1, 1}};

    float output[4][1] = {{ 0 },
                          { 1 },
                          { 1 },
                          { 0 }};

    learn(&xor, input[0], 4, 2,
                output[0], 4, 1);
    
    Matrix result = predict(&xor, input[0], 4, 2);
    
    printf("Results: ");
    print_elements(result);

    return 0;
}