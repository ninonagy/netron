#define N_ALLOC
#include "netron.h"

// Convert '#' to ones and '.' to zeroes
float *character(char *string)
{
    int length = (int)strlen(string);
    float *result = (float *)malloc(length * sizeof(*result));
    
    for(int i = 0; i < length; i++)
    {
        if(string[i] == '#') result[i] = 1;
        if(string[i] == '.') result[i] = 0;
    }

    return result;
}

float map(char c)
{
    if(c == 'a') return 0.1f;
    if(c == 'b') return 0.3f;
    if(c == 'c') return 0.5f;
    return 0;
}

// Put all values in single array
float *format(float **array, int array_count, int element_size)
{
    float *result = (float *)malloc(array_count * element_size * sizeof(*result));
    
    int at = 0;
    for(int i = 0; i < array_count; i++)
    {
        float *source = array[i];
        for(int e = 0; e < element_size; e++)
        {
            result[at++] = source[e];
        }
    }

    return result;
}

void emit(char *str, int iteration, Matrix errors, Matrix *results)
{
    printf("Iteration #%i \n", iteration);
    print_elements(results[3]);
}

int main()
{
    float learning_rate = 0.2f;
    int iterations = 1000;
    int hidden_layers = 2;
    int hidden_units = 4;

    n_netron letters;
 
    // Initialize network
    netron_init(&letters, learning_rate, iterations,
                hidden_layers, hidden_units, emit);

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
    float *input = format(abc, 3, 49);
    float output[3] = { map('a'), map('b'), map('c') };

    learn(&letters, input, 3, 49,
                    output, 3, 1);

    Matrix result = predict(&letters, b, 1, 49);
    printf("Results: ");
    print_elements(result);

    printf("Saving network to file 'letters.txt'...\n");
    write_weights_to_file("letters.txt", &letters, 1);

    float *b_modified = character(
    "######."
    "##....#"
    "#.....#"
    "######."
    "#....##"
    "#.....#"
    "######.");

    printf("Loading network from saved file...\n");
    n_netron *letters_trained = read_weights_from_file("letters.txt");

    result = predict(letters_trained, b_modified, 1, 49);
    printf("Output: ");
    print_elements(result);
    
    return 0;
}