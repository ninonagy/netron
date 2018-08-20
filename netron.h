/*
Based on:
https://github.com/stevenmiller888/mind
https://github.com/ivanseidel/IAMDinosaur
*/

#define _CRT_SECURE_NO_WARNINGS 1

#ifdef N_DLL_EXPORT
#define PROC_DEC __declspec(dllexport)
#else
#define PROC_DEC __declspec(dllimport)
#endif

#include <math.h> // exp, pow
#include <stdlib.h> // calloc, qsort, abs
#include <stdio.h> // printf, FILE
#include <string.h> // memcpy, strlen

#if !defined __cplusplus
typedef int bool;
#define true 1
#define false 0
#endif

#ifndef assert
#define assert(expression) if(!(expression)) { __assert(#expression, __FILE__, __LINE__); }
inline void __assert(const char *expression, const char *file, int line)
{
    fprintf(stderr, "Assertion '%s' failed, file '%s' line '%d'.", expression, file, line);
    // abort();
}
#endif

static unsigned int _n_rz = 362436069;
static unsigned int _n_rw = 521288629;

inline unsigned int n_rand(void)
{
    _n_rz = 36969 * (_n_rz & 65535) + (_n_rz >> 16);
    _n_rw = 18000 * (_n_rw & 65535) + (_n_rw >> 16);
    return (_n_rz << 16) + _n_rw;
}

// return from 0.0f to 1.0f
inline float n_randf(void)
{
    unsigned int u = n_rand();
    return (float)((u + 1.0) * 2.328306435454494e-10);
}

inline int n_randi_range(int min, int max)
{
    return (n_rand() % (max + 1 - min)) + min;
}

inline float n_randf_range(float min, float max)
{
    return n_randi_range((int)(min * 100), (int)(max * 100)) / 100.0f;
}

#ifdef N_ALLOC
#include "alloc.h"
// TODO: Define alloc size in init function?
#define ALLOCATION_SIZE 16
static memory_block netron_memory;
#define n_alloc(size) alloc_size(&netron_memory, size)
#else
#define n_alloc(size) malloc(size)
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    // @Matrix
    typedef struct
    {
        int rows;
        int columns;
        float *elements;
    } n_matrix;

    typedef n_matrix* Matrix;

    static Matrix new_matrix(int rows, int columns);
    static Matrix new_matrix_elements(int rows, int columns, float *values);
    static int element_count(Matrix a);
    static void set_elements(Matrix a, float *values);
    static void print_elements(Matrix a);
    static Matrix multiply(Matrix a, Matrix b);
    static Matrix add(Matrix a, Matrix b);
    static void matrix_add(Matrix a, Matrix b);
    static Matrix subtract(Matrix a, Matrix b);
    static Matrix transpose(Matrix a);
    static Matrix multiply_elements(Matrix a, Matrix b);
    static Matrix multiply_scalar(Matrix a, float s);
    static Matrix transform_sigmoid(Matrix a);
    static Matrix transform_sigmoid_prime(Matrix a);

    Matrix new_matrix(int rows, int columns)
    {
        Matrix result = 0;
        int struct_size = sizeof(*result);
        int element_size = rows * columns * sizeof(float);
        void *ptr = (void *)n_alloc(struct_size + element_size);
        if (!ptr) return 0;

        result = (Matrix)ptr;
        result->rows = rows;
        result->columns = columns;
        result->elements = (float *)((char *)ptr + struct_size);

        // NOTE: Set to zero
        for (int i = 0; i < rows * columns; ++i)
        {
            result->elements[i] = 0;
        }

        return result;
    }

    Matrix new_matrix_elements(int rows, int columns, float *values)
    {
        Matrix result = new_matrix(rows, columns);
        set_elements(result, values);

        return result;
    }

    int element_count(Matrix a) { return (a->rows * a->columns); }

    void set_elements(Matrix a, float *values)
    {
        for (int i = 0; i < element_count(a); ++i)
        {
            a->elements[i] = values[i];
        }
    }

    void print_elements(Matrix a)
    {
        int count = element_count(a);
        printf("{ ");
        for (int i = 0; i < count; i++)
        {
            float value = a->elements[i];
            if (i < count - 1) {
                printf("%0.2f, ", value);
            }
            else {
                printf("%0.2f", value);
            }
        }
        printf(" }\n");
    }

    Matrix multiply(Matrix a, Matrix b)
    {
        Matrix result = new_matrix(a->rows, b->columns);

        int alc = 0;
        for (int x = 0; x < a->rows; x++)
        {
            for (int y = 0; y < b->columns; y++)
            {
                float value = 0;
                for (int z = 0; z < b->rows; z++)
                {
                    value += a->elements[z + x * a->columns] *
                             b->elements[y + z * b->columns];
                }

                result->elements[alc++] = value;
            }
        }

        return result;
    }

    Matrix add(Matrix a, Matrix b)
    {
        // Only possible if both matrices are same size!
        assert(a->rows == b->rows && a->columns == b->columns);
        Matrix result = new_matrix(a->rows, b->columns);

        for (int i = 0; i < element_count(result); ++i)
        {
            result->elements[i] = a->elements[i] + b->elements[i];
        }

        return result;
    }

    // NOTE: If matrices are same size why would we always allocate new one?
    void matrix_add(Matrix a, Matrix b)
    {
        // Only possible if both matrices are same size!
        assert(a->rows == b->rows && a->columns == b->columns);

        for (int i = 0; i < element_count(a); ++i)
        {
            a->elements[i] += b->elements[i];
        }
    }

    Matrix subtract(Matrix a, Matrix b)
    {
        // Only possible if both matrices are same size!
        assert(a->rows == b->rows && a->columns == b->columns);
        Matrix result = new_matrix(a->rows, b->columns);

        for (int i = 0; i < element_count(result); ++i)
        {
            result->elements[i] = a->elements[i] - b->elements[i];
        }

        return result;
    }

    // http://softwareengineering.stackexchange.com/questions/271713/transpose-a-matrix-without-a-buffering-one
    Matrix transpose(Matrix a)
    {
        Matrix result = new_matrix(a->columns, a->rows);

        // TODO: Remove memcpy? 
        memcpy(result->elements, a->elements, element_count(result) * sizeof(float));

        for (int start = 0;
            start <= element_count(a) - 1;
            start++)
        {
            int next = start;
            int i = 0;

            do {
                ++i;
                next = (next % a->rows) * a->columns + next / a->rows;
            } while (next > start);

            if (next >= start && i != 1)
            {
                float tmp = result->elements[start];
                next = start;

                do {
                    i = (next % a->rows) * a->columns + next / a->rows;
                    result->elements[next] = (i == start) ? tmp : result->elements[i];
                    next = i;
                } while (next > start);
            }
        }

        return result;
    }

    Matrix multiply_elements(Matrix a, Matrix b)
    {
        // Only possible if both matrices are same size!
        assert(a->rows == b->rows && a->columns == b->columns);
        Matrix result = new_matrix(a->rows, b->columns);

        for (int i = 0; i < element_count(result); ++i)
        {
            result->elements[i] = a->elements[i] * b->elements[i];
        }

        return result;
    }

    Matrix multiply_scalar(Matrix a, float s)
    {
        for (int i = 0; i < element_count(a); ++i)
        {
            a->elements[i] = a->elements[i] * s;
        }

        return a;
    }

    // TODO: function pointer?
    Matrix transform_sigmoid(Matrix a)
    {
        Matrix result = new_matrix(a->rows, a->columns);

        for (int i = 0; i < element_count(a); ++i)
        {
            float tmp = a->elements[i];
            result->elements[i] = 1.0f / (1 + (float)exp(-tmp));
        }

        return result;
    }

    Matrix transform_sigmoid_prime(Matrix a)
    {
        Matrix result = new_matrix(a->rows, a->columns);

        for (int i = 0; i < element_count(a); ++i)
        {
            float tmp = a->elements[i];
            result->elements[i] = (float)exp(tmp) / (float)pow(1 + (float)exp(tmp), 2);
        }

        return result;
    }

    // @netron
    typedef struct
    {
        int initialized;
        int iterations;
        int hidden_units;
        int hidden_layers;
        int weights_init;
        float learning_rate;
        float fitness;
        Matrix *weights;
    } n_netron;

    typedef n_netron n_genome;

    static void(*netron_emit)(char*, int, Matrix, Matrix*);
    static void(*netron_emit_genome)(n_netron*, Matrix);

    PROC_DEC void netron_init(
        n_netron *netron,
        float learning_rate,
        int iterations,
        int hidden_layers,
        int hidden_units,
        void(*emit)(char*, int, Matrix, Matrix*));

    PROC_DEC void netron_init_genomes(
        n_netron *genomes,
        int genome_units,
        int hidden_layers,
        int hidden_units,
        int input_length,
        int output_length,
        void(*emit)(n_netron*, Matrix));

    PROC_DEC void learn(
        n_netron *netron,
        float *input, int input_count, int input_length,
        float *output, int output_count, int output_length);

    PROC_DEC void run_genome_input(
        n_netron *genome,
        float *input,
        int input_count,
        int input_length);

    PROC_DEC void update_generation(
        n_netron *genomes,
        int genome_units,
        int selection_count,
        float mutation_rate);

    Matrix *forward(n_netron *netron, Matrix input);

    Matrix back(
        n_netron *netron, Matrix *results,
        Matrix input, Matrix output);

    PROC_DEC Matrix predict(
        n_netron *netron,
        float *input,
        int input_count,
        int input_length);

    PROC_DEC void upload(n_netron *netron, Matrix *weights);
    PROC_DEC Matrix *download(n_netron *netron);

    PROC_DEC void write_weights_to_file(const char *file_name, n_netron *array, int array_len);
    PROC_DEC n_netron *read_weights_from_file(const char *file_name);

    int netron_weights(n_netron *netron)
    {
        return (netron->hidden_layers + 1);
    }

    // NOTE: Populate matrix with random numbers between -1 and 1
    void setup_weights(
        n_netron *netron,
        int input_length,
        int output_length)
    {
        int weights_count = netron->hidden_layers + 1;
        netron->weights = (Matrix *)n_alloc(weights_count * sizeof(*netron->weights));

        if (netron->weights)
        {
            // input > hidden
            netron->weights[0] = new_matrix(input_length, netron->hidden_units);
            for (int i = 0; i < input_length * netron->hidden_units; ++i)
            {
                netron->weights[0]->elements[i] = (n_randf() - 0.5f) * 2;
            }

            // hidden > hidden
            for (int i = 1; i < netron->hidden_layers; i++)
            {
                netron->weights[i] = new_matrix(netron->hidden_units, netron->hidden_units);
                for (int e = 0; e < element_count(netron->weights[i]); ++e)
                {
                    netron->weights[i]->elements[e] = (n_randf() - 0.5f) * 2;
                }
            }

            // hidden > output
            netron->weights[netron->hidden_layers] = new_matrix(netron->hidden_units, output_length);
            for (int i = 0; i < netron->hidden_units * output_length; ++i)
            {
                netron->weights[netron->hidden_layers]->elements[i] = (n_randf() - 0.5f) * 2;
            }
        }
    }

    void netron_init(
        n_netron *netron,
        float learning_rate,
        int iterations,
        int hidden_layers,
        int hidden_units,
        void(*emit)(char*, int, Matrix, Matrix*))
    {
        netron->learning_rate = learning_rate;
        netron->iterations = iterations;
        netron->hidden_layers = hidden_layers;
        netron->hidden_units = hidden_units;
        netron->weights_init = false;

        netron_emit = emit;

        netron->initialized = true;

#ifdef N_ALLOC
        int size = megabytes(ALLOCATION_SIZE);
        init_block(&netron_memory, size);
#endif
    }

    // TODO: Add hidden_layers
    // TODO: Decide which parameters to add?
    void netron_init_genomes(
        n_netron *genomes,
        int genome_units,
        int hidden_layers,
        int hidden_units,
        int input_length,
        int output_length,
        void(*emit)(n_netron*, Matrix))
    {
#ifdef N_ALLOC
        int size = megabytes(8 * genome_units);
        init_block(&netron_memory, size);
#endif
        for (int i = 0; i < genome_units; i++)
        {
            n_netron *genome = genomes + i;
            genome->hidden_layers = hidden_layers;
            genome->hidden_units = hidden_units;
            setup_weights(genome, input_length, output_length);
            genome->weights_init = true;
            genome->fitness = 0;

            netron_emit_genome = emit;

            genome->initialized = true;
        }
    }

    // NOTE: Higher fitness value is better
    int compare_higher_fitness(const void *a, const void *b)
    {
        n_netron *genome_a = (n_netron *)a;
        n_netron *genome_b = (n_netron *)b;
        if (genome_a->fitness > genome_b->fitness) return -1;
        else if (genome_a->fitness < genome_b->fitness) return 1;
        else return 0;
    }

    // NOTE: Lower fitness value is better
    int compare_lower_fitness(const void *a, const void *b)
    {
        n_netron *genome_a = (n_netron *)a;
        n_netron *genome_b = (n_netron *)b;
        if (genome_a->fitness < genome_b->fitness) return -1;
        else if (genome_a->fitness > genome_b->fitness) return 1;
        else return 0;
    }

    void select_best_genomes(n_netron *genomes,
        int genome_units,
        int selection_count)
    {
        qsort(genomes, genome_units, sizeof(n_netron), compare_lower_fitness);

        // TODO: Delete genome_units - selection_count
    }

    /*
    Uniform Crossover
    parent 1: 0 0 0 0 0 0 0 0 0
    -     -   - -   -
    parent 2: 1 1 1 1 1 1 1 1 1
    - -   -     -
    child: 0 1 1 0 1 0 0 1 0

    One-point Crossover
    parent 1: 1 0 1 1 0 1 0 1 1
    - - - - -
    parent 2: 0 1 1 0 0 1 0 1 0
    - - - -
    child: 1 0 1 1 0 1 0 1 0
    */
    Matrix *cross_over(n_netron *a, n_netron *b)
    {
        // NOTE: Parents need to have same network layout!
        int weights_count = a->hidden_layers + 1;
        Matrix *result = (Matrix *)n_alloc(weights_count * sizeof(*result));

        for (int i = 0; i < weights_count; i++)
        {
            result[i] = new_matrix(a->weights[i]->rows, a->weights[i]->columns);
        }

        n_netron *parent[2] = { a, b };

        // Swap (50% prob.)
        if (n_randf() > 0.5f)
        {
            parent[0] = b;
            parent[1] = a;
        }

        int weights_elements_count = 0;
        for (int i = 0; i < weights_count; i++)
        {
            weights_elements_count += element_count(result[i]);
        }

        // TODO: Layer specific crossover method?
#if 0
        // One-point Crossover
        int cut_location = (int)(weights_elements_count * n_randf());

        int weight_index = 0;
        int prev_count = 0;
        for (int element_index = 0;
            element_index < weights_elements_count;
            element_index++)
        {
            n_netron *parent = (element_index < cut_location) ? a : b;

            // Find in which weight to put parent value
            if (element_index - prev_count == element_count(result[weight_index]))
            {
                prev_count += element_count(result[weight_index]);
                ++weight_index;
            }

            float *element = result[weight_index]->elements +
                (element_index - prev_count);

            *element = parent->weights[weight_index]->elements[element_index - prev_count];
        }
#else
        // Uniform Crossover
        int weight_index = 0;
        int prev_count = 0;
        for (int element_index = 0;
            element_index < weights_elements_count;
            element_index++)
        {
            int random_index = (n_randf() > 0.5f) ? 0 : 1;
            n_netron *p = parent[random_index];

            // Find in which weight to put parent value
            if (element_index - prev_count == element_count(result[weight_index]))
            {
                prev_count += element_count(result[weight_index]);
                ++weight_index;
            }

            float *element = result[weight_index]->elements +
                (element_index - prev_count);

            *element = p->weights[weight_index]->elements[element_index - prev_count];
        }
#endif
        return result;
    }

    void mutate(n_netron *genome, float mutation_rate)
    {
        // TODO: Support for hidden layers
        int weights_count = genome->hidden_layers + 1;
        Matrix *weights = genome->weights;

        int weights_elements_count = 0;
        for (int i = 0; i < weights_count; i++)
        {
            weights_elements_count += element_count(weights[i]);
        }

        int weight_index = 0;
        int prev_count = 0;
        for (int element_index = 0;
            element_index < weights_elements_count;
            element_index++)
        {
            if (element_index - prev_count == element_count(weights[weight_index]))
            {
                prev_count += element_count(weights[weight_index]);
                ++weight_index;
            }

            // NOTE: Should mutate?
            if (n_randf() > mutation_rate)
            {
                continue;
            }

            float *element = weights[weight_index]->elements +
                (element_index - prev_count);

            *element += *element * (n_randf() - 0.5f) * 3 + (n_randf() - 0.5f);
        }
    }

    float calculate_fitness(Matrix result, Matrix output)
    {
        float value = 0;
        for (int i = 0; i < element_count(output); i++)
        {
            value += abs(output->elements[i] - result->elements[i]);
        }

        return value;
    }

    // TODO: Add states
    void run_genome_input(n_netron *genome,
        float *input,
        int input_count,
        int input_length)
    {
#ifdef N_ALLOC
        temp_memory temp = set_temp_mem(&netron_memory);
#endif

        Matrix result = predict(genome, input, input_count, input_length);
        if (netron_emit_genome) netron_emit_genome(genome, result);

#ifdef N_ALLOC
        end_temp_mem(temp);
#endif
    }

    void run_genome(n_netron *genome,
        Matrix input,
        Matrix output)
    {
#ifdef N_ALLOC
        temp_memory temp = set_temp_mem(&netron_memory);
#endif

        Matrix result = predict(genome, input->elements, input->rows, input->columns);
        genome->fitness = calculate_fitness(result, output);
        if (netron_emit_genome) netron_emit_genome(genome, result);

#ifdef N_ALLOC
        end_temp_mem(temp);
#endif
    }

    // TODO: Add selection count param?
    void update_generation(
        n_netron *genomes,
        int genome_units,
        int selection_count,
        float mutation_rate)
    {
        // NOTE: Select best selection_count genomes based on fitness
        select_best_genomes(genomes, genome_units, selection_count);

        // NOTE: Give mutation to other genomes
        for (int i = selection_count; i < genome_units; ++i)
        {
#ifdef N_ALLOC
            temp_memory temp = set_temp_mem(&netron_memory);
#endif

            // Get selection_count random genomes (best genomes)
            n_netron *genome_a = genomes + (int)((selection_count - 1) * n_randf());
            n_netron *genome_b = genomes + (int)((selection_count - 1) * n_randf());

            Matrix *child_weights = cross_over(genome_a, genome_b);

            n_netron *new_genome = genomes + i;

            // Copy weights from child to new genome
            for (int w = 0; w < new_genome->hidden_layers + 1; ++w)
            {
                set_elements(new_genome->weights[w], child_weights[w]->elements);
            }

            mutate(new_genome, mutation_rate);

#ifdef N_ALLOC
            end_temp_mem(temp);
#endif
        }
    }

    void run_generation(
        n_netron *genomes,
        int genome_units,
        int selection_count,
        float mutation_rate,
        Matrix input,
        Matrix output)
    {
        for (int i = 0; i < genome_units; i++)
        {
            run_genome(genomes + i, input, output);
        }

        update_generation(genomes, genome_units, selection_count, mutation_rate);
    }

    void learn(
        n_netron *netron,
        float *input, int input_count, int input_length,
        float *output, int output_count, int output_length)
    {
        Matrix input_matrix = new_matrix_elements(
            input_count, input_length, input);
        Matrix output_matrix = new_matrix_elements(
            output_count, output_length, output);

        if (!netron->weights_init)
        {
            setup_weights(netron, input_length, output_length);

            netron->weights_init = true;
        }

        for (int i = 0; i < netron->iterations; ++i)
        {
#ifdef N_ALLOC
            temp_memory temp = set_temp_mem(&netron_memory);
#endif

            // NOTE: Get pointers from function
            Matrix *results = forward(netron, input_matrix);

            // if(i >= 447) {
                print_elements(*results);
            // }
            // print_elements(netron_weights);

            Matrix errors = back(netron, results, input_matrix, output_matrix);

            int results_count = (netron->hidden_layers + 1) * 2;
            // TODO: Calculate fitness based on errors (squared sum error)?
            netron->fitness = calculate_fitness(results[results_count - 1], output_matrix);

            if (netron_emit) netron_emit("data", i, errors, results);

#ifdef N_ALLOC
            end_temp_mem(temp);
#endif
        }
    }

    Matrix *forward(n_netron *netron, Matrix input)
    {
        int weights_count = netron->hidden_layers + 1;
        int results_count = weights_count * 2;
        Matrix *results = (Matrix *)n_alloc(results_count * sizeof(*results));
        int result_index = 0;

        // input > hidden
        results[result_index] = multiply(input, netron->weights[0]);
        result_index++;
        results[result_index] = transform_sigmoid(results[0]);
        result_index++;

        // hidden > hidden
        for (int i = 1; i < netron->hidden_layers; i++)
        {
            results[result_index] = multiply(results[result_index - 1], netron->weights[i]);
            result_index++;
            results[result_index] = transform_sigmoid(results[result_index - 1]);
            result_index++;
        }

        // hidden > output
        results[result_index] = multiply(results[result_index - 1], netron->weights[weights_count - 1]);
        result_index++;
        results[result_index] = transform_sigmoid(results[result_index - 1]);

        return results;
    }

    Matrix back(n_netron *netron, Matrix *results,
                Matrix input, Matrix output)
    {
        int weights_count = netron->hidden_layers + 1;
        int results_count = weights_count * 2;

        // output > hidden
        Matrix error = subtract(output, results[results_count - 1]);
        Matrix delta = multiply_elements(transform_sigmoid_prime(results[results_count - 2]), error);
        Matrix changes = multiply_scalar(multiply(transpose(results[1]), delta), netron->learning_rate);
        matrix_add(netron->weights[weights_count - 1], changes);

        // hidden > hidden
        for (int i = 1; i < netron->hidden_layers; i++)
        {
            // delta = dot(multiply(weights[weights.length - i].transpose(), delta), results[results.length - (i + 1)].sum.transform(activatePrime));
            // changes = scalar(multiply(delta, results[results.length - (i + 1)].result.transpose()), learningRate);
            // weights[weights.length - (i + 1)] = add(weights[weights.length - (i + 1)], changes);
            delta = multiply_elements(transform_sigmoid_prime(results[results_count - (i + 1) - 2]),
                multiply(delta, transpose(netron->weights[weights_count - i])));
            changes = multiply_scalar(multiply(transpose(results[results_count - (i + 1) - 1]), delta), netron->learning_rate);
            matrix_add(netron->weights[weights_count - (i + 1)], changes);
        }

        // delta = dot(multiply(weights[1].transpose(), delta), results[0].sum.transform(activatePrime));
        // changes = scalar(multiply(delta, examples.input.transpose()), learningRate);
        // weights[0] = add(weights[0], changes);
        delta = multiply_elements(multiply(delta, transpose(netron->weights[1])), transform_sigmoid_prime(results[0]));
        changes = multiply_scalar(multiply(transpose(input), delta), netron->learning_rate);
        matrix_add(netron->weights[0], changes);

        return error;
    }

    Matrix predict(
        n_netron *netron,
        float *input,
        int input_count,
        int input_length)
    {
        Matrix input_matrix = new_matrix_elements(
            input_count, input_length, input);

        Matrix *results = forward(netron, input_matrix);

        int results_count = (netron->hidden_layers + 1) * 2;
        return results[results_count - 1];
    }

    void upload(n_netron *netron, Matrix *weights)
    {
        // TODO: Check if weights are compatible with network
        // TODO: Allocate space for weights?
        int weights_count = netron->hidden_layers + 1;
        for (int i = 0; i < weights_count; i++)
        {
            netron->weights[i] = weights[i];
        }
    }

    Matrix *download(n_netron *netron)
    {
        return netron->weights;
    }

    void write_weights_to_file(
        const char *file_name,
        n_netron *array,
        int array_len)
    {
        FILE *file = fopen(file_name, "wb");

        if (file)
        {
            int write_size = 0;

            for (int i = 0; i < array_len; ++i)
            {
                int struct_size = sizeof(n_netron);
                write_size += struct_size;

                for (int w = 0; w < netron_weights(array + i); w++)
                {
                    int weight_header_size = sizeof(n_matrix);
                    int weight_element_size = element_count(array[i].weights[w]) * sizeof(float);
                    write_size += weight_header_size + weight_element_size;
                }
            }

            char *data = (char *)malloc(write_size);

            char *dest = data;
            for (int i = 0; i < array_len; ++i)
            {
                int struct_size = sizeof(n_netron);
                char *source = (char *)(array + i);
                memcpy(dest, source, struct_size);
                dest += struct_size;

                for (int w = 0; w < netron_weights(array + i); w++)
                {
                    int matrix_size = sizeof(n_matrix) +
                        element_count(array[i].weights[w]) * sizeof(float);
                    char *source = (char *)array[i].weights[w];
                    memcpy(dest, source, matrix_size);
                    dest += matrix_size;
                }
            }

            fwrite(data, 1, write_size, file);
            fclose(file);
        }
    }

    n_netron *read_weights_from_file(const char *file_name)
    {
        FILE *file = fopen(file_name, "rb");

        char *data = 0;

        size_t file_size = 0;
        if (file)
        {
            fseek(file, 0, SEEK_END);
            file_size = ftell(file);
            fseek(file, 0, SEEK_SET);

            data = (char *)malloc(file_size + 1);
            fread(data, file_size, 1, file);
            data[file_size] = 0;

            fclose(file);
        }

        char *source = data;
        int netron_count = 0;
        n_netron *netron = 0;
        // NOTE: Find how many netrons are in file
        while (source < data + file_size)
        {
            netron = (n_netron *)source;
            int weights_count = netron_weights(netron);
            source += sizeof(n_netron);

            for (int w = 0; w < weights_count; w++)
            {
                Matrix matrix_ptr = (Matrix)source;
                int matrix_size = sizeof(n_matrix) + element_count(matrix_ptr) * sizeof(float);
                source += matrix_size;
            }

            ++netron_count;
        }

        n_netron *result = (n_netron *)malloc(netron_count * sizeof(*result));

        source = data;
        for (int i = 0; i < netron_count; ++i)
        {
            netron = (n_netron *)source;
            memcpy(result + i, netron, sizeof(n_netron));
            source += sizeof(n_netron);

            result[i].weights = (Matrix *)malloc(netron_weights(netron) * sizeof(*netron->weights));

            for (int w = 0; w < netron_weights(netron); w++)
            {
                Matrix matrix_ptr = (Matrix)source;
                source += sizeof(n_matrix);
                float *elements = (float *)source;
                source += element_count(matrix_ptr) * sizeof(float);

                result[i].weights[w] = new_matrix_elements(matrix_ptr->rows, matrix_ptr->columns, elements);
            }
        }

        return result;
    }

#ifdef __cplusplus
}
#endif