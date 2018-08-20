#pragma once

#define kilobytes(value) ((value)*1024LL)
#define megabytes(value) (kilobytes(value)*1024LL)
#define gigabytes(value) (megabytes(value)*1024LL)
#define terabytes(value) (gigabytes(value)*1024LL)

#ifndef assert
#define assert(expression) if(!(expression)) {*(int *)0 = 0;}
#endif

typedef struct
{
    size_t used;
    size_t size;
    char *base;
} memory_block;

typedef struct
{
    memory_block *block;
    size_t used_at;
} temp_memory;

static void
init_block(memory_block *block, size_t size)
{
    block->used = 0;
    block->size = size;
    block->base = (char *)calloc(size, sizeof(char));
}

static temp_memory
set_temp_mem(memory_block *block)
{
    assert(block);
    temp_memory result = {0};
    result.block = block;
    result.used_at = block->used;
    return result;
}

inline void
end_temp_mem(temp_memory temp)
{
    memory_block *block = temp.block;
    assert(block->used <= block->size);
    block->used -= block->used - temp.used_at;
}

#define zero_struct(instance) zero_size(sizeof(instance), (instance))
inline void zero_size(size_t size, void *source)
{
    char *byte = (char *)source;
    while(size--) *byte++ = 0;
}

#define alloc_struct(mem, type) (type *)alloc_(mem, sizeof(type))
#define alloc_array(mem, count, type) (type *)alloc_(mem, count*sizeof(type))
#define alloc_size(mem, size) alloc_(mem, size)

static void *
alloc_(memory_block *mem, int size)
{
    assert(mem->base);
    assert((mem->used + size) <= mem->size);
    void *result = mem->base + mem->used;
    mem->used += size;
    // TODO: not optimized clear!
    zero_size(size, result);

    return result;
}
