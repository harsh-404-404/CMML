#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h> // for size_t

// ------------------------------------------------------------------
//  TYPE DEFINITION
// ------------------------------------------------------------------
typedef struct matrix {
    int row;
    int col;
    float *value;
} matrix;

// ------------------------------------------------------------------
//  0. MEMORY MANAGEMENT
// ------------------------------------------------------------------
void*   malloc_safe(size_t n);

// ------------------------------------------------------------------
//  1. CONSTRUCTORS & DESTRUCTORS
//  (Lifecycle Management)
// ------------------------------------------------------------------
matrix* new_matrix(int row, int col);
matrix* new_random_matrix(int row, int col, float min, float max);
matrix* new_gaussian_matrix(int row, int col);
matrix* eye(int n);
matrix* zeros(int row, int col);
matrix* array_matrix(float *arr, int row, int col);
matrix* copy_matrix(const matrix *old);

void    free_matrix(matrix* m);

// ------------------------------------------------------------------
//  2. ACCESSORS & UTILITIES
//  (Inspection and Debugging)
// ------------------------------------------------------------------
int     get_rows(matrix* m);
int     get_cols(matrix* m);
float   get_matrix_element(matrix* m, int r, int c); 

void    print_matrix(matrix *m);
void    print_shape(matrix* m);
int     matrix_check_equal(matrix* a, matrix* b, float tolerance);

// ------------------------------------------------------------------
//  3. DATA MANIPULATION
//  (Setters, Reshaping, Filling)
// ------------------------------------------------------------------
void    set_matrix(matrix *m, int r, int c, float value);
void    fill_matrix(matrix *m, float value);

// Reshaping
matrix* reshape_matrix(const matrix* m, int new_rows, int new_cols);
int     reshape_matrix_inplace(matrix *m, int new_rows, int new_cols);

// Slicing (Extracting Sub-regions)
matrix* get_row(matrix* m, int r);
matrix* get_col(matrix* m, int c);
matrix* get_slice(matrix* m, int r_start, int r_end, int c_start, int c_end);

// ------------------------------------------------------------------
//  4. ARITHMETIC OPERATIONS
//  (Element-wise Math)
// ------------------------------------------------------------------

// Addition
matrix* add_matrix(const matrix *a, const matrix *b);
int     add_matrix_inplace(matrix *dest, matrix *src);

// Subtraction
matrix* subtract_matrix(const matrix *a, const matrix *b);
int     subtract_matrix_inplace(matrix *dest, matrix *src);

// Element-wise Multiplication (Hadamard)
matrix* hadamard_matrix(const matrix *a, const matrix *b);
int     hadamard_matrix_inplace(matrix *dest, matrix *src);

// Scalar Operations
matrix* scalar_multiply(matrix *a, const float b);
int     scalar_multiply_inplace(matrix *a, const float b);
matrix* scalar_add(matrix *a, const float b);
int     scalar_add_inplace(matrix *a, const float b);

// ------------------------------------------------------------------
//  5. MATRIX ALGEBRA
//  (Dot Products, Transpose, Linear Algebra)
// ------------------------------------------------------------------
matrix* multiply_matrix(const matrix *a, const matrix *b);
float   dot_product(const matrix* a, const matrix* b);
matrix* transpose_matrix(matrix *a);

// Linear Algebra Tools
float   determinant_matrix(const matrix *m);
matrix* inverse_matrix(matrix* a);
matrix* cofactor_matrix(matrix* a);
matrix* adjoint_matrix(matrix* a);
matrix* minor_matrix(matrix* a, int i, int j);

// Elementary Row Operations
void    swap_rows(matrix* m, int r1, int r2);
void    multiply_row(matrix* m, int r, float scalar);
void    add_rows(matrix* m, int target_r, int source_r, float scale);

// ------------------------------------------------------------------
//  6. MACHINE LEARNING UTILITIES
//  (Activation, Broadcasting, Initialization Tools)
// ------------------------------------------------------------------
matrix* apply_function(matrix* a, float (*f)(float));
int     apply_function_inplace(matrix* a, float (*f)(float));

matrix* broadcast_add(matrix* m, matrix* b);
int     broadcast_add_inplace(matrix* m, matrix* b);

int     argmax(matrix* a);


#endif // MATRIX_H