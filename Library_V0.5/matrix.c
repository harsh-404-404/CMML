#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359

#define ERROR_LOG(fmt, ...) fprintf(stderr, "[ERROR] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#define CHECK_NULL_MATRIX(m, ret_val)                                    \
    if ((m) == NULL)                                                     \
    {                                                                    \
        ERROR_LOG("Argument '%s' is NULL in function %s", #m, __func__); \
        return ret_val;                                                  \
    }

// index = (i * total_column + j)

typedef struct matrix
{
    int row;
    int col;
    float *value;
} matrix;

// ------------------------------------------------------------------
//  0. MEMORY MANAGEMENT
// ------------------------------------------------------------------

void *malloc_safe(size_t n)
{
    void *p = malloc(n);
    if (p == NULL)
    {
        ERROR_LOG("Fatal: Out of memory! Failed to allocate %zu bytes.", n);
        exit(1);
    }
    return p;
}

// ------------------------------------------------------------------
//  1. CONSTRUCTORS & DESTRUCTORS
//  (Lifecycle Management)
// ------------------------------------------------------------------

matrix *new_matrix(int a, int b)
{
    matrix *new_matrix = malloc_safe(sizeof(matrix));
    new_matrix->row = a;
    new_matrix->col = b;
    new_matrix->value = malloc_safe(sizeof(float) * a * b);
    return new_matrix;
}

void free_matrix(matrix *m)
{
    if (m == NULL)
    {
        return;
    }
    free(m->value);
    free(m);
}

matrix *new_random_matrix(int a, int b, float min, float max)
{
    matrix *new = new_matrix(a, b);
    new->row = a;
    new->col = b;
    int count = a * b;
    for (int i = 0; i < count; ++i)
    {
        new->value[i] = ((float)rand() / (float)RAND_MAX) * (max - min) + min;
    }
    return new;
}

matrix *new_gaussian_matrix(int a, int b)
{
    matrix *m = new_matrix(a, b);
    int count = a * b;
    for (int i = 0; i < count; ++i)
    {
        m->value[i] = normal_rand();
    }
    return m;
}

{
    matrix *m = new_matrix(n, n);
    fill_matrix(m, 0.0f);
    for (int i = 0; i < n; i++)
    {
        m->value[i * n + i] = 1.0f;
    }
    return m;
}

matrix *zeros(int r, int c)
{
    matrix *m = new_matrix(r, c);
    fill_matrix(m, 0.0f);
    return m;
}

matrix *array_matrix(float *arr, int a, int b)
{
    CHECK_NULL_MATRIX(arr, NULL);
    matrix *c = new_matrix(a, b);
    for (int i = 0; i < a; ++i)
    {
        for (int j = 0; j < b; ++j)
        {
            c->value[i * (b) + j] = arr[i * (b) + j];
        }
    }
    return c;
}

matrix *copy_matrix(const matrix *old)
{
    CHECK_NULL_MATRIX(old, NULL);
    int count = (old->col) * (old->row);
    matrix *new = new_matrix(old->row, old->col);
    memcpy(new->value, old->value, sizeof(float) * count);
    return new;
}

// ------------------------------------------------------------------
//  2. ACCESSORS & UTILITIES
//  (Inspection and Debugging)
// ------------------------------------------------------------------

int get_rows(matrix *m)
{
    CHECK_NULL_MATRIX(m, 0);
    return m->row;
}

int get_cols(matrix *m)
{
    CHECK_NULL_MATRIX(m, 0);
    return m->col;
}

float get_matrix_element(matrix *m, int i, int j)
{
    CHECK_NULL_MATRIX(m, NAN);
    if (i < 0 || i >= m->row || j < 0 || j >= m->col)
    {
        ERROR_LOG("Index out of bounds: (%d, %d). Matrix is (%d, %d)",
                  i, j, m->row, m->col);
        return NAN;
    }
    return m->value[i * m->col + j];
}

void print_matrix(matrix *ptr)
{
    CHECK_NULL_MATRIX(ptr, );
    for (int i = 0; i < ptr->row; ++i)
    {
        for (int j = 0; j < ptr->col; ++j)
        {
            printf("%g  ", ptr->value[i * (ptr->col) + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_shape(matrix *m)
{
    CHECK_NULL_MATRIX(m, );
    printf("(%d x %d)\n", m->row, m->col);
}

int matrix_check_equal(matrix *a, matrix *b, float tolerance)
{
    CHECK_NULL_MATRIX(a, 0);
    CHECK_NULL_MATRIX(b, 0);
    if (a->row != b->row || a->col != b->col)
    {
        return 0;
    }
    int count = a->row * a->col;
    for (int i = 0; i < count; ++i)
    {
        if (fabsf(a->value[i] - b->value[i]) > tolerance)
        {
            return 0;
        }
    }
    return 1;
}

// ------------------------------------------------------------------
//  3. DATA MANIPULATION
//  (Setters, Reshaping, Filling)
// ------------------------------------------------------------------

void set_matrix(matrix *ptr, int a, int b, float value)
{
    CHECK_NULL_MATRIX(ptr, );
    int col = ptr->col;
    int row = ptr->row;
    if (a >= row || a < 0 || b >= col || b < 0)
    {
        ERROR_LOG("Index out of bounds: Setting (%d, %d) on matrix of size (%d, %d)", a, b, row, col);
        return;
    }
    ptr->value[a * (col) + b] = value;
}

void fill_matrix(matrix *ptr, float value)
{
    CHECK_NULL_MATRIX(ptr, );
    int temp = (ptr->col) * (ptr->row);
    for (int i = 0; i < temp; ++i)
    {
        ptr->value[i] = value;
    }
}

matrix *reshape_matrix(const matrix *m, int new_rows, int new_cols)
{
    CHECK_NULL_MATRIX(m, NULL);
    if (new_rows * new_cols != (m->col) * (m->row))
    {
        ERROR_LOG("Invalid matrix Dimension ! can not reshape %d x %d matrix into %d x %d", m->row, m->col, new_rows, new_cols);
        return NULL;
    }
    matrix *new = copy_matrix(m);
    new->row = new_rows;
    new->col = new_cols;
    return new;
}

int reshape_matrix_inplace(matrix *m, int new_rows, int new_cols)
{
    CHECK_NULL_MATRIX(m, -1);
    if (new_rows * new_cols != (m->col) * (m->row))
    {
        ERROR_LOG("Invalid matrix Dimension ! can not reshape %d x %d matrix into %d x %d", m->row, m->col, new_rows, new_cols);
        return -1;
    }
    m->row = new_rows;
    m->col = new_cols;
    return 0;
}

matrix *get_row(matrix *m, int r)
{
    CHECK_NULL_MATRIX(m, NULL);
    if (r >= m->row || r < 0)
    {
        ERROR_LOG("Invalid row index : %d , Expected index >= 0 and index < %d\n", r, m->row);
        return NULL;
    }
    matrix *new = new_matrix(1, m->col);
    memcpy(new->value, &m->value[r * m->col], sizeof(float) * m->col);
    return new;
}

matrix *get_col(matrix *m, int c)
{
    CHECK_NULL_MATRIX(m, NULL);
    if (c >= m->col || c < 0)
    {
        ERROR_LOG("Invalid column index : %d , Expected index >= 0 and index < %d\n", c, m->col);
        return NULL;
    }
    matrix *new = new_matrix(m->row, 1);
    for (int i = 0; i < m->row; ++i)
    {
        new->value[i] = m->value[i * (m->col) + c];
    }
    return new;
}

matrix *get_slice(matrix *m, int r_start, int r_end, int c_start, int c_end)
{
    CHECK_NULL_MATRIX(m, NULL);
    int row = m->row;
    int col = m->col;
    if (r_start >= r_end || c_start >= c_end)
    {
        ERROR_LOG("Invalid slice size (0 or negative). Rows: %d->%d, Cols: %d->%d", r_start, r_end, c_start, c_end);
        return NULL;
    }
    if (r_start < 0 || r_end > row || c_start < 0 || c_end > col)
    {
        ERROR_LOG("Slice out of bounds. Matrix is %dx%d, requested slice ends at %dx%d", row, col, r_end, c_end);
        return NULL;
    }
    matrix *s = new_matrix(r_end - r_start, c_end - c_start);
    for (int i = r_start; i < r_end; ++i)
    {
        memcpy(&s->value[(i - r_start) * s->col], &m->value[i * col + c_start], sizeof(float) * s->col);
    }
    return s;
}

// ------------------------------------------------------------------
//  4. ARITHMETIC OPERATIONS
//  (Element-wise Math)
// ------------------------------------------------------------------

matrix *add_matrix(const matrix *a, const matrix *b)
{
    CHECK_NULL_MATRIX(a, NULL);
    CHECK_NULL_MATRIX(b, NULL);
    if (a->col != b->col || a->row != b->row)
    {
        ERROR_LOG("Matrix dimensions didn't match!\n");
        return NULL;
    }
    matrix *c = new_matrix(a->row, a->col);
    for (int i = 0; i < (a->row) * (a->col); ++i)
    {
        c->value[i] = a->value[i] + b->value[i];
    }
    return c;
}

int add_matrix_inplace(matrix *a, matrix *b)
{
    CHECK_NULL_MATRIX(a, -1);
    CHECK_NULL_MATRIX(b, -1);
    if (a->col != b->col || a->row != b->row)
    {
        ERROR_LOG("Matrix dimensions didn't match!\n");
        return -1;
    }
    for (int i = 0; i < (a->row) * (a->col); ++i)
    {
        a->value[i] += b->value[i];
    }
    return 0;
}

matrix *subtract_matrix(const matrix *a, const matrix *b)
{
    CHECK_NULL_MATRIX(a, NULL);
    CHECK_NULL_MATRIX(b, NULL);
    if (a->col != b->col || a->row != b->row)
    {
        ERROR_LOG("Matrix dimensions didn't match!\n");
        return NULL;
    }
    matrix *c = new_matrix(a->row, a->col);
    int count = (a->row) * (a->col);
    for (int i = 0; i < count; ++i)
    {
        c->value[i] = a->value[i] - b->value[i];
    }
    return c;
}

int subtract_matrix_inplace(matrix *a, matrix *b)
{
    CHECK_NULL_MATRIX(a, -1);
    CHECK_NULL_MATRIX(b, -1);
    if (a->col != b->col || a->row != b->row)
    {
        ERROR_LOG("Matrix dimensions didn't match!\n");
        return -1;
    }
    int count = (a->row) * (a->col);
    for (int i = 0; i < count; ++i)
    {
        a->value[i] -= b->value[i];
    }
    return 0;
}

matrix *hadamard_matrix(const matrix *a, const matrix *b)
{
    CHECK_NULL_MATRIX(a, NULL);
    CHECK_NULL_MATRIX(b, NULL);
    if (a->col != b->col || a->row != b->row)
    {
        ERROR_LOG("Matrix dimensions didn't match!\n");
        return NULL;
    }
    matrix *c = new_matrix(a->row, a->col);
    int count = (a->row) * (a->col);
    for (int i = 0; i < count; ++i)
    {
        c->value[i] = a->value[i] * b->value[i];
    }
    return c;
}

int hadamard_matrix_inplace(matrix *a, matrix *b)
{
    CHECK_NULL_MATRIX(a, -1);
    CHECK_NULL_MATRIX(b, -1);
    if (a->col != b->col || a->row != b->row)
    {
        ERROR_LOG("Matrix dimensions didn't match!\n");
        return -1;
    }
    int count = (a->row) * (a->col);
    for (int i = 0; i < count; ++i)
    {
        a->value[i] *= b->value[i];
    }
    return 0;
}

matrix *scalar_multiply(matrix *a, const float b)
{
    CHECK_NULL_MATRIX(a, NULL);
    matrix *new = new_matrix(a->row, a->col);
    int count = (a->row) * (a->col);
    for (int i = 0; i < count; ++i)
    {
        new->value[i] = a->value[i] * b;
    }
    return new;
}

int scalar_multiply_inplace(matrix *a, const float b)
{
    CHECK_NULL_MATRIX(a, -1);
    int count = (a->row) * (a->col);
    for (int i = 0; i < count; ++i)
    {
        a->value[i] *= b;
    }
    return 0;
}

matrix *scalar_add(matrix *a, const float b)
{
    CHECK_NULL_MATRIX(a, NULL);
    matrix *new = new_matrix(a->row, a->col);
    int count = (a->row) * (a->col);
    for (int i = 0; i < count; ++i)
    {
        new->value[i] = a->value[i] + b;
    }
    return new;
}

int scalar_add_inplace(matrix *a, const float b)
{
    CHECK_NULL_MATRIX(a, -1);
    int count = (a->row) * (a->col);
    for (int i = 0; i < count; ++i)
    {
        a->value[i] += b;
    }
    return 0;
}

// ------------------------------------------------------------------
//  5. MATRIX ALGEBRA
//  (Dot Products, Transpose, Linear Algebra)
// ------------------------------------------------------------------

matrix *multiply_matrix(const matrix *a, const matrix *b)
{
    CHECK_NULL_MATRIX(a, NULL);
    CHECK_NULL_MATRIX(b, NULL);
    if (a->col != b->row)
    {
        ERROR_LOG("Matrix dimensions didn't match!\n");
        return NULL;
    }
    matrix *c = new_matrix(a->row, b->col);
    fill_matrix(c, 0);
    int row1 = a->row;
    int col1 = a->col;
    int row2 = b->row;
    int col2 = b->col;
    for (int i = 0; i < row1; ++i)
    {
        for (int j = 0; j < col1; ++j)
        {
            float r = a->value[i * col1 + j];
            for (int k = 0; k < col2; ++k)
            {
                c->value[i * col2 + k] += r * b->value[j * col2 + k];
            }
        }
    }
    return c;
}

float dot_product(const matrix *a, const matrix *b)
{
    CHECK_NULL_MATRIX(a, NAN);
    CHECK_NULL_MATRIX(b, NAN);
    int a_row = a->row;
    int b_row = b->row;
    int a_col = a->col;
    int b_col = b->col;
    if (a_row != 1 && a_col != 1)
    {
        ERROR_LOG("Dot product input 'a' is not a vector (Dimensions: %dx%d)", a_row, a_col);
        return NAN;
    }
    if (b_row != 1 && b_col != 1)
    {
        ERROR_LOG("Dot product input 'b' is not a vector (Dimensions: %dx%d)", b_row, b_col);
        return NAN;
    }
    if (a_row != b_row || a_col != b_col)
    {
        ERROR_LOG("vectors does not have same dimentions (%dx%d) vs (%dx%d)", a_row, a_col, b_row, b_col);
        return NAN;
    }
    int temp = b_col * b_row;
    float sum = 0;
    for (int i = 0; i < temp; ++i)
    {
        sum += a->value[i] * b->value[i];
    }
    return sum;
}

matrix *transpose_matrix(matrix *a)
{
    CHECK_NULL_MATRIX(a, NULL);
    int row1 = a->row;
    int col1 = a->col;
    matrix *b = new_matrix(col1, row1);
    for (int i = 0; i < row1; ++i)
    {
        for (int j = 0; j < col1; ++j)
        {
            b->value[j * (row1) + i] = a->value[i * (col1) + j];
        }
    }
    return b;
}

float determinant_matrix(const matrix *m)
{
    CHECK_NULL_MATRIX(m, NAN);
    int row = m->row;
    int col = m->col;
    if (row != col)
    {
        ERROR_LOG("Determinants are only defined for squre metrix and given metrix has dimention: (%dx%d)", row, col);
        return NAN;
    }
    matrix *k = copy_matrix(m);
    int end_det = 1;
    for (int i = 0; i < row; ++i)
    {
        float pivot = k->value[i * col + i];
        if (pivot == 0)
        {
            int temp;
            for (temp = i + 1; temp < row; ++temp)
            {
                if (k->value[temp * col + i] != 0)
                {
                    swap_rows(k, temp, i);
                    pivot = k->value[i * col + i];
                    end_det *= (-1);
                    break;
                }
            }
            if (temp == row)
            {
                free_matrix(k);
                return 0;
            }
        }
        for (int j = i + 1; j < row; ++j)
        {
            float entry = k->value[j * col + i];
            add_rows(k, j, i, (-1) * (entry / pivot));
        }
    }
    float det = 1;
    for (int i = 0; i < row; ++i)
    {
        det *= k->value[i * col + i];
    }
    det *= end_det;
    free_matrix(k);
    return det;
}

matrix *inverse_matrix(matrix *a)
{
    CHECK_NULL_MATRIX(a, NULL);
    if (a->row != a->col)
    {
        ERROR_LOG("Inverse matrix is only defined for square matrix but given metrix has dimentin: (%dx%d)", a->row, a->col);
        return NULL;
    }
    float determinant = determinant_matrix(a);
    if (determinant == 0)
    {
        ERROR_LOG("Inverse matrix is only defined for square matrix with non-zero determinant but for given matrix determinant = %d", 0);
        return NULL;
    }
    matrix *adj = adjoint_matrix(a);
    matrix *invr = scalar_multiply(adj, 1 / determinant);
    free_matrix(adj);
    return invr;
}

matrix *cofactor_matrix(matrix *a)
{
    CHECK_NULL_MATRIX(a, NULL);
    int row = a->row;
    int col = a->col;
    if (row != col)
    {
        ERROR_LOG("Cofactors are only defined for squre metrix and given metrix has dimention: (%dx%d)", row, col);
        exit(1);
    }
    matrix *new = new_matrix(a->row, a->col);
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            int sign = ((i + j) % 2 == 0) ? 1 : -1;
            matrix *minor = minor_matrix(a, i, j);
            new->value[i * col + j] = sign * determinant_matrix(minor);
            free_matrix(minor);
        }
    }
    return new;
}

matrix *adjoint_matrix(matrix *a)
{
    CHECK_NULL_MATRIX(a, NULL);
    matrix *cofact = cofactor_matrix(a);
    matrix *adj = transpose_matrix(cofact);
    free_matrix(cofact);
    return adj;
}

matrix *minor_matrix(matrix *a, int i, int j)
{
    CHECK_NULL_MATRIX(a, NULL);
    int row = a->row;
    int col = a->col;
    if (row != col)
    {
        ERROR_LOG("Minor matrices are only defined for squre metrix and given metrix has dimention: (%dx%d)", row, col);
        return NULL;
    }
    if (i > row - 1 || i < 0 || j > col - 1 || j < 0)
    {
        ERROR_LOG("invalid indexes %d, %d", i, j);
        return NULL;
    }
    matrix *minor = new_matrix(row - 1, col - 1);
    int append_count = 0;
    for (int l = 0; l < row; ++l)
    {
        if (l == i)
        {
            continue;
        }
        for (int k = 0; k < col; ++k)
        {
            if (k == j)
            {
                continue;
            }
            minor->value[append_count] = a->value[l * col + k];
            ++append_count;
        }
    }
    return minor;
}

void swap_rows(matrix *m, int r1, int r2)
{
    CHECK_NULL_MATRIX(m, );
    int col = m->col;
    int row = m->row;
    if (r1 >= row - 1 || r1 < 0 || r2 >= row - 1 || r2 < 0)
    {
        ERROR_LOG("invalid indexes %d, %d", r1, r2);
        return;
    }
    if (r1 == r2)
        return;
    size_t bytes = sizeof(float) * col;
    if (col < 10000)
    {
        float temp[col];
        memcpy(temp, &(m->value[r1 * col]), bytes);
        memcpy(&(m->value[r1 * col]), &(m->value[r2 * col]), bytes);
        memcpy(&(m->value[r2 * col]), temp, bytes);
        return;
    }
    float *temp = malloc_safe(bytes);
    memcpy(temp, &(m->value[r1 * col]), bytes);
    memcpy(&(m->value[r1 * col]), &(m->value[r2 * col]), bytes);
    memcpy(&(m->value[r2 * col]), temp, bytes);
    free(temp);
}

void multiply_row(matrix *m, int r, float scalar)
{
    CHECK_NULL_MATRIX(m, );
    int col = m->col;
    if (r >= m->row || r < 0)
    {
        ERROR_LOG("invalid row indexes %d", r);
        return;
    }
    float *row = &m->value[r * m->col];
    for (int i = 0; i < col; ++i)
    {
        row[i] *= scalar;
    }
}

void add_rows(matrix *m, int target_r, int source_r, float scale)
{
    CHECK_NULL_MATRIX(m, );
    int col = m->col;
    int row = m->row;
    if (source_r >= row || source_r < 0 || target_r >= row || target_r < 0)
    {
        ERROR_LOG("invalid row index %d , %d", target_r, source_r);
        return;
    }
    float *t = &m->value[target_r * col];
    float *s = &m->value[source_r * col];
    for (int i = 0; i < col; ++i)
    {
        t[i] += scale * s[i];
    }
}

// ------------------------------------------------------------------
//  6. MACHINE LEARNING UTILITIES
//  (Activation, Broadcasting, Initialization Tools)
// ------------------------------------------------------------------

matrix *apply_function(matrix *a, float (*f)(float))
{
    CHECK_NULL_MATRIX(a, NULL);
    CHECK_NULL_MATRIX(f, NULL);
    int count = a->col * a->row;
    matrix *b = new_matrix(a->row, a->col);
    for (int i = 0; i < count; ++i)
    {
        b->value[i] = f(a->value[i]);
    }
    return b;
}

int apply_function_inplace(matrix *a, float (*f)(float))
{
    CHECK_NULL_MATRIX(a, -1);
    CHECK_NULL_MATRIX(f, -1);
    int count = a->col * a->row;
    for (int i = 0; i < count; ++i)
    {
        a->value[i] = f(a->value[i]);
    }
    return 0;
}

matrix *broadcast_add(matrix *m, matrix *b)
{
    CHECK_NULL_MATRIX(m, NULL);
    CHECK_NULL_MATRIX(b, NULL);
    if (b->col != 1)
    {
        ERROR_LOG("Broadcasting Error: Bias must be a column vector (Nx1). Got (%dx%d)", b->row, b->col);
        return NULL;
    }
    if (b->row != m->row)
    {
        ERROR_LOG("Broadcasting Error: Dimension mismatch. Matrix Rows %d != Bias Rows %d)", m->row, b->row);
        return NULL;
    }
    matrix *c = copy_matrix(m);
    int row = m->row;
    int col = m->col;
    for (int i = 0; i < row; ++i)
    {
        int ph = i * col;
        float bias = b->value[i];
        for (int j = 0; j < col; ++j)
        {
            c->value[ph + j] += bias;
        }
    }
    return c;
}

int broadcast_add_inplace(matrix *m, matrix *b)
{
    CHECK_NULL_MATRIX(m, -1);
    CHECK_NULL_MATRIX(b, -1);
    if (b->col != 1)
    {
        ERROR_LOG("Broadcasting Error: Bias must be a column vector (Nx1). Got (%dx%d)", b->row, b->col);
        return -1;
    }
    if (b->row != m->row)
    {
        ERROR_LOG("Broadcasting Error: Dimension mismatch. Matrix Rows %d != Bias Rows %d)", m->row, b->row);
        return -1;
    }
    int row = m->row;
    int col = m->col;
    for (int i = 0; i < row; ++i)
    {
        int ph = i * col;
        float bias = b->value[i];
        for (int j = 0; j < col; ++j)
        {
            m->value[ph + j] += bias;
        }
    }
    return 0;
}

int argmax(matrix *a)
{
    CHECK_NULL_MATRIX(a, -1);
    int max_index = 0;
    float max_value = a->value[0];
    int total_elements = a->row * a->col;
    for (int i = 1; i < total_elements; ++i)
    {
        if (max_value < a->value[i])
        {
            max_value = a->value[i];
            max_index = i;
        }
    }
    return max_index;
}

float normal_rand()
{
    static int have_extra = 0;
    static float extra;
    if (have_extra)
    {
        have_extra = 0;
        return extra;
    }
    have_extra = 1;
    double num1;
    do
    {
        num1 = (rand() / (double)RAND_MAX);
    } while (num1 < 1e-100);
    double num2 = (rand() / (double)RAND_MAX);
    double number_hold_1 = sqrt(-2 * log(num1));
    double number_hold_2 = 2 * PI * num2;
    extra = (number_hold_1 * sin(number_hold_2));
    return (float)(number_hold_1 * cos(number_hold_2));
}

