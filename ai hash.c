#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <stdint.h>
#include <limits.h>

//条件编译
#define USE_MARKLE_TREE 1// 是否使用Merkle树加速搜索，0表示不使用，1表示使用
#define USE_AFFINE_QUANTIZATION 0// 是否使用仿射量化，0表示不使用，1表示使用
#define USE_MANHATTAN_DISTANCE 1// 是否使用曼哈顿距离，0表示使用欧氏距离，1表示使用曼哈顿距离

#define n 1024 // 矩阵维度
#define num_subspaces 8 // 子空间数量
#define num_centroids_per_subspace 16 // 每个子空间的质心数量
#define sub_dim (n / num_subspaces) // 每个子空间的维度
#define PQ_BITS 4

double get_time();
double** malloc_2d_double(int rows, int cols);
void free_2d_double(double** array, int dim1);
int **malloc_2d_int(int rows, int cols);
uint8_t **malloc_2d_uint8(int rows, int cols);
uint16_t **malloc_2d_uint16(int rows, int cols);
void free_2d_int(int **arr, int dim1);
void free_2d_uint8(uint8_t **arr, int dim1);
void free_2d_uint16(uint16_t **arr, int dim1);
double ***malloc_3d_double(int dim1, int dim2, int dim3);
void free_3d_double(double ***array, int dim1, int dim2);
uint8_t ***malloc_3d_uint8(int dim1, int dim2, int dim3);
void free_3d_uint8(uint8_t ***arr, int dim1, int dim2);
void print2DArray(int rows, int cols, double** arr);
void initialize_matrix(int m, double** mat);
void initialize_zero_matrix(int m, double** mat);
void original_multiply(int number, double** a, double** b, double** c);
void generate_random_matrix(int dim, double **mat);
double distance_calculate(double *A, double *B, int dim);
int int_distance_calculate(double *A, double *B, int dim);
int uint8_distance_calculate(uint8_t *A, uint8_t *B, int dim);
double dot_product(double *A, double *B, int dim);
static double median_from_values(const double *values, int count);
static int extract_prefix_bits(int code, int processed_dims);
void k_means(int scale, double ***center_save, double **A, int subspace_number, int center_number_in_per_subspace);
void make_the_list(int scale, double **A, int **list, double ***center_save, int subspace_number,  int center_number_in_per_subspace);
void built_table(int scale, double **B, double ***final_list, int subspace_number, int center_number_in_per_subspace, double ***center_save);
void generate_result(int scale, int **list, double ***final_list, double **C_PQ, int subspace_number, int center_number_in_per_subspace);
void pq_multiply(int scale, double** A_test, double** A, double** B, double** C_PQ, double *total_time_pq, double** C_original);
double calculate_relative_error(double **reference, double **approx, int dim);
void affine_quantization(int scale, double **A, uint8_t **A_quantized, double *min_val, double *max_val);
void k_means_with_affine_quantization(int scale, double ***center_save, double **A,
                                     int subspace_number, int center_number_in_per_subspace);
void make_markle_tree(double ***center_save, int **markle_tree_point, int subspace_number);
void make_markle_tree_first_dimension(int **center_list, double ***center_save, int **markle_tree_point,
                                      double **markle_tree_1dimension, int subspace_number);
void make_markle_tree_second_dimension(int **center_list, double ***center_save, int **markle_tree_point,
                                       double **markle_tree_2dimension, int subspace_number);
void make_markle_tree_third_dimension(int **center_list, double ***center_save, int **markle_tree_point,
                                      double **markle_tree_3dimension, int subspace_number);
void make_markle_tree_fourth_dimension(int **center_list, double ***center_save, int **markle_tree_point,
                                       double **markle_tree_4dimension, int subspace_number);
void make_the_list_with_markle_tree(int scale, double **A, int **list, double ***center_save, int **center_list,
                                    int subspace_number, int **markle_tree_point,
                                    double **markle_tree_1dimension, double **markle_tree_2dimension,
                                    double **markle_tree_3dimension, double **markle_tree_4dimension);

int main()
{
    srand((unsigned int)time(NULL));
    double **A = malloc_2d_double(n, n), 
           **A_test = malloc_2d_double(n, n), 
           **B = malloc_2d_double(n, n), 
           **C_original = malloc_2d_double(n, n), 
           **C_PQ = malloc_2d_double(n, n);
    double total_time_classic = 0.0;
    initialize_matrix(n, A);
    initialize_matrix(n, A_test);
    initialize_matrix(n, B);
    initialize_zero_matrix(n, C_original);
    initialize_zero_matrix(n, C_PQ);
///////////////////////////////////original multiply//////////////////////////////////////
    // double start = get_time();
    // original_multiply(n, A, B, C_original);
    // total_time_classic += get_time() - start;
    //printf("Classic: Total time = %.6f s\n", total_time_classic);
////////////////////////////////////PQ multiply//////////////////////////////////////
    double total_time_pq = 0.0;
    pq_multiply(n, A_test, A, B, C_PQ, &total_time_pq, C_original);
    // printf("Classic: Total time = %.6f s\n", total_time_classic);
    // printf("PQ: Total time = %.6f s\n", total_time_pq);
    
    // printf("Classic result (first 3x3):\n");
    // print2DArray(3, 3, C_original);
    // printf("PQ result (first 3x3):\n");
    // print2DArray(3, 3, C_PQ);
    // printf("speed up: %.2f x\n", total_time_classic / total_time_pq);
    // double relative_error = calculate_relative_error(C_original, C_PQ, n);
    // printf("Relative Frobenius error = %.6e\n", relative_error);

    // double error = calculate_relative_error(C_original, C_PQ, n);
    // printf("Relative Frobenius error = %.6e\n", error);

    free_2d_double(A, n);
    free_2d_double(A_test, n);
    free_2d_double(B, n);
    free_2d_double(C_original, n);
    free_2d_double(C_PQ, n);
    
    return 0;
}
double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}
double** malloc_2d_double(int rows, int cols)
{
    // 分配行指针数组
    double **array = (double**)malloc(rows * sizeof(double*));
    // 一次性分配所有元素的内存（连续内存块）
    double *data = (double*)calloc(rows * cols, sizeof(double));
    // 将行指针指向对应的内存位置
    for (int i = 0; i < rows; i++)
    {
        array[i] = data + i * cols;
    }
    return array;
}
void free_2d_double(double** arr, int dim1)
{
    if (arr == NULL) return;
    
    if (arr[0] != NULL) {
        free(arr[0]);  // 释放整个连续内存块
    }
    
    free(arr);  // 释放行指针数组
    arr = NULL;
}
int **malloc_2d_int(int rows, int cols)
{
    int **array = (int**)malloc(rows * sizeof(int*));
    for (int i = 0; i < rows; i++)
    {
        array[i] = (int*)malloc(cols * sizeof(int));
    }
    return array;
}
uint8_t **malloc_2d_uint8(int rows, int cols)
{
    uint8_t **array = (uint8_t**)malloc(rows * sizeof(uint8_t*));
    for (int i = 0; i < rows; i++)
    {
        array[i] = (uint8_t*)malloc(cols * sizeof(uint8_t));
    }
    return array;
}
uint16_t **malloc_2d_uint16(int rows, int cols)
{
    uint16_t **array = (uint16_t**)malloc(rows * sizeof(uint16_t*));
    for (int i = 0; i < rows; i++)
    {
        array[i] = (uint16_t*)malloc(cols * sizeof(uint16_t));
    }
    return array;
}
void free_2d_int(int **arr, int dim1)
{
    if (arr == NULL) return;
    
    for (int i = 0; i < dim1; i++)
    {
        if (arr[i] != NULL)
        {
            free(arr[i]);  // 释放第二维
            arr[i] = NULL;
        }
    }
    free(arr);  // 释放第一维
    arr = NULL;
}
void free_2d_uint8(uint8_t **arr, int dim1)
{
    if (arr == NULL) return;
    
    for (int i = 0; i < dim1; i++)
    {
        if (arr[i] != NULL)
        {
            free(arr[i]);  // 释放第二维
            arr[i] = NULL;
        }
    }
    free(arr);  // 释放第一维
    arr = NULL;
}
void free_2d_uint16(uint16_t **arr, int dim1)
{
    if (arr == NULL) return;
    
    for (int i = 0; i < dim1; i++)
    {
        if (arr[i] != NULL)
        {
            free(arr[i]);  // 释放第二维
            arr[i] = NULL;
        }
    }
    free(arr);  // 释放第一维
    arr = NULL;
}
double ***malloc_3d_double(int dim1, int dim2, int dim3)
{
    double ***array = (double***)malloc(dim1 * sizeof(double**));
    for (int i = 0; i < dim1; i++)
    {
        array[i] = (double**)malloc(dim2 * sizeof(double*));
        array[i][0] = (double*)calloc(dim2 * dim3, sizeof(double));
        for (int j = 1; j < dim2; j++)
        {
            array[i][j] = array[i][0] + j * dim3;
        }
    }
    return array;
}
uint8_t ***malloc_3d_uint8(int dim1, int dim2, int dim3)
{
    uint8_t ***array = (uint8_t***)malloc(dim1 * sizeof(uint8_t**));
    for (int i = 0; i < dim1; i++)
    {
        array[i] = (uint8_t**)malloc(dim2 * sizeof(uint8_t*));
        for (int j = 0; j < dim2; j++)
        {
            array[i][j] = (uint8_t*)malloc(dim3 * sizeof(uint8_t));
        }
    }
    return array;
}
void free_3d_double(double ***arr, int dim1, int dim2)
{
    if (arr == NULL) return;
    
    for (int i = 0; i < dim1; i++)
    {
        if (arr[i] != NULL) {
            // 由于第三维是连续分配的，只需要释放第一个指针
            if (arr[i][0] != NULL) {
                free(arr[i][0]);  // 释放整个连续内存块
            }
            free(arr[i]);  // 释放第二维
            arr[i] = NULL;
        }
    }
    free(arr);  // 释放第一维
    arr = NULL;
}
void free_3d_uint8(uint8_t ***arr, int dim1, int dim2)
{
    if (arr == NULL) return;
    
    for (int i = 0; i < dim1; i++)
    {
        if (arr[i] != NULL) {
            for (int j = 0; j < dim2; j++)
            {
                if (arr[i][j] != NULL)
                {
                    free(arr[i][j]);  // 释放第三维
                    arr[i][j] = NULL;
                }
            }
            free(arr[i]);  // 释放第二维
            arr[i] = NULL;
        }
    }
    free(arr);  // 释放第一维
    arr = NULL;
}
void print2DArray(int rows, int cols, double** arr)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f\t", arr[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}
void initialize_matrix(int m, double** mat)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            //mat[i][j] = 10000.0 * (double)rand() / RAND_MAX;
            mat[i][j] = ((double)rand() / RAND_MAX) * 20.0; // [-10, 10]
        }
    }
}
void initialize_zero_matrix(int m, double** mat)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            mat[i][j] = 0.0;
        }
    }
}
void generate_random_matrix(int dim, double **mat)
{
    initialize_matrix(dim, mat);
}
void original_multiply(int number, double** a, double** b, double** c)
{
    for (int i = 0; i < number; i++)
    {
        for (int k = 0; k < number; k++)
        {
            double temp = a[i][k];
            for (int j = 0; j < number; j++)
            {
                c[i][j] += temp * b[k][j];
                //c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}
double distance_calculate(double *A, double *B, int dim)
{
#if USE_MANHATTAN_DISTANCE == 0
    double dist1 = 0.0;
    for (int d = 0; d < dim; d++)
    {
        double diff = A[d] - B[d];
        dist1 += diff * diff;//欧式距离平方
    }
    return sqrt(dist1);
#else
    double dist2 = 0.0;
    for (int d = 0; d < dim; d++)
    {
        double diff = A[d] - B[d];
        dist2 += fabs(diff);//曼哈顿距离
    }
    return dist2;
#endif
}
int int_distance_calculate(double *A, double *B, int dim)
{
    //int dist1 = 0;
    int dist2 = 0;
    for (int d = 0; d < dim; d++)
    {
        int diff = (int)(A[d]) - (int)(B[d]);
        //dist1 += diff * diff;//欧式距离平方
        dist2 += fabs(diff);//曼哈顿距离
    }
    //dist2 = (int)sqrt((double)dist1);
    return dist2;
}
int uint8_distance_calculate(uint8_t *A, uint8_t *B, int dim)
{
    int dist = 0;
    for (int d = 0; d < dim; d++)
    {
        int diff = (int)A[d] - (int)B[d];
        dist += abs(diff);  // 曼哈顿距离
    }
    return dist;
}
static double median_from_values(const double *values, int count)
{
    if (count <= 0)
    {
        return 0.0;
    }
    double sorted[num_centroids_per_subspace];
    for (int i = 0; i < count; i++)
    {
        sorted[i] = values[i];
    }
    for (int i = 1; i < count; i++)
    {
        double key = sorted[i];
        int j = i - 1;
        while (j >= 0 && sorted[j] > key)
        {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = key;
    }
    if (count % 2 == 0)
    {
        return 0.5 * (sorted[count / 2 - 1] + sorted[count / 2]);
    }
    return sorted[count / 2];
}
static int extract_prefix_bits(int code, int processed_dims)
{
    if (processed_dims <= 0)
    {
        return 0;
    }
    int shift = PQ_BITS - processed_dims;
    int mask = (1 << processed_dims) - 1;
    return (code >> shift) & mask;
}
double dot_product(double *A, double *B, int dim)
{
    double dot = 0.0;
    for (int d = 0; d < dim; d++)
    {
        dot += A[d] * B[d];
    }
    return dot;
}
void affine_quantization(int scale, double **A, uint8_t **A_quantized, double *min_val, double *max_val)
{//                      矩阵行数       矩阵A    量化后矩阵A_quantized
    *min_val = INFINITY;
    *max_val = -INFINITY;
    //找到矩阵A中的最小值和最大值
    for(int i = 0;i < scale;i++)
    {
        for(int j = 0;j < scale;j++)
        {
            if(A[i][j] < *min_val)
                *min_val = A[i][j];
            if(A[i][j] > *max_val)
                *max_val = A[i][j];
        }
    }
    double range = *max_val - *min_val;
    //进行量化
    for(int i = 0;i < scale;i++)
    {
        for(int j = 0;j < scale;j++)
        {
            A_quantized[i][j] = (uint8_t)((A[i][j] - *min_val)/range * 255);
        }
    }
}
void k_means(int scale, double ***center_save, double **A, int subspace_number, int center_number_in_per_subspace)
{//           矩阵行数                                 矩阵A        子空间数量          每个子空间质心数量
    //最高迭代次数
    int max_iterations = 60;
    //每个子空间的维度
    int sub_dimension = scale / subspace_number;
    // 随机初始化质心,每个子空间独立初始化,不重复选择
    for(int subspace = 0;subspace < subspace_number;subspace++)
    {
        int used_indices[scale];
        memset(used_indices, 0, sizeof(used_indices));
        for(int k = 0;k < center_number_in_per_subspace;k++)
        {
            int rand_row = rand() % scale;
            while(used_indices[rand_row])
            {
                rand_row = rand() % scale;
            }
            used_indices[rand_row] = 1;
            for(int d = 0;d < sub_dimension;d++)
            {
                center_save[subspace][k][d] = A[rand_row][subspace * sub_dimension + d];
            }
        }
        //k-means迭代，在某个子空间内进行
        int *labels = (int*)malloc(scale * sizeof(int));
        //labels[i]表示第 i 个样本所属的质心索引，范围 [0, center_number_in_per_subspace-1]，或者说，[0, 15]
        double **add = malloc_2d_double(center_number_in_per_subspace, sub_dimension);
        //add[c][d]表示第 c 个质心的第 d 维的累加和
        int *counts = (int*)calloc(center_number_in_per_subspace, sizeof(int));
        //counts[c]表示第 c 个质心被分配到的数据点数量
        for(int iter = 0;iter < max_iterations;iter++)
        {
            //分配步骤,第一次
            for(int i = 0;i < scale;i++)
            {
                double min_dist = INFINITY;
                int best_center = -1;
                for(int c = 0;c < center_number_in_per_subspace;c++)
                {
                    double dist = distance_calculate(&A[i][subspace * sub_dimension], center_save[subspace][c], sub_dimension);
                    if(dist < min_dist)
                    {
                        min_dist = dist;
                        best_center = c;
                    }
                }
                labels[i] = best_center;
            }
            //更新，清零计数器和累加器
            memset(counts, 0, center_number_in_per_subspace * sizeof(int));
            for(int c = 0;c < center_number_in_per_subspace;c++)
            {
                memset(add[c], 0, sub_dimension * sizeof(double));
            }
            //获取新的簇中心
            for(int i = 0;i < scale;i++)
            {
                int c = labels[i];
                counts[c]++;
                for(int d = 0;d < sub_dimension;d++)
                {
                    add[c][d] += A[i][subspace * sub_dimension + d];//根据所属簇累加
                }
            }
            double shift = 0.0;
            for(int c = 0;c < center_number_in_per_subspace;c++)
            {
                //如果某簇没有分配到任何点，则随机重新选择一个点作为质心
                if(counts[c] == 0)
                {
                    int rand_row = rand() % scale;
                    for(int d = 0;d < sub_dimension;d++)
                    {
                        center_save[subspace][c][d] = A[rand_row][subspace * sub_dimension + d];
                        shift = INFINITY;
                    }
                    continue;
                }
                //获得更新后的簇中心
                for(int d = 0;d < sub_dimension;d++)
                {
                    shift += fabs((add[c][d] / counts[c]) - center_save[subspace][c][d]);
                    center_save[subspace][c][d] = add[c][d] / counts[c];
                }
            }
            if(shift < 1e-6)
                break;
        }
        free(labels);
        free_2d_double(add, center_number_in_per_subspace);
        free(counts);
    }
}
void k_means_with_affine_quantization(int scale, double ***center_save, double **A, 
                                     int subspace_number, int center_number_in_per_subspace)
{
    printf("Starting affine quantization k-means...\n");
    
    uint8_t **A_quantized = malloc_2d_uint8(scale, scale);
    double min_val, max_val;
    
    // 对矩阵A进行仿射量化
    affine_quantization(scale, A, A_quantized, &min_val, &max_val);
    double range = max_val - min_val;
    
    printf("Quantization range: min=%.6f, max=%.6f, range=%.6f\n", min_val, max_val, range);
    
    int max_iterations = 60;
    int sub_dimension = scale / subspace_number;
    
    for(int subspace = 0; subspace < subspace_number; subspace++)
    {
        printf("Processing subspace %d/%d\n", subspace+1, subspace_number);
        
        // 在量化空间进行k-means
        uint8_t **centroids_quant = malloc_2d_uint8(center_number_in_per_subspace, sub_dimension);
        
        // 随机选择初始质心（在量化空间）
        int used_indices[scale];
        memset(used_indices, 0, sizeof(used_indices));
        for(int k = 0; k < center_number_in_per_subspace; k++)
        {
            int rand_row = rand() % scale;
            while(used_indices[rand_row])
            {
                rand_row = (rand_row + 1) % scale;
            }
            used_indices[rand_row] = 1;
            
            for(int d = 0; d < sub_dimension; d++)
            {
                centroids_quant[k][d] = A_quantized[rand_row][subspace * sub_dimension + d];
            }
        }
        
        int *labels = (int*)malloc(scale * sizeof(int));
        uint16_t **add = malloc_2d_uint16(center_number_in_per_subspace, sub_dimension);
        int *counts = (int*)calloc(center_number_in_per_subspace, sizeof(int));
        
        for(int iter = 0; iter < max_iterations; iter++)
        {
            // 分配步骤：在量化空间计算距离
            for(int i = 0; i < scale; i++)
            {
                int min_dist = INT_MAX;
                int best_center = -1;
                
                for(int c = 0; c < center_number_in_per_subspace; c++)
                {
                    // 使用正确的uint8_t距离计算
                    int dist = uint8_distance_calculate(&A_quantized[i][subspace * sub_dimension], 
                                                       centroids_quant[c], sub_dimension);
                    
                    if(dist < min_dist)
                    {
                        min_dist = dist;
                        best_center = c;
                    }
                }
                labels[i] = best_center;
            }
            
            // 更新步骤：清零计数器和累加器
            memset(counts, 0, center_number_in_per_subspace * sizeof(int));
            for(int c = 0; c < center_number_in_per_subspace; c++)
            {
                memset(add[c], 0, sub_dimension * sizeof(uint16_t));
            }
            
            // 累加属于每个质心的向量
            for(int i = 0; i < scale; i++)
            {
                int c = labels[i];
                counts[c]++;
                for(int d = 0; d < sub_dimension; d++)
                {
                    add[c][d] += A_quantized[i][subspace * sub_dimension + d];
                }
            }
            
            // 更新质心并计算shift
            int total_shift = 0;
            for(int c = 0; c < center_number_in_per_subspace; c++)
            {
                if(counts[c] == 0)
                {
                    // 重新初始化空簇
                    int rand_row = rand() % scale;
                    for(int d = 0; d < sub_dimension; d++)
                    {
                        centroids_quant[c][d] = A_quantized[rand_row][subspace * sub_dimension + d];
                    }
                    total_shift += 100;  // 大的shift值
                    continue;
                }
                
                // 更新量化空间的质心
                for(int d = 0; d < sub_dimension; d++)
                {
                    uint8_t old_centroid = centroids_quant[c][d];
                    uint8_t new_centroid = (uint8_t)((add[c][d] + counts[c]/2) / counts[c]);  // 四舍五入
                    
                    total_shift += abs((int)new_centroid - (int)old_centroid);
                    centroids_quant[c][d] = new_centroid;
                }
            }
            
            // 检查收敛
            if(total_shift < 10)  // 宽松的收敛条件
            {
                printf("Subspace %d converged after %d iterations, shift=%d\n", 
                       subspace+1, iter+1, total_shift);
                break;
            }
            
            if(iter == max_iterations - 1)
            {
                printf("Subspace %d reached max iterations, shift=%d\n", 
                       subspace+1, total_shift);
            }
        }
        
        // 将量化空间的质心转换回原始空间
        for(int c = 0; c < center_number_in_per_subspace; c++)
        {
            for(int d = 0; d < sub_dimension; d++)
            {
                double original_value = min_val + (centroids_quant[c][d] / 255.0) * range;
                center_save[subspace][c][d] = original_value;
            }
        }
        
        // 释放内存
        free(labels);
        free_2d_uint16(add, center_number_in_per_subspace);
        free(counts);
        free_2d_uint8(centroids_quant, center_number_in_per_subspace);
    }
    
    free_2d_uint8(A_quantized, scale);
    printf("Affine quantization k-means completed.\n");
}
void make_markle_tree(double ***center_save, int **markle_tree_point, int subspace_number)
{//                          质心存储矩阵      markle_tree点存储矩阵          子空间数量
    for(int sub = 0;sub < subspace_number;sub++)//遍历每个子空间
    {
        //初始化markle_tree
        for(int i = 0;i < 4;i++)
        {
            markle_tree_point[sub][i] = -1;
        }
        double importance[sub_dim];
        //importance[dim]存储维度 dim 在该子空间中对分裂的贡献度
        for(int dim = 0; dim < sub_dim; dim++)
        {
            double mean = 0.0;
            for(int c = 0; c < num_centroids_per_subspace; c++)
            {
                mean += center_save[sub][c][dim];
            }
            mean /= num_centroids_per_subspace;
            double score = 0.0;
            for(int c = 0; c < num_centroids_per_subspace; c++)
            {
                score += fabs(center_save[sub][c][dim] - mean);
            }
            importance[dim] = score;
        }
        //选择方差最大的四个维度作为分割节点
        for(int k = 0; k < PQ_BITS; k++)
        {
            double max_val = -1.0;
            int max_index = 0;
            for(int dim = 0; dim < sub_dim; dim++)
            {
                if(importance[dim] > max_val)
                {
                    max_val = importance[dim];
                    max_index = dim;
                }
            }
            markle_tree_point[sub][k] = max_index;
            importance[max_index] = -1.0;
        }
    }
}
void make_markle_tree_first_dimension(int **center_list, double ***center_save, int **markle_tree_point, double **markle_tree_1dimension, int subspace_number)
{//                                                             质心存储矩阵      markle_tree点存储矩阵          markle_tree存储矩阵          子空间数量
    for(int sub = 0; sub < subspace_number; sub++)
    {
        for(int i = 0; i < num_centroids_per_subspace; i++)
        {
            center_list[sub][i] = 0;
        }
        int dim_index = markle_tree_point[sub][0];
        double values[num_centroids_per_subspace];
        for(int i = 0; i < num_centroids_per_subspace; i++)
        {
            values[i] = center_save[sub][i][dim_index];
        }
        double median = median_from_values(values, num_centroids_per_subspace);
        markle_tree_1dimension[sub][0] = median; //根节点的中位数阈值
        for(int i = 0; i < num_centroids_per_subspace; i++)
        {
            if(center_save[sub][i][dim_index] >= median)
            {
                center_list[sub][i] |= (1 << 3);
            }
        }
    }
}
void make_markle_tree_second_dimension(int **center_list, double ***center_save, int **markle_tree_point, double **markle_tree_2dimension, int subspace_number)
{//                                                             质心存储矩阵      markle_tree点存储矩阵          markle_tree存储矩阵          子空间数量
    for(int sub = 0; sub < subspace_number; sub++)
    {
        int dim_index = markle_tree_point[sub][1];
        for(int group = 0; group < 2; group++)
        {
            //只在父节点属于 group 的簇中计算二层阈值
            double values[num_centroids_per_subspace];
            int count = 0;
            for(int i = 0; i < num_centroids_per_subspace; i++)
            {
                int prefix = extract_prefix_bits(center_list[sub][i], 1);
                if(prefix == group)
                {
                    values[count++] = center_save[sub][i][dim_index];
                }
            }
            if(count == 0)
            {
                markle_tree_2dimension[sub][group] = 0.0;
                continue;
            }
            double median = median_from_values(values, count);
            markle_tree_2dimension[sub][group] = median;
            for(int i = 0; i < num_centroids_per_subspace; i++)
            {
                int prefix = extract_prefix_bits(center_list[sub][i], 1);
                if(prefix == group && center_save[sub][i][dim_index] >= median)
                {
                    center_list[sub][i] |= (1 << 2);
                }
            }
        }
    }
}
void make_markle_tree_third_dimension(int **center_list, double ***center_save, int **markle_tree_point, double **markle_tree_3dimension, int subspace_number)
{//                                                             质心存储矩阵      markle_tree点存储矩阵          markle_tree存储矩阵          子空间数量
    for(int sub = 0; sub < subspace_number; sub++)
    {
        int dim_index = markle_tree_point[sub][2];
        for(int group = 0; group < 4; group++)
        {
            //第三层根据前两位前缀 (00~11) 继续细分
            double values[num_centroids_per_subspace];
            int count = 0;
            for(int i = 0; i < num_centroids_per_subspace; i++)
            {
                int prefix = extract_prefix_bits(center_list[sub][i], 2);
                if(prefix == group)
                {
                    values[count++] = center_save[sub][i][dim_index];
                }
            }
            if(count == 0)
            {
                markle_tree_3dimension[sub][group] = 0.0;
                continue;
            }
            double median = median_from_values(values, count);
            markle_tree_3dimension[sub][group] = median;
            for(int i = 0; i < num_centroids_per_subspace; i++)
            {
                int prefix = extract_prefix_bits(center_list[sub][i], 2);
                if(prefix == group && center_save[sub][i][dim_index] >= median)
                {
                    center_list[sub][i] |= (1 << 1);
                }
            }
        }
    }
}
void make_markle_tree_fourth_dimension(int **center_list, double ***center_save, int **markle_tree_point, double **markle_tree_4dimension, int subspace_number)
{//                                                             质心存储矩阵      markle_tree点存储矩阵          markle_tree存储矩阵          子空间数量
    for(int sub = 0; sub < subspace_number; sub++)
    {
        int dim_index = markle_tree_point[sub][3];
        for(int group = 0; group < 8; group++)
        {
            //第四层对应 8 个叶子分支
            double values[num_centroids_per_subspace];
            int count = 0;
            for(int i = 0; i < num_centroids_per_subspace; i++)
            {
                int prefix = extract_prefix_bits(center_list[sub][i], 3);
                if(prefix == group)
                {
                    values[count++] = center_save[sub][i][dim_index];
                }
            }
            if(count == 0)
            {
                markle_tree_4dimension[sub][group] = 0.0;
                continue;
            }
            double median = median_from_values(values, count);
            markle_tree_4dimension[sub][group] = median;
            for(int i = 0; i < num_centroids_per_subspace; i++)
            {
                int prefix = extract_prefix_bits(center_list[sub][i], 3);
                if(prefix == group && center_save[sub][i][dim_index] >= median)
                {
                    center_list[sub][i] |= 1;
                }
            }
        }
    }
}
void make_the_list_with_markle_tree(int scale, double **A, int **list, double ***center_save, int **center_list, int subspace_number, int **markle_tree_point,
                                    double **markle_tree_1dimension, double **markle_tree_2dimension, double **markle_tree_3dimension, double **markle_tree_4dimension)
{
    int A_sign[scale][subspace_number];
    memset(A_sign, 0, sizeof(A_sign));
    int subspace_dim = scale / subspace_number;
    for(int sub = 0;sub < subspace_number;sub++)//遍历每个子空间
    {
        for(int row = 0;row < scale;row++)//遍历每一行,为每一行分配质心
        {
            int base = sub * subspace_dim;
            //沿着markle tree四层比较，得到编码 0~15
            if(A[row][base + markle_tree_point[sub][0]] >= markle_tree_1dimension[sub][0])
            {
                A_sign[row][sub] |= (1 << 3);
            }
            int prefix1 = extract_prefix_bits(A_sign[row][sub], 1);
            if(A[row][base + markle_tree_point[sub][1]] >= markle_tree_2dimension[sub][prefix1])
            {
                A_sign[row][sub] |= (1 << 2);
            }
            int prefix2 = extract_prefix_bits(A_sign[row][sub], 2);
            if(A[row][base + markle_tree_point[sub][2]] >= markle_tree_3dimension[sub][prefix2])
            {
                A_sign[row][sub] |= (1 << 1);
            }
            int prefix3 = extract_prefix_bits(A_sign[row][sub], 3);
            if(A[row][base + markle_tree_point[sub][3]] >= markle_tree_4dimension[sub][prefix3])
            {
                A_sign[row][sub] |= 1;
            }
            //直接匹配等价编码的质心索引
            int best_c = -1;
            for(int c = 0;c < num_centroids_per_subspace;c++)
            {
                if(center_list[sub][c] == A_sign[row][sub])
                {
                    best_c = c;
                    break;
                }
            }
            if(best_c == -1)
            {
                //若四叉树未命中，则退化为暴力距离搜索
                double min_dist = INFINITY;
                for(int c = 0;c < num_centroids_per_subspace;c++)
                {
                    double dist = distance_calculate(&A[row][base], center_save[sub][c], subspace_dim);
                    if(dist < min_dist)
                    {
                        min_dist = dist;
                        best_c = c;
                    }
                }
            }
            list[row][sub] = best_c;
        }
    }
}
void make_the_list(int scale, double **A, int **list, double ***center_save, int subspace_number,  int center_number_in_per_subspace)
{//                 矩阵行数      矩阵A    编码结果存储矩阵                             子空间数量          每个子空间质心数量
    int sub_dimension = scale / subspace_number;
    for(int sub = 0;sub < subspace_number;sub++)//遍历每个子空间
    {
        for(int row = 0;row < scale;row++)//遍历每一行,为每一行分配质心
        {
            //寻找最近质心
            double min_dist = INFINITY;
            int best_c = -1;
            for(int c = 0;c < center_number_in_per_subspace;c++)
            {
                double dist = distance_calculate(&A[row][sub * sub_dimension], center_save[sub][c], sub_dimension);
                if(dist < min_dist)
                {
                    min_dist = dist;
                    best_c = c;
                }
            }
            list[row][sub] = best_c;
        }
    }
}
void built_table(int scale, double **B, double ***final_list, int subspace_number, int center_number_in_per_subspace, double ***center_save)
{//                 矩阵行数      矩阵B        查找表存储矩阵                             子空间数量          每个子空间质心数量
    double **BT = malloc_2d_double(n, n);
    //先转置矩阵B，方便按列访问
    for(int i = 0;i < scale;i++)
    {
        for(int j = 0;j < scale;j++)
        {
            BT[j][i] = B[i][j];
        }
    }
    int subspace_dim = scale / subspace_number;
    for(int sub = 0;sub < subspace_number;sub++)//遍历每个子空间
    {
        for(int col = 0;col < scale;col++)//遍历矩阵B的每一列
        {
            for(int c = 0;c < center_number_in_per_subspace;c++)//遍历每个质心
            {
                //计算质心与列向量的点积
                double dot = dot_product(center_save[sub][c], &BT[col][sub * subspace_dim], subspace_dim);
                final_list[col][sub][c] = dot;
                //final_list[col][sub][c]表示矩阵 B 的第 col 列与第 sub 个子空间中第 c 个质心的点积结果。
            }
        }
    }
}
void generate_result(int scale, int **list, double ***final_list, double **C_PQ, int subspace_number, int center_number_in_per_subspace)
{//                 矩阵行数    编码结果存储矩阵     查找表存储矩阵         结果矩阵          子空间数量          每个子空间质心数量
    for(int row = 0;row < scale;row++)//遍历每一行
    {
        for(int col = 0;col < scale;col++)//遍历每一列
        {
            double sum = 0.0;
            for(int sub = 0;sub < subspace_number;sub++)//遍历每个子空间
            {
                int c = list[row][sub];//获取该行在该子空间对应的质心索引
                sum += final_list[col][sub][c];//查表累加
            }
            C_PQ[row][col] = sum;
        }
    }
}
double calculate_relative_error(double **reference, double **approx, int dim)
{//reference为精确结果，approx为近似结果
    double diff_norm = 0.0;
    double ref_norm = 0.0;
    for(int i = 0;i < dim;i++)
    {
        for(int j = 0;j < dim;j++)
        {
            double diff = reference[i][j] - approx[i][j];
            diff_norm += diff * diff;
            ref_norm += reference[i][j] * reference[i][j];
        }
    }
    if(ref_norm == 0.0)
    {
        return diff_norm == 0.0 ? 0.0 : INFINITY;
    }
    return sqrt(diff_norm / ref_norm);
}
void pq_multiply(int scale, double **A_test, double** A, double** B, double** C_PQ, double *total_time_pq, double** C_original)
{
    double total_time_classic = 0.0;
    double ***center_save = malloc_3d_double(num_subspaces, num_centroids_per_subspace, sub_dim);
    //center_save[sub][k][d] 表示第 sub 个子空间中第 k 个质心的第 d 维的值。
    int **center_list = malloc_2d_int(num_subspaces, num_centroids_per_subspace);
    //center_list[sub][k] 表示第 sub 个子空间中第 k 个质心的索引。
    int **list = malloc_2d_int(n, num_subspaces);
    //list[row][sub]表示矩阵 A 的第 row 行在第 sub 个子空间中被分配到的质心索引。
    double ***final_list = malloc_3d_double(n, num_subspaces, num_centroids_per_subspace);
    //final_list[col][sub][k] 表示矩阵 B 的第 col 列与第 sub 个子空间中第 k 个质心的点积结果。
    //构建markle tree所需的矩阵,markle_tree存储每个子空间节点的中值，markle_tree_point存储对应的分割维度
#if USE_MARKLE_TREE == 1
    double **markle_tree_1dimension = malloc_2d_double(num_subspaces, 1);
    double **markle_tree_2dimension = malloc_2d_double(num_subspaces, 2);
    double **markle_tree_3dimension = malloc_2d_double(num_subspaces, 4);
    double **markle_tree_4dimension = malloc_2d_double(num_subspaces, 8);
    int **markle_tree_point = malloc_2d_int(num_subspaces, 4);
#endif
    //进行 PQ 乘法的各个步骤
    //对训练集A_test进行k-means聚类，得到质心
    printf("Starting k-means clustering...\n");
    double start_time = get_time();
#if USE_AFFINE_QUANTIZATION == 0
    k_means(scale, center_save, A_test, num_subspaces, num_centroids_per_subspace);
#else
    k_means_with_affine_quantization(scale, center_save, A_test, num_subspaces, num_centroids_per_subspace);
#endif
    printf("K-means clustering completed.\n");
    //构建查找表
    printf("Starting built_table...\n");
    built_table(scale, B, final_list, num_subspaces, num_centroids_per_subspace, center_save);
    printf("built_table completed.\n");
    *total_time_pq = get_time() - start_time;
    printf("Time for k-means and built_table: %.6f seconds\n", *total_time_pq);
    
    //生成编码列表，将矩阵A的行映射到质心，依据距离最近原则
    printf("Starting make_the_list...\n");
    start_time = get_time();
#if USE_MARKLE_TREE == 0
    make_the_list(scale, A, list, center_save, num_subspaces, num_centroids_per_subspace);
#else
    //基于Markle Tree加速生成编码列表
    make_markle_tree(center_save, markle_tree_point, num_subspaces);
    make_markle_tree_first_dimension(center_list, center_save, markle_tree_point, markle_tree_1dimension, num_subspaces);
    make_markle_tree_second_dimension(center_list, center_save, markle_tree_point, markle_tree_2dimension, num_subspaces);
    make_markle_tree_third_dimension(center_list, center_save, markle_tree_point, markle_tree_3dimension, num_subspaces);
    make_markle_tree_fourth_dimension(center_list, center_save, markle_tree_point, markle_tree_4dimension, num_subspaces);
    *total_time_pq = get_time();
    make_the_list_with_markle_tree(scale, A, list, center_save, center_list, num_subspaces, markle_tree_point,
                                   markle_tree_1dimension, markle_tree_2dimension, markle_tree_3dimension, markle_tree_4dimension); 
    printf("make_the_list completed.\n");
#endif
    //生成最终结果矩阵
    printf("Starting generate_result...\n");
    generate_result(scale, list, final_list, C_PQ, num_subspaces, num_centroids_per_subspace);
    printf("generate_result completed.\n");
    *total_time_pq = get_time() - start_time;
    printf("Time for make_the_list and generate_result: %.6f seconds\n", *total_time_pq);

    double start = get_time();
    original_multiply(n, A, B, C_original);
    total_time_classic += get_time() - start;
    printf("Time for classic multiplication: %.6f seconds\n", total_time_classic);

    printf("speed up: %.2f x\n", total_time_classic / *total_time_pq);

    double relative_error = calculate_relative_error(C_original, C_PQ, n);
    printf("Relative error between original and PQ result: %.6f\n", relative_error);

    int times = 5;
    printf("Running %d additional tests to average PQ time...\n", times);
    for(int i = 0; i < times; i++)
    {
        generate_random_matrix(n, A);
        double start = get_time();
        original_multiply(n, A, B, C_original);
        total_time_classic = get_time() - start;
        start = get_time();
#if USE_MARKLE_TREE == 0
    make_the_list(scale, A, list, center_save, num_subspaces, num_centroids_per_subspace);
#else
    make_the_list_with_markle_tree(scale, A, list, center_save, center_list, num_subspaces, markle_tree_point,
                                   markle_tree_1dimension, markle_tree_2dimension, markle_tree_3dimension, markle_tree_4dimension);
#endif
        generate_result(scale, list, final_list, C_PQ, num_subspaces, num_centroids_per_subspace);
        *total_time_pq = get_time() - start;
        printf("speed up: %.2f x\n", total_time_classic / *total_time_pq);
    }

    //释放内存
    free_3d_double(center_save, num_subspaces, num_centroids_per_subspace);
    free_2d_int(list, n);
    free_3d_double(final_list, n, num_subspaces);
#if USE_MARKLE_TREE == 1
    free_2d_double(markle_tree_1dimension, num_subspaces);
    free_2d_double(markle_tree_2dimension, num_subspaces);
    free_2d_double(markle_tree_3dimension, num_subspaces);
    free_2d_double(markle_tree_4dimension, num_subspaces);
    free_2d_int(markle_tree_point, num_subspaces);
    free_2d_int(center_list, num_subspaces);
#endif
}