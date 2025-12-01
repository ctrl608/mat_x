#ifndef LIB_H
#define LIB_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
typedef double mtype;
//-----------函数声明----------
mtype** create_matrix(int n);//创建n x n矩阵,返回指向矩阵的指针
void initialize_matrix(int n, mtype** mat);//初始化n x n矩阵,元素为随机mtype
void free_matrix(int n, mtype** mat);//释放n x n矩阵的内存

void classic(int scale, mtype** m1, mtype** m2, mtype** ans);
void strassen(int scale, mtype** m1, mtype** m2, mtype** ans);
void CW(int scale, mtype **m1, mtype **m2, mtype **ans);

void train_centroids(int scale, int num_subspaces, int num_centroids_per_subspace
                        , mtype ***centroids, mtype **mat);
void encode(int scale, mtype **mat ,int num_subspaces, int num_centroids_per_subspace,
        mtype ***centroids,int **codebook);
void build_table(int scale, mtype **matrix_B,int num_subspaces, 
        int num_centroids_per_subspace,mtype ***centroids,mtype ***table);

void compute_result(int scale, int num_subspaces, int num_centroids_per_subspace,int **codebook,
                    mtype ***lookup_table,mtype **result);
void print10(int scale, mtype **mat);
void compute_error(int scale, mtype **approx, mtype **exact);
void debug_pq_mismatch(int scale, mtype **A, mtype **B, mtype **approx, mtype **exact,
                       int num_subspaces, mtype ***centroids, int **codebook, mtype ***lookup_table);

//-----------------------------
#define LIMIT 64  // Strassen和CW算法的递归阈值

/* -------------------- 内存管理函数 -------------------- */
mtype** create_matrix(int n) {
    mtype** mat = (mtype**)malloc(n * sizeof(mtype*));
    if (!mat) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        mat[i] = (mtype*)malloc(n * sizeof(mtype));
        if (!mat[i]) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(EXIT_FAILURE);
        }
    }
    return mat;
}

void free_matrix(int n, mtype** mat) {
    for (int i = 0; i < n; i++) {
        free(mat[i]);
    }
    free(mat);
}

void initialize_matrix(int n, mtype** mat) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i][j] = ((mtype)rand() / RAND_MAX)*20.0-10.0;//此处如果不-10就完全正常
        }
    }
}

/* -------------------- 经典矩阵乘法 -------------------- */
void classic(int scale, mtype** m1, mtype** m2, mtype** ans) {
    for (int i = 0; i < scale; i++) {
        for (int j = 0; j < scale; j++) {
            ans[i][j] = 0.0;
            for (int k = 0; k < scale; k++) {
                ans[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
}

/* -------------------- Strassen算法 -------------------- */
void strassen(int scale, mtype** m1, mtype** m2, mtype** ans) {
    if (scale <= LIMIT) {
        classic(scale, m1, m2, ans);
        return;
    }
    
    int half = scale / 2;
    
    // 创建临时矩阵
    mtype** T1 = create_matrix(half);
    mtype** T2 = create_matrix(half);
    mtype** T3 = create_matrix(half);
    mtype** T4 = create_matrix(half);
    mtype** T5 = create_matrix(half);
    mtype** T6 = create_matrix(half);
    mtype** T7 = create_matrix(half);
    mtype** temp1 = create_matrix(half);
    mtype** temp2 = create_matrix(half);
    
    // T1 = (A11 + A22) * (B11 + B22)
    for (int i = 0; i < half; ++i) {
        for (int j = 0; j < half; ++j) {
            temp1[i][j] = m1[i][j] + m1[i + half][j + half];
            temp2[i][j] = m2[i][j] + m2[i + half][j + half];
        }
    }
    strassen(half, temp1, temp2, T1);
    
    // T2 = (A21 + A22) * B11
    for (int i = 0; i < half; ++i) {
        for (int j = 0; j < half; ++j) {
            temp1[i][j] = m1[i + half][j] + m1[i + half][j + half];
            temp2[i][j] = m2[i][j];
        }
    }
    strassen(half, temp1, temp2, T2);
    
    // T3 = A11 * (B12 - B22)
    for (int i = 0; i < half; ++i) {
        for (int j = 0; j < half; ++j) {
            temp1[i][j] = m1[i][j];
            temp2[i][j] = m2[i][j + half] - m2[i + half][j + half];
        }
    }
    strassen(half, temp1, temp2, T3);
    
    // T4 = A22 * (B21 - B11)
    for (int i = 0; i < half; ++i) {
        for (int j = 0; j < half; ++j) {
            temp1[i][j] = m1[i + half][j + half];
            temp2[i][j] = m2[i + half][j] - m2[i][j];
        }
    }
    strassen(half,temp1,temp2,T4);
    // T5 = (A11 + A12) * B22
    for(int i=0; i<half; ++i){
        for(int j=0; j<half; ++j){
            temp1[i][j]=m1[i][j]+m1[i][j+half];
            temp2[i][j]=m2[i+half][j+half];
        }
    }strassen(half,temp1,temp2,T5);
    // T6 = (A21 - A11) * (B11 + B12)
    for(int i=0; i<half; ++i){
        for(int j=0; j<half; ++j){
            temp1[i][j]=m1[i+half][j]-m1[i][j];
            temp2[i][j]=m2[i][j]+m2[i][j+half];
        }
    }strassen(half,temp1,temp2,T6);
    // T7 = (A12 - A22) * (B21 + B22)
    for(int i=0; i<half; ++i){
        for(int j=0; j<half; ++j){
            temp1[i][j]=m1[i][j+half]-m1[i+half][j+half];
            temp2[i][j]=m2[i+half][j]+m2[i+half][j+half];
        }
    }strassen(half,temp1,temp2,T7);
        // 直接合并计算结果矩阵的四个子块,节省内存
    for (int i=0; i<scale ;++i){
        for (int j=0;j<scale ; ++j){
            if (i<half){
                if(j<half){
                    ans[i][j]=T1[i][j]+T4[i][j]-T5[i][j]+T7[i][j];// C11 = T1 + T4 - T5 + T7
                }else{
                    ans[i][j]=T3[i][j-half]+T5[i][j-half];// C12 = T3 + T5
                }
            }else{
                if(j<half){
                    ans[i][j]=T2[i-half][j]+T4[i-half][j];// C21 = T2 + T4
                }else{
                    ans[i][j]=T1[i-half][j-half]-T2[i-half][j-half]+T3[i-half][j-half]+T6[i-half][j-half];// C22 = T1 - T2 + T3 + T6
                }
            }
        }
    }
    free_matrix(half, T1);
    free_matrix(half, T2);
    free_matrix(half, T3);
    free_matrix(half, T4);
    free_matrix(half, T5);
    free_matrix(half, T6);
    free_matrix(half, T7);
    free_matrix(half, temp1);
    free_matrix(half, temp2);
}
void CW(int scale, mtype **m1, mtype **m2, mtype **ans) {
    if(scale <= LIMIT) {
        classic(scale, m1, m2, ans);
        return;
    }
    
    int half = scale / 2;
    
    // 分配临时矩阵
    mtype **S1 = create_matrix(half);
    mtype **S2 = create_matrix(half);
    mtype **S3 = create_matrix(half);
    mtype **S4 = create_matrix(half);
    mtype **T1 = create_matrix(half);
    mtype **T2 = create_matrix(half);
    mtype **T3 = create_matrix(half);
    mtype **T4 = create_matrix(half);
    mtype **R1 = create_matrix(half);
    mtype **R2 = create_matrix(half);
    mtype **R3 = create_matrix(half);
    mtype **R4 = create_matrix(half);
    mtype **R5 = create_matrix(half);
    mtype **R6 = create_matrix(half);
    mtype **R7 = create_matrix(half);
    mtype **C1 = create_matrix(half);
    mtype **C2 = create_matrix(half);
    mtype **C3 = create_matrix(half);
    mtype **C4 = create_matrix(half);
    mtype **C5 = create_matrix(half);
    mtype **C6 = create_matrix(half);
    mtype **C7 = create_matrix(half);
    mtype **A11 = create_matrix(half);
    mtype **B11 = create_matrix(half);
    mtype **A12 = create_matrix(half);
    mtype **B21 = create_matrix(half);
    mtype **B22 = create_matrix(half);
    mtype **A22 = create_matrix(half);
    for(int i=0; i<half; ++i){
        for(int j=0; j<half; ++j){
            A12[i][j]=m1[i][j+half];
            A22[i][j]=m1[i+half][j+half];
            B21[i][j]=m2[i+half][j];
            B22[i][j]=m2[i+half][j+half];
            A11[i][j] = m1[i][j];
            B11[i][j] = m2[i][j];
        }
    }
    for(int i=0; i<half; ++i){
        for(int j=0; j<half; ++j){
            S1[i][j]=m1[i+half][j]+m1[i+half][j+half];
        }
    }
    for(int i=0; i<half; ++i){
        for(int j=0; j<half; ++j){
            S2[i][j]=S1[i][j]-m1[i][j];
        }
    }
    for(int i=0; i<half; ++i){
        for(int j=0; j<half; ++j){
            S3[i][j]=m1[i][j]-m1[i+half][j];
        }
    }   
    for(int i=0; i<half; ++i){
        for(int j=0; j<half; ++j){
            S4[i][j]=m1[i][j+half]-S2[i][j];
        }
    }   
    for(int i=0; i<half; ++i){
        for(int j=0; j<half; ++j){
            T1[i][j]=m2[i][j+half]-m2[i][j];
            T2[i][j]=m2[i+half][j+half]-T1[i][j];
        }
    }
    for(int i=0; i<half; ++i){
        for(int j=0; j<half; ++j){
            T3[i][j]=m2[i+half][j+half]-m2[i][j+half];
            T4[i][j]=T2[i][j]-m2[i+half][j];
        }
    }
    CW(half,A11,B11,R1);
    CW(half,A12,B21,R2);
    CW(half,S4,B22,R3);
    CW(half,A22,T4,R4);
    CW(half,S1,T1,R5);
    CW(half,S2,T2,R6);
    CW(half,S3,T3,R7);
    for(int i=0; i<half; ++i){
        for(int j=0; j<half; ++j){
            C1[i][j]=R1[i][j]+R2[i][j];
            C2[i][j]=R1[i][j]+R6[i][j];
        }
    }
    for(int i=0; i<half; ++i){
        for(int j=0; j<half; ++j){
            C3[i][j]=C2[i][j]+R7[i][j];
            C4[i][j]=C2[i][j]+R5[i][j];
        }
    }
    for(int i=0; i<half; ++i){
        for(int j=0; j<half; ++j){
            C5[i][j]=C4[i][j]+R3[i][j];
            C6[i][j]=C3[i][j]-R4[i][j];
            C7[i][j]=C3[i][j]+R5[i][j];
        }
    }

    /* 正确组装结果矩阵 ans */
    for (int i = 0; i < scale; ++i) {
        for (int j = 0; j < scale; ++j) {
            if (i < half) {
                if (j < half) {
                    ans[i][j] = C1[i][j];
                } else {
                    ans[i][j] = C5[i][j - half];
                }
            } else {
                if (j < half) {
                    ans[i][j] = C6[i - half][j];
                } else {
                    ans[i][j] = C7[i - half][j - half];
                }
            }
        }
    }

    // 释放所有临时矩阵
    free_matrix(half, S1);
    free_matrix(half, S2);
    free_matrix(half, S3);
    free_matrix(half, S4);
    free_matrix(half, T1);
    free_matrix(half, T2);
    free_matrix(half, T3);
    free_matrix(half, T4);
    free_matrix(half, R1);
    free_matrix(half, R2);
    free_matrix(half, R3);
    free_matrix(half, R4);
    free_matrix(half, R5);
    free_matrix(half, R6);
    free_matrix(half, R7);
    free_matrix(half, C1);
    free_matrix(half, C2);
    free_matrix(half, C3);
    free_matrix(half, C4);
    free_matrix(half, C5);
    free_matrix(half, C6);
    free_matrix(half, C7);
    free_matrix(half, A11);
    free_matrix(half, B11);
    free_matrix(half, A12);
    free_matrix(half, B21);
    free_matrix(half, B22);
    free_matrix(half, A22);
    return;
}
/* -------------------- 训练质心 (K-means) -------------------- */
void train_centroids(int scale, int num_subspaces, int num_centroids_per_subspace, 
                     mtype ***centroids, mtype **mat) {
    // if (num_subspaces <= 0 || num_centroids_per_subspace <= 0 || scale % num_subspaces != 0) {
    //     fprintf(stderr, "train_centroids: invalid arguments\n");
    //     exit(EXIT_FAILURE);
    // }

    int sub_dim = scale / num_subspaces;
    int max_iters = 60;

    for (int sub = 0; sub < num_subspaces; ++sub) {
        for (int c = 0; c < num_centroids_per_subspace; ++c) {
            int rand_row = rand() % scale;
            for (int d = 0; d < sub_dim; ++d) {
                centroids[sub][c][d] = mat[rand_row][sub * sub_dim + d];
            }
        }

        int *labels = (int*)malloc((size_t)scale * sizeof(int));
        //if (!labels) { fprintf(stderr, "labels malloc failed\n"); exit(EXIT_FAILURE); }

        mtype **acc = (mtype**)malloc((size_t)num_centroids_per_subspace * sizeof(mtype*));
        //if (!acc) { fprintf(stderr, "acc malloc failed\n"); exit(EXIT_FAILURE); }
        for (int c = 0; c < num_centroids_per_subspace; ++c) {
            acc[c] = (mtype*)calloc((size_t)sub_dim, sizeof(mtype));
            //if (!acc[c]) { fprintf(stderr, "acc[%d] calloc failed\n", c); exit(EXIT_FAILURE); }
        }
        int *counts = (int*)calloc((size_t)num_centroids_per_subspace, sizeof(int));
        //if (!counts) { fprintf(stderr, "counts calloc failed\n"); exit(EXIT_FAILURE); }

        for (int iter = 0; iter < max_iters; ++iter) {
            for (int i = 0; i < scale; ++i) {
                mtype min_dist = INFINITY;
                int best_c = 0;
                for (int c = 0; c < num_centroids_per_subspace; ++c) {
                    mtype dist = 0.0;
                    for (int d = 0; d < sub_dim; ++d) {
                        mtype diff = mat[i][sub * sub_dim + d] - centroids[sub][c][d];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_c = c;
                    }
                }
                labels[i] = best_c;
            }

            memset(counts, 0, (size_t)num_centroids_per_subspace * sizeof(int));
            for (int c = 0; c < num_centroids_per_subspace; ++c) {
                memset(acc[c], 0, (size_t)sub_dim * sizeof(mtype));
            }

            for (int i = 0; i < scale; ++i) {
                int c = labels[i];
                counts[c]++;
                for (int d = 0; d < sub_dim; ++d) {
                    acc[c][d] += mat[i][sub * sub_dim + d];
                }
            }

            mtype max_shift = 0.0;

            for (int c = 0; c < num_centroids_per_subspace; ++c) {
                if (counts[c] == 0) {
                    int rand_row = rand() % scale;
                    for (int d = 0; d < sub_dim; ++d)
                        centroids[sub][c][d] = mat[rand_row][sub * sub_dim + d];
                    max_shift = INFINITY;
                    continue;
                }
                for (int d = 0; d < sub_dim; ++d) {
                    mtype new_value = acc[c][d] / counts[c];
                    mtype shift = fabs(new_value - centroids[sub][c][d]);
                    if (shift > max_shift) max_shift = shift;
                    centroids[sub][c][d] = new_value;
                }
            }

            if (max_shift < 1e-6) break;
        }

        for (int c = 0; c < num_centroids_per_subspace; ++c) free(acc[c]);
        free(acc);
        free(counts);
        free(labels);
    }
}

/* -------------------- 编码 -------------------- */
void encode(int scale, mtype **mat ,int num_subspaces, int num_centroids_per_subspace,
        mtype ***centroids,
        int **codebook){ // 引入codebook,存储每个簇对应的质心索引
    const int sub_dims = scale / num_subspaces; // 每个子空间的维度
    for (int row = 0; row < scale; ++row){ // 遍历每一行
        for (int sub = 0; sub < num_subspaces; ++sub){ // 遍历该行在每个子空间对应的"子行"

            mtype min_dist = INFINITY; // 初始化最小距离为无穷大
            int best_centroid = -1; // 初始化最佳质心索引
            for (int centroid = 0; centroid < num_centroids_per_subspace; ++centroid){
                mtype dist = 0.0; // 计算该子行与质心的欧式距离平方
                for (int dim = 0; dim < sub_dims; ++dim){
                    mtype diff = mat[row][sub*sub_dims + dim] - centroids[sub][centroid][dim];
                    dist += diff*diff;
                }
                if (dist < min_dist){
                    min_dist = dist;
                    best_centroid = centroid;
                }
            }
            codebook[row][sub] = best_centroid; // 写回codebook
        }
    }
}

/* -------------------- 构建查找表 -------------------- */
void build_table(int scale, mtype **matrix_B,
                 int num_subspaces, int num_centroids_per_subspace,
                 mtype ***centroids,
                 mtype ***table) {

    int sub_dim = scale / num_subspaces;

    for (int col = 0; col < scale; ++col) {
        for (int sub = 0; sub < num_subspaces; ++sub) {
            for (int k = 0; k < num_centroids_per_subspace; ++k) {
                mtype dot = 0.0;
                for (int d = 0; d < sub_dim; ++d) {
                    int row_idx = sub * sub_dim + d;
                    dot += centroids[sub][k][d] * matrix_B[row_idx][col];
                }
                table[col][sub][k] = dot;
            }
        }
    }
}

/* -------------------- 使用查找表计算结果 -------------------- */
void compute_result(int scale, int num_subspaces, int num_centroids_per_subspace,
                    int **codebook,
                    mtype ***lookup_table,
                    mtype **result) {

    for (int row = 0; row < scale; ++row){
        for (int col = 0; col < scale; ++col){
            mtype sum = 0.0;
            for (int sub = 0; sub < num_subspaces; ++sub){
                int centroid_index = codebook[row][sub];
                sum += lookup_table[col][sub][centroid_index];
            }
            result[row][col] = sum;
        }
    }
}

/* -------------------- 打印前10列 -------------------- */
void print10(int scale, mtype **mat){
    for (int j = 0; j < 10 && j < scale; ++j)
        printf("%f ", mat[0][j]);
    printf("\n");
}

/* -------------------- 计算误差 -------------------- */
void compute_error(int scale, mtype **approx, mtype **exact){
    mtype mse = 0.0;       // 均方误差
    mtype max_err = 0.0;   // 最大绝对误差

    for(int i = 0; i < scale; ++i){
        for(int j = 0; j < scale; ++j){
            mtype diff = approx[i][j] - exact[i][j];
            mse += diff * diff;
            if(fabs(diff) > max_err)
                max_err = fabs(diff);
        }
    }
    mse /= (scale * scale);

    printf("MSE (mean squared error): %lf\n", mse);
    printf("Max absolute error: %lf\n", max_err);
}







#endif