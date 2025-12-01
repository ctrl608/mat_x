#include "lib.h"
#include <sys/time.h>

// 计时函数
mtype get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

int main() {
    printf("=== 矩阵乘法算法性能比较 ===\n");
    srand((unsigned int)time(NULL));

    int scale = 256;  // 可调整矩阵大小
    int num_trials = 1;

    printf("矩阵大小: %d x %d\n", scale, scale);
    printf("测试次数: %d\n\n", num_trials);

    // 分配内存
    printf("正在分配内存...\n");
    mtype **A = create_matrix(scale);
    mtype **B = create_matrix(scale);
    mtype **C_classic = create_matrix(scale);
    mtype **C_strassen = create_matrix(scale);
    mtype **C_cw = create_matrix(scale);
    mtype **C_pq = create_matrix(scale);
    printf("内存分配完成！\n\n");

    // 初始化矩阵 A 和 B
    printf("正在初始化矩阵...\n");
    initialize_matrix(scale, A);
    initialize_matrix(scale, B);
    printf("矩阵初始化完成！\n\n");
    print10(scale, A);
    // -------------------- 经典算法 --------------------
    printf("开始测试经典算法...\n");
    mtype total_time_classic = 0.0;
    for (int t = 0; t < num_trials; t++) {
        if (t % 100 == 0) printf("经典算法进度: %d/%d\n", t, num_trials);
        mtype start = get_time();
        classic(scale, A, B, C_classic);
        total_time_classic += get_time() - start;
    }
    printf("经典算法测试完成！\n");
    printf("Classic: 平均时间 = %.6f s\n\n", total_time_classic / num_trials);

    // -------------------- Strassen --------------------
    printf("开始测试Strassen算法...\n");
    mtype total_time_strassen = 0.0;
    for (int t = 0; t < num_trials; t++) {
        if (t % 100 == 0) printf("Strassen算法进度: %d/%d\n", t, num_trials);
        mtype start = get_time();
        strassen(scale, A, B, C_strassen);
        total_time_strassen += get_time() - start;
    }
    printf("Strassen算法测试完成！\n");
    printf("Strassen: 平均时间 = %.6f s\n\n", total_time_strassen / num_trials);

    // -------------------- CW --------------------
    printf("开始测试CW算法...\n");
    mtype total_time_cw = 0.0;
    for (int t = 0; t < num_trials; t++) {
        if (t % 100 == 0) printf("CW算法进度: %d/%d\n", t, num_trials);
        mtype start = get_time();
        CW(scale, A, B, C_cw);
        total_time_cw += get_time() - start;
    }
    printf("CW算法测试完成！\n");
    printf("CW: 平均时间 = %.6f s\n\n", total_time_cw / num_trials);

    // -------------------- PQ 近似算法 --------------------
    printf("开始设置PQ算法参数...\n");
    int num_subspaces = 8;
    int num_centroids_per_subspace = 64;

    printf("子空间数量: %d\n", num_subspaces);
    printf("每个子空间的质心数: %d\n", num_centroids_per_subspace);

    // 为 PQ 分配内存
    printf("正在为PQ算法分配内存...\n");
    mtype ***centroids = (mtype***)malloc(num_subspaces * sizeof(mtype**));
    for (int s = 0; s < num_subspaces; s++) {
        centroids[s] = (mtype**)malloc(num_centroids_per_subspace * sizeof(mtype*));
        for (int c = 0; c < num_centroids_per_subspace; c++) {
            centroids[s][c] = (mtype*)malloc((scale / num_subspaces) * sizeof(mtype));
        }
    }

    int **codebook = (int**)malloc(scale * sizeof(int*));
    for (int i = 0; i < scale; i++) {
        codebook[i] = (int*)malloc(num_subspaces * sizeof(int));
    }

    mtype ***lookup_table = (mtype***)malloc(scale * sizeof(mtype**));
    for (int c = 0; c < scale; c++) {
        lookup_table[c] = (mtype**)malloc(num_subspaces * sizeof(mtype*));
        for (int s = 0; s < num_subspaces; s++) {
            lookup_table[c][s] = (mtype*)malloc(num_centroids_per_subspace * sizeof(mtype));
        }
    }
    printf("PQ算法内存分配完成！\n");

    // 训练质心
    printf("正在训练质心...\n");
    train_centroids(scale, num_subspaces, num_centroids_per_subspace, centroids, A);
    printf("质心训练完成！\n");

    // 编码
    printf("正在进行编码...\n");
    encode(scale, A, num_subspaces, num_centroids_per_subspace, centroids, codebook);
    printf("编码完成！\n");

    
    // 计时 PQ 计算
    printf("开始测试PQ算法...\n");
    

    // 构建查找表
    printf("正在构建查找表...\n");
    build_table(scale, B, num_subspaces, num_centroids_per_subspace, centroids, lookup_table);
    printf("查找表构建完成！\n");

    mtype total_time_pq = 0.0;
    for (int t = 0; t < num_trials; t++) {
        if (t % 100 == 0) printf("PQ算法进度: %d/%d\n", t, num_trials);
        mtype start = get_time();
        compute_result(scale, num_subspaces, num_centroids_per_subspace, codebook, lookup_table, C_pq);
        total_time_pq += get_time() - start;
    }
    printf("PQ算法测试完成！\n");
    printf("PQ: 平均时间 = %.6f s\n\n", total_time_pq / num_trials);

    // -------------------- 准确性比较 --------------------
    printf("=== 准确性比较 ===\n");
    printf("(以经典算法结果为基准)\n\n");
    
    printf("Classic vs Strassen:\n");
    compute_error(scale, C_strassen, C_classic);
    printf("\n");

    printf("Classic vs CW:\n");
    compute_error(scale, C_cw, C_classic);
    printf("\n");

    printf("Classic vs PQ:\n");
    compute_error(scale, C_pq, C_classic);
    printf("\n");
    printf("经典算法结果前10列:\n");
    print10(scale, C_classic);
    printf("PQ算法结果前10列:\n");
    print10(scale, C_pq);

    // -------------------- 性能总结 --------------------
    printf("=== 性能总结 ===\n");
    printf("经典算法: %.6f s\n", total_time_classic / num_trials);
    printf("Strassen: %.6f s (%.2fx)\n", total_time_strassen / num_trials, total_time_classic / total_time_strassen);
    printf("CW算法:   %.6f s (%.2fx)\n", total_time_cw / num_trials, total_time_classic / total_time_cw);
    printf("PQ算法:   %.6f s (%.2fx)\n", total_time_pq / num_trials, total_time_classic / total_time_pq);

    // -------------------- 释放内存 --------------------
    printf("\n正在释放内存...\n");
    free_matrix(scale, A);
    free_matrix(scale, B);
    free_matrix(scale, C_classic);
    free_matrix(scale, C_strassen);
    free_matrix(scale, C_cw);
    free_matrix(scale, C_pq);

    for (int s = 0; s < num_subspaces; s++) {
    for (int c = 0; c < num_centroids_per_subspace; c++) {
        free(centroids[s][c]);
        }
    free(centroids[s]);
    }
    free(centroids);

    for (int i = 0; i < scale; i++) {
        free(codebook[i]);
    }
    free(codebook);

    for (int c = 0; c < scale; c++) {
        for (int s = 0; s < num_subspaces; s++) {
            free(lookup_table[c][s]);
        }
        free(lookup_table[c]);
    }
    free(lookup_table);

    printf("内存释放完成！\n");
    printf("=== 程序运行结束 ===\n");

    return 0;
}