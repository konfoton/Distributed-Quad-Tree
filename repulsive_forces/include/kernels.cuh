#pragma once
#include <cuda_runtime.h>
#include "objects.cuh"


__global__ void calculate_bounding_box(node* node, plane* plane);
__global__ void clear_kernel(tree* tree);
__global__ void build_tree(float* points, int number_of_points, tree* tree,root* root);
__global__ void clear_kernel_two(float* average, int* count_of_points, int number_of_cells);
__global__ void summarize_kernel(float* points, float* average, int* count_of_points, tree* tree, int number_of_points, int number_of_cells);
__global__ void prepare_to_send_levels(tree* tree, float* average_of_points, int* count_of_points, float* result_average, int* result_count_of_points,
  int number_of_iter, int number_of_layers, int number_of_points);
__global__ void apply_sumamary_across_nodes(tree* tree, float* average_of_points, int* count_of_points, float* result_average, int* result_count_of_points,
  int number_of_iter, int number_of_layers, int number_of_points);
__global__ void ClearKernelthree(int* count, tree* tree);
__global__ void SortNodes(int* count, int* sorted, float* points, int* count_of_points, tree* tree, int number_of_points, int number_of_cells);
__global__ void traverse_tree(tree* tree, root* root, float itolsqd, float epssqd, int* sorted, float* average, int* count_of_points, 
    int number_of_cells, int number_of_points, float* points, float* gradient);
