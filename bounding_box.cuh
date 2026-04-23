#pragma once
#include "objects.cuh"


#define max_threads 256

__global__ void calculate_bounding_box(node* node, plane* plane){
   __shared__ float minx[max_threads], miny[max_threads], maxx[max_threads], maxy[max_threads];
   int inc = max_threads * gridDim.x;
   int i = threadIdx.x;

   float val_minx = node->points[0];
   float val_miny = node->points[1];
   float val_maxx = node->points[0];
   float val_maxy = node->points[1];

   for(int j = i; j < node->number_of_points; j += inc){
    val_minx = fminf(val_minx, node->points[j * 2]);
    val_miny = fminf(val_miny, node->points[j * 2 + 1]);
    val_maxx = fmaxf(val_maxx, node->points[j * 2]); 
    val_maxy = fmaxf(val_maxy, node->points[j * 2 + 1]); 
   }
   minx[i] = val_minx;
   miny[i] = val_miny;
   maxx[i] = val_maxx;
   maxy[i] = val_maxy;

   __syncthreads();

   for(int j = max_threads / 2; j > 0; j /= 2){
    if(i < j){
        minx[i] = fminf(minx[i], minx[i+j]);
        miny[i] = fminf(miny[i], miny[i+j]);
        maxx[i] = fmaxf(maxx[i], maxx[i+j]);
        maxy[i] = fmaxf(maxy[i], maxy[i+j]);
    }
    __syncthreads();
   }
   
   if(i == 0){
        plane->minx[blockIdx.x] = minx[0];
        plane->miny[blockIdx.x] = miny[0];
        plane->maxx[blockIdx.x] = maxx[0];
        plane->maxy[blockIdx.x] = maxy[0];
   }

   unsigned int number_of_blocks = gridDim.x - 1;
   if(number_of_blocks == atomicInc(&plane->current_number_of_blocks, number_of_blocks)){
    for(int i = 1; i < gridDim.x; i++){
        plane->minx[0] = fminf(plane->minx[0], plane->minx[i]);
        plane->miny[0] = fminf(plane->miny[0], plane->miny[i]);
        plane->maxx[0] = fmaxf(plane->maxx[0], plane->maxx[i]);
        plane->maxy[0] = fmaxf(plane->maxy[0], plane->maxy[i]);
    }
   }
    
}




__global__ void build_tree(float* points, int number_of_points, tree* tree ){
    return;
}



/*

d: distance to a current center of mass
r_cell: size of a cell
f: final_force

TREE TRAVERSAL
for each points yi we will be summing forces

start_from_root_node
if it is a cell 
    if r_cell/d < thetha then
        we add f += N_cell * (y_i - y_cell) / (1 + ||y_i - y_cell||^2)^2
    else
        go deeper
else
    f += (y_i - y_current) / (1 + ||y_i - y_current||^2)^2 

*/

__global__ void traverse_tree(){
 return;
}