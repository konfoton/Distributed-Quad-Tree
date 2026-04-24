#pragma once
#include "objects.cuh"


#define max_threads 256
#define MAXDEPTH 32


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

__global__ void clear_kernel(tree* tree){
    int i = threadIdx.x; + blockIdx.x * blockDim.x;
    int inc = gridDim.x * blockDim.x;
    for(int j = i; j < tree->number_of_cells; j += inc){
        tree->is_body[i] = false;
        tree->cells[i] = -1;
    }
}

/*
we build tree itera




*/
__global__ void build_tree(float* points, int number_of_points, tree* tree, root* root){

    int i = threadIdx.x; + blockIdx.x * blockDim.x;

    int inc = gridDim.x * blockDim.x;

    float x, y, dx, dy, r;

    int step;
    int child;


    x = root.x;
    y = root.y;
    r = root.radius;
    step = 0;

    int child = tree->number_of_cells - 1;

    while(i < number_of_points){
       
        while(child > number_of_points){

            r *= 0.5f;

            dx = -r;
            dy = -r;

            /*
            3 - 11 - SE
            2 - 10 - SW 
            0 - 00 - NW
            1 - 01 - NE
            */
            if(points[i * 2] > x){
                step |= 1;
                dx = r;
            }
            if(points[i * 2 + 1] > y){
                step |= 2;
                dy = r;
            }

            x += dx;
            y += dy;
            child = tree->cells[child * 4 + step];
            step = 0;
        }
        if (child != -2) { 
            locked = n*8+j;
      if (ch == -1) {
        if (-1 == atomicCAS((int*)&childd[locked], -1, i)) { 
          i += inc;  
          skip = 1;
        }
      } else {  // there already is a body at this position
        if (ch == atomicCAS((int*)&childd[locked], ch, -2)) {  // try to lock
          patch = -1;
          const float4 chp = posMassd[ch];
          // create new cell(s) and insert the old and new bodies
          do {
            depth++;
            if (depth > MAXDEPTH) {printf("ERROR: maximum depth exceeded (bodies are too close together)\n"); asm("trap;");}

            cell = atomicSub((int*)&bottomd, 1) - 1;
            if (cell <= nbodiesd) {printf("ERROR: out of cell memory\n"); asm("trap;");}

            if (patch != -1) {
              childd[n*8+j] = cell;
            }
            patch = max(patch, cell);

            j = 0;
            if (x < chp.x) j = 1;
            if (y < chp.y) j |= 2;
            if (z < chp.z) j |= 4;
            childd[cell*8+j] = ch;

            n = cell;
            r *= 0.5f;
            dx = dy = dz = -r;
            j = 0;
            if (x < p.x) {j = 1; dx = r;}
            if (y < p.y) {j |= 2; dy = r;}
            if (z < p.z) {j |= 4; dz = r;}
            x += dx;
            y += dy;
            z += dz;

            ch = childd[n*8+j];
            // repeat until the two bodies are different children
          } while (ch >= 0);
        
}

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