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
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int inc = gridDim.x * blockDim.x;
    for(int j = i; j < tree->number_of_cells; j += inc){
        tree->is_body[i] = false;
        tree->cells[i] = -1;
    }
}

/*
we build tree itera




*/
__global__ void build_tree(float* points, int number_of_points, tree* tree, root* root)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    int inc = gridDim.x * blockDim.x;

    float x, y, dx, dy, r;

    int step, child, depth;


    x = root->x;
    y = root->y;
    r = root->radius;
    step = 0;
    depth = 0;

    int n = tree->number_of_cells - 1;
    child = n;

    while(i < number_of_points)
    {
       
        // starting traversing from root
        // TODO after being blocked we can skip and start from
        // where we ended
        while(child > number_of_points){

            n = child;
            step = 0;
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
            child = tree->cells[n * 4 + step];
        }




        if (child != -2) 
        { 
            int locked = n * 4 + step;
            if (child == -1) 
            {
                if (-1 == atomicCAS(&tree->cells[locked], -1, i))
                { 
                i += inc;  
                }
            } else 
            {  
                if (child == atomicCAS(&tree->cells[locked], child, -2)) 
                {
                    int patch = -1;
                    int second_point = child;
                    int old_cell = -1;
            
                do {
                    depth++;
                    
                    // we are out of depth
                    if (depth > MAXDEPTH) {printf("ERROR: maximum depth exceeded (bodies are too close together)\n"); asm("trap;");}



                    // getting another free cell from a pool
                    int cell = atomicSub(tree->number_of_free_cells, 1) - 1;

                    if(patch == -1){
                        patch = cell;
                    } else {
                        tree->cells[old_cell + step] = cell;
                    }

                    // we are of of pool
                    if (cell <= number_of_points) {printf("ERROR: out of cell memory\n"); asm("trap;");}

                    
                    step = 0;
                    if (points[second_point * 2] > x) step |= 1;
                    if (points[second_point * 2 * 1] > y) step |= 2;
                    tree->cells[cell*4+step] = second_point;


                    r *= 0.5f;
                    dx = dy = -r;


                    step = 0;
                    if (points[i * 2] > x) {step |= 1; dx = r;}
                    if (points[i * 2 + 1] > y) {step |= 2; dy = r;}


                    x += dx;
                    y += dy;

                    n = cell*4+step; 
                    child = tree->cells[n];

                    old_cell = cell;

                    // repeat until the two bodies are different children
                } while (child >= 0);

                tree->cells[n] = i;
                tree->cells[locked] = patch;
                i += inc;

                }
            }
        } 
        // it is performence boost because if thread is 
        // locked it may 
        // revolve around and here it just
        // blocked until it is resolved 
        __syncthreads();
    }
}

__global__ void summarize_kernel(){
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