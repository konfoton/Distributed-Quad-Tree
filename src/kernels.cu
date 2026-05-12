#include <memory>
#include "objects.cuh"
#include "kernels.cuh"
#include <stdio.h>
#define max_threads 256
#define MAXDEPTH 32
#define WARPSIZE 32

__global__ void calculate_bounding_box(node* node, plane* plane) {
  __shared__ float minx[max_threads], miny[max_threads], maxx[max_threads],
      maxy[max_threads];
  int inc = max_threads * gridDim.x;
  int i = threadIdx.x;

  float val_minx = node->points[0];
  float val_miny = node->points[1];
  float val_maxx = node->points[0];
  float val_maxy = node->points[1];

  for (int j = i; j < node->number_of_points; j += inc) {
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

  for (int j = max_threads / 2; j > 0; j /= 2) {
    if (i < j) {
      minx[i] = fminf(minx[i], minx[i + j]);
      miny[i] = fminf(miny[i], miny[i + j]);
      maxx[i] = fmaxf(maxx[i], maxx[i + j]);
      maxy[i] = fmaxf(maxy[i], maxy[i + j]);
    }
    __syncthreads();
  }

  if (i == 0) {
    plane->minx[blockIdx.x] = minx[0];
    plane->miny[blockIdx.x] = miny[0];
    plane->maxx[blockIdx.x] = maxx[0];
    plane->maxy[blockIdx.x] = maxy[0];
  }

  unsigned int number_of_blocks = gridDim.x - 1;
  if (number_of_blocks ==
      atomicInc(&plane->current_number_of_blocks, number_of_blocks)) {
    for (int i = 1; i < gridDim.x; i++) {
      plane->minx[0] = fminf(plane->minx[0], plane->minx[i]);
      plane->miny[0] = fminf(plane->miny[0], plane->miny[i]);
      plane->maxx[0] = fmaxf(plane->maxx[0], plane->maxx[i]);
      plane->maxy[0] = fmaxf(plane->maxy[0], plane->maxy[i]);
    }
  }
}

__global__ void clear_kernel(tree* tree) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int inc = gridDim.x * blockDim.x;
  for (int j = i; j < tree->number_of_cells; j += inc) { 
    tree->cells[i] = -1;
    tree->cells[i + 1] = -1;
    tree->cells[i + 2] = -1;
    tree->cells[i + 3] = -1;
  }
}

/*
we build tree iteratively
possible entry of a tree->cells
if -1 then empty
if -2 then locked
if > number_of_points then cell
if < numbber_of_points then point
*/
__global__ void build_tree(float* points, int number_of_points, tree* tree,
                           root* root) {
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

  while (i < number_of_points) {
    /* TODO optimization may be done becase each iteratino we traverse from the
     * beggining*/
    n = tree->number_of_cells - 1;
    child = n;
    x = root->x;
    y = root->y;
    r = root->radius;
    step = 0;
    depth = 0;
    /*
    starting traversing from root
    TODO after being blocked we can skip and start from
    where we ended
    */
    while (child > number_of_points) {
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
      if (points[i * 2] > x) {
        step |= 1;
        dx = r;
      }
      if (points[i * 2 + 1] > y) {
        step |= 2;
        dy = r;
      }

      x += dx;
      y += dy;
      child = __ldcg(&tree->cells[n * 4 + step]);
    }

    if (child != -2) {
      int locked = n * 4 + step;
      if (child == -1) {
        if (-1 == atomicCAS(&tree->cells[locked], -1, i)) {
          i += inc;
        }
      } else {
        if (child == atomicCAS(&tree->cells[locked], child, -2)) {
          int patch = -1;
          int second_point = child;
          int old_cell = -1;

          do {
            depth++;

            // we are out of depth
            if (depth > MAXDEPTH) {
              printf(
                  "ERROR: maximum depth exceeded (bodies are too close "
                  "together)\n");
              asm("trap;");
            }

            // getting another free cell from a pool
            int cell = atomicSub(tree->number_of_free_cells, 1) - 1;

            if (patch == -1) {
              patch = cell;
            } else {
              tree->cells[old_cell * 4 + step] = cell;
            }

            // we are of of pool
            if (cell <= number_of_points) {
              printf("ERROR: out of cell memory\n");
              asm("trap;");
            }

            step = 0;
            if (points[second_point * 2] > x) step |= 1;
            if (points[second_point * 2 + 1] > y) step |= 2;
            tree->cells[cell * 4 + step] = second_point;

            r *= 0.5f;
            dx = dy = -r;

            step = 0;
            if (points[i * 2] > x) {
              step |= 1;
              dx = r;
            }
            if (points[i * 2 + 1] > y) {
              step |= 2;
              dy = r;
            }

            x += dx;
            y += dy;

            n = cell * 4 + step;
            child = tree->cells[n];

            old_cell = cell;

            // repeat until the two bodies are different children
          } while (child >= 0);

          tree->cells[n] = i;
          __threadfence();
          tree->cells[locked] = patch; // possible making it visible
          i += inc;
        }
      }
    }
    // it is performence boost because if thread is
    // locked it may
    // revolve around and here it just
    // blocked until it is resolved
    // __syncthreads();
  }
}


__global__ void clear_kernel_two(float* average, int* count_of_points, int number_of_cells) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int inc = gridDim.x * blockDim.x;
  for (int j = i; j < number_of_cells; j += inc) {
    count_of_points[i] = -1; 
  }
}

/*
2 * (number_of_points + number_of_cells) is the average
(number_of_points + number_of_cells) is the count of points
*/
__global__ void summarize_kernel(float* points, float* average, int* count_of_points, tree* tree, int number_of_points, int number_of_cells){
  int i, j, k, ch, inc, cnt, bottom;
  float m, cm, px, py, pz;
  __shared__ int child[max_threads * 8];

  bottom = *(tree->number_of_free_cells);
  inc = blockDim.x * gridDim.x;
  k = threadIdx.x + blockIdx.x * blockDim.x;
  while (k < bottom) k += inc;

  while (k <= number_of_cells) {
    if (count_of_points[k] < 0.0f) {
      for (i = 0; i < 4; i++) {
        ch = tree->cells[k*4+i];
        child[i*max_threads + threadIdx.x] = ch;  // cache children
        if ((ch >= number_of_points) && ((count_of_points[ch]) < 0)) {
          break;
        }
      }
      if (i == 4) {
        // all children are ready
        px = 0.0f;
        py = 0.0f;
        cnt = 0;
        float temp_count;
        for (i = 0; i < 4; i++) {
          ch = child[i*max_threads+threadIdx.x];
          if (ch >= 0) {
            float chx, chy;
            if(ch >= number_of_points){
              chx = average[ch * 2];
              chy = average[ch * 2 + 1];
            } else {
              chx = points[ch * 2];
              chy = points[ch * 2 + 1];
            }
            if (ch >= number_of_points) {
              cnt += count_of_points[ch];
              temp_count = count_of_points[ch];
            } else {
              cnt++;
              temp_count = 1;
            }
            // add child's contribution
            float coefficient = temp_count;
            px += chx * coefficient;
            py += chy * coefficient;
          }
        }
        float coefficient = 1.0f / cnt; 
        average[k * 2] =  px * coefficient;
        average[k * 2 + 1] = py * coefficient;
        __threadfence();
        count_of_points[k] = cnt;
        k += inc; 
      }
    }
  }
}

/*
We want to have exact reduced copy of first k levels
it is M = 4^0 + 4^1 + ... + 4^k cells
to imlement it effecively i propose to for each node create buffer of size M
with some predetermined pattern adn then to allreduce and then each node update
its tree to create this buffer i propose to spawn 4^k threads then each thread
process k nodes each thread see its quaternary represtation id and then process
all prefixes for instance for k = 5 and thread 03213 it process 0, 03, 032,
0321, 03213 there is a bijection between prefix and walk on a tree the last
problem is how to access appopriate idex all we have to do is for sequecne of
length m and value j index is 4^0 + 4^1 + ... + 4^(m-1) + j
*/
__global__ void prepare_to_send_levels(tree* tree, float* average_of_points, int* count_of_points, float* result_average, int* result_count_of_points,
  int number_of_iter, int number_of_layers, int number_of_points) {
    int n = tree->number_of_cells - 1; 
    int inc = blockDim.x * gridDim.x;
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int start = 1;
    int mask = 0x3;
    int ch = n;
    int sum = 0;
    int encoding;
    int copy = i;
    while(i < number_of_iter){
     for(int j = 0; j <= number_of_layers; j++){
        if(ch < number_of_points){
          break;
        }
        if(i < start){
          result_average[(sum + i) * 2] =  average_of_points[2 * ch + 0] * count_of_points[ch];
          result_average[(sum + i) * 2 + 1] =  average_of_points[2 * ch + 1] * count_of_points[ch];
          result_count_of_points[sum + i] = count_of_points[ch];
        }
        encoding = copy & mask;
        ch = tree->cells[ch * 4 + encoding];
        copy >>= 2;  
        sum += start; 
        start *= 4;
     }
     i += inc;
    }
}

// number_of_layers = 1 => root + one level => 5
__global__ void apply_sumamary_across_nodes(tree* tree, float* average_of_points, int* count_of_points, float* result_average, int* result_count_of_points,
  int number_of_iter, int number_of_layers, int number_of_points){
    int n = tree->number_of_cells - 1; 
    int inc = blockDim.x * gridDim.x;
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int start = 1;
    int mask = 0x3;
    int ch = n;
    int sum = 0;
    int encoding;
    int copy = i; 
    while(i < number_of_iter){
     for(int j = 0; j <= number_of_layers; j++){ 
        if(ch < number_of_points){
          break;
        }  
        if(i < start){
          average_of_points[2 * ch + 0] = result_average[(sum + i) * 2] / result_count_of_points[sum + i];
          average_of_points[2 * ch + 1] = result_average[(sum + i) * 2 + 1] / result_count_of_points[sum + i];
          count_of_points[ch] = result_count_of_points[sum + i];
        }
        encoding = copy & mask;
        ch = tree->cells[ch * 4 + encoding];
        copy >>= 2; 
        sum += start; 
        start *= 4;
     }
     i += inc;
    }
}

/*
start has size number_of_cells
sorted has size number of points

*/
__global__ void ClearKernelthree(int* count, tree* tree){
  int bottom = *(tree->number_of_free_cells);
  int inc = blockDim.x * gridDim.x;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while(i < bottom) i += inc;
  if(i == tree->number_of_cells - 1){
    count[i] = 0;
    return;
  }
  while(i < tree->number_of_cells){
    count[i] = -1;
    i += inc;
  }
  


}
__global__ void SortNodes(int* count, int* sorted, float* points, int* count_of_points, tree* tree, int number_of_points, int number_of_cells){
  int i, j, k, ch, dec, start, bottom;

  bottom = *(tree->number_of_free_cells);
  dec = blockDim.x * gridDim.x;
  k = number_of_cells + 1 - dec + threadIdx.x + blockIdx.x * blockDim.x - 1;

  while (k >= bottom) {
    start = count[k];
    if (start >= 0) {
      for (i = 0; i < 4; i++) {
        ch = tree->cells[k*4+i];
        if (ch >= 0) {
          if (ch >= number_of_points) {
            count[ch] = start;  
            start += count_of_points[ch];
          } else {
            sorted[start] = ch;  
            start++;
          }
        }
      }
      k -= dec; 
    }
    __syncthreads(); 
  }
}

/*
r/distance < thetha 
<=>

r^2/ (thetha)^2 < distance^2


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
__global__ void traverse_tree(tree* tree, root* root, float itolsqd, float epssqd, int* sorted, float* average, int* count_of_points, int number_of_cells, int number_of_points, float* points, float* gradient) {
  int i, j, k, n, depth, base, sbase, diff, pd, nd;
  float ax, ay, az, dx, dy, dz, tmp, x_cell, y_cell, count;
  __shared__ int pos[MAXDEPTH * max_threads/WARPSIZE], node[MAXDEPTH * max_threads/WARPSIZE];
  __shared__ float dq[MAXDEPTH * max_threads/WARPSIZE];

  if (0 == threadIdx.x) {
    tmp = root->radius * 2;
    dq[0] = tmp * tmp * itolsqd;
    for (i = 1; i < MAXDEPTH; i++) {
      dq[i] = dq[i - 1] * 0.25f;
      dq[i - 1] += epssqd;
    }
    dq[i - 1] += epssqd;
  }
  __syncthreads();

  base = threadIdx.x / WARPSIZE;
  sbase = base * WARPSIZE;
  j = base * MAXDEPTH;

  diff = threadIdx.x - sbase;

  if (diff < MAXDEPTH) {
    dq[diff+j] = dq[diff];
  }
  __syncthreads();

  for (k = threadIdx.x + blockIdx.x * blockDim.x; k < number_of_points; k += blockDim.x * gridDim.x) {
    i = sorted[k];

    float x_point = points[2 * i];
    float y_point = points[2 * i + 1];

    ax = 0.0f;
    ay = 0.0f;

    depth = j;
    if (sbase == threadIdx.x) {
      pos[j] = 0;
      node[j] = (number_of_cells - 1) * 4;
    }
    do {
      // stack is not empty
      pd = pos[depth];
      nd = node[depth];
      while (pd < 4) {

        n = tree->cells[nd + pd];
        pd++;

        if (n >= 0) {

          if(n >= number_of_points){
            x_cell = average[2 * n];
            y_cell = average[2 * n + 1];
            count = count_of_points[n];
          } else {
            x_cell = points[2 * n];
            y_cell = points[2 * n + 1];
            count = 1;
          }

          dx = x_point - x_cell;
          dy = y_point - y_cell;
          tmp = dx*dx + (dy*dy + epssqd);  
          if ((n < number_of_points) || __all_sync(0xffffffff, tmp >= dq[depth])) { 
            tmp = count / powf((1 + tmp), 2);
            ax += dx * tmp;
            ay += dy * tmp;
          } else {
            if (sbase == threadIdx.x) {
              pos[depth] = pd;
              node[depth] = nd;
            }
            depth++;
            pd = 0;
            nd = n * 4;
          }
        } else {
          pd = 4; 
        }
      }
      depth--;
    } while (depth >= j);

    gradient[k * 2] = ax;
    gradient[k * 2 + 1] = ay;
  }
}