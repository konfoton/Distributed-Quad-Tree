#pragma once

typedef struct node {
  unsigned int number_of_points;
  float* points;
} node;

typedef struct plane {
  unsigned int current_number_of_blocks = 0;
  float* minx;
  float* miny;
  float* maxx;
  float* maxy;
} plane;

/*

quadrant indexing used by build_tree:
  bit 0 (value 1) is set when point.x > center.x  (east)
  bit 1 (value 2) is set when point.y > center.y  (north)

  step 0 = 00 = SW   (west, south)
  step 1 = 01 = SE   (east, south)
  step 2 = 10 = NW   (west, north)
  step 3 = 11 = NE   (east, north)

children of cell i live at cells[i*4 + 0 .. i*4 + 3]:
  cells[i*4 + 0] -> SW
  cells[i*4 + 1] -> SE
  cells[i*4 + 2] -> NW
  cells[i*4 + 3] -> NE

slot sentinels:
  cells[k] = -1          slot is free
  cells[k] = -2          slot is locked (insertion in progress)
*/

typedef struct tree {
  unsigned int number_of_cells;
  unsigned int* number_of_free_cells;
  int* cells;
} tree;

typedef struct root {
  float x, y, radius;
} root;

/*
each cell has info about its center of mass and
number of elements
*/
typedef struct center_of_mass {
  float* center;
  unsigned int* number_of_elements;
} center_of_mass;

class builder {
 public:
  tree* create_tree(unsigned int number_of_cells) {
    int* cells;

    cudaMalloc(&cells, sizeof(unsigned int) * number_of_cells);

    tree* object;
    tree temp;

    temp.cells = cells;
    temp.number_of_cells = number_of_cells;

    cudaMalloc(&object, sizeof(tree));
    cudaMemcpy(object, &temp, sizeof(temp), cudaMemcpyHostToDevice);

    return object;
  }

  node* create_node(float* h_data, unsigned int number_of_points) {
    float* d_data;

    cudaMalloc(&d_data, sizeof(float) * number_of_points * 2);
    cudaMemcpy(d_data, h_data, sizeof(float) * number_of_points * 2,
               cudaMemcpyHostToDevice);

    node* object;
    node temp;

    temp.points = d_data;
    temp.number_of_points = number_of_points;

    cudaMalloc(&object, sizeof(node));
    cudaMemcpy(object, &temp, sizeof(temp), cudaMemcpyHostToDevice);

    return object;
  }
  plane* create_plane(unsigned int number_of_blocks) {
    float *minx, *miny, *maxx, *maxy;
    cudaMalloc(&minx, number_of_blocks * sizeof(float));
    cudaMalloc(&miny, number_of_blocks * sizeof(float));
    cudaMalloc(&maxx, number_of_blocks * sizeof(float));
    cudaMalloc(&maxy, number_of_blocks * sizeof(float));

    plane* object;
    plane temp;

    temp.minx = minx;
    temp.miny = miny;
    temp.maxx = maxx;
    temp.maxy = maxy;

    cudaMalloc(&object, sizeof(plane));
    cudaMemcpy(object, &temp, sizeof(temp), cudaMemcpyHostToDevice);

    return object;
  }
};