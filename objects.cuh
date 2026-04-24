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

3 - 11 - SE
2 - 10 - SW 
0 - 00 - NW
1 - 01 - NE

cells[i * 4 + 0] 
cells[i * 4 + 1] 
cells[i * 4 + 2] 
cells[i * 4 + 3] 

cells[i] = -2 it is locked
cells[i] = -1 it is free
cells[i] != -1 and != -2 and is_body[i] = true it is body index 
cells[i] != -1 and != -2 and is_body[i] = false it is cell index

*/

typedef struct tree {
    unsigned int number_of_cells;
    unsigned int* number_of_free_cells;
    bool* is_body;
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
        tree* create_tree(unsigned int number_of_cells){
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

        node* create_node(float* h_data, unsigned int number_of_points){
            float* d_data; 

            cudaMalloc(&d_data, sizeof(float) * number_of_points * 2);
            cudaMemcpy(d_data, h_data, sizeof(float) * number_of_points * 2, cudaMemcpyHostToDevice);

            node* object; 
            node temp;

            temp.points = d_data;
            temp.number_of_points = number_of_points; 

            cudaMalloc(&object, sizeof(node));
            cudaMemcpy(object, &temp, sizeof(temp), cudaMemcpyHostToDevice);

            return object;
        }
        plane* create_plane(unsigned int number_of_blocks){
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