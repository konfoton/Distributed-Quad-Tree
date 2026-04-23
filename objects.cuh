#pragma once





typedef struct node {
    int number_of_points;
    float* points;
} node;

typedef struct plane {
    int current_number_of_blocks = 0;
    float* minx;
    float* miny;
    float* maxx;
    float* maxy;
} plane;



/*
cells[i * 4 + 1] stores NW
cells[i * 4 + 2] stores NE
cells[i * 4 + 3] stores SW
cells[i * 4 + 4] stores SE
is_body checks if it is cell or a body
*/

typedef struct tree {
    int number_of_cells;
    bool* is_body;
    int* cells;
} tree;




/*
each cell has info about its center of mass and 
number of elements
*/
typedef struct center_of_mass {
    float* center;
    int* number_of_elements; 
} center_of_mass; 



class builder {
    public:
        tree* create_tree(int number_of_cells){
            bool* is_body; 
            int* cells;

            cudaMalloc(&is_body, sizeof(bool) * number_of_cells);
            cudaMalloc(&cells, sizeof(int) * number_of_cells);

            tree* object;
            tree temp;

            temp.is_body = is_body;
            temp.cells = cells;
            temp.number_of_cells = number_of_cells;

            cudaMalloc(&object, sizeof(tree));
            cudaMemcpy(object, &temp, sizeof(temp), cudaMemcpyHostToDevice);
            
            return object;
        }

        node* create_node(float* h_data, int number_of_points){
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
        plane* create_plane(int number_of_blocks){
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
            temp.maxx = maxx;
            temp.maxy = maxy;
            
            cudaMalloc(&object, sizeof(object));
            cudaMemcpy(object, &temp, sizeof(temp), cudaMemcpyHostToDevice);

            return object;
        }
        
        

}