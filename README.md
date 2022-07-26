# Template Search in Images using CUDA
This repo contains the implementation of a template search algorithm in images compatible for GPU programming using CUDA. 

Solution of Assignment 4 of the course COL380- Introduction to Parallel and Distributed Programming.

## Template Search
Given a Data RGB image (call it L) and a small query RGB image (call it Q). The task is to locate the query image Q approximately in the data image L. 

Note that the query image Q need not be upright with respect to the data image L. There may be a rotated copy of query image Q in data image L. Thus, a match is specified by the X, Y row-column numbers of the lower-left corner of the query image Q in the data image L and its counter-clockwise rotation in degree of the base of query image Q. 

To simplify the problem, the query image is rotated from -45° to +45° in steps of 45°. Image coordinates are (0,0) on the lower left. The required output is a series of <X, Y, degree> triplets. Please note (X, Y) represents row number and column number from the bottom left of Data image L.

Read more [here](./A4_PS.pdf)

## How to run the Code?
Clone the git repo and enter the Code folder. Execute:
```
make all
```
This command compiles all the required files.
```
./run.sh <path_of_data_image> <path_of_query_image> <threshold_1> <threshold_2> <n as in top n>
```
This command executes the search program. The threshold 1 and 2 values are required for RMSD calculations and filtering respectively. n is required for outputing top n matched points.