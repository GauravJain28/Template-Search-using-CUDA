#include <bits/stdc++.h>
#include <iostream>
#include <cuda_runtime.h>
using namespace std;


__global__ void kernel(int* data_img, int* query_img, float* d_ans, float th1, float th2, int m, int n, int mq, int nq, float imqsum){
    
    //thread id
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    //start point
    int x0 = tid % n;
    int y0 = tid / n;

    //sin45 and cos45
    float sin45 = 0.70710678118;
    float cos45 = 0.70710678118;

    for(int orientation = 0; orientation<3;orientation++){
        d_ans[y0*n*3 + x0*3 + orientation] = FLT_MAX;
    
        float hcomp_x,hcomp_y;
        float vcomp_x,vcomp_y;

        float x1,y1,x2,y2,x3,y3;
        int a0,b0,a1,b1,a2,b2,a3,b3;

        //0-degree
        if (orientation == 0){
            x1 = x0;
            y1 = y0 + mq-1;

            x2 = x0 + nq-1;
            y2 = y1;

            x3 = x2;
            y3 = y0;

            a0 = (int) x0;
            b0 = (int) y0;

            a1 = a0;
            b1 = (int) y1;

            a2 = (int) x2;
            b2 = b1;

            a3 = a2;
            b3 = b0;

            hcomp_x = 1;
            vcomp_x = 0;

            hcomp_y = 0;
            vcomp_y = 1;
        }

        // -45 degree
        else if(orientation == 1){
            x1 = x0 + (mq-1)*sin45;
            y1 = y0 + (mq-1)*cos45;

            x2 = x0 + (nq-1)*cos45 + (mq-1)*sin45;
            y2 = y0 + (mq-1)*cos45 - (nq-1)*sin45;

            x3 = x0 + (nq-1)*cos45;
            y3 = y0 - (nq-1)*sin45;

            a0 = (int) x0;
            b0 = (int) y3;

            a1 = a0;
            b1 = (int) y1;

            a2 = (int) x2;
            b2 = b1;

            a3 = a2;
            b3 = b0;

            hcomp_x = cos45;
            vcomp_x = sin45;

            hcomp_y = -sin45;
            vcomp_y = cos45;

        }

        // 45 degree
        else{
            x1 = x0 - (mq-1)*sin45;
            y1 = y0 + (mq-1)*cos45;

            x2 = x0 + (nq-1)*cos45 - (mq-1)*sin45;
            y2 = y0 + (mq-1)*cos45 + (nq-1)*sin45;

            x3 = x0 + (nq-1)*cos45;
            y3 = y0 + (nq-1)*sin45;

            a0 = (int) x1;
            b0 = (int) y0;

            a1 = a0;
            b1 = (int) y2;

            a2 = (int) x3;
            b2 = b1;

            a3 = a2;
            b3 = b0;

            hcomp_x = cos45;
            vcomp_x = -sin45;

            hcomp_y = sin45;
            vcomp_y = cos45;
        }

        //out of bounds check
        if(b0 < 0 || a0 < 0 || b2 > m-1 || a2 > n-1){
            //return;
        }
        else{

        //filtering
        float imsum = 0;

        for (int i = b0; i <= b1; i++){
            for (int j= a0 ; j <= a3; j++){
                float temp = 0;
                temp += data_img[i*n*3 + j*3];
                temp += data_img[i*n*3 + j*3 + 1];
                temp += data_img[i*n*3 + j*3 + 2];
                //temp = temp/(3*(a3-a0+1)*(b1-b0+1));
                imsum += temp;

            }
        }

        imsum = imsum/(3*(a3-a0+1)*(b1-b0+1));
        imsum = abs(imsum - imqsum);

        // std::cout << "Average: " << imsum << endl;

        if (imsum >= th2){
            //return;
        }
        else{

        //rmsd

        float rmd = 0.0;

        for (int i = 0; i < mq; i++){
            for (int j = 0; j < nq; j++){
                float x =  (x0 + hcomp_x*j + vcomp_x*i);
                float y =  (y0 + hcomp_y*j + vcomp_y*i);

                int bx0 = (int) x;
                int by0 = (int) y;

                int bx1 = bx0;
                int by1 = by0 + 1;

                int bx2 = bx0 + 1;
                int by2 = by1;

                int bx3 = bx2;
                int by3 = by0;

                float gx = x - bx0;
                float gy = y - by0;

                float red = data_img[by0*n*3 + bx0*3]*(1-gx)*(1-gy) 
                          + data_img[by1*n*3 + bx1*3]*(1-gx)*(gy)
                          + data_img[by2*n*3 + bx2*3]*(gx)*(gy)
                          + data_img[by3*n*3 + bx3*3]*(gx)*(1-gy);

                float green = data_img[by0*n*3 + bx0*3+1]*(1-gx)*(1-gy) 
                          + data_img[by1*n*3 + bx1*3+1]*(1-gx)*(gy)
                          + data_img[by2*n*3 + bx2*3+1]*(gx)*(gy)
                          + data_img[by3*n*3 + bx3*3+1]*(gx)*(1-gy);

                float blue = data_img[by0*n*3 + bx0*3+2]*(1-gx)*(1-gy) 
                          + data_img[by1*n*3 + bx1*3+2]*(1-gx)*(gy)
                          + data_img[by2*n*3 + bx2*3+2]*(gx)*(gy)
                          + data_img[by3*n*3 + bx3*3+2]*(gx)*(1-gy);

                float qred = query_img[i*nq*3 + j*3];
                float qgreen = query_img[i*nq*3 + j*3 + 1];
                float qblue = query_img[i*nq*3 + j*3 + 2];

                rmd += (qred-red)*(qred-red) + (qgreen - green)*(qgreen - green) + (qblue-blue)*(qblue-blue);

            }
        }

        rmd = rmd/(3*nq*mq);

        rmd = sqrt(rmd);
        d_ans[y0*n*3 + x0*3 + orientation] = rmd;
            }    
        }
    }
}





int main(int argc, char const *argv[]){

    string data_img_p = argv[1];
    string query_img_p = argv[2];
    int num = stoi(argv[5]);
    float th1 = stof(argv[3]);
    float th2 = stof(argv[4]);

    ifstream data_file; 
    data_file.open(data_img_p,ios::in);

    int data_m;
    int data_n;

    string word;
    data_file >> word;
    data_m = stoi(word);
    data_file >> word;
    data_n = stoi(word);

    int* data_img = new int[data_m*data_n*3];

    for(int i=0;i<data_m;i++){
        for(int j=0;j<data_n*3;j++){
            data_file >> word;
            data_img[(data_m-i-1)*data_n*3 + j] = stoi(word);
        }
    }
    
    ifstream query_file; 
    query_file.open(query_img_p,ios::in);

    int query_m;
    int query_n;

    query_file >> word;
    query_m = stoi(word);
    query_file >> word;
    query_n = stoi(word);

    int* query_img = new int[query_m*query_n*3];

    for(int i=0;i<query_m;i++){
        for(int j=0;j<query_n*3;j++){
            query_file >> word;
            query_img[(query_m-i-1)*query_n*3 + j] = stoi(word);
        }
    }

    float imqsum = 0;

    for (int i = 0; i < query_m*query_n*3; i++){
        imqsum += query_img[i];
    }
    imqsum = imqsum/(query_m*query_n*3);

    // cout<<data_img[(840)*1200*3 + 900*3]<<" "<<query_img[0]<<endl;

    size_t data_size = 3*data_m*data_n*sizeof(int);
    size_t query_size = 3*query_m*query_n*sizeof(int);
    size_t ans_size = 3*data_m*data_n*sizeof(float);

    float* ans = (float*)malloc(ans_size);
    
    //cuda memory allocation
    int* d_data_img;
    int* d_query_img;
    float* d_ans;

    cudaMalloc(&d_data_img, data_size);
    cudaMalloc(&d_query_img, query_size);
    cudaMalloc(&d_ans, ans_size);
    
    cudaMemcpy(d_data_img, data_img, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_img, query_img, query_size, cudaMemcpyHostToDevice);

    int blkDim = (data_m*data_n)/1024 + 1;
    int thdDim = 1024;

    kernel<<<blkDim,thdDim>>>(d_data_img,d_query_img,d_ans,th1,th2,data_m,data_n,query_m,query_n,imqsum);
    //cudaDeviceSynchronize();
    
    //float* ans = (float*)malloc(data_size);
    cudaMemcpy(ans, d_ans, ans_size, cudaMemcpyDeviceToHost);
  
    //std::cout << ans[841*1200*3+3*900] << " " << ans[1] << " "<< ans[2] << endl;
    
    
    float minrmsd = FLT_MAX;
    int minrow,mincol,minor;
    
    //std::cout<<"started\n";
    
    for(int i = 0; i < data_m;i++){
        for (int j = 0; j < data_n; j++){
            if(ans[i*data_n*3 + j*3]<minrmsd){
                    minrow = i;
                    mincol = j;
                    minor = 0;
                    minrmsd = ans[i*data_n*3 + j*3];
            }
            if(ans[i*data_n*3 + j*3+1]<minrmsd){
                    minrow = i;
                    mincol = j;
                    minor = 1;
                    minrmsd = ans[i*data_n*3 + j*3+1];
            }
            if(ans[i*data_n*3 + j*3+2]<minrmsd){
                    minrow = i;
                    mincol = j;
                    minor = 2;
                    minrmsd = ans[i*data_n*3 + j*3+2];
            }
        }
    }
    //std::cout<<"ended\n";
    
    
    int angle[3] = {0,-45,45};
    ofstream myfile;
    myfile.open ("output.txt");
    myfile << minrow<<" "<<mincol<<" "<<angle[minor]<<"\n";
    
    myfile.close();
    
            
    
}