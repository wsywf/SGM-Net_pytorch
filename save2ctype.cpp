#include <iostream>
#include <math.h>
#include <fstream>
#include <string>   

using namespace std;
extern "C" {
void show_matrix(float *matrix, int lens,int img_name)

   {  
    cout<<lens<<endl;
    cout<<img_name<<endl;

    int len=lens;
    int imag_name = img_name;

    sprintf(path, "/test_file/mid_out_cost%d.txt",imag_name);
    std::ofstream  ofs(path, std::ios::binary | std::ios::out);
    ofs.write((const char*)matrix, sizeof(float) * lens);
    ofs.close();
    //show(matrix[1],matrix[2]);
    cout<<  matrix[0]<<endl<<matrix[6]<<endl;
   }
}

