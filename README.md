# SGM-Net_pytorch
SGM-Net re-implemented with pytorch.
I‘m trying to re-implement the closed-source SGM-Net project proposed by <SGM-Nets: Semi-global matching with neural networks> with pytorch.For of some reasons, I could just partly release the program with only path-cost.

Contact: [wsywf@bupt.edu.cn](mailto:wsywf@bupt.edu.cn). Any questions or discussions are welcomed!  

## Usage

If you want to train the SGM-Net,you might need a initial cost_volume same as the traditional stereo-maching task.  
1.You can use the mc-cnn project(https://github.com/jzbontar/mc-cnn)to get the initial cost,just like the original paper refered.

2.You can also get a ct-cost_colume with a provided python demo file ./calculate_ct_cost/cal_ct.py,after you set the data paths.

./dataloader.py --------------- To set the dataset,the datasets needed left_image and disp_image.

./SGMNet.py ------------------- The SGM-NET modle.

./train.py -------------------- Train the SGM-NET.

./loss/sgm_pathloss.py -------- The source file to calculate the path-cost and manually get the backward grad with Dynamic Programming stragety.

./test.py --------------------- To get the p1p2 params with the trained model.

If you want to use the params to post-procedure with c++,you can set the save_path in save2ctype.cpp and then use command     g++ -fPIC -shared -o libsave.so save2ctype.cpp  to build a dynamic link library and use it to get a c_type-params-volume.
                               
