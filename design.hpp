#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <vector>
#include <cmath>
//#include <hls_math.h>

using namespace std;

// 0. train/test size, batch size and class amount
#define NTRAIN 2
#define NTEST 2
#define BATCHSIZE 2
#define BATCHFACTOR 2
#define BATCHPART (BATCHSIZE)/(BATCHFACTOR)
#define CLASS 10
#define CLASSFACTOR 1
#define CLASSPART (CLASS)/(CLASSFACTOR)
#define ETA 0.001
#define EPOCH 1

// 1. conv2d
#define INPUTX1 32
#define INPUTY1 32
#define INPUTCHANNEL1 3
#define INPUTCHANNELFACTOR1 1
#define INPUTCHANNELPART1 (INPUTCHANNEL1)/(INPUTCHANNELFACTOR1)

#define KERNELX1 5
#define KERNELY1 5
#define FILTER1 2
#define FILTERFACTOR1 1
#define FILTERPART1 (FILTER1)/(FILTERFACTOR1)
#define STRIDEX1 1
#define STRIDEY1 1
#define PADDINGX1 0
#define PADDINGY1 0
#define DILATIONX1 1
#define DILATIONY1 1

#define OUTPUTX1 ((INPUTX1)-(DILATIONX1)*((KERNELX1)-1)-1+2*(PADDINGX1))/(STRIDEX1) + 1
#define OUTPUTY1 ((INPUTY1)-(DILATIONY1)*((KERNELY1)-1)-1+2*(PADDINGY1))/(STRIDEY1) + 1
#define OUTPUTCHANNEL1 (FILTER1)
#define OUTPUTCHANNELFACTOR1 (FILTERFACTOR1)
#define OUTPUTCHANNELPART1 (FILTERPART1)

// 2. pool2d
#define INPUTX2 (OUTPUTX1)
#define INPUTY2 (OUTPUTY1)
#define INPUTCHANNEL2 (OUTPUTCHANNEL1)
#define INPUTCHANNELFACTOR2 (OUTPUTCHANNELFACTOR1)
#define INPUTCHANNELPART2 (OUTPUTCHANNELPART1)

#define POOLX2 2
#define POOLY2 2
#define POOLDIM2 2

#define OUTPUTX2 (INPUTX2)/(POOLX2)
#define OUTPUTY2 (INPUTY2)/(POOLY2)
#define OUTPUTCHANNEL2 (INPUTCHANNEL2)
#define OUTPUTCHANNELFACTOR2 (INPUTCHANNELFACTOR2)
#define OUTPUTCHANNELPART2 (INPUTCHANNELPART2)

// 3. conv2d
#define INPUTX3 (OUTPUTX2)
#define INPUTY3 (OUTPUTY2)
#define INPUTCHANNEL3 (OUTPUTCHANNEL2)
#define INPUTCHANNELFACTOR3 (OUTPUTCHANNELFACTOR2)
#define INPUTCHANNELPART3 (OUTPUTCHANNELPART2)

#define KERNELX3 5
#define KERNELY3 5
#define FILTER3 2
#define FILTERFACTOR3 1
#define FILTERPART3 (FILTER3)/(FILTERFACTOR3)
#define STRIDEX3 1
#define STRIDEY3 1
#define PADDINGX3 0
#define PADDINGY3 0
#define DILATIONX3 1
#define DILATIONY3 1

#define OUTPUTX3 ((INPUTX3)-(DILATIONX3)*((KERNELX3)-1)-1+2*(PADDINGX3))/(STRIDEX3) + 1
#define OUTPUTY3 ((INPUTY3)-(DILATIONY3)*((KERNELY3)-1)-1+2*(PADDINGY3))/(STRIDEY3) + 1
#define OUTPUTCHANNEL3 (FILTER3)
#define OUTPUTCHANNELFACTOR3 (FILTERFACTOR3)
#define OUTPUTCHANNELPART3 (FILTERPART3)

// 4. dense
#define INPUTX4 (OUTPUTX3)
#define INPUTY4 (OUTPUTY3)
#define INPUTCHANNEL4 (OUTPUTCHANNEL3)
#define INPUTCHANNELFACTOR4 (OUTPUTCHANNELFACTOR3)
#define INPUTCHANNELPART4 (OUTPUTCHANNELPART3)

#define OUTPUTX4 1
#define OUTPUTY4 1
#define LENGTH4 64
#define LENGTHFACTOR4 1
#define LENGTHPART4 (LENGTH4)/(LENGTHFACTOR4)


// 5. dense
#define INPUTX5 (OUTPUTX4)
#define INPUTY5 (OUTPUTY4)
#define INPUTLENGTH5 (LENGTH4)
#define INPUTLENGTHFACTOR5 (LENGTHFACTOR4)
#define INPUTLENGTHPART5 (LENGTH4)/(LENGTHFACTOR4)

#define OUTPUTX5 1
#define OUTPUTY5 1
#define LENGTH5 (CLASS)
#define LENGTHFACTOR5 (CLASSFACTOR)
#define LENGTHPART5 (CLASS)/(CLASSFACTOR)


//in out values
#define LAYERAMT 4

#define WEIGHT0LEN (FILTERFACTOR1)*(INPUTCHANNELFACTOR1)*(FILTERPART1)*(INPUTCHANNELPART1)*(KERNELX1)*(KERNELY1)
#define WEIGHT1LEN (FILTERFACTOR3)*(INPUTCHANNELFACTOR3)*(FILTERPART3)*(INPUTCHANNELPART3)*(KERNELX3)*(KERNELY3)
#define WEIGHT2LEN (INPUTCHANNELFACTOR4)*(LENGTHFACTOR4)*(INPUTCHANNELPART4)*(INPUTX4)*(INPUTY4)*(LENGTHPART4)
#define WEIGHT3LEN (INPUTLENGTHFACTOR5)*(LENGTHFACTOR5)*(INPUTLENGTHPART5)*(INPUTX5)*(INPUTY5)*(LENGTHPART5)
#define WEIGHTMAXLEN max((WEIGHT0LEN), max((WEIGHT1LEN), max((WEIGHT2LEN), (WEIGHT3LEN))))

#define BIAS0LEN (FILTERFACTOR1)*(FILTERPART1)
#define BIAS1LEN (FILTERFACTOR3)*(FILTERPART3)
#define BIAS2LEN (LENGTHFACTOR4)*(LENGTHPART4)
#define BIAS3LEN (LENGTHFACTOR5)*(LENGTHPART5)
#define BIASMAXLEN max((BIAS0LEN), max((BIAS1LEN), max((BIAS2LEN), (BIAS3LEN))))

#define WEIGHT_MAXD1 max(max(max((FILTERFACTOR1), (FILTERFACTOR3)), (INPUTCHANNELFACTOR4)), (INPUTLENGTHFACTOR5))
#define WEIGHT_MAXD2 max(max(max((INPUTCHANNELFACTOR1), (INPUTCHANNELFACTOR3)), (LENGTHFACTOR4)), (LENGTHFACTOR5))
#define WEIGHT_MAXD3 max(max(max((FILTERPART1), (FILTERPART3)), (INPUTCHANNELPART4)), (INPUTLENGTHPART5))
#define WEIGHT_MAXD4 max(max(max((INPUTCHANNELPART1), (INPUTCHANNELPART3)), (INPUTX4)), (INPUTX5))
#define WEIGHT_MAXD5 max(max(max((KERNELX1), (KERNELX3)), (INPUTY4)), (INPUTY5))
#define WEIGHT_MAXD6 max(max(max((KERNELY1), (KERNELY3)), (LENGTHPART4)), (LENGTHPART5))
#define BIAS_MAXD1 max(max(max((FILTERFACTOR1), (FILTERFACTOR3)), (LENGTHFACTOR4)), (LENGTHFACTOR5))
#define BIAS_MAXD2 max(max(max((FILTERPART1), (FILTERPART3)), (LENGTHPART4)), (LENGTHPART5))






/*
//FIXED_POINT
//typedef ap_fixed<8, 0> type_input_img;
typedef ap_int<4> type_input_cls;
typedef short type_index;
typedef ap_fixed<16, 2, AP_RND_CONV, AP_SAT> type_weight;
typedef ap_fixed<24, 6, AP_RND_CONV, AP_SAT> type_inter;
//typedef ap_fixed<8, 0, AP_RND_CONV, AP_SAT> type_out;
*/

/*
//DOUBLE
typedef double type_input_img;
typedef short type_input_cls;
typedef short type_index;
typedef double type_weight;
typedef double type_inter;
typedef double type_out;
*/


//FLOAT
/*typedef float type_input_img;
typedef int type_input_cls;
typedef int type_index;
typedef float type_weight;
typedef float type_inter;
typedef float type_out;*/






/*void train(float I[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1],
		int y[BATCHFACTOR][CLASSFACTOR][BATCHPART][CLASSPART][OUTPUTX5][OUTPUTY5],
		float POUT[BATCHFACTOR][CLASSFACTOR][BATCHPART][CLASSPART][OUTPUTX5][OUTPUTY5],
		float F1[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float bF1[FILTERFACTOR1][FILTERPART1],
		float F3[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float bF3[FILTERFACTOR3][FILTERPART3],
		float W4[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float bW4[LENGTHFACTOR4][LENGTHPART4],
		float W5[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float bW5[LENGTHFACTOR5][LENGTHPART5],
		//type_index step,
		float dF1_opt[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float dbF1_opt[FILTERFACTOR1][FILTERPART1],
		float dF3_opt[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float dbF3_opt[FILTERFACTOR3][FILTERPART3],
		float dW4_opt[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float dbW4_opt[LENGTHFACTOR4][LENGTHPART4],
		float dW5[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float dbW5[LENGTHFACTOR5][LENGTHPART5]);*/


void train(float I[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1],
		int y[BATCHFACTOR][CLASSFACTOR][BATCHPART][CLASSPART][OUTPUTX5][OUTPUTY5],
		float POUT[BATCHFACTOR][CLASSFACTOR][BATCHPART][CLASSPART][OUTPUTX5][OUTPUTY5],
		float weight_in[LAYERAMT][WEIGHTMAXLEN],
		float bias_in[LAYERAMT][BIASMAXLEN],
		//type_index step,
		float weight_out[LAYERAMT][WEIGHTMAXLEN],
		float bias_out[LAYERAMT][BIASMAXLEN]);






