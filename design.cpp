#include "design.hpp"

/*

Weights and ADAM momentums

*/

/*float F1[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1];
float bF1[FILTERFACTOR1][FILTERPART1];
float F3[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3];
float bF3[FILTERFACTOR3][FILTERPART3];
float W4[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4];
float bW4[LENGTHFACTOR4][LENGTHPART4];*/

/*static float mF1[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1] = {0};
static float vF1[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1] = {0};
static float mbF1[FILTERFACTOR1][FILTERPART1] = {0};
static float vbF1[FILTERFACTOR1][FILTERPART1] = {0};

static float mF3[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3] = {0};
static float vF3[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3] = {0};
static float mbF3[FILTERFACTOR3][FILTERPART3] = {0};
static float vbF3[FILTERFACTOR3][FILTERPART3] = {0};

static float mW4[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4] = {0};
static float vW4[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4] = {0};
static float mbW4[LENGTHFACTOR4][LENGTHPART4] = {0};
static float vbW4[LENGTHFACTOR4][LENGTHPART4] = {0};

static float mW5[INPUTFACTOR5][LENGTHFACTOR5][INPUTPART5][LENGTHPART5] = {0};
static float vW5[INPUTFACTOR5][LENGTHFACTOR5][INPUTPART5][LENGTHPART5] = {0};
static float mbW5[LENGTHFACTOR5][LENGTHPART5] = {0};
static float vbW5[LENGTHFACTOR5][LENGTHPART5] = {0};
*/

/*

Activation and approximate math functions

*/

float relu(float x){

	if (x < 0){
		x = 0;
	}

	return x;

}

float relu_drv(float x){

	if (x <= 0){
		x = 0;
	}
	else{
		x = 1;
	}

	return x;

}

/*float exp_func(float x){
	float x_2 = x*x;
	float x_3 = x_2*x;

	return 1+ x + x_2/2 + x_3/6;
}*/

float exp_func(float x){

	return exp(x);
}

/*

1: 2D Convolution

*/

void conv2d_1_in(float I[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1],
		float F1[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float I_in[BATCHFACTOR][INPUTCHANNELFACTOR1][FILTERFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1],
		float F1_in[FILTERFACTOR1][INPUTCHANNELFACTOR1][BATCHFACTOR][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float I_back_tmp[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1]){

	int i, j, k, l, m, n, p, q, r, s;

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<INPUTY1 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL
							float I_val = I[m][n][i][j][k][l];
							I_back_tmp[m][n][i][j][k][l] = I_val;

							for (p=0 ; p<FILTERFACTOR1 ; ++p){
							#pragma HLS UNROLL
								I_in[m][n][p][i][j][k][l] = I_val;
							}
						}
					}
				}
			}
		}
	}


	for (i=0 ; i<FILTERPART1 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY1 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR1 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL
							float F1_val = F1[m][n][i][j][k][l];

							for (p=0 ; p<BATCHFACTOR ; ++p){
							#pragma HLS UNROLL
								F1_in[m][n][p][i][j][k][l] = F1_val;
							}
						}
					}
				}
			}
		}
	}

}

void conv2d_1_calc(float I_part[BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1],
		float F1_part[FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float P1_part[BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1]){

	int i, j, k, l, m, n, p, q;
	float I_tmp[BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1];
	float F1_tmp[FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1];

	for (i=0 ; i<FILTERPART1 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY1 ; ++l){
#pragma HLS PIPELINE off
					F1_tmp[i][j][k][l] = F1_part[i][j][k][l];
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<INPUTY1 ; ++l){
#pragma HLS PIPELINE off
					I_tmp[i][j][k][l] = I_part[i][j][k][l];
				}
			}
		}
	}

	for (i=0; i<BATCHPART ; ++i){
		for (j=0 ; j<OUTPUTCHANNELPART1 ; ++j){
			for (k=0 ; k<OUTPUTX1 ; ++k){
				for (l=0 ; l<OUTPUTY1 ; ++l){
					float P1_val = 0;

					for (m=0 ; m<INPUTCHANNELPART1 ; ++m){
						for (n=0 ; n<KERNELX1 ; ++n){
							for (p=0 ; p<KERNELY1 ; ++p){
#pragma HLS PIPELINE off
								int x = k*STRIDEX1 - PADDINGX1 + n*DILATIONX1;
								int y = l*STRIDEY1 - PADDINGY1 + p*DILATIONY1;
								//int x_d = x / DILATIONX1;
								//int y_d = y / DILATIONY1;
								if (x>-1 && x<INPUTX1 && y>-1 && y<INPUTY1){

									P1_val += I_tmp[i][m][x][y] * F1_tmp[j][m][n][p];


								}
							}
						}
					}

					P1_part[i][j][k][l] = P1_val;

					//if (i==0 && j==0 && k==0 && l==0){
					//	cout<<"P1_part[0][0][0][0]: "<<P1_part[0][0][0][0]<<endl;
					//}

				}
			}
		}
	}

}

void conv2d_1_out(float P1_in[BATCHFACTOR][FILTERFACTOR1][INPUTCHANNELFACTOR1][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1],
		float bF1[FILTERFACTOR1][FILTERPART1],
		float P1_act[BATCHFACTOR][OUTPUTCHANNELFACTOR1][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1],
		float P1_nonact_back[BATCHFACTOR][OUTPUTCHANNELFACTOR1][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1],
		float I_back_tmp[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1],
		float I_back[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1]){

	int i, j, k, l, m, n, p, q;
	float bF1_temp[FILTERFACTOR1][FILTERPART1];
#pragma HLS ARRAY_PARTITION variable=bF1_temp complete dim=1


	lp1:for (i=0 ; i<FILTERPART1 ; ++i){
		lp2:for (j=0 ; j<FILTERFACTOR1 ; ++j){
		#pragma HLS UNROLL
			bF1_temp[j][i] = bF1[j][i];
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<OUTPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY1 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<OUTPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL
							float P1_val = 0;

							for (p=0 ; p<INPUTCHANNELFACTOR1 ; ++p){
							#pragma HLS UNROLL
								P1_val += P1_in[m][n][p][i][j][k][l];
							}

							P1_val += bF1_temp[n][j];

							P1_nonact_back[m][n][i][j][k][l] = P1_val;
							P1_act[m][n][i][j][k][l] = relu(P1_val);


						}
					}
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<INPUTY1 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL
							I_back[m][n][i][j][k][l] = I_back_tmp[m][n][i][j][k][l];
						}
					}
				}
			}
		}
	}

}

void conv2d_1(float I[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1],
		float F1[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float bF1[FILTERFACTOR1][FILTERPART1],
		float P1_act[BATCHFACTOR][OUTPUTCHANNELFACTOR1][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1],
		float I_back[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1],
		float P1_nonact_back[BATCHFACTOR][OUTPUTCHANNELFACTOR1][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1]){
#pragma HLS DATAFLOW

	int i, j, k;

	float I_in1[BATCHFACTOR][INPUTCHANNELFACTOR1][FILTERFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1];
#pragma HLS STREAM variable=I_in1 type=fifo
#pragma HLS ARRAY_PARTITION variable=I_in1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=I_in1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=I_in1 complete dim=3
	float F1_in1[FILTERFACTOR1][INPUTCHANNELFACTOR1][BATCHFACTOR][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1];
#pragma HLS STREAM variable=F1_in1 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=F1_in1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=F1_in1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=F1_in1 complete dim=3
	float I_back_tmp[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1];
#pragma HLS STREAM variable=I_back_tmp type=fifo
#pragma HLS ARRAY_PARTITION variable=I_back_tmp complete dim=1
#pragma HLS ARRAY_PARTITION variable=I_back_tmp complete dim=2
	float P1_in[BATCHFACTOR][FILTERFACTOR1][INPUTCHANNELFACTOR1][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1];
#pragma HLS STREAM variable=P1_in type=fifo
#pragma HLS ARRAY_PARTITION variable=P1_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=P1_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=P1_in complete dim=3

	conv2d_1_in(I, F1, I_in1, F1_in1, I_back_tmp);

	for (i=0 ; i<BATCHFACTOR ; ++i){
	#pragma HLS UNROLL
		for (j=0 ; j<INPUTCHANNELFACTOR1 ; ++j){
		#pragma HLS UNROLL
			for (k=0 ; k<FILTERFACTOR1 ; ++k){
			#pragma HLS UNROLL
				conv2d_1_calc(I_in1[i][j][k], F1_in1[k][j][i], P1_in[i][k][j]);
			}
		}
	}
	/*conv2d_1_calc(I_in1[0][0][0], F1_in1[0][0][0], P1_in[0][0][0]);
	conv2d_1_calc(I_in1[1][0][0], F1_in1[0][0][1], P1_in[1][0][0]);*/ //BF2FF1FF1

	conv2d_1_out(P1_in, bF1, P1_act, P1_nonact_back, I_back_tmp, I_back);

}

/*

2: 2D MaxPool

*/

void maxpool2d_2_calc(float P1_part[BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1],
		float P2_part[BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2],
		float P2_back_part[BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2],
		int P2_poolindex_back_part[BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2][POOLDIM2]){

	int i, j, k, l, m;
	float P1_tmp[BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1];

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<OUTPUTCHANNELPART2 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY1 ; ++l){
#pragma HLS PIPELINE off
					P1_tmp[i][j][k][l] = P1_part[i][j][k][l];
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
		for (j=0 ; j<OUTPUTCHANNELPART2 ; ++j){
			for (k=0 ; k<OUTPUTX2 ; ++k){
				for (l=0 ; l<OUTPUTY2 ; ++l){
#pragma HLS PIPELINE off
					float P2_val = P1_tmp[i][j][k*POOLX2][l*POOLY2];
					int point[POOLDIM2];
					point[0] = 0;
					point[1] = 0;

					if (P1_tmp[i][j][k*POOLX2][l*POOLY2+1] > P2_val){
						P2_val = P1_tmp[i][j][k*POOLX2][l*POOLY2+1];
						point[0] = 0;
						point[1] = 1;
					}
					if (P1_tmp[i][j][k*POOLX2+1][l*POOLY2] > P2_val){
						P2_val = P1_tmp[i][j][k*POOLX2+1][l*POOLY2];
						point[0] = 1;
						point[1] = 0;
					}
					if (P1_tmp[i][j][k*POOLX2+1][l*POOLY2+1] > P2_val){
						P2_val = P1_tmp[i][j][k*POOLX2+1][l*POOLY2+1];
						point[0] = 1;
						point[1] = 1;
					}

					P2_part[i][j][k][l] = P2_val;
					P2_back_part[i][j][k][l] = P2_val;
					for (m=0 ; m<POOLDIM2 ; ++m){
						P2_poolindex_back_part[i][j][k][l][m] = point[m];
					}
				}
			}
		}
	}

}

void maxpool2d_2(float P1_act[BATCHFACTOR][OUTPUTCHANNELFACTOR1][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1],
		float P2_act[BATCHFACTOR][OUTPUTCHANNELFACTOR2][BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2],
		float P2_act_back[BATCHFACTOR][OUTPUTCHANNELFACTOR2][BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2],
		int P2_poolindex_back[BATCHFACTOR][OUTPUTCHANNELFACTOR2][BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2][POOLDIM2]){

	int i, j;

	for (i=0 ; i<BATCHFACTOR ; ++i){
	#pragma HLS UNROLL
		for (j=0 ; j<OUTPUTCHANNELFACTOR1 ; ++j){
		#pragma HLS UNROLL
			maxpool2d_2_calc(P1_act[i][j], P2_act[i][j], P2_act_back[i][j], P2_poolindex_back[i][j]);
		}
	}

}

/*

3: 2D Convolution

*/

void conv2d_3_in(float P2[BATCHFACTOR][OUTPUTCHANNELFACTOR2][BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2],
		float F3[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float P2_in[BATCHFACTOR][OUTPUTCHANNELFACTOR2][FILTERFACTOR3][BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2],
		float F3_in[FILTERFACTOR3][INPUTCHANNELFACTOR3][BATCHFACTOR][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float F3_back_tmp[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3]){

	int i, j, k, l, m, n, p, q, r, s;

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<OUTPUTCHANNELPART2 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX2 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY2 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<OUTPUTCHANNELFACTOR2 ; ++n){
						#pragma HLS UNROLL
							float P2_val = P2[m][n][i][j][k][l];

							for (p=0 ; p<FILTERFACTOR3 ; ++p){
							#pragma HLS UNROLL
								P2_in[m][n][p][i][j][k][l] = P2_val;
							}
						}
					}
				}
			}
		}
	}


	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY3 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR3 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL
							float F3_val = F3[m][n][i][j][k][l];
							F3_back_tmp[m][n][i][j][k][l] = F3_val;

							for (p=0 ; p<BATCHFACTOR ; ++p){
							#pragma HLS UNROLL
								F3_in[m][n][p][i][j][k][l] = F3_val;
							}
						}
					}
				}
			}
		}
	}

}

void conv2d_3_calc(float P2_part[BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2],
		float F3_part[FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float P3_part[BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3]){

	int i, j, k, l, m, n, p, q;
	float P2_tmp[BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2];
	float F3_tmp[FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3];

	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY3 ; ++l){
#pragma HLS PIPELINE off
					F3_tmp[i][j][k][l] = F3_part[i][j][k][l];
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<OUTPUTCHANNELPART2 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX2 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY2 ; ++l){
#pragma HLS PIPELINE off
					P2_tmp[i][j][k][l] = P2_part[i][j][k][l];
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
		for (j=0 ; j<OUTPUTCHANNELPART3 ; ++j){
			for (k=0 ; k<OUTPUTX3 ; ++k){
				for (l=0 ; l<OUTPUTY3 ; ++l){
					float P3_val = 0;

					for (m=0 ; m<INPUTCHANNELPART3 ; ++m){
						for (n=0 ; n<KERNELX3 ; ++n){
							for (p=0 ; p<KERNELY3 ; ++p){
#pragma HLS PIPELINE off
								int x = k*STRIDEX3 - PADDINGX3 + n*DILATIONX3;
								int y = l*STRIDEY3 - PADDINGY3 + p*DILATIONX3;
								//int x_d = x / DILATIONX3;
								//int y_d = y / DILATIONY3;
								if (x>-1 && x<INPUTX3 && y>-1 && y<INPUTY3){
									P3_val += P2_tmp[i][m][x][y] * F3_tmp[j][m][n][p];
								}
							}
						}
					}

					P3_part[i][j][k][l] = P3_val;
				}
			}
		}
	}

}

void conv2d_3_out(float P3_in[BATCHFACTOR][OUTPUTCHANNELFACTOR3][INPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float bF3[FILTERFACTOR3][FILTERPART3],
		float F3_back_tmp[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float P3_act[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float P3_act_back[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float P3_nonact_back[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float F3_back[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3]){

	int i, j, k, l, m, n, p, q;
	float bF3_temp[FILTERFACTOR3][FILTERPART3];
#pragma HLS ARRAY_PARTITION variable=bF3_temp complete dim=1

	lp1:for (i=0 ; i<FILTERPART3 ; ++i){
		lp2:for (j=0 ; j<FILTERFACTOR3 ; ++j){
		#pragma HLS UNROLL
			bF3_temp[j][i] = bF3[j][i];
		}
	}

	lp3:for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		lp4:for (j=0 ; j<OUTPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			lp5:for (k=0 ; k<OUTPUTX3 ; ++k){
#pragma HLS PIPELINE off
				lp6:for (l=0 ; l<OUTPUTY3 ; ++l){
#pragma HLS PIPELINE off
					lp7:for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						lp8:for (n=0 ; n<OUTPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL
							float P3_val = 0;

							lp9:for (p=0 ; p<INPUTCHANNELFACTOR3 ; ++p){
							#pragma HLS UNROLL
								P3_val += P3_in[m][n][p][i][j][k][l];
							}

							P3_val += bF3_temp[n][j];

							P3_nonact_back[m][n][i][j][k][l] = P3_val;

							float P3_val_act = relu(P3_val);
							P3_act[m][n][i][j][k][l] = P3_val_act;
							P3_act_back[m][n][i][j][k][l] = P3_val_act;

							//cout<<"P3_act["<<m<<"]["<<n<<"]["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<P3_act[m][n][i][j][k][l]<<endl;
							//cout<<"bF3_temp["<<n<<"]["<<j<<"]: "<<bF3_temp[n][j]<<endl;

						}
					}
				}
			}
		}
	}

	lp10:for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		lp11:for (j=0 ; j<INPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			lp12:for (k=0 ; k<KERNELX3 ; ++k){
#pragma HLS PIPELINE off
				lp13:for (l=0 ; l<KERNELY3 ; ++l){
#pragma HLS PIPELINE off
					lp14:for (m=0 ; m<FILTERFACTOR3 ; ++m){
					#pragma HLS UNROLL
						lp15:for (n=0 ; n<INPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL
							F3_back[m][n][i][j][k][l] = F3_back_tmp[m][n][i][j][k][l];
						}
					}
				}
			}
		}
	}

}

void conv2d_3(float P2_act[BATCHFACTOR][OUTPUTCHANNELFACTOR2][BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2],
		float F3[FILTERFACTOR3][OUTPUTCHANNELFACTOR2][FILTERPART3][OUTPUTCHANNELPART2][KERNELX3][KERNELY3],
		float bF3[FILTERFACTOR3][FILTERPART3],
		float P3_act[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float P3_act_back[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float P3_nonact_back[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float F3_back[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3]){
#pragma HLS DATAFLOW

	int i, j, k;

	float P2_in1[BATCHFACTOR][OUTPUTCHANNELFACTOR2][FILTERFACTOR3][BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2];
#pragma HLS STREAM variable=P2_in1 type=fifo
#pragma HLS ARRAY_PARTITION variable=P2_in1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=P2_in1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=P2_in1 complete dim=3
	float F3_in1[FILTERFACTOR3][INPUTCHANNELFACTOR3][BATCHFACTOR][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3];
#pragma HLS STREAM variable=F3_in1 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=F3_in1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=F3_in1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=F3_in1 complete dim=3
	float F3_back_tmp[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3];
#pragma HLS STREAM variable=F3_back_tmp type=fifo
#pragma HLS ARRAY_PARTITION variable=F3_back_tmp complete dim=1
#pragma HLS ARRAY_PARTITION variable=F3_back_tmp complete dim=2
	float P3_in1[BATCHFACTOR][OUTPUTCHANNELFACTOR3][INPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3];
#pragma HLS STREAM variable=P3_in1 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=P3_in1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=P3_in1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=P3_in1 complete dim=3

	conv2d_3_in(P2_act, F3, P2_in1, F3_in1, F3_back_tmp);

	for (i=0 ; i<BATCHFACTOR ; ++i){
	#pragma HLS UNROLL
		for (j=0 ; j<INPUTCHANNELFACTOR3 ; ++j){
		#pragma HLS UNROLL
			for (k=0 ; k<FILTERFACTOR3 ; ++k){
			#pragma HLS UNROLL
				conv2d_3_calc(P2_in1[i][j][k], F3_in1[k][j][i], P3_in1[i][k][j]);
			}
		}
	}
	/*conv2d_3_calc(P2_in1[0][0][0], F3_in1[0][0][0], P3_in1[0][0][0]);
	conv2d_3_calc(P2_in1[1][0][0], F3_in1[0][0][1], P3_in1[1][0][0]);*/

	conv2d_3_out(P3_in1, bF3, F3_back_tmp, P3_act, P3_act_back, P3_nonact_back, F3_back);


}


/*

4: Dense

*/

void dense_4_in(float P3_act[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float W4[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float P3_act_in[BATCHFACTOR][OUTPUTCHANNELFACTOR3][LENGTHFACTOR4][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float W4_in[INPUTCHANNELFACTOR4][LENGTHFACTOR4][BATCHFACTOR][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float W4_back_tmp[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4]){

	int i, j, k, l, m, n, p;

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<OUTPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY3 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<OUTPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL
							float P3_val = P3_act[m][n][i][j][k][l];

							for (p=0 ; p<LENGTHFACTOR4 ; ++p){
							#pragma HLS UNROLL
								P3_act_in[m][n][p][i][j][k][l] = P3_val;
							}
						}
					}
				}
			}
		}
	}

	for (i=0 ; i<INPUTCHANNELPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART4 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTCHANNELFACTOR4 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL
							float W4_val = W4[m][n][i][j][k][l];
							W4_back_tmp[m][n][i][j][k][l] = W4_val;

							for (p=0 ; p<BATCHFACTOR ; ++p){
							#pragma HLS UNROLL
								W4_in[m][n][p][i][j][k][l] = W4_val;
							}

						}
					}
				}
			}
		}
	}


}

void dense_4_calc(float P3_part[BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float W4_part[INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float P4_part[BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4]){

	int i, j, k, l, m, n, p;

	float W4_tmp[INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4];
	float P3_tmp[BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3];

	for (i=0 ; i<INPUTCHANNELPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART4 ; ++l){
#pragma HLS PIPELINE off
					W4_tmp[i][j][k][l] = W4_part[i][j][k][l];
				}
			}
		}
	}

	for(i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<OUTPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY3 ; ++l){
#pragma HLS PIPELINE off
					P3_tmp[i][j][k][l] = P3_part[i][j][k][l];
				}
			}
		}
	}

	for(i=0 ; i<BATCHPART ; ++i){
		for (j=0 ; j<LENGTHPART4 ; ++j){
			for(k=0 ; k<OUTPUTX4 ; ++k){
				for (l=0 ; l<OUTPUTY4 ; ++l){
#pragma HLS PIPELINE off
					float P4_val = 0;

					for (m=0 ; m<OUTPUTCHANNELPART3 ; ++m){
						for (n=0 ; n<OUTPUTX3 ; ++n){
							for (p=0 ; p<OUTPUTY3 ; ++p){
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE off
#pragma HLS ALLOCATION operation instances=mul limit=1
								P4_val += P3_tmp[i][m][n][p] * W4_tmp[m][n][p][j];
							}
						}
					}

					P4_part[i][j][k][l] = P4_val;
				}
			}
		}
	}

}

void dense_4_out(float P4_in[BATCHFACTOR][LENGTHFACTOR4][OUTPUTCHANNELFACTOR3][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float bW4[LENGTHFACTOR4][LENGTHPART4],
		float W4_back_tmp[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float P4_act[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float P4_act_back[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float P4_nonact_back[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float W4_back[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4]){

	int i, j, k, l, m, n, p;
	float bW4_tmp[LENGTHFACTOR4][LENGTHPART4];
#pragma HLS ARRAY_PARTITION variable=bW4_tmp complete dim=1

	//cout<<"Design Checkpoint 4_3_1"<<endl;
	for (i=0 ; i<LENGTHPART4 ; ++i){
		for (j=0 ; j<LENGTHFACTOR4 ; ++j){
		#pragma HLS UNROLL
			bW4_tmp[j][i] = bW4[j][i];
		}
	}

	//cout<<"Design Checkpoint 4_3_2"<<endl;
	for (i=0 ; i<BATCHPART ; ++i){
		for (j=0 ; j<LENGTHPART4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY4 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL

							float P4_val = 0;

							for (p=0 ; p<OUTPUTCHANNELFACTOR3 ; ++p){
							#pragma HLS UNROLL
								P4_val += P4_in[m][n][p][i][j][k][l];
							}

							P4_val += bW4_tmp[n][j];

							P4_nonact_back[m][n][i][j][k][l] = P4_val;

							float P4_val_act = relu(P4_val);
							P4_act[m][n][i][j][k][l] = P4_val_act;
							P4_act_back[m][n][i][j][k][l] = P4_val_act;

							//cout<<"P4_act["<<m<<"]["<<n<<"]["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<P4_act[m][n][i][j][k][l]<<endl;

						}
					}
				}
			}
		}
	}

	//cout<<"Design Checkpoint 4_3_4"<<endl;
	for (i=0 ; i<INPUTCHANNELPART4 ; ++i){
		for (j=0 ; j<INPUTX4 ; ++j){
			for (k=0 ; k<INPUTY4 ; ++k){
				for (l=0 ; l<LENGTHPART4 ; ++l){
					for (m=0 ; m<INPUTCHANNELFACTOR4 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL
							W4_back[m][n][i][j][k][l] = W4_back_tmp[m][n][i][j][k][l];
						}
					}
				}
			}
		}
	}

}

void dense_4(float P3_act[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float W4[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float bW4[LENGTHFACTOR4][LENGTHPART4],
		float P4_act[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float P4_act_back[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float P4_nonact_back[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float W4_back[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4]){

#pragma HLS DATAFLOW

	int i, j, k, l;

	float P3_in2[BATCHFACTOR][OUTPUTCHANNELFACTOR3][LENGTHFACTOR4][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3];
#pragma HLS STREAM variable=P3_in2 type=fifo
#pragma HLS ARRAY_PARTITION variable=P3_in2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=P3_in2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=P3_in2 complete dim=3
	float W4_in1[INPUTCHANNELFACTOR4][LENGTHFACTOR4][BATCHFACTOR][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4];
#pragma HLS STREAM variable=W4_in1 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=W4_in1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=W4_in1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=W4_in1 complete dim=3
	float P4_in1[BATCHFACTOR][LENGTHFACTOR4][OUTPUTCHANNELFACTOR3][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4];
#pragma HLS STREAM variable=P4_in1 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=P4_in1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=P4_in1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=P4_in1 complete dim=3
	float W4_back_tmp[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4];
#pragma HLS STREAM variable=W4_back_tmp type=fifo
#pragma HLS ARRAY_PARTITION variable=W4_back_tmp complete dim=1
#pragma HLS ARRAY_PARTITION variable=W4_back_tmp complete dim=2


	//cout<<"Design Checkpoint 4_1"<<endl;
	dense_4_in(P3_act, W4, P3_in2, W4_in1, W4_back_tmp);

	//cout<<"Design Checkpoint 4_2"<<endl;
	for (i=0 ; i<BATCHFACTOR ; ++i){
	#pragma HLS UNROLL
		for (j=0 ; j<INPUTCHANNELFACTOR4 ; ++j){
		#pragma HLS UNROLL
			for (k=0 ; k<LENGTHFACTOR4 ; ++k){
			#pragma HLS UNROLL
				dense_4_calc(P3_in2[i][j][k], W4_in1[j][k][i], P4_in1[i][k][j]);
			}
		}
	}
	/*dense_4_calc(P3_in2[0][0][0], W4_in1[0][0][0], P4_in1[0][0][0]);
	dense_4_calc(P3_in2[1][0][0], W4_in1[0][0][1], P4_in1[1][0][0]);*/

	//cout<<"Design Checkpoint 4_3"<<endl;
	dense_4_out(P4_in1, bW4, W4_back_tmp, P4_act, P4_act_back, P4_nonact_back, W4_back);


}



/*

5: Dense

*/

void dense_5_in(float P4_act[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float W5[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float P4_act_in[BATCHFACTOR][LENGTHFACTOR4][LENGTHFACTOR5][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float W5_in[INPUTLENGTHFACTOR5][LENGTHFACTOR5][BATCHFACTOR][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float W5_back_tmp[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5]){

	int i, j, k, l, m, n, p;

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY4 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL
							float P4_val = P4_act[m][n][i][j][k][l];

							for (p=0 ; p<LENGTHFACTOR5 ; ++p){
							#pragma HLS UNROLL
								P4_act_in[m][n][p][i][j][k][l] = P4_val;
							}
						}
					}
				}
			}
		}
	}

	for (i=0 ; i<INPUTLENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTLENGTHFACTOR5 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL
							float W5_val = W5[m][n][i][j][k][l];
							W5_back_tmp[m][n][i][j][k][l] = W5_val;

							for (p=0 ; p<BATCHFACTOR ; ++p){
							#pragma HLS UNROLL
								W5_in[m][n][p][i][j][k][l] = W5_val;
							}

						}
					}
				}
			}
		}
	}


}

void dense_5_calc(float P4_part[BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float W5_part[INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float P5_part[BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5]){

	int i, j, k, l, m, n, p;

	float W5_tmp[INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5];
	float P4_tmp[BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4];

	for (i=0 ; i<INPUTLENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART5 ; ++l){
#pragma HLS PIPELINE off
					W5_tmp[i][j][k][l] = W5_part[i][j][k][l];
				}
			}
		}
	}

	for(i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX4 ; ++k){
	#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY4 ; ++l){
		#pragma HLS PIPELINE off
					P4_tmp[i][j][k][l] = P4_part[i][j][k][l];
				}
			}
		}
	}

	for(i=0 ; i<BATCHPART ; ++i){
		for (j=0 ; j<LENGTHPART5 ; ++j){
#pragma HLS PIPELINE off
			for(k=0 ; k<OUTPUTX5 ; ++k){
				for (l=0 ; l<OUTPUTY5 ; ++l){
					float P5_val = 0;

					for (m=0 ; m<INPUTLENGTHPART5 ; ++m){
						for (n=0 ; n<INPUTX5 ; ++n){
							for (p=0 ; p<INPUTY5 ; ++p){
#pragma HLS LOOP_FLATTEN off
#pragma HLS ALLOCATION operation instances=fadd limit=1
#pragma HLS ALLOCATION operation instances=fmul limit=1
								P5_val += P4_tmp[i][m][n][p] * W5_tmp[m][n][p][j];
							}
						}
					}

					P5_part[i][j][k][l] = P5_val;

				}
			}
		}
	}

}

void dense_5_out(float P5_in[BATCHFACTOR][LENGTHFACTOR5][LENGTHFACTOR4][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float bW5[LENGTHFACTOR5][LENGTHPART5],
		float W5_back_tmp[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float POUT[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float P5_act_back[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float W5_back[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5]){

	int i, j, k, l, m, n, p;
	float P5_exp_sum[BATCHFACTOR][BATCHPART];
#pragma HLS ARRAY_PARTITION variable=P5_exp_sum complete dim=1
	float P5_tmp[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5];
#pragma HLS ARRAY_PARTITION variable=P5_tmp complete dim=1
#pragma HLS ARRAY_PARTITION variable=P5_tmp complete dim=2
	float bW5_tmp[LENGTHFACTOR5][LENGTHPART5];
#pragma HLS ARRAY_PARTITION variable=bW5_tmp complete dim=1

	//cout<<"Design Checkpoint 4_3_1"<<endl;
	for (i=0 ; i<LENGTHPART5 ; ++i){
		for (j=0 ; j<LENGTHFACTOR5 ; ++j){
		#pragma HLS UNROLL
			bW5_tmp[j][i] = bW5[j][i];
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
		for (j=0 ; j<BATCHFACTOR ; ++j){
		#pragma HLS UNROLL
			P5_exp_sum[j][i] = 0;
		}
	}

	//cout<<"Design Checkpoint 4_3_2"<<endl;
	for (i=0 ; i<BATCHPART ; ++i){
		for (j=0 ; j<LENGTHPART5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL
							float P5_val = 0;


							for (p=0 ; p<LENGTHFACTOR4 ; ++p){
							#pragma HLS UNROLL
								P5_val += P5_in[m][n][p][i][j][k][l];
							}


							P5_val += bW5_tmp[n][j];


							float P5_val_exp = exp_func(P5_val);

							P5_exp_sum[m][i] += P5_val_exp;
							P5_tmp[m][n][i][j][k][l] = P5_val;

						}
					}
				}
			}
		}
	}

	//cout<<"Design Checkpoint 4_3_3"<<endl;
	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL
							//cout<<"P4_exp_sum["<<k<<"]["<<i<<"]: "<<P4_exp_sum[k][i]<<endl;
							float P5_val = exp_func(P5_tmp[m][n][i][j][k][l]) / P5_exp_sum[m][i];
							POUT[m][n][i][j][k][l] = P5_val;
							//cout<<k<<", "<<l<<", "<<i<<", "<<j<<", "<<P4_tmp[k][l][i][j]<<" | "<<P4_val<<endl;
							P5_act_back[m][n][i][j][k][l] = P5_val;

						}
					}
				}
			}
		}

	}

	//cout<<"Design Checkpoint 4_3_4"<<endl;
	for (i=0 ; i<INPUTLENGTHPART5 ; ++i){
		for (j=0 ; j<INPUTX5 ; ++j){
			for (k=0 ; k<INPUTY5 ; ++k){
				for (l=0 ; l<LENGTHPART5 ; ++l){
					for (m=0 ; m<INPUTLENGTHFACTOR5 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL
							W5_back[m][n][i][j][k][l] = W5_back_tmp[m][n][i][j][k][l];
						}
					}
				}
			}
		}
	}

}

void dense_5(float P4_act[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float W5[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float bW5[LENGTHFACTOR5][LENGTHPART5],
		float POUT[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float P5_act_back[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float W5_back[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5]){

#pragma HLS DATAFLOW

	int i, j, k, l;

	float P4_in2[BATCHFACTOR][LENGTHFACTOR4][LENGTHFACTOR5][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4];
#pragma HLS STREAM variable=P4_in2 type=fifo
#pragma HLS ARRAY_PARTITION variable=P4_in2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=P4_in2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=P4_in2 complete dim=3
	float W5_in1[INPUTLENGTHFACTOR5][LENGTHFACTOR5][BATCHFACTOR][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5];
#pragma HLS STREAM variable=W5_in1 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=W5_in1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=W5_in1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=W5_in1 complete dim=3
	float P5_in[BATCHFACTOR][LENGTHFACTOR5][LENGTHFACTOR4][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5];
#pragma HLS STREAM variable=P5_in type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=P5_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=P5_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=P5_in complete dim=3
	float W5_back_tmp[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5];
#pragma HLS STREAM variable=W5_back_tmp type=fifo
#pragma HLS ARRAY_PARTITION variable=W5_back_tmp complete dim=1
#pragma HLS ARRAY_PARTITION variable=W5_back_tmp complete dim=2


	//cout<<"Design Checkpoint 4_1"<<endl;
	dense_5_in(P4_act, W5, P4_in2, W5_in1, W5_back_tmp);


	//cout<<"Design Checkpoint 4_2"<<endl;
	for (i=0 ; i<BATCHFACTOR ; ++i){
	#pragma HLS UNROLL
		for (j=0 ; j<LENGTHFACTOR4 ; ++j){
		#pragma HLS UNROLL
			for (k=0 ; k<LENGTHFACTOR5 ; ++k){
			#pragma HLS UNROLL
				dense_5_calc(P4_in2[i][j][k], W5_in1[j][k][i], P5_in[i][k][j]);
			}
		}
	}
	/*dense_5_calc(P4_in2[0][0][0], W5_in1[0][0][0], P5_in[0][0][0]);
	dense_5_calc(P4_in2[1][0][0], W5_in1[0][0][1], P5_in[1][0][0]);*/

	//cout<<"Design Checkpoint 4_3"<<endl;
	dense_5_out(P5_in, bW5, W5_back_tmp, POUT, P5_act_back, W5_back);


}

/*

5B: dL5

*/

void dL5_func(float P5_act_back[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		int y[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float dL5[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5]){

	int i, j, k, l, m, n;

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL
#pragma HLS ALLOCATION operation instances=fsub limit=1
							dL5[m][n][i][j][k][l] = P5_act_back[m][n][i][j][k][l] - y[m][n][i][j][k][l];
							//cout<<"P5_act_back["<<m<<"]["<<n<<"]["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<P5_act_back[m][n][i][j][k][l]<<endl;

						}
					}
				}
			}
		}
	}
	//cout<<endl;
}


/*

5B: dW5

*/

void dW5_func_in(float dL5[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float P4_act_back[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float dL5_in[BATCHFACTOR][LENGTHFACTOR5][LENGTHFACTOR4][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float P4_in[BATCHFACTOR][LENGTHFACTOR4][LENGTHFACTOR5][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float dL5_back_tmp[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5]){


	int i, j, k, l, m, n, p;

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL
							float dL5_val = dL5[m][n][i][j][k][l];
							dL5_back_tmp[m][n][i][j][k][l] = dL5_val;
							for (p=0 ; p<LENGTHFACTOR4 ; ++p){
							#pragma HLS UNROLL
								dL5_in[m][n][p][i][j][k][l] = dL5_val;
							}
						}
					}
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY4 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL
							float P4_val = P4_act_back[m][n][i][j][k][l];
							for (p=0 ; p<CLASSFACTOR ; ++p){
							#pragma HLS UNROLL
								P4_in[m][n][p][i][j][k][l] = P4_val;
							}
						}
					}
				}
			}
		}
	}

}

void dW5_func_calc(float dL5_part[BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float P4_part[BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float dW5_part[INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5]){

	int i, j, k, l, m, n, p;
	float dL5_tmp[BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5];
	float P4_tmp[BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4];

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY5 ; ++l){
#pragma HLS PIPELINE off
					dL5_tmp[i][j][k][l] = dL5_part[i][j][k][l];
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY4 ; ++l){
#pragma HLS PIPELINE off
					P4_tmp[i][j][k][l] = P4_part[i][j][k][l];
				}
			}
		}
	}

	for (i=0 ; i<INPUTLENGTHPART5 ; ++i){
		for (j=0 ; j<INPUTX5 ; ++j){
			for (k=0 ; k<INPUTY5 ; ++k){
				for (l=0 ; l<LENGTHPART5 ; ++l){
#pragma HLS PIPELINE off
					float dW5_val = 0;

					for (m=0 ; m<BATCHPART ; ++m){
						for (n=0 ; n<OUTPUTX5 ; ++n){
							for (p=0 ; p<OUTPUTY5 ; ++p){
								dW5_val += dL5_tmp[m][l][n][p] * P4_tmp[m][i][j][k];
							}
						}
					}

					dW5_part[i][j][k][l] = dW5_val;
				}
			}
		}
	}

}

void dW5_func_out(float dW5_in[LENGTHFACTOR5][LENGTHFACTOR5][BATCHFACTOR][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float dL5_back_tmp[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float dW5[LENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float dbW5[LENGTHFACTOR5][LENGTHPART5],
		float dL5_back[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5]){

	int i, j, k, l, m, n, p;

	float dbW5_tmp[LENGTHFACTOR5][LENGTHPART5];
#pragma HLS ARRAY_PARTITION variable=dbW5_tmp complete dim=1

	for (i=0 ; i<INPUTLENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTLENGTHFACTOR5 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL
							float dW5_val = 0;
							for (p=0 ; p<BATCHFACTOR ; ++p){
							#pragma HLS UNROLL
								dW5_val += dW5_in[m][n][p][i][j][k][l];
							}
							dW5[m][n][i][j][k][l] = dW5_val;

							//cout<<"dW4["<<m<<"]["<<n<<"]["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<dW4[m][n][i][j][k][l]<<endl;

							//NanControl
							/*if (dW4[m][n][i][j][k][l] != dW4[m][n][i][j][k][l]) {
								cout<<"dW4_func_out"<<endl;
								return;
							}*/
						}
					}
				}
			}
		}
	}

	for (j=0 ; j<LENGTHPART5 ; ++j){
#pragma HLS PIPELINE off
		for (l=0 ; l<LENGTHFACTOR5 ; ++l){
		#pragma HLS UNROLL
			dbW5_tmp[l][j] = 0;
		}
	}



	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY5 ; ++l){
		#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL
							float dL5_val = dL5_back_tmp[m][n][i][j][k][l];

							dL5_back[m][n][i][j][k][l] = dL5_val;
							//cout<<"dL4_back["<<k<<"]["<<l<<"]["<<i<<"]["<<j<<"]: "<<dL4_back[k][l][i][j]<<endl;
							/*if (i==0 && k==0 && dbW4_tmp[l][j]!=0){
								dbW4_tmp[l][j] = 0;
							}*/
							//dbW4_tmp[l][j] += dL4_val;
							dbW5_tmp[n][j] += dL5_val;
						}
					}
				}
			}
		}
	}

	for (i=0 ; i<LENGTHPART5 ; ++i){
		for (j=0 ; j<LENGTHFACTOR5 ; ++j){
		#pragma HLS UNROLL
			dbW5[j][i] = dbW5_tmp[j][i];

			//cout<<"dbW4["<<j<<"]["<<i<<"]: "<<dbW4[j][i]<<endl;

		}

	}


}

void dW5_func(float dL5[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float P4_act_back[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float dW5[LENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float dbW5[LENGTHFACTOR5][LENGTHPART5],
		float dL5_back[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5]){

#pragma HLS DATAFLOW

	int i, j, k;

	float dL5_in1[BATCHFACTOR][LENGTHFACTOR5][LENGTHFACTOR4][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5];
#pragma HLS STREAM variable=dL5_in1 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dL5_in1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL5_in1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=dL5_in1 complete dim=3
	float P4_in3[BATCHFACTOR][LENGTHFACTOR4][LENGTHFACTOR5][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4];
#pragma HLS STREAM variable=P4_in3 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=P4_in3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=P4_in3 complete dim=2
#pragma HLS ARRAY_PARTITION variable=P4_in3 complete dim=3
	float dL5_back_tmp[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5];
#pragma HLS STREAM variable=dL5_back_tmp type=fifo
#pragma HLS ARRAY_PARTITION variable=dL5_back_tmp complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL5_back_tmp complete dim=2
	float dW5_in[LENGTHFACTOR4][LENGTHFACTOR5][BATCHFACTOR][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5];
#pragma HLS STREAM variable=dW5_in type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dW5_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=dW5_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=dW5_in complete dim=3

	dW5_func_in(dL5, P4_act_back, dL5_in1, P4_in3, dL5_back_tmp);
	for (i=0 ; i<BATCHFACTOR ; ++i){
	#pragma HLS UNROLL
		for (j=0 ; j<LENGTHFACTOR5 ; ++j){
		#pragma HLS UNROLL
			for (k=0 ; k<LENGTHFACTOR4 ; ++k){
			#pragma HLS UNROLL
				dW5_func_calc(dL5_in1[i][j][k], P4_in3[i][k][j], dW5_in[k][j][i]);
			}
		}
	}
	/*dW5_func_calc(dL5_in1[0][0][0], P4_in3[0][0][0], dW5_in[0][0][0]);
	dW5_func_calc(dL5_in1[1][0][0], P4_in3[1][0][0], dW5_in[0][0][1]);*/
	dW5_func_out(dW5_in, dL5_back_tmp, dW5, dbW5, dL5_back);

}



/*

5B: dL4

*/


void dL4_func_in(float dL5_back[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float W5_back[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float dL5_in[BATCHFACTOR][LENGTHFACTOR5][LENGTHFACTOR4][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float W5_in[INPUTLENGTHFACTOR5][LENGTHFACTOR5][BATCHFACTOR][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5]){

	int i, j, k, l, m, n, p;

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL
							float dL5_val = dL5_back[m][n][i][j][k][l];

							for (p=0 ; p<LENGTHFACTOR4 ; ++p){
							#pragma HLS UNROLL
								dL5_in[m][n][p][i][j][k][l] = dL5_val;
							}
						}
					}
				}
			}
		}
	}

	for (i=0 ; i<INPUTLENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTLENGTHFACTOR5 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL
							float W5_val = W5_back[m][n][i][j][k][l];

							for (p=0 ; p<BATCHFACTOR ; ++p){
							#pragma HLS UNROLL
								W5_in[m][n][p][i][j][k][l] = W5_val;
							}
						}
					}
				}
			}

		}
	}

}

void dL4_func_calc(float dL5_in[BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float W5_in[INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float dL4_in[BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4]){

	int i, j, k, l, m, n, p;

	float dL5_tmp[BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5];
	float W5_tmp[INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5];

	for (i=0 ; i<INPUTLENGTHPART5 ; ++i){
		for (j=0 ; j<INPUTX5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART5 ; ++l){
#pragma HLS PIPELINE off
					W5_tmp[i][j][k][l] = W5_in[i][j][k][l];
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY5 ; ++l){
#pragma HLS PIPELINE off
						dL5_tmp[i][j][k][l] = dL5_in[i][j][k][l];
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
		for (j=0 ; j<LENGTHPART4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX4 ; ++k){
				for (l=0 ; l<OUTPUTY4 ; ++l){

					float dL4_val = 0;

					for (m=0 ; m<LENGTHPART5 ; ++m){
						for (n=0 ; n<OUTPUTX5 ; ++n){
							for (p=0 ; p<OUTPUTY5 ; ++p){
#pragma HLS ALLOCATION operation instances=fadd limit=1
#pragma HLS ALLOCATION operation instances=fmul limit=1
#pragma HLS LOOP_FLATTEN off
								dL4_val += W5_tmp[j][k][l][m] * dL5_tmp[i][m][n][p];
							}
						}
					}
					dL4_in[i][j][k][l] = dL4_val;
				}
			}
		}
	}

}

void dL4_func_out(float dL4_in[BATCHFACTOR][LENGTHFACTOR4][LENGTHFACTOR5][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float P4_nonact_back[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float dL4[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4]){

	int i, j, k, l, m, n, p;

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY4 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL
#pragma HLS ALLOCATION operation instances=fadd limit=1
#pragma HLS ALLOCATION operation instances=fmul limit=1
							float dL4_val = 0;

							for (p=0 ; p<LENGTHFACTOR5 ; ++p){
							#pragma HLS UNROLL
								dL4_val += dL4_in[m][n][p][i][j][k][l];
							}

							dL4[m][n][i][j][k][l] = dL4_val * relu_drv(P4_nonact_back[m][n][i][j][k][l]);
							//cout<<"dL3["<<m<<"]["<<n<<"]["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<dL3[m][n][i][j][k][l]<<endl;

							//cout<<"dL3["<<m<<"]["<<n<<"]["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<dL3_val<<" * "<<relu_drv(P3_nonact_back[m][n][i][j][k][l])<<" ("<<P3_nonact_back[m][n][i][j][k][l]<<")"<<endl;

							//cout<<"dL3["<<m<<"]["<<n<<"]["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<dL3[m][n][i][j][k][l]<<endl;

							//NanControl
							/*if (dL3[m][n][i][j][k][l] != dL3[m][n][i][j][k][l]) {
								cout<<"dL3_func_out"<<endl;
								return;
							}*/

						}
					}
				}
			}
		}
	}

}

void dL4_func(float dL5_back[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float P4_nonact_back[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float W5_back[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float dL4[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4]){

#pragma HLS DATAFLOW

	int i, j, k;

	float dL5_in2[BATCHFACTOR][LENGTHFACTOR5][LENGTHFACTOR4][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5];
#pragma HLS STREAM variable=dL5_in2 type=fifo
#pragma HLS ARRAY_PARTITION variable=dL5_in2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL5_in2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=dL5_in2 complete dim=3
	float W5_in2[INPUTLENGTHFACTOR5][LENGTHFACTOR5][BATCHFACTOR][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5];
#pragma HLS STREAM variable=W5_in2 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=W5_in2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=W5_in2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=W5_in2 complete dim=3
	float dL4_in1[BATCHFACTOR][LENGTHFACTOR4][LENGTHFACTOR5][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4];
#pragma HLS STREAM variable=dL4_in1 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dL4_in1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL4_in1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=dL4_in1 complete dim=3

	dL4_func_in(dL5_back, W5_back, dL5_in2, W5_in2);
	for (i=0 ; i<BATCHFACTOR ; ++i){
	#pragma HLS UNROLL
		for (j=0 ; j<LENGTHFACTOR5 ; ++j){
		#pragma HLS UNROLL
			for (k=0 ; k<LENGTHFACTOR4 ; ++k){
			#pragma HLS UNROLL
				dL4_func_calc(dL5_in2[i][j][k], W5_in2[k][j][i], dL4_in1[i][k][j]);
			}
		}
	}
	/*dL4_func_calc(dL5_in2[0][0][0], W5_in2[0][0][0], dL4_in1[0][0][0]);
	dL4_func_calc(dL5_in2[1][0][0], W5_in2[0][0][1], dL4_in1[1][0][0]);*/
	dL4_func_out(dL4_in1, P4_nonact_back, dL4);
}



/*

4B: dW4

*/

void dW4_func_in(float dL4[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float P3_act_back[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float dL4_in[BATCHFACTOR][LENGTHFACTOR4][OUTPUTCHANNELFACTOR3][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float P3_in[BATCHFACTOR][OUTPUTCHANNELFACTOR3][LENGTHFACTOR4][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float dL4_back_tmp[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4]){


	int i, j, k, l, m, n, p;

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY4 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL
							float dL4_val = dL4[m][n][i][j][k][l];
							dL4_back_tmp[m][n][i][j][k][l] = dL4_val;
							for (p=0 ; p<OUTPUTCHANNELFACTOR3 ; ++p){
							#pragma HLS UNROLL
								dL4_in[m][n][p][i][j][k][l] = dL4_val;
							}
						}
					}
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<OUTPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY3 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<OUTPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL
							float P3_val = P3_act_back[m][n][i][j][k][l];
							for (p=0 ; p<LENGTHFACTOR4 ; ++p){
							#pragma HLS UNROLL
								P3_in[m][n][p][i][j][k][l] = P3_val;
							}
						}
					}
				}
			}
		}
	}

}

void dW4_func_calc(float dL4_part[BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float P3_part[BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float dW4_part[INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4]){

	int i, j, k, l, m, n, p;
	float dL4_tmp[BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4];
	float P3_tmp[BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3];

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY4 ; ++l){
#pragma HLS PIPELINE off
					dL4_tmp[i][j][k][l] = dL4_part[i][j][k][l];
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<OUTPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY3 ; ++l){
#pragma HLS PIPELINE off
					P3_tmp[i][j][k][l] = P3_part[i][j][k][l];
				}
			}
		}
	}

	for (i=0 ; i<INPUTCHANNELPART4 ; ++i){
		for (j=0 ; j<INPUTX4 ; ++j){
			for (k=0 ; k<INPUTY4 ; ++k){
				for (l=0 ; l<LENGTHPART4 ; ++l){
#pragma HLS PIPELINE off
					float dW4_val = 0;

					for (m=0 ; m<BATCHPART ; ++m){
						for (n=0 ; n<OUTPUTX4 ; ++n){
							for (p=0 ; p<OUTPUTY4 ; ++p){
								dW4_val += dL4_tmp[m][l][n][p] * P3_tmp[m][i][j][k];
							}
						}
					}

					dW4_part[i][j][k][l] = dW4_val;
				}
			}
		}
	}

}

void dW4_func_out(float dW4_in[INPUTCHANNELFACTOR4][LENGTHFACTOR4][BATCHFACTOR][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float dL4_back_tmp[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float dW4[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float dbW4[LENGTHFACTOR4][LENGTHPART4],
		float dL4_back[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4]){

	int i, j, k, l, m, n, p;

	float dbW4_tmp[LENGTHFACTOR4][LENGTHPART4];
#pragma HLS ARRAY_PARTITION variable=dbW4_tmp complete dim=1

	for (i=0 ; i<INPUTCHANNELPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART4 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTCHANNELFACTOR4 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL
							float dW4_val = 0;
							for (p=0 ; p<BATCHFACTOR ; ++p){
							#pragma HLS UNROLL
								dW4_val += dW4_in[m][n][p][i][j][k][l];
							}
							dW4[m][n][i][j][k][l] = dW4_val;

							//cout<<"dW4["<<m<<"]["<<n<<"]["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<dW4[m][n][i][j][k][l]<<endl;

							//NanControl
							/*if (dW4[m][n][i][j][k][l] != dW4[m][n][i][j][k][l]) {
								cout<<"dW4_func_out"<<endl;
								return;
							}*/
						}
					}
				}
			}
		}
	}

	for (j=0 ; j<LENGTHPART4 ; ++j){
#pragma HLS PIPELINE off
		for (l=0 ; l<LENGTHFACTOR4 ; ++l){
		#pragma HLS UNROLL
			dbW4_tmp[l][j] = 0;
		}
	}



	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY4 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL
							float dL4_val = dL4_back_tmp[m][n][i][j][k][l];

							dL4_back[m][n][i][j][k][l] = dL4_val;
							//cout<<"dL4_back["<<k<<"]["<<l<<"]["<<i<<"]["<<j<<"]: "<<dL4_back[k][l][i][j]<<endl;
							/*if (i==0 && k==0 && dbW4_tmp[l][j]!=0){
								dbW4_tmp[l][j] = 0;
							}*/
							//dbW4_tmp[l][j] += dL4_val;
							dbW4_tmp[n][j] += dL4_val;
						}
					}
				}
			}
		}
	}

	for (i=0 ; i<LENGTHPART4 ; ++i){
		for (j=0 ; j<LENGTHFACTOR4 ; ++j){
		#pragma HLS UNROLL
			dbW4[j][i] = dbW4_tmp[j][i];

			//cout<<"dbW4["<<j<<"]["<<i<<"]: "<<dbW4[j][i]<<endl;

		}

	}


}

void dW4_func(float dL4[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float P3_act_back[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float dW4[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float dbW4[LENGTHFACTOR4][LENGTHPART4],
		float dL4_back[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4]){

#pragma HLS DATAFLOW

	int i, j, k;

	float dL4_in2[BATCHFACTOR][LENGTHFACTOR4][OUTPUTCHANNELFACTOR3][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4];
#pragma HLS STREAM variable=dL4_in2 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dL4_in2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL4_in2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=dL4_in2 complete dim=3
	float P3_in3[BATCHFACTOR][OUTPUTCHANNELFACTOR3][LENGTHFACTOR4][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3];
#pragma HLS STREAM variable=P3_in3 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=P3_in3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=P3_in3 complete dim=2
#pragma HLS ARRAY_PARTITION variable=P3_in3 complete dim=3
	float dL4_back_tmp[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4];
#pragma HLS STREAM variable=dL4_back_tmp type=fifo
#pragma HLS ARRAY_PARTITION variable=dL4_back_tmp complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL4_back_tmp complete dim=2
	float dW4_in[INPUTCHANNELFACTOR4][LENGTHFACTOR4][BATCHFACTOR][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4];
#pragma HLS STREAM variable=dW4_in type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dW4_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=dW4_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=dW4_in complete dim=3

	dW4_func_in(dL4, P3_act_back, dL4_in2, P3_in3, dL4_back_tmp);
	for (i=0 ; i<BATCHFACTOR ; ++i){
	#pragma HLS UNROLL
		for (j=0 ; j<LENGTHFACTOR4 ; ++j){
		#pragma HLS UNROLL
			for (k=0 ; k<INPUTCHANNELFACTOR4 ; ++k){
			#pragma HLS UNROLL
				dW4_func_calc(dL4_in2[i][j][k], P3_in3[i][k][j], dW4_in[k][j][i]);
			}
		}
	}
	/*dW4_func_calc(dL4_in2[0][0][0], P3_in3[0][0][0], dW4_in[0][0][0]);
	dW4_func_calc(dL4_in2[1][0][0], P3_in3[1][0][0], dW4_in[0][0][1]);*/
	dW4_func_out(dW4_in, dL4_back_tmp, dW4, dbW4, dL4_back);

}





/*

3B: dL3

*/

void dL3_func_in(float dL4_back[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float W4_back[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float dL4_in[BATCHFACTOR][LENGTHFACTOR4][OUTPUTCHANNELFACTOR3][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float W4_in[INPUTCHANNELFACTOR4][LENGTHFACTOR4][BATCHFACTOR][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4]){

	int i, j, k, l, m, n, p;

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY4 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL
							float dL4_val = dL4_back[m][n][i][j][k][l];

							for (p=0 ; p<OUTPUTCHANNELFACTOR3 ; ++p){
							#pragma HLS UNROLL
								dL4_in[m][n][p][i][j][k][l] = dL4_val;
							}
						}
					}
				}
			}
		}
	}

	for (i=0 ; i<INPUTCHANNELPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART4 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTCHANNELFACTOR4 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL
							float W4_val = W4_back[m][n][i][j][k][l];

							for (p=0 ; p<BATCHFACTOR ; ++p){
							#pragma HLS UNROLL
								W4_in[m][n][p][i][j][k][l] = W4_val;
							}
						}
					}
				}
			}

		}
	}

}

void dL3_func_calc(float dL4_in[BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float W4_in[INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float dL3_in[BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3]){

	int i, j, k, l, m, n, p;

	float dL4_tmp[BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4];
	float W4_tmp[INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4];

	for (i=0 ; i<INPUTCHANNELPART4 ; ++i){
		for (j=0 ; j<INPUTX4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART4 ; ++l){
#pragma HLS PIPELINE off
					W4_tmp[i][j][k][l] = W4_in[i][j][k][l];
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY4 ; ++l){
#pragma HLS PIPELINE off
					dL4_tmp[i][j][k][l] = dL4_in[i][j][k][l];
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<OUTPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY3 ; ++l){
#pragma HLS PIPELINE off
					float dL3_val = 0;

					for (m=0 ; m<LENGTHPART4 ; ++m){
#pragma HLS PIPELINE off
						for (n=0 ; n<OUTPUTX4 ; ++n){
#pragma HLS PIPELINE off
							for (p=0 ; p<OUTPUTY4 ; ++p){
#pragma HLS ALLOCATION operation instances=fadd limit=1
#pragma HLS ALLOCATION operation instances=fmul limit=1
#pragma HLS LOOP_FLATTEN off

								dL3_val += W4_tmp[j][k][l][m] * dL4_tmp[i][m][n][p];
							}
						}
					}

					dL3_in[i][j][k][l] = dL3_val;
				}
			}
		}
	}

}

void dL3_func_out(float dL3_in[BATCHFACTOR][OUTPUTCHANNELFACTOR3][LENGTHFACTOR4][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float P3_nonact_back[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float dL3[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3]){

	int i, j, k, l, m, n, p;

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<OUTPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY3 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<OUTPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL
#pragma HLS ALLOCATION operation instances=fadd limit=1
#pragma HLS ALLOCATION operation instances=fmul limit=1
							float dL3_val = 0;

							for (p=0 ; p<LENGTHFACTOR4 ; ++p){
							#pragma HLS UNROLL
								dL3_val += dL3_in[m][n][p][i][j][k][l];
							}

							dL3[m][n][i][j][k][l] = dL3_val * relu_drv(P3_nonact_back[m][n][i][j][k][l]);
							//cout<<"dL3["<<m<<"]["<<n<<"]["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<dL3[m][n][i][j][k][l]<<endl;

							//cout<<"dL3["<<m<<"]["<<n<<"]["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<dL3_val<<" * "<<relu_drv(P3_nonact_back[m][n][i][j][k][l])<<" ("<<P3_nonact_back[m][n][i][j][k][l]<<")"<<endl;

							//cout<<"dL3["<<m<<"]["<<n<<"]["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<dL3[m][n][i][j][k][l]<<endl;

							//NanControl
							/*if (dL3[m][n][i][j][k][l] != dL3[m][n][i][j][k][l]) {
								cout<<"dL3_func_out"<<endl;
								return;
							}*/

						}
					}
				}
			}
		}
	}

}

void dL3_func(float dL4_back[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4],
		float P3_nonact_back[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float W4_back[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float dL3[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3]){

#pragma HLS DATAFLOW

	int i, j, k;

	float dL4_in3[BATCHFACTOR][LENGTHFACTOR4][OUTPUTCHANNELFACTOR3][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4];
#pragma HLS STREAM variable=dL4_in3 type=fifo
#pragma HLS ARRAY_PARTITION variable=dL4_in3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL4_in3 complete dim=2
#pragma HLS ARRAY_PARTITION variable=dL4_in3 complete dim=3
	float W4_in2[INPUTCHANNELFACTOR4][LENGTHFACTOR4][BATCHFACTOR][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4];
#pragma HLS STREAM variable=W4_in2 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=W4_in2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=W4_in2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=W4_in2 complete dim=3
	float dL3_in1[BATCHFACTOR][OUTPUTCHANNELFACTOR3][LENGTHFACTOR4][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3];
#pragma HLS STREAM variable=dL3_in1 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dL3_in1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL3_in1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=dL3_in1 complete dim=3

	dL3_func_in(dL4_back, W4_back, dL4_in3, W4_in2);
	for (i=0 ; i<BATCHFACTOR ; ++i){
	#pragma HLS UNROLL
		for (j=0 ; j<LENGTHFACTOR4 ; ++j){
		#pragma HLS UNROLL
			for (k=0 ; k<INPUTCHANNELFACTOR4 ; ++k){
			#pragma HLS UNROLL
				dL3_func_calc(dL4_in3[i][j][k], W4_in2[k][j][i], dL3_in1[i][k][j]);
			}
		}
	}
	/*dL3_func_calc(dL4_in3[0][0][0], W4_in2[0][0][0], dL3_in1[0][0][0]);
	dL3_func_calc(dL4_in3[1][0][0], W4_in2[0][0][1], dL3_in1[1][0][0]);*/
	dL3_func_out(dL3_in1, P3_nonact_back, dL3);
}

/*

3B: dF3

*/

void dF3_func_in(float dL3[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float P2_act_back[BATCHFACTOR][OUTPUTCHANNELFACTOR2][BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2],
		float dL3_in[BATCHFACTOR][OUTPUTCHANNELFACTOR3][INPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float P2_in[BATCHFACTOR][OUTPUTCHANNELFACTOR2][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2],
		float dL3_back_tmp[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3]){

	int i, j, k, l, m, n, p, q, r, s;

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<OUTPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY3 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<OUTPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL
							float dL3_val = dL3[m][n][i][j][k][l];
							dL3_back_tmp[m][n][i][j][k][l] = dL3_val;
							//cout<<dL3_val<<endl;

							for (p=0 ; p<INPUTCHANNELFACTOR3 ; ++p){
							#pragma HLS UNROLL
								dL3_in[m][n][p][i][j][k][l] = dL3_val;
							}
						}
					}
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<OUTPUTCHANNELPART2 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX2 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY2 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<OUTPUTCHANNELFACTOR2 ; ++n){
						#pragma HLS UNROLL
							float dP2_val = P2_act_back[m][n][i][j][k][l];
							for (p=0 ; p<OUTPUTCHANNELFACTOR3 ; ++p){
							#pragma HLS UNROLL
								P2_in[m][n][p][i][j][k][l] = dP2_val;
							}
						}
					}
				}
			}
		}
	}

}

void dF3_func_calc(float dL3_part[BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float P2_part[BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2],
		float dF3_part[FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3]){

	int i, j, k, l, m, n, p, q;
	float dL3_tmp[BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3];
	float P2_tmp[BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2];


	for (i=0 ; i<BATCHPART ; ++i){
		for (j=0 ; j<OUTPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX3 ; ++k){
				for (l=0 ; l<OUTPUTY3 ; ++l){
					dL3_tmp[i][j][k][l] = dL3_part[i][j][k][l];
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
		for (j=0 ; j<OUTPUTCHANNELPART2 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX2 ; ++k){
				for (l=0 ; l<OUTPUTY2 ; ++l){
					P2_tmp[i][j][k][l] = P2_part[i][j][k][l];
				}
			}
		}
	}

	for (i=0 ; i<FILTERPART3 ; ++i){
		for (j=0 ; j<INPUTCHANNELPART3 ; ++j){
			for (k=0 ; k<KERNELX3 ; ++k){
				for (l=0 ; l<KERNELY3 ; ++l){
					float F3_val = 0;

					for (m=0 ; m<BATCHPART ; ++m){
						for (n=0 ; n<OUTPUTX3 ; ++n){
							for (p=0 ; p<OUTPUTY3 ; ++p){
#pragma HLS PIPELINE off
#pragma HLS LOOP_FLATTEN off
#pragma HLS ALLOCATION operation instances=mul limit=1
#pragma HLS ALLOCATION operation instances=sub limit=1
								int x_x = k*DILATIONX3 - PADDINGX3 + n*STRIDEX3;
								int y_x = l*DILATIONY3 - PADDINGY3 + p*STRIDEY3;
								//int x_dy = n / DILATIONX3;
								//int y_dy = p / DILATIONY3;
								if (x_x>-1 && x_x<OUTPUTX2 && y_x>-1 && y_x<OUTPUTY2){
									F3_val += P2_tmp[m][j][x_x][y_x] * dL3_tmp[m][i][n][p];
									//cout<<"P2_tmp["<<m<<"]["<<j<<"]["<<x_x<<"]["<<y_x<<"]: "<<P2_tmp[m][j][x_x][y_x]<<endl;
									//cout<<"dL3_tmp["<<m<<"]["<<i<<"]["<<x_dy<<"]["<<y_dy<<"]: "<<dL3_tmp[m][i][x_dy][y_dy]<<endl;
									//cout<<"P2_tmp["<<m<<"]["<<j<<"]["<<x_x<<"]["<<y_x<<"] * dL3_tmp["<<m<<"]["<<i<<"]["<<x_dy<<"]["<<y_dy<<"]: "<<P2_tmp[m][j][x_x][y_x] * dL3_tmp[m][i][x_dy][y_dy]<<endl;
								}
							}
						}
					}

					dF3_part[i][j][k][l] = F3_val;

					//cout<<"dF3_part["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<dF3_part[i][j][k][l]<<endl;
				}
			}
		}
	}

}

void dF3_func_out(float dF3_in[FILTERFACTOR3][INPUTCHANNELFACTOR3][BATCHFACTOR][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float dL3_back_tmp[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float dF3[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float dbF3[FILTERFACTOR3][FILTERPART3],
		float dL3_back[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3]){

#pragma HLS ALLOCATION operation instances=sub limit=1

	int i, j, k, l, m, n, p, q;
	float dbF3_tmp[FILTERFACTOR3][FILTERPART3];
#pragma HLS ARRAY_PARTITION variable=dbF3_tmp complete dim=1

	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY3 ; ++l){
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR3 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL
							float dF3_val = 0;

							for (p=0 ; p<BATCHFACTOR ; ++p){
							#pragma HLS UNROLL
								dF3_val += dF3_in[m][n][p][i][j][k][l];
							}

							dF3[m][n][i][j][k][l] = dF3_val;

							//cout<<"dF3["<<m<<"]["<<n<<"]["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<dF3[m][n][i][j][k][l]<<endl;

							//NanControl
							/*if (dF3[m][n][i][j][k][l] != dF3[m][n][i][j][k][l]) {
								cout<<"dF3_func_out"<<endl;
								return;
							}*/
						}
					}
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<OUTPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY3 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<OUTPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL
							float dL3_val = dL3_back_tmp[m][n][i][j][k][l];
							//cout<<m<<", "<<i<<": "<<dL3_val<<endl;
							dL3_back[m][n][i][j][k][l] = dL3_val;

							if (i==0 && k==0 && l==0 && m==0 && dbF3_tmp[n][j]!=0){
								dbF3_tmp[n][j] = 0;
							}
							dbF3_tmp[n][j] += dL3_val;
						}
					}
				}
			}
		}
	}

	for (i=0 ; i<FILTERPART3 ; ++i){
		for (j=0 ; j<FILTERFACTOR3 ; ++j){
		#pragma HLS UNROLL
			dbF3[j][i] = dbF3_tmp[j][i];
			//cout<<"dbF3["<<j<<"]["<<i<<"]: "<<dbF3[j][i]<<endl;
		}
	}


}

void dF3_func(float dL3[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		float P2_act_back[BATCHFACTOR][OUTPUTCHANNELFACTOR2][BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2],
		float dF3[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float dbF3[FILTERFACTOR3][FILTERPART3],
		float dL3_back[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3]){

#pragma HLS DATAFLOW

	int i, j, k, l, m;

	float dL3_in2[BATCHFACTOR][OUTPUTCHANNELFACTOR3][INPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3];
#pragma HLS STREAM variable=dL3_in2 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dL3_in2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL3_in2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=dL3_in2 complete dim=3
	float P2_in2[BATCHFACTOR][OUTPUTCHANNELFACTOR2][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2];
#pragma HLS STREAM variable=P2_in2 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=P2_in2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=P2_in2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=P2_in2 complete dim=3
	float dL3_back_tmp[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3];
#pragma HLS STREAM variable=dL3_back_tmp type=fifo
#pragma HLS ARRAY_PARTITION variable=dL3_back_tmp complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL3_back_tmp complete dim=2
	float dF3_in[FILTERFACTOR3][INPUTCHANNELFACTOR3][BATCHFACTOR][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3];
#pragma HLS STREAM variable=dF3_in type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dF3_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=dF3_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=dF3_in complete dim=3

	dF3_func_in(dL3, P2_act_back, dL3_in2, P2_in2, dL3_back_tmp);
	for (i=0 ; i<BATCHFACTOR ; ++i){
	#pragma HLS UNROLL
		for (j=0 ; j<OUTPUTCHANNELFACTOR3 ; ++j){
		#pragma HLS UNROLL
			for (k=0 ; k<INPUTCHANNELFACTOR3 ; ++k){
			#pragma HLS UNROLL
				dF3_func_calc(dL3_in2[i][j][k], P2_in2[i][k][j], dF3_in[j][k][i]);
			}
		}
	}
	/*dF3_func_calc(dL3_in2[0][0][0], P2_in2[0][0][0], dF3_in[0][0][0]);
	dF3_func_calc(dL3_in2[1][0][0], P2_in2[1][0][0], dF3_in[0][0][1]);*/
	dF3_func_out(dF3_in, dL3_back_tmp, dF3, dbF3, dL3_back);

}

/*

1B: dL1

*/

void dL1_func_in(float dL3_back[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		int P2_poolindex[BATCHFACTOR][OUTPUTCHANNELFACTOR2][BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2][POOLDIM2],
		float F3_back[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float dL3_in[BATCHFACTOR][OUTPUTCHANNELFACTOR3][INPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		int P2_poolindex_in[BATCHFACTOR][OUTPUTCHANNELFACTOR2][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2][POOLDIM2],
		float F3_in[FILTERFACTOR3][INPUTCHANNELFACTOR3][BATCHFACTOR][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3]
		){

	int i, j, k, l, m, n, p, q, r;

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<OUTPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY3 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<OUTPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL
							float dL3_val = dL3_back[m][n][i][j][k][l];
							for (p=0 ; p<INPUTCHANNELFACTOR3 ; ++p){
							#pragma HLS UNROLL
								dL3_in[m][n][p][i][j][k][l] = dL3_val;
							}
						}
					}
				}
			}
		}
	}




	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY3 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR3 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL
							float F3_val = F3_back[m][n][i][j][k][l];
							for (p=0 ; p<BATCHFACTOR ; ++p){
							#pragma HLS UNROLL
								F3_in[m][n][p][i][j][k][l] = F3_val;
							}
						}
					}
				}
			}
		}
	}


	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<OUTPUTCHANNELPART2 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX2 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY2 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<POOLDIM2 ; ++m){
#pragma HLS PIPELINE off
						for (n=0 ; n<BATCHFACTOR ; ++n){
						#pragma HLS UNROLL
							for (p=0 ; p<OUTPUTCHANNELFACTOR2 ; ++p){
							#pragma HLS UNROLL
								int P2_poolindex_val = P2_poolindex[n][p][i][j][k][l][m];
								for (q=0 ; q<OUTPUTCHANNELFACTOR3 ; ++q){
								#pragma HLS UNROLL
									P2_poolindex_in[n][p][q][i][j][k][l][m] = P2_poolindex_val;
								}
							}
						}
					}
				}
			}
		}
	}



}

void dL1_func_calc(float dL3_part[BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		int P2_poolindex_part[BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2][POOLDIM2],
		float F3_part[FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float dL1_part[BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1]){
//#pragma HLS ALLOCATION instances=AddSub limit=5 core

	int i, j, k, l, m, n, p, q, r;
	float dL3_tmp[BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3];
	float F3_tmp[FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3];
	int P2_poolindex_tmp[BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2][POOLDIM2];


	for (i=0 ; i<FILTERPART3 ; ++i){
		for (j=0 ; j<INPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX3 ; ++k){
				for (l=0 ; l<KERNELY3 ; ++l){
					F3_tmp[i][j][k][l] = F3_part[i][j][k][l];
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
		for (j=0 ; j<OUTPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX3 ; ++k){
				for (l=0 ; l<OUTPUTY3 ; ++l){
					dL3_tmp[i][j][k][l] = dL3_part[i][j][k][l];
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
		for (j=0 ; j<OUTPUTCHANNELPART2 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX2 ; ++k){
				for (l=0 ; l<OUTPUTY2 ; ++l){
					for (m=0 ; m<POOLDIM2 ; ++m){
						P2_poolindex_tmp[i][j][k][l][m] = P2_poolindex_part[i][j][k][l][m];
					}
				}
			}
		}
	}


	for (i=0 ; i<BATCHPART ; ++i){
		for (j=0 ; j<INPUTCHANNELPART3 ; ++j){
			for (k=0 ; k<OUTPUTX1 ; ++k){
				for (l=0 ; l<OUTPUTY1 ; ++l){
					float dL1_val = 0;

					for (m=0 ; m<OUTPUTCHANNELPART3 ; ++m){
						for (n=0 ; n<KERNELX3 ; ++n){
							for (p=0 ; p<KERNELY3 ; ++p){
#pragma HLS PIPELINE off
#pragma HLS LOOP_FLATTEN off
#pragma HLS ALLOCATION operation instances=mul limit=1
#pragma HLS ALLOCATION operation instances=sub limit=1
#pragma HLS ALLOCATION operation instances=div limit=1
								int x_x = (k/POOLX2)*1 - (KERNELX3-1+PADDINGX1) + n*1;
								int y_x = (l/POOLY2)*1 - (KERNELY3-1+PADDINGY1) + p*1;
								int x_p = P2_poolindex_tmp[i][j][k/POOLX2][l/POOLY2][0];
								int y_p = P2_poolindex_tmp[i][j][k/POOLX2][l/POOLY2][1];
								//cout<<x_x<<", "<<y_x<<", "<<x_p<<", "<<y_p<<endl;
								//int x_dy = n / DILATIONX3;
								//int y_dy = p / DILATIONY3;
								if (x_x>-1 && x_x<OUTPUTX3 && x_x%STRIDEX3==0 && x_p==k%POOLX2 && y_x>-1 && y_x<OUTPUTY3 && y_x%STRIDEY3==0 && y_p==l%POOLY2){
									//F3_val += P2_tmp[m][j][x_x][y_x] * dL3_tmp[m][i][x_x][y_x];
									dL1_val += dL3_tmp[i][m][x_x][y_x] * F3_tmp[m][j][KERNELX3-1-n][KERNELY3-1-p];
									//cout<<"dL3_tmp["<<i<<"]["<<m<<"]["<<x_x<<"]["<<y_x<<"]: "<<dL3_tmp[i][m][x_x][y_x]<<endl;
									//cout<<"F3_tmp["<<m<<"]["<<j<<"]["<<KERNELX3-1-n<<"]["<<KERNELY3-1-p<<"]: "<<F3_tmp[m][j][KERNELX3-1-n][KERNELY3-1-p]<<endl;
									//cout<<"dL3_tmp["<<i<<"]["<<m<<"]["<<x_x<<"]["<<y_x<<"] * F3_tmp["<<m<<"]["<<j<<"]["<<KERNELX3-1-n<<"]["<<KERNELY3-1-p<<"]: "<<dL3_tmp[i][m][x_x][y_x] * F3_tmp[m][j][KERNELX3-1-n][KERNELY3-1-p]<<endl;

								}
							}
						}
					}

					dL1_part[i][j][k][l] = dL1_val;

					//cout<<"dL1_part["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<dL1_part[i][j][k][l]<<endl;
				}
			}
		}
	}


}

void dL1_func_out(float dL1_in[BATCHFACTOR][OUTPUTCHANNELFACTOR1][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1],
		float P1_nonact_back[BATCHFACTOR][OUTPUTCHANNELFACTOR1][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1],
		float dL1[BATCHFACTOR][OUTPUTCHANNELFACTOR1][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1]){

	int i, j, k, l, m, n, p, q, r;

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<OUTPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY1 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<OUTPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL
#pragma HLS ALLOCATION operation instances=fmul limit=1
							float dL1_val = 0;
							for (p=0 ; p<OUTPUTCHANNELFACTOR3 ; ++p){
							#pragma HLS UNROLL
#pragma HLS ALLOCATION operation instances=fadd limit=1
								dL1_val += dL1_in[m][n][p][i][j][k][l];
							}
							dL1[m][n][i][j][k][l] = dL1_val * relu_drv(P1_nonact_back[m][n][i][j][k][l]);

							//cout<<"dL1_val: "<<dL1_val<<endl;
							//cout<<"dL1["<<m<<"]["<<n<<"]["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<dL1[m][n][i][j][k][l]<<endl;

							//NanControl
							/*if (dL1[m][n][i][j][k][l] != dL1[m][n][i][j][k][l]) {
								cout<<"dL1_func_out"<<endl;
								return;
							}*/
						}
					}
				}
			}
		}
	}


}

void dL1_func(float dL3_back[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3],
		int P2_poolindex_back[BATCHFACTOR][OUTPUTCHANNELFACTOR2][BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2][POOLDIM2],
		float P1_nonact_back[BATCHFACTOR][OUTPUTCHANNELFACTOR1][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1],
		float F3_back[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float dL1[BATCHFACTOR][OUTPUTCHANNELFACTOR1][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1]){

#pragma HLS DATAFLOW

	int i, j, k, l, m;

	float dL3_in3[BATCHFACTOR][OUTPUTCHANNELFACTOR3][INPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3];
#pragma HLS STREAM variable=dL3_in3 type=fifo
#pragma HLS ARRAY_PARTITION variable=dL3_in3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL3_in3 complete dim=2
#pragma HLS ARRAY_PARTITION variable=dL3_in3 complete dim=3
	int P2_poolindex_in[BATCHFACTOR][OUTPUTCHANNELFACTOR2][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2][POOLDIM2];
#pragma HLS STREAM variable=P2_poolindex_in type=fifo
#pragma HLS ARRAY_PARTITION variable=P2_poolindex_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=P2_poolindex_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=P2_poolindex_in complete dim=3
	float F3_in2[FILTERFACTOR3][INPUTCHANNELFACTOR3][BATCHFACTOR][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3];
#pragma HLS STREAM variable=F3_in2 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=F3_in2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=F3_in2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=F3_in2 complete dim=3
	float dL1_in1[BATCHFACTOR][OUTPUTCHANNELFACTOR1][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1];
#pragma HLS STREAM variable=dL1_in1 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dL1_in1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL1_in1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=dL1_in1 complete dim=3

	dL1_func_in(dL3_back, P2_poolindex_back, F3_back, dL3_in3, P2_poolindex_in, F3_in2);
	for (i=0 ; i<BATCHFACTOR ; ++i){
	#pragma HLS UNROLL
		for (j=0 ; j<OUTPUTCHANNELFACTOR3 ; ++j){
		#pragma HLS UNROLL
			for (k=0 ; k<INPUTCHANNELFACTOR3 ; ++k){
			#pragma HLS UNROLL
				dL1_func_calc(dL3_in3[i][j][k], P2_poolindex_in[i][k][j], F3_in2[j][k][i], dL1_in1[i][k][j]);
			}
		}
	}
	/*dL1_func_calc(dL3_in3[0][0][0], P2_poolindex_in[0][0][0],  F3_in2[0][0][0], dL1_in1[0][0][0]);
	dL1_func_calc(dL3_in3[1][0][0], P2_poolindex_in[1][0][0], F3_in2[0][0][1], dL1_in1[1][0][0]);*/
	dL1_func_out(dL1_in1, P1_nonact_back, dL1);

}

/*

1B: dF1

*/

void dF1_func_in(float dL1[BATCHFACTOR][OUTPUTCHANNELFACTOR1][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1],
		float I_back[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1],
		float dbF1_tmp[FILTERFACTOR1][FILTERPART1],
		float dL1_in[BATCHFACTOR][OUTPUTCHANNELFACTOR1][INPUTCHANNELFACTOR1][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1],
		float I_back_in[BATCHFACTOR][INPUTCHANNELFACTOR1][OUTPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1]
				){

	int i, j, k, l, m, n, p, q, r, s;
	float dbF1_tmp2[FILTERFACTOR1][FILTERPART1];

	for (i=0 ; i<BATCHPART ; ++i){
		for (j=0 ; j<OUTPUTCHANNELPART1 ; ++j){
			for (k=0 ; k<OUTPUTX1 ; ++k){
				for (l=0 ; l<OUTPUTY1 ; ++l){
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<OUTPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL
							float dL1_val = dL1[m][n][i][j][k][l];

							if (i==0 && k==0 && l==0 && m==0 && dbF1_tmp2[n][j]!=0){
								dbF1_tmp2[n][j] = 0;
							}
							dbF1_tmp2[n][j] += dL1_val;

							for (p=0 ; p<INPUTCHANNELFACTOR1 ; ++p){
							#pragma HLS UNROLL
								dL1_in[m][n][p][i][j][k][l] = dL1_val;
							}
						}
					}
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
			for (k=0 ; k<INPUTX1 ; ++k){
				for (l=0 ; l<INPUTY1 ; ++l){
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL
							float I_val = I_back[m][n][i][j][k][l];

							for (p=0 ; p<OUTPUTCHANNELFACTOR1 ; ++p){
							#pragma HLS UNROLL
								I_back_in[m][n][p][i][j][k][l] = I_val;
							}
						}
					}
				}
			}
		}
	}



	for (i=0 ; i<FILTERPART1 ; ++i){
		for (j=0 ; j<FILTERFACTOR1 ; ++j){
		#pragma HLS UNROLL
			dbF1_tmp[j][i] = dbF1_tmp2[j][i];
		}
	}

}

void dF1_func_calc(float dL1_part[BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1],
		float I_back_part[BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1],
		float dF1_part[FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1]){

	int i, j, k, l, m, n, p, q;
	float dL1_tmp[BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1];
	float I_tmp[BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1];

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<OUTPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY1 ; ++l){
#pragma HLS PIPELINE off
					dL1_tmp[i][j][k][l] = dL1_part[i][j][k][l];
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<INPUTY1 ; ++l){
#pragma HLS PIPELINE off
					I_tmp[i][j][k][l] = I_back_part[i][j][k][l];
				}
			}
		}
	}




	for (i=0 ; i<FILTERPART1 ; ++i){
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
			for (k=0 ; k<KERNELX1 ; ++k){
				for (l=0 ; l<KERNELY1 ; ++l){
					float F1_val = 0;

					for (m=0 ; m<BATCHPART ; ++m){
						for (n=0 ; n<OUTPUTX1 ; ++n){
							for (p=0 ; p<OUTPUTY1 ; ++p){
#pragma HLS PIPELINE off
#pragma HLS LOOP_FLATTEN off
#pragma HLS ALLOCATION operation instances=mul limit=1
#pragma HLS ALLOCATION operation instances=sub limit=1
								int x_x = k*DILATIONX1 - PADDINGX1 + n*STRIDEX1;
								int y_x = l*DILATIONY1 - PADDINGY1 + p*STRIDEY1;
								//int x_dy = n / DILATIONX3;
								//int y_dy = p / DILATIONY3;
								if (x_x>-1 && x_x<INPUTX1 && y_x>-1 && y_x<INPUTY1){
									//F3_val += P2_tmp[m][j][x_x][y_x] * dL3_tmp[m][i][n][p];
									F1_val += I_tmp[m][j][x_x][y_x] * dL1_tmp[m][i][n][p];
									//cout<<"I_tmp["<<m<<"]["<<j<<"]["<<x_x<<"]["<<y_x<<"]: "<<I_tmp[m][j][x_x][y_x]<<endl;
									//cout<<"dL1_tmp["<<m<<"]["<<i<<"]["<<n<<"]["<<p<<"]: "<<dL1_tmp[m][i][n][p]<<endl;
									//cout<<"I_tmp["<<m<<"]["<<j<<"]["<<x_x<<"]["<<y_x<<"] * dL1_tmp["<<m<<"]["<<i<<"]["<<n<<"]["<<p<<"]: "<<I_tmp[m][j][x_x][y_x] * dL1_tmp[m][i][n][p]<<endl;
								}
							}
						}
					}

					//dF3_part[i][j][k][l] = F3_val;
					dF1_part[i][j][k][l] = F1_val;

					//cout<<"dF1_part["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<dF1_part[i][j][k][l]<<endl;
				}
			}
		}
	}

}

void dF1_func_out(float dF1_in[FILTERFACTOR1][INPUTCHANNELFACTOR1][BATCHFACTOR][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float dbF1_tmp[FILTERFACTOR1][FILTERPART1],
		float dF1[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float dbF1[FILTERFACTOR1][FILTERPART1]){

	int i, j, k, l, m, n, p, q;

	for (i=0 ; i<FILTERPART1 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY1 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR1 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL
							float dF1_val = 0;

							for (p=0 ; p<BATCHFACTOR ; ++p){
							#pragma HLS UNROLL
								dF1_val += dF1_in[m][n][p][i][j][k][l];
							}

							dF1[m][n][i][j][k][l] = dF1_val;

							//NanControl
							/*if (dF1[m][n][i][j][k][l] != dF1[m][n][i][j][k][l]) {
								cout<<"dF1_func_out"<<endl;
								return;
							}*/
						}
					}
				}
			}
		}
	}

	for (i=0 ; i<FILTERPART1 ; ++i){
		for (j=0 ; j<FILTERFACTOR1 ; ++j){
		#pragma HLS UNROLL
			dbF1[j][i] = dbF1_tmp[j][i];
		}
	}

}

void dF1_func(float dL1[BATCHFACTOR][OUTPUTCHANNELFACTOR1][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1],
		float I_back[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1],
		float dF1[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float dbF1[FILTERFACTOR1][FILTERPART1]){

#pragma HLS DATAFLOW

	int i, j, k, l, m, n;
	float dL1_in2[BATCHFACTOR][OUTPUTCHANNELFACTOR1][INPUTCHANNELFACTOR1][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1];
#pragma HLS STREAM variable=dL1_in2 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dL1_in2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL1_in2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=dL1_in2 complete dim=3
	float I_back_in[BATCHFACTOR][INPUTCHANNELFACTOR1][OUTPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1];
#pragma HLS STREAM variable=I_back_in type=fifo
#pragma HLS ARRAY_PARTITION variable=I_back_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=I_back_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=I_back_in complete dim=3
	float dF1_in[FILTERFACTOR1][INPUTCHANNELFACTOR1][BATCHFACTOR][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1];
#pragma HLS STREAM variable=dF1_in type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dF1_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=dF1_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=dF1_in complete dim=3
	float dbF1_tmp[FILTERFACTOR1][FILTERPART1];
#pragma HLS STREAM variable=dbF1_tmp type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dbF1_tmp complete dim=1

	dF1_func_in(dL1, I_back, dbF1_tmp, dL1_in2, I_back_in);
	for (i=0 ; i<BATCHFACTOR ; ++i){
	#pragma HLS UNROLL
		for (j=0 ; j<OUTPUTCHANNELFACTOR1 ; ++j){
		#pragma HLS UNROLL
			for (k=0 ; k<INPUTCHANNELFACTOR1 ; ++k){
			#pragma HLS UNROLL
				dF1_func_calc(dL1_in2[i][j][k], I_back_in[i][k][j], dF1_in[j][k][i]);
			}
		}
	}
	/*dF1_func_calc(dL1_in2[0][0][0], I_back_in[0][0][0], dF1_in[0][0][0]);
	dF1_func_calc(dL1_in2[1][0][0], I_back_in[1][0][0], dF1_in[0][0][1]);*/
	dF1_func_out(dF1_in, dbF1_tmp, dF1, dbF1);

}

/*

Optimizer

*/
/*
void optimizer(float POUT_in[BATCHFACTOR][CLASSFACTOR][BATCHPART][CLASSPART],
		float dF1[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float dbF1[FILTERFACTOR1][FILTERPART1],
		float dF3[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float dbF3[FILTERFACTOR3][FILTERPART3],
		float dW4[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float dbW4[LENGTHFACTOR4][LENGTHPART4],
		int step,
		float POUT[BATCHFACTOR][CLASSFACTOR][BATCHPART][CLASSPART],
		float dF1_opt[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float dbF1_opt[FILTERFACTOR1][FILTERPART1],
		float dF3_opt[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float dbF3_opt[FILTERFACTOR3][FILTERPART3],
		float dW4_opt[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float dbW4_opt[LENGTHFACTOR4][LENGTHPART4]){

	int step_tmp = step;
	int i, j, k, l, m, n, p, q, r;
	float eta = (float) ETA;
	float batch_size = (float) BATCHSIZE;

	for (i=0 ; i<BATCHPART ; ++i){
		for (j=0 ; j<CLASSPART ; ++j){
			for (k=0 ; k<BATCHFACTOR ; ++k){
			#pragma HLS UNROLL
				for (l=0 ; l<CLASSFACTOR ; ++l){
				#pragma HLS UNROLL
					POUT[k][l][i][j] = POUT_in[k][l][i][j];
				}
			}
		}
	}

	for (i=0 ; i<FILTERPART1 ; ++i){
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
			for (k=0 ; k<KERNELX1 ; ++k){
				for (l=0 ; l<KERNELY1 ; ++l){
					for (m=0 ; m<FILTERFACTOR1 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL
							dF1_opt[m][n][i][j][k][l] = eta * dF1[m][n][i][j][k][l] / batch_size;

						}
					}
				}
			}
		}
	}

	for (i=0 ; i<FILTERPART1 ; ++i){
		for (j=0 ; j<FILTERFACTOR1 ; ++j){
		#pragma HLS UNROLL
			dbF1_opt[j][i] = eta * dbF1[j][i] / batch_size;

		}
	}

	for (i=0 ; i<FILTERPART3 ; ++i){
		for (j=0 ; j<INPUTCHANNELPART3 ; ++j){
			for (k=0 ; k<KERNELX3 ; ++k){
				for (l=0 ; l<KERNELY3 ; ++l){
					for (m=0 ; m<FILTERFACTOR3 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL
							dF3_opt[m][n][i][j][k][l] = eta * dF3[m][n][i][j][k][l] / batch_size;

						}
					}
				}
			}
		}
	}

	for (i=0 ; i<FILTERPART3 ; ++i){
		for (j=0 ; j<FILTERFACTOR3 ; ++j){
		#pragma HLS UNROLL
			dbF3_opt[j][i] = eta * dbF3[j][i] / batch_size;

		}
	}

	for (i=0 ; i<INPUTCHANNELPART4 ; ++i){
		for (j=0 ; j<INPUTX4 ; ++j){
			for (k=0 ; k<INPUTY4 ; ++k){
				for (l=0 ; l<LENGTHPART4 ; ++l){
					for (m=0 ; m<INPUTCHANNELFACTOR4 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL
							dW4_opt[m][n][i][j][k][l] = eta * dW4[m][n][i][j][k][l] / batch_size;
						}

					}
				}
			}
		}
	}

	for (i=0 ; i<LENGTHPART4 ; ++i){
		for (j=0 ; j<LENGTHFACTOR4 ; ++j){
		#pragma HLS UNROLL
			dbW4_opt[j][i] = eta * dbW4[j][i] / batch_size;

		}
	}

}
*/


void inFunc(float I[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1],
		int y[BATCHFACTOR][CLASSFACTOR][BATCHPART][CLASSPART][OUTPUTX5][OUTPUTY5],
		float F1[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float bF1[FILTERFACTOR1][FILTERPART1],
		float F3[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float bF3[FILTERFACTOR3][FILTERPART3],
		float W4[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float bW4[LENGTHFACTOR4][LENGTHPART4],
		float W5[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float bW5[LENGTHFACTOR5][LENGTHPART5],
		float I_in[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1],
		int y_in[BATCHFACTOR][CLASSFACTOR][BATCHPART][CLASSPART][OUTPUTX5][OUTPUTY5],
		float F1_in[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float bF1_in[FILTERFACTOR1][FILTERPART1],
		float F3_in[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float bF3_in[FILTERFACTOR3][FILTERPART3],
		float W4_in[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float bW4_in[LENGTHFACTOR4][LENGTHPART4],
		float W5_in[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float bW5_in[LENGTHFACTOR5][LENGTHPART5]){

	int i, j, k, l, m, n, o, p;

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<INPUTY1 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL

							I_in[m][n][i][j][k][l] = I[m][n][i][j][k][l];

						}
					}
				}
			}
		}
	}

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<CLASSPART ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<CLASSFACTOR ; ++n){
						#pragma HLS UNROLL


							y_in[m][n][i][j][k][l] = y[m][n][i][j][k][l];

						}
					}
				}
			}
		}
	}


	for (i=0 ; i<FILTERPART1 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY1 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR1 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL

							F1_in[m][n][i][j][k][l] = F1[m][n][i][j][k][l];

						}
					}
				}
			}
		}
	}

	for (i=0 ; i<FILTERPART1 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<FILTERFACTOR1 ; ++j){
		#pragma HLS UNROLL

			bF1_in[j][i] = bF1[j][i];

		}
	}


	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY3 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR3 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL

							F3_in[m][n][i][j][k][l] = F3[m][n][i][j][k][l];

						}
					}
				}
			}
		}
	}


	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<FILTERFACTOR3 ; ++j){
		#pragma HLS UNROLL

			bF3_in[j][i] = bF3[j][i];

		}
	}


	for (i=0 ; i<INPUTCHANNELPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART4 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTCHANNELFACTOR4 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL

							W4_in[m][n][i][j][k][l] = W4[m][n][i][j][k][l];

						}
					}
				}
			}
		}
	}


	for (i=0 ; i<LENGTHPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHFACTOR4 ; ++j){
		#pragma HLS UNROLL

			bW4_in[j][i] = bW4[j][i];

		}
	}


	for (i=0 ; i<INPUTLENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTLENGTHFACTOR5 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL

							W5_in[m][n][i][j][k][l] = W5[m][n][i][j][k][l];

						}
					}
				}
			}
		}
	}


	for (i=0 ; i<LENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHFACTOR5 ; ++j){
		#pragma HLS UNROLL

			bW5_in[j][i] = bW5[j][i];

		}
	}


}



void outFunc(float POUT_out[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float dF1_out[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float dbF1_out[FILTERFACTOR1][FILTERPART1],
		float dF3_out[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float dbF3_out[FILTERFACTOR3][FILTERPART3],
		float dW4_out[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float dbW4_out[LENGTHFACTOR4][LENGTHPART4],
		float dW5_out[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float dbW5_out[LENGTHFACTOR5][LENGTHPART5],
		float POUT[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float dF1[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float dbF1[FILTERFACTOR1][FILTERPART1],
		float dF3[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float dbF3[FILTERFACTOR3][FILTERPART3],
		float dW4[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float dbW4[LENGTHFACTOR4][LENGTHPART4],
		float dW5[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float dbW5[LENGTHFACTOR5][LENGTHPART5]){



	int i, j, k, l, m, n, o, p;


	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL


							POUT[m][n][i][j][k][l] = POUT_out[m][n][i][j][k][l];

						}
					}
				}
			}
		}
	}


	for (i=0 ; i<FILTERPART1 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY1 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR1 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL


							/*if (i==0 && j==0 && k==0 && l==0 && m==0 && n==0){
								cout<<"dF1_out[0][0][0][0][0][0]: "<<dF1_out[0][0][0][0][0][0]<<endl;
							}*/
							dF1[m][n][i][j][k][l] = dF1_out[m][n][i][j][k][l];

							//cout<<"dF1["<<m<<"]["<<n<<"]["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<dF1[m][n][i][j][k][l]<<endl;

						}
					}
				}
			}
		}
	}

	for (i=0 ; i<FILTERPART1 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<FILTERFACTOR1 ; ++j){
		#pragma HLS UNROLL

			dbF1[j][i] = dbF1_out[j][i];

			//cout<<"dbF1["<<j<<"]["<<i<<"]: "<<dbF1[j][i]<<endl;

		}
	}


	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY3 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR3 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL

							dF3[m][n][i][j][k][l] = dF3_out[m][n][i][j][k][l];

							//cout<<"dF3["<<m<<"]["<<n<<"]["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<dF3[m][n][i][j][k][l]<<endl;

						}
					}
				}
			}
		}
	}


	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<FILTERFACTOR3 ; ++j){
		#pragma HLS UNROLL

			dbF3[j][i] = dbF3_out[j][i];

			//cout<<"dbF3["<<j<<"]["<<i<<"]: "<<dbF3[j][i]<<endl;

		}
	}


	for (i=0 ; i<INPUTCHANNELPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART4 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTCHANNELFACTOR4 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL

							dW4[m][n][i][j][k][l] = dW4_out[m][n][i][j][k][l];

							//cout<<"dW4["<<m<<"]["<<n<<"]["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<dW4[m][n][i][j][k][l]<<endl;

						}
					}
				}
			}
		}
	}


	for (i=0 ; i<LENGTHPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHFACTOR4 ; ++j){
		#pragma HLS UNROLL

			dbW4[j][i] = dbW4_out[j][i];

		}
	}


	for (i=0 ; i<INPUTLENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTLENGTHFACTOR5 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL

							dW5[m][n][i][j][k][l] = dW5_out[m][n][i][j][k][l];

							//cout<<"dW4["<<m<<"]["<<n<<"]["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]: "<<dW4[m][n][i][j][k][l]<<endl;

						}
					}
				}
			}
		}
	}


	for (i=0 ; i<LENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHFACTOR5 ; ++j){
		#pragma HLS UNROLL

			dbW5[j][i] = dbW5_out[j][i];

		}
	}


}




void inFunc2_1(float I[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1],
		 int y[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		 float I_in[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1],
		 int y_in[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5]){


	int i, j, k, l, m, n, o, p, idx;

	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<INPUTY1 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL

							I_in[m][n][i][j][k][l] = I[m][n][i][j][k][l];

						}
					}
				}
			}
		}
	}


	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL

							y_in[m][n][i][j][k][l] = y[m][n][i][j][k][l];
						}

					}
				}
			}
		}
	}



}



void inFunc2_2(float weight_in[LAYERAMT][WEIGHTMAXLEN],
		 float bias_in[LAYERAMT][BIASMAXLEN],
		 float weight_in_temp_0[WEIGHT0LEN],
		 float weight_in_temp_1[WEIGHT1LEN],
		 float weight_in_temp_2[WEIGHT2LEN],
		 float weight_in_temp_3[WEIGHT3LEN],
		 float bias_in_temp_0[BIAS0LEN],
		 float bias_in_temp_1[BIAS1LEN],
		 float bias_in_temp_2[BIAS2LEN],
		 float bias_in_temp_3[BIAS3LEN]){

	int i, j, k, l, m, n, o, p, idx;


	for (i=0 ; i<LAYERAMT ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<WEIGHTMAXLEN ; ++j){
#pragma HLS PIPELINE off

			float w_i;

			w_i = weight_in[i][j];

			if (i==0){
				if (j<WEIGHT0LEN){
					weight_in_temp_0[j] = w_i;
				}
			}
			else if (i==1){
				if (j<WEIGHT1LEN){
					weight_in_temp_1[j] = w_i;
				}
			}
			else if (i==2){
				if (j<WEIGHT2LEN){
					weight_in_temp_2[j] = w_i;
				}
			}
			else if (i==3){
				if (j<WEIGHT3LEN){
					weight_in_temp_3[j] = w_i;
				}
			}

		}

	}


	for (i=0 ; i<LAYERAMT ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<BIASMAXLEN ; ++j){
#pragma HLS PIPELINE off

			float b_i;

			b_i = bias_in[i][j];

			if (i==0){
				if (j<BIAS0LEN){
					bias_in_temp_0[j] = b_i;
				}
			}
			else if (i==1){
				if (j<BIAS1LEN){
					bias_in_temp_1[j] = b_i;
				}
			}
			else if (i==2){
				if (j<BIAS2LEN){
					bias_in_temp_2[j] = b_i;
				}
			}
			else if (i==3){
				if (j<BIAS3LEN){
					bias_in_temp_3[j] = b_i;
				}
			}

		}

	}






}



void inFunc2_3(float weight_in[WEIGHT0LEN],
		 float bias_in[BIAS0LEN],
		 float F1_in[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		 float bF1_in[FILTERFACTOR1][FILTERPART1]){

	int i, j, k, l, m, n, o, p, idx;

	//idx = 0;
	for (i=0 ; i<FILTERPART1 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY1 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR1 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTCHANNELPART1*KERNELX1*KERNELY1*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+j*KERNELX1*KERNELY1*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+k*KERNELY1*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+l*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+m*INPUTCHANNELFACTOR1
									+n;

							F1_in[m][n][i][j][k][l] = weight_in[idx];
							//idx += 1;

						}
					}
				}
			}
		}
	}


	//idx = 0;
	for (i=0 ; i<FILTERPART1 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<FILTERFACTOR1 ; ++j){
		#pragma HLS UNROLL

			idx = i*FILTERFACTOR1+j;

			bF1_in[j][i] = bias_in[idx];
			//idx += 1;

		}
	}


}




void inFunc2_4(float weight_in[WEIGHT1LEN],
		 float bias_in[BIAS1LEN],
		 float F3_in[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		 float bF3_in[FILTERFACTOR3][FILTERPART3]){

	int i, j, k, l, m, n, o, p, idx;

	//idx = 0;
	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY3 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR3 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTCHANNELPART3*KERNELX3*KERNELY3*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+j*KERNELX3*KERNELY3*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+k*KERNELY3*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+l*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+m*INPUTCHANNELFACTOR3
									+n;

							F3_in[m][n][i][j][k][l] = weight_in[idx];
							//idx += 1;

						}
					}
				}
			}
		}
	}

	//idx = 0;
	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<FILTERFACTOR3 ; ++j){
		#pragma HLS UNROLL

			idx = i*FILTERFACTOR3+j;

			bF3_in[j][i] = bias_in[idx];
			//idx += 1;

		}
	}


}



void inFunc2_5(float weight_in[WEIGHT2LEN],
		 float bias_in[BIAS2LEN],
		 float W4_in[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		 float bW4_in[LENGTHFACTOR4][LENGTHPART4]){

	int i, j, k, l, m, n, o, p, idx;

	//idx = 0;
	for (i=0 ; i<INPUTCHANNELPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART4 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTCHANNELFACTOR4 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTX4*INPUTY4*LENGTHPART4*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+j*INPUTY4*LENGTHPART4*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+k*LENGTHPART4*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+l*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+m*LENGTHFACTOR4
									+n;

							W4_in[m][n][i][j][k][l] = weight_in[idx];
							//idx += 1;

						}
					}
				}
			}
		}
	}


	//idx = 0;
	for (i=0 ; i<LENGTHPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHFACTOR4 ; ++j){
		#pragma HLS UNROLL

			idx = i*LENGTHFACTOR4+j;

			bW4_in[j][i] = bias_in[idx];
			//idx += 1;

		}
	}


}



void inFunc2_6(float weight_in[WEIGHT3LEN],
		 float bias_in[BIAS3LEN],
		 float W5_in[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		 float bW5_in[LENGTHFACTOR5][LENGTHPART5]){

	int i, j, k, l, m, n, o, p, idx;

	//idx = 0;
	for (i=0 ; i<INPUTLENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTLENGTHFACTOR5 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTX5*INPUTY5*LENGTHPART5*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+j*INPUTY5*LENGTHPART5*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+k*LENGTHPART5*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+l*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+m*LENGTHFACTOR5
									+n;

							W5_in[m][n][i][j][k][l] = weight_in[idx];
							//idx += 1;

						}
					}
				}
			}
		}
	}

	//idx = 0;
	for (i=0 ; i<LENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHFACTOR5 ; ++j){
		#pragma HLS UNROLL

			idx = i*LENGTHFACTOR5+j;

			bW5_in[j][i] = bias_in[idx];
			//idx += 1;

		}
	}


}


/*void inFunc2_11(float weight_in[LAYERAMT][WEIGHTMAXLEN],
		 float bias_in[LAYERAMT][BIASMAXLEN],
		 float F1_in[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		 float bF1_in[FILTERFACTOR1][FILTERPART1],
		 float F3_in[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		 float bF3_in[FILTERFACTOR3][FILTERPART3],
		 float W4_in[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		 float bW4_in[LENGTHFACTOR4][LENGTHPART4],
		 float W5_in[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		 float bW5_in[LENGTHFACTOR5][LENGTHPART5]){

	int i, j, k, l, m, n, o, p, idx;

	float weight_in_tmp[LAYERAMT][WEIGHTMAXLEN];
	float bias_in_tmp[LAYERAMT][BIASMAXLEN];

	for (i=0 ; i<LAYERAMT ; ++i){
		for (j=0 ; j<WEIGHTMAXLEN ; ++j){
			weight_in_tmp[i][j] = weight_in[i][j];
		}
	}

	for (i=0 ; i<LAYERAMT ; ++i){
		for (j=0 ; j<BIASMAXLEN ; ++j){
			bias_in_tmp[i][j] = bias_in[i][j];
		}
	}

	//idx = 0;
	for (i=0 ; i<FILTERPART1 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY1 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR1 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTCHANNELPART1*KERNELX1*KERNELY1*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+j*KERNELX1*KERNELY1*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+k*KERNELY1*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+l*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+m*INPUTCHANNELFACTOR1
									+n;

							F1_in[m][n][i][j][k][l] = weight_in_tmp[idx];
							//idx += 1;

						}
					}
				}
			}
		}
	}


	//idx = 0;
	for (i=0 ; i<FILTERPART1 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<FILTERFACTOR1 ; ++j){
		#pragma HLS UNROLL

			idx = i*FILTERFACTOR1+j;

			bF1_in[j][i] = bias_in_tmp[idx];
			//idx += 1;

		}
	}

	//idx = 0;
	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY3 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR3 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTCHANNELPART3*KERNELX3*KERNELY3*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+j*KERNELX3*KERNELY3*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+k*KERNELY3*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+l*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+m*INPUTCHANNELFACTOR3
									+n;

							F3_in[m][n][i][j][k][l] = weight_in_tmp[idx];
							//idx += 1;

						}
					}
				}
			}
		}
	}

	//idx = 0;
	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<FILTERFACTOR3 ; ++j){
		#pragma HLS UNROLL

			idx = i*FILTERFACTOR3+j;

			bF3_in[j][i] = bias_in_tmp[idx];
			//idx += 1;

		}
	}


	//idx = 0;
	for (i=0 ; i<INPUTCHANNELPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART4 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTCHANNELFACTOR4 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTX4*INPUTY4*LENGTHPART4*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+j*INPUTY4*LENGTHPART4*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+k*LENGTHPART4*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+l*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+m*LENGTHFACTOR4
									+n;

							W4_in[m][n][i][j][k][l] = weight_in_tmp[idx];
							//idx += 1;

						}
					}
				}
			}
		}
	}


	//idx = 0;
	for (i=0 ; i<LENGTHPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHFACTOR4 ; ++j){
		#pragma HLS UNROLL

			idx = i*LENGTHFACTOR4+j;

			bW4_in[j][i] = bias_in_tmp[idx];
			//idx += 1;

		}
	}


	//idx = 0;
	for (i=0 ; i<INPUTLENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTLENGTHFACTOR5 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTX5*INPUTY5*LENGTHPART5*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+j*INPUTY5*LENGTHPART5*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+k*LENGTHPART5*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+l*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+m*LENGTHFACTOR5
									+n;

							W5_in[m][n][i][j][k][l] = weight_in_tmp[idx];
							//idx += 1;

						}
					}
				}
			}
		}
	}

	//idx = 0;
	for (i=0 ; i<LENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHFACTOR5 ; ++j){
		#pragma HLS UNROLL

			idx = i*LENGTHFACTOR5+j;

			bW5_in[j][i] = bias_in_tmp[idx];
			//idx += 1;

		}
	}


}*/


void inFunc2(float I[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1],
			 int y[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
			 float weight_in[LAYERAMT][WEIGHTMAXLEN],
			 float bias_in[LAYERAMT][BIASMAXLEN],
			 float I_in[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1],
			 int y_in[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
			 float F1_in[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
			 float bF1_in[FILTERFACTOR1][FILTERPART1],
			 float F3_in[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
			 float bF3_in[FILTERFACTOR3][FILTERPART3],
			 float W4_in[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
			 float bW4_in[LENGTHFACTOR4][LENGTHPART4],
			 float W5_in[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
			 float bW5_in[LENGTHFACTOR5][LENGTHPART5]){


//#pragma HLS DATAFLOW

/*	int i, j, k, l, m, n, o, p, idx;


	float weight_in_temp[LAYERAMT][WEIGHTMAXLEN];
#pragma HLS STREAM variable=weight_in_temp type=fifo
#pragma HLS ARRAY_PARTITION variable=weight_in_temp complete dim=1
	float bias_in_temp[LAYERAMT][BIASMAXLEN];
#pragma HLS STREAM variable=bias_in_temp type=fifo
#pragma HLS ARRAY_PARTITION variable=bias_in_temp complete dim=1*/


	int i, j, k, l, m, n, o, p, idx;

	float weight_in_tmp_0[WEIGHT0LEN];
//#pragma HLS STREAM variable=weight_in_tmp_0 type=fifo
	float weight_in_tmp_1[WEIGHT1LEN];
//#pragma HLS STREAM variable=weight_in_tmp_1 type=fifo
	float weight_in_tmp_2[WEIGHT2LEN];
//#pragma HLS STREAM variable=weight_in_tmp_2 type=fifo
	float weight_in_tmp_3[WEIGHT3LEN];
//#pragma HLS STREAM variable=weight_in_tmp_3 type=fifo
	float bias_in_tmp_0[BIAS0LEN];
//#pragma HLS STREAM variable=bias_in_tmp_0 type=fifo
	float bias_in_tmp_1[BIAS1LEN];
//#pragma HLS STREAM variable=bias_in_tmp_1 type=fifo
	float bias_in_tmp_2[BIAS2LEN];
//#pragma HLS STREAM variable=bias_in_tmp_2 type=fifo
	float bias_in_tmp_3[BIAS3LEN];
//#pragma HLS STREAM variable=bias_in_tmp_3 type=fifo



	inFunc2_1(I, y, I_in, y_in);
	inFunc2_2(weight_in, bias_in,
			weight_in_tmp_0, weight_in_tmp_1, weight_in_tmp_2, weight_in_tmp_3,
			bias_in_tmp_0, bias_in_tmp_1, bias_in_tmp_2, bias_in_tmp_3);
	inFunc2_3(weight_in_tmp_0, bias_in_tmp_0, F1_in, bF1_in);
	inFunc2_4(weight_in_tmp_1, bias_in_tmp_1, F3_in, bF3_in);
	inFunc2_5(weight_in_tmp_2, bias_in_tmp_2, W4_in, bW4_in);
	inFunc2_6(weight_in_tmp_3, bias_in_tmp_3, W5_in, bW5_in);
	/*inFunc2_3(weight_in[0], bias_in[0], F1_in, bF1_in);
	inFunc2_4(weight_in[1], bias_in[1], F3_in, bF3_in);
	inFunc2_5(weight_in[2], bias_in[2], W4_in, bW4_in);
	inFunc2_6(weight_in[3], bias_in[3], W5_in, bW5_in);*/

	//inFunc2_11(weight_in, bias_in, F1_in, bF1_in, F3_in, bF3_in, W4_in, bW4_in, W5_in, bW5_in);



/*

	int i, j, k, l, m, n, o, p, idx;

	float weight_in_tmp[LAYERAMT][WEIGHTMAXLEN];
	float bias_in_tmp[LAYERAMT][BIASMAXLEN];



	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<INPUTY1 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL

							I_in[m][n][i][j][k][l] = I[m][n][i][j][k][l];

						}
					}
				}
			}
		}
	}


	for (i=0 ; i<BATCHPART ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHPART5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<OUTPUTX5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<OUTPUTY5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<BATCHFACTOR ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL

							y_in[m][n][i][j][k][l] = y[m][n][i][j][k][l];
						}

					}
				}
			}
		}
	}




	for (i=0 ; i<LAYERAMT ; ++i){
		for (j=0 ; j<WEIGHTMAXLEN ; ++j){
			weight_in_tmp[i][j] = weight_in[i][j];
		}
	}

	for (i=0 ; i<LAYERAMT ; ++i){
		for (j=0 ; j<BIASMAXLEN ; ++j){
			bias_in_tmp[i][j] = bias_in[i][j];
		}
	}

	//idx = 0;
	for (i=0 ; i<FILTERPART1 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY1 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR1 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTCHANNELPART1*KERNELX1*KERNELY1*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+j*KERNELX1*KERNELY1*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+k*KERNELY1*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+l*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+m*INPUTCHANNELFACTOR1
									+n;

							F1_in[m][n][i][j][k][l] = weight_in_tmp[0][idx];
							//idx += 1;

						}
					}
				}
			}
		}
	}


	//idx = 0;
	for (i=0 ; i<FILTERPART1 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<FILTERFACTOR1 ; ++j){
		#pragma HLS UNROLL

			idx = i*FILTERFACTOR1+j;

			bF1_in[j][i] = bias_in_tmp[0][idx];
			//idx += 1;

		}
	}

	//idx = 0;
	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY3 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR3 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTCHANNELPART3*KERNELX3*KERNELY3*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+j*KERNELX3*KERNELY3*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+k*KERNELY3*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+l*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+m*INPUTCHANNELFACTOR3
									+n;

							F3_in[m][n][i][j][k][l] = weight_in_tmp[1][idx];
							//idx += 1;

						}
					}
				}
			}
		}
	}

	//idx = 0;
	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<FILTERFACTOR3 ; ++j){
		#pragma HLS UNROLL

			idx = i*FILTERFACTOR3+j;

			bF3_in[j][i] = bias_in_tmp[1][idx];
			//idx += 1;

		}
	}


	//idx = 0;
	for (i=0 ; i<INPUTCHANNELPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART4 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTCHANNELFACTOR4 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTX4*INPUTY4*LENGTHPART4*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+j*INPUTY4*LENGTHPART4*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+k*LENGTHPART4*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+l*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+m*LENGTHFACTOR4
									+n;

							W4_in[m][n][i][j][k][l] = weight_in_tmp[2][idx];
							//idx += 1;

						}
					}
				}
			}
		}
	}


	//idx = 0;
	for (i=0 ; i<LENGTHPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHFACTOR4 ; ++j){
		#pragma HLS UNROLL

			idx = i*LENGTHFACTOR4+j;

			bW4_in[j][i] = bias_in_tmp[2][idx];
			//idx += 1;

		}
	}


	//idx = 0;
	for (i=0 ; i<INPUTLENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTLENGTHFACTOR5 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTX5*INPUTY5*LENGTHPART5*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+j*INPUTY5*LENGTHPART5*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+k*LENGTHPART5*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+l*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+m*LENGTHFACTOR5
									+n;

							W5_in[m][n][i][j][k][l] = weight_in_tmp[3][idx];
							//idx += 1;

						}
					}
				}
			}
		}
	}

	//idx = 0;
	for (i=0 ; i<LENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHFACTOR5 ; ++j){
		#pragma HLS UNROLL

			idx = i*LENGTHFACTOR5+j;

			bW5_in[j][i] = bias_in_tmp[3][idx];
			//idx += 1;

		}
	}

*/








}






void outFunc2_1(float POUT_out[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float POUT[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5]){

	int i, j, k, l, m, n, o, p;

	for (i=0 ; i<BATCHPART ; ++i){
	#pragma HLS PIPELINE off
			for (j=0 ; j<LENGTHPART5 ; ++j){
	#pragma HLS PIPELINE off
				for (k=0 ; k<OUTPUTX5 ; ++k){
	#pragma HLS PIPELINE off
					for (l=0 ; l<OUTPUTY5 ; ++l){
	#pragma HLS PIPELINE off
						for (m=0 ; m<BATCHFACTOR ; ++m){
						#pragma HLS UNROLL
							for (n=0 ; n<LENGTHFACTOR5 ; ++n){
							#pragma HLS UNROLL


								POUT[m][n][i][j][k][l] = POUT_out[m][n][i][j][k][l];


							}
						}
					}
				}
			}
		}

}



void outFunc2_2(float dF1_out[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float dbF1_out[FILTERFACTOR1][FILTERPART1],
		float weight_out[WEIGHT0LEN],
		float bias_out[BIAS0LEN]){

	int i, j, k, l, m, n, o, p, idx;


	for (i=0 ; i<WEIGHT0LEN ; ++i){
#pragma HLS PIPELINE off
		weight_out[i] = 0;
	}

	for (i=0 ; i<BIAS0LEN ; ++i){
#pragma HLS PIPELINE off
		bias_out[i] = 0;
	}

	for (i=0 ; i<FILTERPART1 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY1 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR1 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTCHANNELPART1*KERNELX1*KERNELY1*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+j*KERNELX1*KERNELY1*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+k*KERNELY1*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+l*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+m*INPUTCHANNELFACTOR1
									+n;

							weight_out[idx] = dF1_out[m][n][i][j][k][l];


						}
					}
				}
			}
		}
	}

	//idx = 0;
	for (i=0 ; i<FILTERPART1 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<FILTERFACTOR1 ; ++j){
		#pragma HLS UNROLL

			idx = i*FILTERFACTOR1+j;

			bias_out[idx] = dbF1_out[j][i];
			//idx += 1;

		}
	}


}



void outFunc2_3(float dF3_out[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float dbF3_out[FILTERFACTOR3][FILTERPART3],
		float weight_out[WEIGHT1LEN],
		float bias_out[BIAS1LEN]){

	int i, j, k, l, m, n, o, p, idx;

	for (i=0 ; i<WEIGHT1LEN ; ++i){
#pragma HLS PIPELINE off
		weight_out[i] = 0;
	}

	for (i=0 ; i<BIAS1LEN ; ++i){
#pragma HLS PIPELINE off
		bias_out[i] = 0;
	}



	//idx = 0;
	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY3 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR3 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTCHANNELPART3*KERNELX3*KERNELY3*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+j*KERNELX3*KERNELY3*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+k*KERNELY3*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+l*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+m*INPUTCHANNELFACTOR3
									+n;

							weight_out[idx] = dF3_out[m][n][i][j][k][l];
							//idx += 1;

						}
					}
				}
			}
		}
	}


	//idx = 0;
	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<FILTERFACTOR3 ; ++j){
		#pragma HLS UNROLL

			idx = i*FILTERFACTOR3+j;

			bias_out[idx] = dbF3_out[j][i];
			//idx += 1;

		}
	}


}



void outFunc2_4(float dW4_out[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float dbW4_out[LENGTHFACTOR4][LENGTHPART4],
		float weight_out[WEIGHT2LEN],
		float bias_out[BIAS2LEN]){

	int i, j, k, l, m, n, o, p, idx;


	for (i=0 ; i<WEIGHT2LEN ; ++i){
#pragma HLS PIPELINE off
		weight_out[i] = 0;
	}

	for (i=0 ; i<BIAS2LEN ; ++i){
#pragma HLS PIPELINE off
		bias_out[i] = 0;
	}

	//idx = 0;
	for (i=0 ; i<INPUTCHANNELPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART4 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTCHANNELFACTOR4 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTX4*INPUTY4*LENGTHPART4*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+j*INPUTY4*LENGTHPART4*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+k*LENGTHPART4*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+l*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+m*LENGTHFACTOR4
									+n;

							weight_out[idx] = dW4_out[m][n][i][j][k][l];
							//idx += 1;

						}
					}
				}
			}
		}
	}

	//idx = 0;
	for (i=0 ; i<LENGTHPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHFACTOR4 ; ++j){
		#pragma HLS UNROLL

			idx = i*LENGTHFACTOR4+j;

			bias_out[idx] = dbW4_out[j][i];
			//idx += 1;

		}
	}


}


void outFunc2_5(float dW5_out[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float dbW5_out[LENGTHFACTOR5][LENGTHPART5],
		float weight_out[WEIGHT3LEN],
		float bias_out[BIAS3LEN]){

	int i, j, k, l, m, n, o, p, idx;


	for (i=0 ; i<WEIGHT3LEN ; ++i){
#pragma HLS PIPELINE off
		weight_out[i] = 0;
	}

	for (i=0 ; i<BIAS3LEN ; ++i){
#pragma HLS PIPELINE off
		bias_out[i] = 0;
	}

	//idx = 0;
	for (i=0 ; i<INPUTLENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTLENGTHFACTOR5 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTX5*INPUTY5*LENGTHPART5*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+j*INPUTY5*LENGTHPART5*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+k*LENGTHPART5*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+l*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+m*LENGTHFACTOR5
									+n;
							weight_out[idx] = dW5_out[m][n][i][j][k][l];
							//idx += 1;

						}
					}
				}
			}
		}
	}

	//idx = 0;
	for (i=0 ; i<LENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHFACTOR5 ; ++j){
		#pragma HLS UNROLL

			idx = i*LENGTHFACTOR5+j;

			bias_out[idx] = dbW5_out[j][i];
			//idx += 1;

		}
	}


}



void outFunc2_6(float POUT_out[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float weight_out_temp_0[WEIGHT0LEN],
		float weight_out_temp_1[WEIGHT1LEN],
		float weight_out_temp_2[WEIGHT2LEN],
		float weight_out_temp_3[WEIGHT3LEN],
		float bias_out_temp_0[BIAS0LEN],
		float bias_out_temp_1[BIAS1LEN],
		float bias_out_temp_2[BIAS2LEN],
		float bias_out_temp_3[BIAS3LEN],
		float POUT[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float weight_out[LAYERAMT][WEIGHTMAXLEN],
		float bias_out[LAYERAMT][BIASMAXLEN]){

	int i, j, k, l, m, n, o, p, idx;


	for (i=0 ; i<BATCHPART ; ++i){
	#pragma HLS PIPELINE off
			for (j=0 ; j<LENGTHPART5 ; ++j){
	#pragma HLS PIPELINE off
				for (k=0 ; k<OUTPUTX5 ; ++k){
	#pragma HLS PIPELINE off
					for (l=0 ; l<OUTPUTY5 ; ++l){
	#pragma HLS PIPELINE off
						for (m=0 ; m<BATCHFACTOR ; ++m){
						#pragma HLS UNROLL
							for (n=0 ; n<LENGTHFACTOR5 ; ++n){
							#pragma HLS UNROLL


								POUT[m][n][i][j][k][l] = POUT_out[m][n][i][j][k][l];


							}
						}
					}
				}
			}
		}




	for (i=0 ; i<LAYERAMT ; ++i){
#pragma HLS PIPELINE off

		for (j=0 ; j<WEIGHTMAXLEN ; ++j){
#pragma HLS PIPELINE off
			float w_o;

			if (i==0){
				if (j<WEIGHT0LEN){
					w_o = weight_out_temp_0[j];
				}
				else{
					w_o = 0;
				}
			}
			else if (i==1){
				if (j<WEIGHT1LEN){
					w_o = weight_out_temp_1[j];
				}
				else{
					w_o = 0;
				}
			}
			else if (i==2){
				if (j<WEIGHT2LEN){
					w_o = weight_out_temp_2[j];
				}
				else{
					w_o = 0;
				}
			}
			else if (i==3){
				if (j<WEIGHT3LEN){
					w_o = weight_out_temp_3[j];
				}
				else{
					w_o = 0;
				}
			}

			weight_out[i][j] = w_o;
		}
	}
	for (i=0 ; i<LAYERAMT ; ++i){
#pragma HLS PIPELINE off

		for (j=0 ; j<BIASMAXLEN ; ++j){
#pragma HLS PIPELINE off
			float b_o;

			if (i==0){
				if (j<BIAS0LEN){
					b_o = bias_out_temp_0[j];
				}
				else{
					b_o = 0;
				}
			}
			else if (i==1){
				if (j<BIAS1LEN){
					b_o = bias_out_temp_1[j];
				}
				else{
					b_o = 0;
				}
			}
			else if (i==2){
				if (j<BIAS2LEN){
					b_o = bias_out_temp_2[j];
				}
				else{
					b_o = 0;
				}
			}
			else if (i==3){
				if (j<BIAS3LEN){
					b_o = bias_out_temp_3[j];
				}
				else{
					b_o = 0;
				}
			}

			bias_out[i][j] = b_o;
		}
	}



}



/*void outFunc2_11(float dF1_out[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float dbF1_out[FILTERFACTOR1][FILTERPART1],
		float dF3_out[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float dbF3_out[FILTERFACTOR3][FILTERPART3],
		float dW4_out[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float dbW4_out[LENGTHFACTOR4][LENGTHPART4],
		float dW5_out[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float dbW5_out[LENGTHFACTOR5][LENGTHPART5],
		float weight_out[LAYERAMT][WEIGHTMAXLEN],
		float bias_out[LAYERAMT][BIASMAXLEN]){


	int i, j, k, l, m, n, o, p, idx;

	float weight_out_tmp[LAYERAMT][WEIGHTMAXLEN];
	float bias_out_tmp[LAYERAMT][BIASMAXLEN];

	for (i=0 ; i<FILTERPART1 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY1 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR1 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTCHANNELPART1*KERNELX1*KERNELY1*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+j*KERNELX1*KERNELY1*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+k*KERNELY1*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+l*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+m*INPUTCHANNELFACTOR1
									+n;

							weight_out_tmp[idx] = dF1_out[m][n][i][j][k][l];


						}
					}
				}
			}
		}
	}

	//idx = 0;
	for (i=0 ; i<FILTERPART1 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<FILTERFACTOR1 ; ++j){
		#pragma HLS UNROLL

			idx = i*FILTERFACTOR1+j;

			bias_out_tmp[idx] = dbF1_out[j][i];
			//idx += 1;

		}
	}


	//idx = 0;
	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY3 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR3 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTCHANNELPART3*KERNELX3*KERNELY3*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+j*KERNELX3*KERNELY3*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+k*KERNELY3*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+l*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+m*INPUTCHANNELFACTOR3
									+n;

							weight_out_tmp[idx] = dF3_out[m][n][i][j][k][l];
							//idx += 1;

						}
					}
				}
			}
		}
	}


	//idx = 0;
	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<FILTERFACTOR3 ; ++j){
		#pragma HLS UNROLL

			idx = i*FILTERFACTOR3+j;

			bias_out_tmp[idx] = dbF3_out[j][i];
			//idx += 1;

		}
	}


	//idx = 0;
	for (i=0 ; i<INPUTCHANNELPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART4 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTCHANNELFACTOR4 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTX4*INPUTY4*LENGTHPART4*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+j*INPUTY4*LENGTHPART4*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+k*LENGTHPART4*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+l*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+m*LENGTHFACTOR4
									+n;

							weight_out_tmp[idx] = dW4_out[m][n][i][j][k][l];
							//idx += 1;

						}
					}
				}
			}
		}
	}

	//idx = 0;
	for (i=0 ; i<LENGTHPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHFACTOR4 ; ++j){
		#pragma HLS UNROLL

			idx = i*LENGTHFACTOR4+j;

			bias_out_tmp[idx] = dbW4_out[j][i];
			//idx += 1;

		}
	}

	//idx = 0;
	for (i=0 ; i<INPUTLENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTLENGTHFACTOR5 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTX5*INPUTY5*LENGTHPART5*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+j*INPUTY5*LENGTHPART5*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+k*LENGTHPART5*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+l*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+m*LENGTHFACTOR5
									+n;
							weight_out_tmp[idx] = dW5_out[m][n][i][j][k][l];
							//idx += 1;

						}
					}
				}
			}
		}
	}

	//idx = 0;
	for (i=0 ; i<LENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHFACTOR5 ; ++j){
		#pragma HLS UNROLL

			idx = i*LENGTHFACTOR5+j;

			bias_out_tmp[idx] = dbW5_out[j][i];
			//idx += 1;

		}
	}


}*/


void outFunc2(float POUT_out[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float dF1_out[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float dbF1_out[FILTERFACTOR1][FILTERPART1],
		float dF3_out[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float dbF3_out[FILTERFACTOR3][FILTERPART3],
		float dW4_out[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float dbW4_out[LENGTHFACTOR4][LENGTHPART4],
		float dW5_out[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float dbW5_out[LENGTHFACTOR5][LENGTHPART5],
		float POUT[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float weight_out[LAYERAMT][WEIGHTMAXLEN],
		float bias_out[LAYERAMT][BIASMAXLEN]){

//#pragma HLS DATAFLOW


/*	int i, j, k, l, m, n, o, p, idx;

	float weight_out_temp[LAYERAMT][WEIGHTMAXLEN];
#pragma HLS STREAM variable=weight_out_temp type=fifo
#pragma HLS ARRAY_PARTITION variable=weight_out_temp complete dim=1
	float bias_out_temp[LAYERAMT][BIASMAXLEN];
#pragma HLS STREAM variable=bias_out_temp type=fifo
#pragma HLS ARRAY_PARTITION variable=bias_out_temp complete dim=1*/


	int i, j, k, l, m, n, o, p, idx;

	float weight_out_tmp_0[WEIGHT0LEN];
//#pragma HLS STREAM variable=weight_out_tmp_0 type=fifo
	float weight_out_tmp_1[WEIGHT1LEN];
//#pragma HLS STREAM variable=weight_out_tmp_1 type=fifo
	float weight_out_tmp_2[WEIGHT2LEN];
//#pragma HLS STREAM variable=weight_out_tmp_2 type=fifo
	float weight_out_tmp_3[WEIGHT3LEN];
//#pragma HLS STREAM variable=weight_out_tmp_3 type=fifo
	float bias_out_tmp_0[BIAS0LEN];
//#pragma HLS STREAM variable=bias_out_tmp_0 type=fifo
	float bias_out_tmp_1[BIAS1LEN];
//#pragma HLS STREAM variable=bias_out_tmp_1 type=fifo
	float bias_out_tmp_2[BIAS2LEN];
//#pragma HLS STREAM variable=bias_out_tmp_2 type=fifo
	float bias_out_tmp_3[BIAS3LEN];
//#pragma HLS STREAM variable=bias_out_tmp_3 type=fifo


	//outFunc2_1(POUT_out, POUT);
	outFunc2_2(dF1_out, dbF1_out, weight_out_tmp_0, bias_out_tmp_0);
	outFunc2_3(dF3_out, dbF3_out, weight_out_tmp_1, bias_out_tmp_1);
	outFunc2_4(dW4_out, dbW4_out, weight_out_tmp_2, bias_out_tmp_2);
	outFunc2_5(dW5_out, dbW5_out, weight_out_tmp_3, bias_out_tmp_3);
	outFunc2_6(POUT_out,
			   weight_out_tmp_0, weight_out_tmp_1, weight_out_tmp_2, weight_out_tmp_3,
			   bias_out_tmp_0, bias_out_tmp_1, bias_out_tmp_2, bias_out_tmp_3,
			   POUT,
			   weight_out, bias_out);
	/*outFunc2_2(dF1_out, dbF1_out, weight_out[0], bias_out[0]);
	outFunc2_3(dF3_out, dbF3_out, weight_out[1], bias_out[1]);
	outFunc2_4(dW4_out, dbW4_out, weight_out[2], bias_out[2]);
	outFunc2_5(dW5_out, dbW5_out, weight_out[3], bias_out[3]);*/

	//outFunc2_11(dF1_out, dbF1_out, dF3_out, dbF3_out, dW4_out, dbW4_out, dW5_out, dbW5_out, weight_out, bias_out);


	/*for (i=0 ; i<BATCHPART ; ++i){
	#pragma HLS PIPELINE off
			for (j=0 ; j<LENGTHPART5 ; ++j){
	#pragma HLS PIPELINE off
				for (k=0 ; k<OUTPUTX5 ; ++k){
	#pragma HLS PIPELINE off
					for (l=0 ; l<OUTPUTY5 ; ++l){
	#pragma HLS PIPELINE off
						for (m=0 ; m<BATCHFACTOR ; ++m){
						#pragma HLS UNROLL
							for (n=0 ; n<LENGTHFACTOR5 ; ++n){
							#pragma HLS UNROLL


								POUT[m][n][i][j][k][l] = POUT_out[m][n][i][j][k][l];


							}
						}
					}
				}
			}
		}

	for (i=0 ; i<FILTERPART1 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART1 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX1 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY1 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR1 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR1 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTCHANNELPART1*KERNELX1*KERNELY1*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+j*KERNELX1*KERNELY1*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+k*KERNELY1*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+l*FILTERFACTOR1*INPUTCHANNELFACTOR1
									+m*INPUTCHANNELFACTOR1
									+n;

							weight_out_tmp[0][idx] = dF1_out[m][n][i][j][k][l];


						}
					}
				}
			}
		}
	}

	//idx = 0;
	for (i=0 ; i<FILTERPART1 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<FILTERFACTOR1 ; ++j){
		#pragma HLS UNROLL

			idx = i*FILTERFACTOR1+j;

			bias_out_tmp[0][idx] = dbF1_out[j][i];
			//idx += 1;

		}
	}


	//idx = 0;
	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTCHANNELPART3 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<KERNELX3 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<KERNELY3 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<FILTERFACTOR3 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<INPUTCHANNELFACTOR3 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTCHANNELPART3*KERNELX3*KERNELY3*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+j*KERNELX3*KERNELY3*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+k*KERNELY3*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+l*FILTERFACTOR3*INPUTCHANNELFACTOR3
									+m*INPUTCHANNELFACTOR3
									+n;

							weight_out_tmp[1][idx] = dF3_out[m][n][i][j][k][l];
							//idx += 1;

						}
					}
				}
			}
		}
	}


	//idx = 0;
	for (i=0 ; i<FILTERPART3 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<FILTERFACTOR3 ; ++j){
		#pragma HLS UNROLL

			idx = i*FILTERFACTOR3+j;

			bias_out_tmp[1][idx] = dbF3_out[j][i];
			//idx += 1;

		}
	}


	//idx = 0;
	for (i=0 ; i<INPUTCHANNELPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX4 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY4 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART4 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTCHANNELFACTOR4 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR4 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTX4*INPUTY4*LENGTHPART4*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+j*INPUTY4*LENGTHPART4*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+k*LENGTHPART4*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+l*INPUTCHANNELFACTOR4*LENGTHFACTOR4
									+m*LENGTHFACTOR4
									+n;

							weight_out_tmp[2][idx] = dW4_out[m][n][i][j][k][l];
							//idx += 1;

						}
					}
				}
			}
		}
	}

	//idx = 0;
	for (i=0 ; i<LENGTHPART4 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHFACTOR4 ; ++j){
		#pragma HLS UNROLL

			idx = i*LENGTHFACTOR4+j;

			bias_out_tmp[2][idx] = dbW4_out[j][i];
			//idx += 1;

		}
	}

	//idx = 0;
	for (i=0 ; i<INPUTLENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<INPUTX5 ; ++j){
#pragma HLS PIPELINE off
			for (k=0 ; k<INPUTY5 ; ++k){
#pragma HLS PIPELINE off
				for (l=0 ; l<LENGTHPART5 ; ++l){
#pragma HLS PIPELINE off
					for (m=0 ; m<INPUTLENGTHFACTOR5 ; ++m){
					#pragma HLS UNROLL
						for (n=0 ; n<LENGTHFACTOR5 ; ++n){
						#pragma HLS UNROLL

							idx = i*INPUTX5*INPUTY5*LENGTHPART5*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+j*INPUTY5*LENGTHPART5*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+k*LENGTHPART5*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+l*INPUTLENGTHFACTOR5*LENGTHFACTOR5
									+m*LENGTHFACTOR5
									+n;
							weight_out_tmp[3][idx] = dW5_out[m][n][i][j][k][l];
							//idx += 1;

						}
					}
				}
			}
		}
	}

	//idx = 0;
	for (i=0 ; i<LENGTHPART5 ; ++i){
#pragma HLS PIPELINE off
		for (j=0 ; j<LENGTHFACTOR5 ; ++j){
		#pragma HLS UNROLL

			idx = i*LENGTHFACTOR5+j;

			bias_out_tmp[3][idx] = dbW5_out[j][i];
			//idx += 1;

		}
	}


	for (i=0 ; i<LAYERAMT ; ++i){
		for (j=0 ; j<WEIGHTMAXLEN ; ++j){
			weight_out[i][j] = weight_out_tmp[i][j];
		}
	}

	for (i=0 ; i<LAYERAMT ; ++i){
		for (j=0 ; j<BIASMAXLEN ; ++j){
			bias_out[i][j] = bias_out_tmp[i][j];
		}
	}*/


}






void train(float I[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1],
		int y[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		float POUT[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5],
		/*float F1[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float bF1[FILTERFACTOR1][FILTERPART1],
		float F3[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float bF3[FILTERFACTOR3][FILTERPART3],
		float W4[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float bW4[LENGTHFACTOR4][LENGTHPART4],
		float W5[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float bW5[LENGTHFACTOR5][LENGTHPART5],*/
		float weight_in[LAYERAMT][WEIGHTMAXLEN],
		float bias_in[LAYERAMT][BIASMAXLEN],
		//int step,
		/*float dF1_opt[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float dbF1_opt[FILTERFACTOR1][FILTERPART1],
		float dF3_opt[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float dbF3_opt[FILTERFACTOR3][FILTERPART3],
		float dW4_opt[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float dbW4_opt[LENGTHFACTOR4][LENGTHPART4]*/
		/*float dF1[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1],
		float dbF1[FILTERFACTOR1][FILTERPART1],
		float dF3[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3],
		float dbF3[FILTERFACTOR3][FILTERPART3],
		float dW4[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4],
		float dbW4[LENGTHFACTOR4][LENGTHPART4],
		float dW5[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5],
		float dbW5[LENGTHFACTOR5][LENGTHPART5]*/
		float weight_out[LAYERAMT][WEIGHTMAXLEN],
		float bias_out[LAYERAMT][BIASMAXLEN]
		){
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=y complete dim=1
#pragma HLS ARRAY_PARTITION variable=y complete dim=2
#pragma HLS ARRAY_PARTITION variable=POUT complete dim=1
#pragma HLS ARRAY_PARTITION variable=POUT complete dim=2
/*#pragma HLS ARRAY_PARTITION variable=dF1_opt complete dim=1
#pragma HLS ARRAY_PARTITION variable=dF1_opt complete dim=2
#pragma HLS ARRAY_PARTITION variable=dbF1_opt complete dim=1
#pragma HLS ARRAY_PARTITION variable=dF3_opt complete dim=1
#pragma HLS ARRAY_PARTITION variable=dF3_opt complete dim=2
#pragma HLS ARRAY_PARTITION variable=dbF3_opt complete dim=1
#pragma HLS ARRAY_PARTITION variable=dW4_opt complete dim=1
#pragma HLS ARRAY_PARTITION variable=dW4_opt complete dim=2
#pragma HLS ARRAY_PARTITION variable=dbW4_opt complete dim=1*/

/*#pragma HLS ARRAY_PARTITION variable=dF1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dF1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=dbF1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dF3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dF3 complete dim=2
#pragma HLS ARRAY_PARTITION variable=dbF3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dW4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dW4 complete dim=2
#pragma HLS ARRAY_PARTITION variable=dbW4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dW5 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dW5 complete dim=2
#pragma HLS ARRAY_PARTITION variable=dbW5 complete dim=1


#pragma HLS ARRAY_PARTITION variable=F1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=F1 complete dim=2

#pragma HLS ARRAY_PARTITION variable=bF1 complete dim=1

#pragma HLS ARRAY_PARTITION variable=F3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=F3 complete dim=2

#pragma HLS ARRAY_PARTITION variable=bF3 complete dim=1

#pragma HLS ARRAY_PARTITION variable=W4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=W4 complete dim=2

#pragma HLS ARRAY_PARTITION variable=bW4 complete dim=1

#pragma HLS ARRAY_PARTITION variable=W5 complete dim=1
#pragma HLS ARRAY_PARTITION variable=W5 complete dim=2

#pragma HLS ARRAY_PARTITION variable=bW5 complete dim=1*/


/*#pragma HLS ARRAY_PARTITION variable=weight_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias_out complete dim=1*/



/*
#pragma HLS STREAM variable=I dim=1
#pragma HLS STREAM variable=I dim=2
#pragma HLS STREAM variable=y dim=1
#pragma HLS STREAM variable=y dim=2
#pragma HLS STREAM variable=POUT dim=1
#pragma HLS STREAM variable=POUT dim=2
#pragma HLS STREAM variable=dF1 dim=1
#pragma HLS STREAM variable=dF1 dim=2
#pragma HLS STREAM variable=dbF1 dim=1
#pragma HLS STREAM variable=dF3 dim=1
#pragma HLS STREAM variable=dF3 dim=2
#pragma HLS STREAM variable=dbF3 dim=1
#pragma HLS STREAM variable=dW4 dim=1
#pragma HLS STREAM variable=dW4 dim=2
#pragma HLS STREAM variable=dbW4 dim=1
*/


/*#pragma HLS INTERFACE axis register both port=dbW4_opt
#pragma HLS INTERFACE axis register both port=dW4_opt
#pragma HLS INTERFACE axis register both port=dbF3_opt
#pragma HLS INTERFACE axis register both port=dF3_opt
#pragma HLS INTERFACE axis register both port=dbF1_opt
#pragma HLS INTERFACE axis register both port=dF1_opt*/
/*#pragma HLS INTERFACE axis port=dbW5
#pragma HLS INTERFACE axis port=dW5
#pragma HLS INTERFACE axis port=dbW4
#pragma HLS INTERFACE axis port=dW4
#pragma HLS INTERFACE axis port=dbF3
#pragma HLS INTERFACE axis port=dF3
#pragma HLS INTERFACE axis port=dbF1
#pragma HLS INTERFACE axis port=dF1*/
//#pragma HLS INTERFACE s_axilite port=step
/*#pragma HLS INTERFACE axis port=bW5
#pragma HLS INTERFACE axis port=W5
#pragma HLS INTERFACE axis port=bW4
#pragma HLS INTERFACE axis port=W4
#pragma HLS INTERFACE axis port=bF3
#pragma HLS INTERFACE axis port=F3
#pragma HLS INTERFACE axis port=bF1
#pragma HLS INTERFACE axis port=F1
#pragma HLS INTERFACE axis port=POUT
#pragma HLS INTERFACE axis port=y
#pragma HLS INTERFACE axis port=I*/

#pragma HLS INTERFACE axis port=bias_out
#pragma HLS INTERFACE axis port=weight_out
#pragma HLS INTERFACE axis port=bias_in
#pragma HLS INTERFACE axis port=weight_in
#pragma HLS INTERFACE axis port=POUT
#pragma HLS INTERFACE axis port=y
#pragma HLS INTERFACE axis port=I


//#pragma HLS INTERFACE mode=s_axilite port=return bundle=control

#pragma HLS DATAFLOW

	float P1_act[BATCHFACTOR][OUTPUTCHANNELFACTOR1][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1];
#pragma HLS STREAM variable=P1_act type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=P1_act complete dim=1
#pragma HLS ARRAY_PARTITION variable=P1_act complete dim=2
	float I_back[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1];
#pragma HLS STREAM variable=I_back type=fifo
#pragma HLS ARRAY_PARTITION variable=I_back complete dim=1
#pragma HLS ARRAY_PARTITION variable=I_back complete dim=2
	float P1_nonact_back[BATCHFACTOR][OUTPUTCHANNELFACTOR1][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1];
#pragma HLS STREAM variable=P1_nonact_back type=fifo
#pragma HLS ARRAY_PARTITION variable=P1_nonact_back complete dim=1
#pragma HLS ARRAY_PARTITION variable=P1_nonact_back complete dim=2


	float P2_act[BATCHFACTOR][OUTPUTCHANNELFACTOR2][BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2];
#pragma HLS STREAM variable=P2_act type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=P2_act complete dim=1
#pragma HLS ARRAY_PARTITION variable=P2_act complete dim=2
	float P2_act_back[BATCHFACTOR][OUTPUTCHANNELFACTOR2][BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2];
#pragma HLS STREAM variable=P2_act_back type=fifo
#pragma HLS ARRAY_PARTITION variable=P2_act_back complete dim=1
#pragma HLS ARRAY_PARTITION variable=P2_act_back complete dim=2
	int P2_poolindex_back[BATCHFACTOR][OUTPUTCHANNELFACTOR2][BATCHPART][OUTPUTCHANNELPART2][OUTPUTX2][OUTPUTY2][POOLDIM2];
#pragma HLS STREAM variable=P2_poolindex_back type=fifo
#pragma HLS ARRAY_PARTITION variable=P2_poolindex_back complete dim=1
#pragma HLS ARRAY_PARTITION variable=P2_poolindex_back complete dim=2

	float P3_act[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3];
#pragma HLS STREAM variable=P3_act type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=P3_act complete dim=1
#pragma HLS ARRAY_PARTITION variable=P3_act complete dim=2
	float P3_act_back[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3];
#pragma HLS STREAM variable=P3_act_back type=fifo
#pragma HLS ARRAY_PARTITION variable=P3_act_back complete dim=1
#pragma HLS ARRAY_PARTITION variable=P3_act_back complete dim=2
	float P3_nonact_back[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3];
#pragma HLS STREAM variable=P3_nonact_back type=fifo
#pragma HLS ARRAY_PARTITION variable=P3_nonact_back complete dim=1
#pragma HLS ARRAY_PARTITION variable=P3_nonact_back complete dim=2
	float F3_back[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3];
#pragma HLS STREAM variable=F3_back type=fifo
#pragma HLS ARRAY_PARTITION variable=F3_back complete dim=1
#pragma HLS ARRAY_PARTITION variable=F3_back complete dim=2


	float P4_act[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4];
#pragma HLS STREAM variable=P4_act type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=P4_act complete dim=1
#pragma HLS ARRAY_PARTITION variable=P4_act complete dim=2
	float P4_act_back[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4];
#pragma HLS STREAM variable=P4_act_back type=fifo
#pragma HLS ARRAY_PARTITION variable=P4_act_back complete dim=1
#pragma HLS ARRAY_PARTITION variable=P4_act_back complete dim=2
	float P4_nonact_back[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4];
#pragma HLS STREAM variable=P4_nonact_back type=fifo
#pragma HLS ARRAY_PARTITION variable=P4_nonact_back complete dim=1
#pragma HLS ARRAY_PARTITION variable=P4_nonact_back complete dim=2
	float W4_back[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4];
#pragma HLS STREAM variable=W4_back type=fifo
#pragma HLS ARRAY_PARTITION variable=W4_back complete dim=1
#pragma HLS ARRAY_PARTITION variable=W4_back complete dim=2

	float P5_act[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5];
#pragma HLS STREAM variable=P5_act type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=P5_act complete dim=1
#pragma HLS ARRAY_PARTITION variable=P5_act complete dim=2
	float P5_act_back[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5];
#pragma HLS STREAM variable=P5_act_back type=fifo
#pragma HLS ARRAY_PARTITION variable=P5_act_back complete dim=1
#pragma HLS ARRAY_PARTITION variable=P5_act_back complete dim=2
	float W5_back[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5];
#pragma HLS STREAM variable=W5_back type=fifo
#pragma HLS ARRAY_PARTITION variable=W5_back complete dim=1
#pragma HLS ARRAY_PARTITION variable=W5_back complete dim=2



	float dL5[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5];
#pragma HLS STREAM variable=dL5 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dL5 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL5 complete dim=2
	float dL5_back[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5];
#pragma HLS STREAM variable=dL5_back type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dL5_back complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL5_back complete dim=2


	float dL4[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4];
#pragma HLS STREAM variable=dL4 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dL4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL4 complete dim=2
	float dL4_back[BATCHFACTOR][LENGTHFACTOR4][BATCHPART][LENGTHPART4][OUTPUTX4][OUTPUTY4];
#pragma HLS STREAM variable=dL4_back type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dL4_back complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL4_back complete dim=2

	float dL3[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3];
#pragma HLS STREAM variable=dL3 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dL3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL3 complete dim=2
	float dL3_back[BATCHFACTOR][OUTPUTCHANNELFACTOR3][BATCHPART][OUTPUTCHANNELPART3][OUTPUTX3][OUTPUTY3];
#pragma HLS STREAM variable=dL3_back type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dL3_back complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL3_back complete dim=2

	float dL1[BATCHFACTOR][OUTPUTCHANNELFACTOR1][BATCHPART][OUTPUTCHANNELPART1][OUTPUTX1][OUTPUTY1];
#pragma HLS STREAM variable=dL1 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dL1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dL1 complete dim=2

/*	float dF1[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1];
#pragma HLS STREAM variable=dF1 dim=1
#pragma HLS STREAM variable=dF1 dim=2
#pragma HLS ARRAY_PARTITION variable=dF1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dF1 complete dim=2
	float dbF1[FILTERFACTOR1][FILTERPART1];
#pragma HLS STREAM variable=dbF1 dim=1
#pragma HLS ARRAY_PARTITION variable=dbF1 complete dim=1
	float dF3[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3];
#pragma HLS STREAM variable=dF3 dim=1
#pragma HLS STREAM variable=dF3 dim=2
#pragma HLS ARRAY_PARTITION variable=dF3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dF3 complete dim=2
	float dbF3[FILTERFACTOR3][FILTERPART3];
#pragma HLS STREAM variable=dbF3 dim=1
#pragma HLS ARRAY_PARTITION variable=dbF3 complete dim=1
	float dW4[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4];
#pragma HLS STREAM variable=dW4 dim=1
#pragma HLS STREAM variable=dW4 dim=2
#pragma HLS ARRAY_PARTITION variable=dW4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dW4 complete dim=2
	float dbW4[LENGTHFACTOR4][LENGTHPART4];
#pragma HLS STREAM variable=dbW4 dim=1
#pragma HLS ARRAY_PARTITION variable=dbW4 complete dim=1*/


	float I_in2[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1];
#pragma HLS STREAM variable=I_in2 type=fifo
#pragma HLS ARRAY_PARTITION variable=I_in2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=I_in2 complete dim=2
	int y_in[BATCHFACTOR][CLASSFACTOR][BATCHPART][CLASSPART][OUTPUTX5][OUTPUTY5];
#pragma HLS STREAM variable=y_in type=fifo
#pragma HLS ARRAY_PARTITION variable=y_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=y_in complete dim=2
	float POUT_out[BATCHFACTOR][LENGTHFACTOR5][BATCHPART][LENGTHPART5][OUTPUTX5][OUTPUTY5];
#pragma HLS STREAM variable=POUT_out type=fifo
#pragma HLS ARRAY_PARTITION variable=POUT_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=POUT_out complete dim=2
	float F1_in2[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1];
#pragma HLS STREAM variable=F1_in2 type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=F1_in2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=F1_in2 complete dim=2
	float bF1_in[FILTERFACTOR1][FILTERPART1];
#pragma HLS STREAM variable=bF1_in type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=bF1_in complete dim=1
	float F3_in3[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3];
#pragma HLS STREAM variable=F3_in3 type=fifo
#pragma HLS ARRAY_PARTITION variable=F3_in3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=F3_in3 complete dim=2
	float bF3_in[FILTERFACTOR3][FILTERPART3];
#pragma HLS STREAM variable=bF3_in type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=bF3_in complete dim=1
	float W4_in3[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4];
#pragma HLS STREAM variable=W4_in3 type=fifo
#pragma HLS ARRAY_PARTITION variable=W4_in3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=W4_in3 complete dim=2
	float bW4_in[LENGTHFACTOR4][LENGTHPART4];
#pragma HLS STREAM variable=bW4_in type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=bW4_in complete dim=1
	float W5_in3[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5];
#pragma HLS STREAM variable=W5_in3 type=fifo
#pragma HLS ARRAY_PARTITION variable=W5_in3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=W5_in3 complete dim=2
	float bW5_in[LENGTHFACTOR5][LENGTHPART5];
#pragma HLS STREAM variable=bW5_in type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=bW5_in complete dim=1
	float dF1_out[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1];
#pragma HLS STREAM variable=dF1_out type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dF1_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=dF1_out complete dim=2
	float dbF1_out[FILTERFACTOR1][FILTERPART1];
#pragma HLS STREAM variable=dbF1_out type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dbF1_out complete dim=1
	float dF3_out[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3];
#pragma HLS STREAM variable=dF3_out type=fifo
#pragma HLS ARRAY_PARTITION variable=dF3_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=dF3_out complete dim=2
	float dbF3_out[FILTERFACTOR3][FILTERPART3];
#pragma HLS STREAM variable=dbF3_out type=fifo depth=6
#pragma HLS ARRAY_PARTITION variable=dbF3_out complete dim=1
	float dW4_out[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4];
#pragma HLS STREAM variable=dW4_out type=fifo
#pragma HLS ARRAY_PARTITION variable=dW4_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=dW4_out complete dim=2
	float dbW4_out[LENGTHFACTOR4][LENGTHPART4];
#pragma HLS STREAM variable=dbW4_out type=fifo
#pragma HLS ARRAY_PARTITION variable=dbW4_out complete dim=1
	float dW5_out[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5];
#pragma HLS STREAM variable=dW5_out type=fifo
#pragma HLS ARRAY_PARTITION variable=dW5_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=dW5_out complete dim=2
	float dbW5_out[LENGTHFACTOR5][LENGTHPART5];
#pragma HLS STREAM variable=dbW5_out type=fifo
#pragma HLS ARRAY_PARTITION variable=dbW5_out complete dim=1


	/*inFunc(I, y, F1, bF1, F3, bF3, W4, bW4, W5, bW5,
		   I_in, y_in, F1_in, bF1_in, F3_in, bF3_in, W4_in, bW4_in, W5_in, bW5_in);*/

	//std::cout<<"Design Checkpoint 0"<<endl;
	inFunc2(I, y, weight_in, bias_in,
			   I_in2, y_in, F1_in2, bF1_in, F3_in3, bF3_in, W4_in3, bW4_in, W5_in3, bW5_in);

	//std::cout<<"Design Checkpoint 1"<<endl;
	conv2d_1(I_in2, F1_in2, bF1_in, P1_act, I_back, P1_nonact_back);

	//std::cout<<"Design Checkpoint 2"<<endl;
	maxpool2d_2(P1_act, P2_act, P2_act_back, P2_poolindex_back);

	//std::cout<<"Design Checkpoint 3"<<endl;
	conv2d_3(P2_act, F3_in3, bF3_in, P3_act, P3_act_back, P3_nonact_back, F3_back);

	//std::cout<<"Design Checkpoint 4"<<endl;
	dense_4(P3_act, W4_in3, bW4_in, P4_act, P4_act_back, P4_nonact_back, W4_back);

	//std::cout<<"Design Checkpoint 5"<<endl;
	dense_5(P4_act, W5_in3, bW5_in, POUT_out, P5_act_back, W5_back);

	//std::cout<<"Design Checkpoint 6"<<endl;
	dL5_func(P5_act_back, y_in, dL5);

	//std::cout<<"Design Checkpoint 7"<<endl;
	dW5_func(dL5, P4_act_back, dW5_out, dbW5_out, dL5_back);

	//std::cout<<"Design Checkpoint 8"<<endl;
	dL4_func(dL5_back, P4_nonact_back, W5_back, dL4);

	//std::cout<<"Design Checkpoint 9"<<endl;
	dW4_func(dL4, P3_act_back, dW4_out, dbW4_out, dL4_back);

	//std::cout<<"Design Checkpoint 10"<<endl;
	dL3_func(dL4_back, P3_nonact_back, W4_back, dL3);

	//std::cout<<"Design Checkpoint 11"<<endl;
	dF3_func(dL3, P2_act_back, dF3_out, dbF3_out, dL3_back);

	//std::cout<<"Design Checkpoint 12"<<endl;
	dL1_func(dL3_back, P2_poolindex_back, P1_nonact_back, F3_back, dL1);

	//std::cout<<"Design Checkpoint 13"<<endl;
	dF1_func(dL1, I_back, dF1_out, dbF1_out);

	//cout<<"Design Checkpoint 14"<<endl;
	//optimizer(POUT_in, dF1, dbF1, dF3, dbF3, dW4, dbW4, step, POUT, dF1_opt, dbF1_opt, dF3_opt, dbF3_opt, dW4_opt, dbW4_opt);

	/*outFunc(POUT_out, dF1_out, dbF1_out, dF3_out, dbF3_out, dW4_out, dbW4_out, dW5_out, dbW5_out,
			POUT, dF1, dbF1, dF3, dbF3, dW4, dbW4, dW5, dbW5);*/

	//std::cout<<"Design Checkpoint 14"<<endl;
	outFunc2(POUT_out, dF1_out, dbF1_out, dF3_out, dbF3_out, dW4_out, dbW4_out, dW5_out, dbW5_out,
				POUT, weight_out, bias_out);


}
