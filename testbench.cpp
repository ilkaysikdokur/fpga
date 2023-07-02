#include "design.hpp"
#include <random>
#include <algorithm>





double random_double(std::mt19937 &e2){
    int a = e2() >> 5;
    int b = e2() >> 6;
    double value = (a * 67108864.0 + b) / 9007199254740992.0;

    return value;
}

double gauss = 0;
int has_gauss = 0;

double random_normal(std::mt19937 &e2, double loc, double scale){

    double val;

    if (has_gauss) {
        const double tmp = gauss;
        gauss = 0;
        has_gauss = 0;
        val = tmp;
    }
    else {
        double f, x1, x2, r2;

        do {
            x1 = 2.0*random_double(e2) - 1.0;
            x2 = 2.0*random_double(e2) - 1.0;
            r2 = x1*x1 + x2*x2;
        }
        while (r2 >= 1.0 || r2 == 0.0);


        f = sqrt(-2.0*log(r2)/r2);

        gauss = f*x1;
        has_gauss = 1;
        val = f*x2;
    }


    return loc + scale*val;
}




int main(){

	int i, j, k, l, m, n;

	

	//cout<<"Testbench Checkpoint 1"<<endl;


		//std::random_device rd;
	//std::mt19937 gen(rd());
	std::mt19937 gen(2022);
	//std::normal_distribution<double> d(0, 0.1);


	//float weight_in[LAYERAMT][WEIGHT_MAXD1][WEIGHT_MAXD2][WEIGHT_MAXD3][WEIGHT_MAXD4][WEIGHT_MAXD5][WEIGHT_MAXD6];
	//float bias_in[LAYERAMT][BIAS_MAXD1][BIAS_MAXD2];

	float weight_in[LAYERAMT][WEIGHTMAXLEN];
	float bias_in[LAYERAMT][BIASMAXLEN];
	int idx;

	float F1[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1];
	float bF1[FILTERFACTOR1][FILTERPART1];
	float F3[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3];
	float bF3[FILTERFACTOR3][FILTERPART3];
	float W4[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4];
	float bW4[LENGTHFACTOR4][LENGTHPART4];
	float W5[INPUTLENGTHFACTOR5][LENGTHFACTOR5][INPUTLENGTHPART5][INPUTX5][INPUTY5][LENGTHPART5];
	float bW5[LENGTHFACTOR5][LENGTHPART5];


	idx = 0;

	for (k=0 ; k<FILTERPART1 ; ++k){
		for (l=0 ; l<INPUTCHANNELPART1 ; ++l){
			for (m=0 ; m<KERNELX1 ; ++m){
				for (n=0 ; n<KERNELY1 ; ++n){
						for (i=0 ; i<FILTERFACTOR1 ; ++i){
							for (j=0 ; j<INPUTCHANNELFACTOR1 ; ++j){
							F1[i][j][k][l][m][n] = random_normal(gen, 0, 0.1);//d(gen);
							weight_in[0][idx] = F1[i][j][k][l][m][n];
							idx += 1;
						}
					}
				}
			}
		}
	}

	idx = 0;
	
	for (j=0 ; j<FILTERPART1 ; ++j){
		for (i=0 ; i<FILTERFACTOR1 ; ++i){
			bF1[i][j] = random_normal(gen, 0, 0.1);//d(gen);
			bias_in[0][idx] = bF1[i][j];
			idx += 1;
		}
	}

	idx = 0;
	
	for (k=0 ; k<FILTERPART3 ; ++k){
		for (l=0 ; l<INPUTCHANNELPART3 ; ++l){
			for (m=0 ; m<KERNELX3 ; ++m){
				for (n=0 ; n<KERNELY3 ; ++n){
					for (i=0 ; i<FILTERFACTOR3 ; ++i){
						for (j=0 ; j<INPUTCHANNELFACTOR3 ; ++j){
							F3[i][j][k][l][m][n] = random_normal(gen, 0, 0.1);//d(gen);
							weight_in[1][idx] = F3[i][j][k][l][m][n];
							idx += 1;
						}
					}
				}
			}
		}
	}

	idx = 0;
	
	for (j=0 ; j<FILTERPART3 ; ++j){
		for (i=0 ; i<FILTERFACTOR3 ; ++i){
			bF3[i][j] = random_normal(gen, 0, 0.1);//d(gen);
			bias_in[1][idx] = bF3[i][j];
			idx += 1;
		}
	}

	idx = 0;
	
	for (k=0 ; k<INPUTCHANNELPART4 ; ++k){
		for (l=0 ; l<INPUTX4 ; ++l){
			for (m=0 ; m<INPUTY4 ; ++m){
				for (n=0 ; n<LENGTHPART4 ; ++n){
					for (i=0 ; i<INPUTCHANNELFACTOR4 ; ++i){
						for (j=0 ; j<LENGTHFACTOR4 ; ++j){
							W4[i][j][k][l][m][n] = random_normal(gen, 0, 0.1);//d(gen);
							weight_in[2][idx] = W4[i][j][k][l][m][n];
							idx += 1;
						}
					}
				}
			}
		}
	}

	idx = 0;
	for (j=0 ; j<LENGTHPART4 ; ++j){
		for (i=0 ; i<LENGTHFACTOR4 ; ++i){
			bW4[i][j] = random_normal(gen, 0, 0.1);//d(gen);
			bias_in[2][idx] = bW4[i][j];
			idx += 1;
		}
	}

	idx = 0;
	
	for (k=0 ; k<INPUTLENGTHPART5 ; ++k){
		for (l=0 ; l<INPUTX5 ; ++l){
			for (m=0 ; m<INPUTY5 ; ++m){
				for (n=0 ; n<LENGTHPART5 ; ++n){
					for (i=0 ; i<INPUTLENGTHFACTOR5 ; ++i){
						for (j=0 ; j<LENGTHFACTOR5 ; ++j){
							W5[i][j][k][l][m][n] = random_normal(gen, 0, 0.1);//d(gen);
							weight_in[3][idx] = W5[i][j][k][l][m][n];
							idx += 1;
						}
					}
				}
			}
		}
	}

	idx = 0;
	
	for (j=0 ; j<LENGTHPART5 ; ++j){
		for (i=0 ; i<LENGTHFACTOR5 ; ++i){
			bW5[i][j] = random_normal(gen, 0, 0.1);//d(gen);
			bias_in[3][idx] = bW5[i][j];
			idx += 1;
		}
	}

	/*cout<<"F1["<<0<<"]["<<0<<"]["<<0<<"]["<<0<<"]["<<0<<"]["<<0<<"]: "<<F1[0][0][0][1][0][2]<<endl;
	cout<<"F3["<<0<<"]["<<0<<"]["<<0<<"]["<<0<<"]["<<0<<"]["<<0<<"]: "<<F3[0][0][1][0][1][1]<<endl;
	cout<<"W4["<<0<<"]["<<0<<"]["<<0<<"]["<<0<<"]["<<0<<"]["<<0<<"]: "<<W4[0][0][0][0][6][210]<<endl;
	cout<<"W5["<<0<<"]["<<0<<"]["<<0<<"]["<<0<<"]["<<0<<"]["<<0<<"]: "<<W5[0][0][126][0][0][5]<<endl;*/


	//cout<<"Testbench Checkpoint 2"<<endl;

	int epochIndex, nIndex;

	float I[BATCHFACTOR][INPUTCHANNELFACTOR1][BATCHPART][INPUTCHANNELPART1][INPUTX1][INPUTY1];
	int y[BATCHFACTOR][CLASSFACTOR][BATCHPART][CLASSPART][OUTPUTX5][OUTPUTY5];
	float POUT[BATCHFACTOR][CLASSFACTOR][BATCHPART][CLASSPART][OUTPUTX5][OUTPUTY5];

	float dF1_opt[FILTERFACTOR1][INPUTCHANNELFACTOR1][FILTERPART1][INPUTCHANNELPART1][KERNELX1][KERNELY1];
	float dbF1_opt[FILTERFACTOR1][FILTERPART1];
	float dF3_opt[FILTERFACTOR3][INPUTCHANNELFACTOR3][FILTERPART3][INPUTCHANNELPART3][KERNELX3][KERNELY3];
	float dbF3_opt[FILTERFACTOR3][FILTERPART3];
	float dW4_opt[INPUTCHANNELFACTOR4][LENGTHFACTOR4][INPUTCHANNELPART4][INPUTX4][INPUTY4][LENGTHPART4];
	float dbW4_opt[LENGTHFACTOR4][LENGTHPART4];
	float dW5_opt[LENGTHFACTOR4][LENGTHFACTOR5][LENGTHPART4][INPUTX5][INPUTY5][LENGTHPART5];
	float dbW5_opt[LENGTHFACTOR5][LENGTHPART5];

	/*float weight_out[LAYERAMT][WEIGHT_MAXD1][WEIGHT_MAXD2][WEIGHT_MAXD3][WEIGHT_MAXD4][WEIGHT_MAXD5][WEIGHT_MAXD6];
	float bias_out[LAYERAMT][BIAS_MAXD1][BIAS_MAXD2];*/

	float weight_out[LAYERAMT][WEIGHTMAXLEN];
	float bias_out[LAYERAMT][BIASMAXLEN];


	char imageRow[1+4*INPUTX1*INPUTY1*INPUTCHANNELFACTOR1*INPUTCHANNELPART1];

	long rowBeginningPositions[NTRAIN] = {0};

	//cout<<"Testbench Checkpoint 2_1"<<endl;

	FILE* f_train = fopen("cifar10_train.csv", "r+");

	//cout<<"Testbench Checkpoint 2_2"<<endl;

	for (nIndex = 1; nIndex <= NTRAIN; ++nIndex) {
		rowBeginningPositions[nIndex - 1] = ftell(f_train);
		fscanf(f_train, "%s\n", imageRow);
	}

	//cout<<"Testbench Checkpoint 2_3"<<endl;

	fclose(f_train);

	//cout<<"Testbench Checkpoint 3"<<endl;



	//Control var for checking delta values
	bool mismatch = false;

	for (epochIndex = 0; epochIndex < EPOCH ; ++epochIndex){


		FILE* f_train = fopen("cifar10_train.csv","r+");

		vector<int> indices(NTRAIN);
		for (i = 0; i < NTRAIN; ++i) //indices[i] = i;
		//shuffle(indices.begin(), indices.end(), default_random_engine(time(NULL)));
		shuffle(indices.begin(), indices.end(), default_random_engine(2022));

		for (nIndex = 1; nIndex <= NTRAIN ; ++nIndex){
			//fseek(f_train, rowBeginningPositions[indices[nIndex-1]], SEEK_SET);

			fscanf(f_train, "%s\n", imageRow);
			char* token;
			int ind = 0;
			token = strtok(imageRow, ",");
			for (i=0 ; i<CLASS ; ++i){
				y[((nIndex-1)/(BATCHPART))%BATCHFACTOR][i/CLASSPART][(nIndex-1)%(BATCHPART)][i%CLASSPART][0][0] = 0;
			}
			y[((nIndex-1)/(BATCHPART))%BATCHFACTOR][atoi(token)/CLASSPART][(nIndex-1)%(BATCHPART)][atoi(token)%CLASSPART][0][0] = 1;
			token = strtok(NULL, ",");
			while (token != NULL){
				I[((nIndex-1)/(BATCHPART))%BATCHFACTOR][(ind/(INPUTX1*INPUTY1))/(INPUTCHANNELPART1)][(nIndex-1)%(BATCHPART)][(ind/(INPUTX1*INPUTY1))%(INPUTCHANNELPART1)][(ind/INPUTX1)%INPUTY1][ind%INPUTY1] = atoi(token)/255.0;
				//cout<<"I["<<((nIndex-1)/(BATCHPART))%BATCHFACTOR<<"]["<<(ind/(INPUTX1*INPUTY1))/(INPUTCHANNELPART1)<<"]["<<(nIndex-1)%(BATCHPART)<<"]["<<(ind/(INPUTX1*INPUTY1))%(INPUTCHANNELPART1)<<"]["<<(ind/INPUTX1)%INPUTY1<<"]["<<ind%INPUTY1<<"]: "<<I[((nIndex-1)/(BATCHPART))%BATCHFACTOR][(ind/(INPUTX1*INPUTY1))/(INPUTCHANNELPART1)][(nIndex-1)%(BATCHPART)][(ind/(INPUTX1*INPUTY1))%(INPUTCHANNELPART1)][(ind/INPUTX1)%INPUTY1][ind%INPUTY1]<<endl;
				ind += 1;
				token = strtok(NULL, ",");
			}



			if (nIndex%BATCHSIZE != 0)
				continue;


			int step = (nIndex/BATCHSIZE)+1;

			//cout<<"Testbench Checkpoint 4"<<endl;

			//train(I, cls);
			//gradient(I, cls, POUT, dL_dW5);
			/*train(I, y, POUT,
					F1, bF1,
					F3, bF3,
					W4, bW4,
					W5, bW5,
					//step,
					dF1_opt, dbF1_opt,
					dF3_opt, dbF3_opt,
					dW4_opt, dbW4_opt,
					dW5_opt, dbW5_opt);*/

			train(I, y, POUT,
					weight_in, bias_in,
					//step,
					weight_out, bias_out);


			char d_value_str[20];
			float d_value;
			FILE* f_d_org;

			f_d_org = fopen("dF1_org.csv", "r+");

			idx = 0;
			for (k=0 ; k<FILTERPART1 ; ++k){
				for (l=0 ; l<INPUTCHANNELPART1 ; ++l){
					for (m=0 ; m<KERNELX1 ; ++m){
						for (n=0 ; n<KERNELY1 ; ++n){
							for (i=0 ; i<FILTERFACTOR1 ; ++i){
								for (j=0 ; j<INPUTCHANNELFACTOR1 ; ++j){
									fscanf(f_d_org, "%s\n", d_value_str);
									d_value = atof(d_value_str)*BATCHSIZE;
									dF1_opt[i][j][k][l][m][n] = weight_out[0][idx];
									idx += 1;
									if (abs(d_value-dF1_opt[i][j][k][l][m][n])>0.1){
										cout<<"dF1["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]["<<m<<"]["<<n<<"] Real Value: "<<d_value<<", Calculated Value: "<<dF1_opt[i][j][k][l][m][n]<<endl;
										mismatch = true;
									}
								}
							}
						}
					}
				}
			}

			fclose(f_d_org);

			f_d_org = fopen("dF3_org.csv", "r+");

			idx = 0;
			for (k=0 ; k<FILTERPART3 ; ++k){
				for (l=0 ; l<INPUTCHANNELPART3 ; ++l){
					for (m=0 ; m<KERNELX3 ; ++m){
						for (n=0 ; n<KERNELY3 ; ++n){
							for (i=0 ; i<FILTERFACTOR3 ; ++i){
								for (j=0 ; j<INPUTCHANNELFACTOR3 ; ++j){
									fscanf(f_d_org, "%s\n", d_value_str);
									d_value = atof(d_value_str)*BATCHSIZE;
									dF3_opt[i][j][k][l][m][n] = weight_out[1][idx];
									idx += 1;
									if (abs(d_value-dF3_opt[i][j][k][l][m][n])>0.1){
										cout<<"dF3["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]["<<m<<"]["<<n<<"] Real Value: "<<d_value<<", Calculated Value: "<<dF3_opt[i][j][k][l][m][n]<<endl;
										mismatch = true;
									}
								}
							}
						}
					}
				}
			}

			fclose(f_d_org);

			f_d_org = fopen("dW4_org.csv", "r+");

			idx = 0;
			for (k=0 ; k<INPUTCHANNELPART4 ; ++k){
				for (l=0 ; l<INPUTX4 ; ++l){
					for (m=0 ; m<INPUTY4 ; ++m){
						for (n=0 ; n<LENGTHPART4 ; ++n){
							for (i=0 ; i<INPUTCHANNELFACTOR4 ; ++i){
								for (j=0 ; j<LENGTHFACTOR4 ; ++j){
									fscanf(f_d_org, "%s\n", d_value_str);
									d_value = atof(d_value_str)*BATCHSIZE;
									dW4_opt[i][j][k][l][m][n] = weight_out[2][idx];
									idx += 1;
									if (abs(d_value-dW4_opt[i][j][k][l][m][n])>0.1){
										cout<<"dW4["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]["<<m<<"]["<<n<<"] Real Value: "<<d_value<<", Calculated Value: "<<dW4_opt[i][j][k][l][m][n]<<endl;
										mismatch = true;
									}
								}
							}
						}
					}
				}
			}

			fclose(f_d_org);


			f_d_org = fopen("dW5_org.csv", "r+");

			idx = 0;
			for (k=0 ; k<LENGTHPART4 ; ++k){
				for (l=0 ; l<INPUTX5 ; ++l){
					for (m=0 ; m<INPUTY5 ; ++m){
						for (n=0 ; n<LENGTHPART5 ; ++n){
							for (i=0 ; i<LENGTHFACTOR4 ; ++i){
								for (j=0 ; j<LENGTHFACTOR5 ; ++j){
									fscanf(f_d_org, "%s\n", d_value_str);
									d_value = atof(d_value_str)*BATCHSIZE;
									dW5_opt[i][j][k][l][m][n] = weight_out[3][idx];
									idx += 1;
									if (abs(d_value-dW5_opt[i][j][k][l][m][n])>0.1){
										cout<<"dW5["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"]["<<m<<"]["<<n<<"] Real Value: "<<d_value<<", Calculated Value: "<<dW5_opt[i][j][k][l][m][n]<<endl;
										mismatch = true;
									}
								}
							}
						}
					}
				}
			}

			fclose(f_d_org);



			f_d_org = fopen("dbF1_org.csv", "r+");

			idx = 0;
			for (j=0 ; j<FILTERPART1 ; ++j){
				for (i=0 ; i<FILTERFACTOR1 ; ++i){
					fscanf(f_d_org, "%s\n", d_value_str);
					d_value = atof(d_value_str)*BATCHSIZE;
					dbF1_opt[i][j] = bias_out[0][idx];
					idx += 1;
					if (abs(d_value-dbF1_opt[i][j])>0.1){
						cout<<"dbF1["<<i<<"]["<<j<<"] Real Value: "<<d_value<<", Calculated Value: "<<dbF1_opt[i][j]<<endl;
						mismatch = true;
					}
				}
			}

			fclose(f_d_org);

			f_d_org = fopen("dbF3_org.csv", "r+");

			idx = 0;
			for (j=0 ; j<FILTERPART3 ; ++j){
				for (i=0 ; i<FILTERFACTOR3 ; ++i){
					fscanf(f_d_org, "%s\n", d_value_str);
					d_value = atof(d_value_str)*BATCHSIZE;
					dbF3_opt[i][j] = bias_out[1][idx];
					idx += 1;
					if (abs(d_value-dbF3_opt[i][j])>0.1){
						cout<<"dbF3["<<i<<"]["<<j<<"] Real Value: "<<d_value<<", Calculated Value: "<<dbF3_opt[i][j]<<endl;
						mismatch = true;
					}
				}
			}

			fclose(f_d_org);

			f_d_org = fopen("dbW4_org.csv", "r+");

			idx = 0;
			for (j=0 ; j<LENGTHPART4 ; ++j){
				for (i=0 ; i<LENGTHFACTOR4 ; ++i){
					fscanf(f_d_org, "%s\n", d_value_str);
					d_value = atof(d_value_str)*BATCHSIZE;
					dbW4_opt[i][j] = bias_out[2][idx];
					idx += 1;
					if (abs(d_value-dbW4_opt[i][j])>0.1){
						cout<<"dbW4["<<i<<"]["<<j<<"] Real Value: "<<d_value<<", Calculated Value: "<<dbW4_opt[i][j]<<endl;
						mismatch = true;
					}
				}
			}

			fclose(f_d_org);

			f_d_org = fopen("dbW5_org.csv", "r+");

			idx = 0;
			for (j=0 ; j<LENGTHPART5 ; ++j){
				for (i=0 ; i<LENGTHFACTOR5 ; ++i){
					fscanf(f_d_org, "%s\n", d_value_str);
					d_value = atof(d_value_str)*BATCHSIZE;
					dbW5_opt[i][j] = bias_out[3][idx];
					idx += 1;
					if (abs(d_value-dbW5_opt[i][j])>0.1){
						cout<<"dbW5["<<i<<"]["<<j<<"] Real Value: "<<d_value<<", Calculated Value: "<<dbW5_opt[i][j]<<endl;
						mismatch = true;
					}
				}
			}

			fclose(f_d_org);


			//cout<<"Testbench Checkpoint 5"<<endl;

			//cout<<"F1[0][0][0][0][0][0] = "<<F1[0][0][0][0][0][0]<<endl;
			//cout<<"F3[0][0][0][0][0][0] = "<<F3[0][0][0][0][0][0]<<endl;
			//cout<<"W4[0][0][0][0][0][0] = "<<W4[0][0][0][0][0][0]<<endl;
			/*cout<<"POUT[0][0][0][0] = "<<POUT[0][0][0][0]<<endl;
			cout<<"POUT[1][0][0][7] = "<<POUT[1][0][0][7]<<endl;
			cout<<"dF1_opt[0][0][0][0][0][0] = "<<dF1_opt[0][0][0][0][0][0]<<endl;
			cout<<"dF3_opt[0][0][0][0][0][0] = "<<dF3_opt[0][0][0][0][0][0]<<endl;
			cout<<"dW4_opt[0][0][0][0][0][0] = "<<dW4_opt[0][0][0][0][0][0]<<endl;
			cout<<endl;*/

			//-----------------------------------------------------------------------
			//UPDATE
			for (i=0 ; i<FILTERFACTOR1 ; ++i){
				for (j=0 ; j<INPUTCHANNELFACTOR1 ; ++j){
					for (k=0 ; k<FILTERPART1 ; ++k){
						for (l=0 ; l<INPUTCHANNELPART1 ; ++l){
							for (m=0 ; m<KERNELX1 ; ++m){
								for (n=0 ; n<KERNELY1 ; ++n){
									F1[i][j][k][l][m][n] -= ETA*dF1_opt[i][j][k][l][m][n]/BATCHSIZE;
								}
							}
						}
					}
				}
			}

			for (i=0 ; i<FILTERFACTOR1 ; ++i){
				for (j=0 ; j<FILTERPART1 ; ++j){
					bF1[i][j] -= ETA*dbF1_opt[i][j]/BATCHSIZE;
				}
			}

			for (i=0 ; i<FILTERFACTOR3 ; ++i){
				for (j=0 ; j<INPUTCHANNELFACTOR3 ; ++j){
					for (k=0 ; k<FILTERPART3 ; ++k){
						for (l=0 ; l<INPUTCHANNELPART3 ; ++l){
							for (m=0 ; m<KERNELX3 ; ++m){
								for (n=0 ; n<KERNELY3 ; ++n){
									F3[i][j][k][l][m][n] -= ETA*dF3_opt[i][j][k][l][m][n]/BATCHSIZE;
								}
							}
						}
					}
				}
			}

			for (i=0 ; i<FILTERFACTOR3 ; ++i){
				for (j=0 ; j<FILTERPART3 ; ++j){
					bF3[i][j] -= ETA*dbF3_opt[i][j]/BATCHSIZE;
				}
			}

			for (i=0 ; i<INPUTCHANNELFACTOR4 ; ++i){
				for (j=0 ; j<LENGTHFACTOR4 ; ++j){
					for (k=0 ; k<INPUTCHANNELPART4 ; ++k){
						for (l=0 ; l<INPUTX4 ; ++l){
							for (m=0 ; m<INPUTY4 ; ++m){
								for (n=0 ; n<LENGTHPART4 ; ++n){
									W4[i][j][k][l][m][n] -= ETA*dW4_opt[i][j][k][l][m][n]/BATCHSIZE;
								}
							}
						}
					}
				}
			}

			for (i=0 ; i<LENGTHFACTOR4 ; ++i){
				for (j=0 ; j<LENGTHPART4 ; ++j){
					bW4[i][j] -= ETA*dbW4_opt[i][j]/BATCHSIZE;
				}
			}
			//-----------------------------------------------------------------------
			//UPDATE

			if (nIndex % 1000 == 0){
				printf("Epoch: %d, nIndex: %d\n", epochIndex+1, nIndex);
			}
			//printf("Epoch: %d, nIndex: %d\n", epochIndex+1, nIndex);
		}

		if (mismatch) {cout<<"There is a mismatch between calculated gradients and original gradients!"<<endl; return 1;}
		else {cout<<"Calculated and original gradients completely match!"<<endl; return 0;}


		//cout<<"F1[0][0][0][0][0][0] = "<<F1[0][0][0][0][0][0]<<endl;
		//cout<<"F3[0][0][0][0][0][0] = "<<F3[0][0][0][0][0][0]<<endl;
		//cout<<"W4[0][0][0][0][0][0] = "<<W4[0][0][0][0][0][0]<<endl;

		//Testing
		/*printf("TESTING\n");
		double trueCnt = 0.0, falseCnt = 0.0;


		FILE* f_test = fopen("cifar10_test.csv", "r+");

		for (nIndex = 1; nIndex <= NTEST ; ++nIndex){
			fscanf(f_test, "%s\n", imageRow);
			char* token;
			int ind = 0;
			token = strtok(imageRow, ",");
			for (i=0 ; i<CLASS ; ++i){
				y[((nIndex-1)/(BATCHPART))%BATCHFACTOR][i/CLASSPART][(nIndex-1)%(BATCHPART)][i%CLASSPART] = 0;
			}
			y[((nIndex-1)/(BATCHPART))%BATCHFACTOR][atoi(token)/CLASSPART][(nIndex-1)%(BATCHPART)][atoi(token)%CLASSPART] = 1;
			//cout<<"Class: "<<atoi(token)<<endl;
			//cout<<"yIndex: "<<((nIndex-1)/(BATCHPART))%BATCHFACTOR<<", "<<atoi(token)/CLASSPART<<", "<<(nIndex-1)%(BATCHPART)<<", "<<atoi(token)%CLASSPART<<endl;
			token = strtok(NULL, ",");
			while (token != NULL){
				I[((nIndex-1)/(BATCHPART))%BATCHFACTOR][(ind/(INPUTX1*INPUTY1))/(INPUTCHANNELPART1)][(nIndex-1)%(BATCHPART)][(ind/(INPUTX1*INPUTY1))%(INPUTCHANNELPART1)][(ind/INPUTX1)%INPUTY1][ind%INPUTY1] = atoi(token)/255.0;
				//cout<<nIndex-1<<", "<<((nIndex-1)/(BATCHPART))%BATCHFACTOR<<","<<(nIndex-1)%(BATCHPART)<<endl;
				ind += 1;
				token = strtok(NULL, ",");
			}

			if (nIndex%BATCHSIZE != 0)
				continue;

			//cout<<I[0][0][0][0][0]<<endl;
			//cout<<I[1][0][0][0][0]<<endl<<endl;

			int step = (nIndex/BATCHSIZE)+1;

			//train(I, cls);
			//gradient(I, cls, POUT, dL_dW5);
			train(I, y, POUT,
					F1, bF1,
					F3, bF3,
					W4, bW4,
					//step,
					dF1_opt, dbF1_opt,
					dF3_opt, dbF3_opt,
					dW4_opt, dbW4_opt);

			for (i=0 ; i<BATCHSIZE ; ++i){
				int classImage = 0;
				double val = -1.0;
				int valInd = -1;
				for (j=0 ; j<CLASS ; ++j){
					//cout<<POUT[i%(BATCHFACTOR)][j/(CLASSPART)][i/(BATCHFACTOR)][j%(CLASSPART)]<<" ";
					if (POUT[i%(BATCHFACTOR)][j/(CLASSPART)][i/(BATCHFACTOR)][j%(CLASSPART)]>=val){
						val = POUT[i%(BATCHFACTOR)][j/(CLASSPART)][i/(BATCHFACTOR)][j%(CLASSPART)];
						valInd = j;
					}
					//cout<<y[i/(BATCHPART)][j/(CLASSPART)][i%(BATCHFACTOR)][j%(CLASSPART)]<<" ";
					//cout<<" | "<<i/(BATCHPART)<<", "<<j/(CLASSPART)<<", "<<i%(BATCHFACTOR)<<", "<<j%(CLASSPART)<<" | ";
					if (y[i%(BATCHFACTOR)][j/(CLASSPART)][i/(BATCHFACTOR)][j%(CLASSPART)] == 1){
						classImage = j;
					}
				}
				//cout<<endl;

				//cout<<"classImage = "<<classImage<<endl;

				//int classImage = y[i%BATCHFACTOR][j/CLASSPART][i/BATCHFACTOR][j%CLASSPART];

				if (valInd == classImage){
					trueCnt += 1;
				}
				else
					falseCnt += 1;


			}

		}

		printf("Accuracy: %f\n\n", (1.0*trueCnt)/(trueCnt + falseCnt));


		fclose(f_test);*/
		//fclose(f_train);

	}



	return 0;

}

