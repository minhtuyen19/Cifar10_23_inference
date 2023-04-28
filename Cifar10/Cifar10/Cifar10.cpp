#include<iostream>
#include<fstream>
#include<string>
#include<stdio.h>
#include<math.h>

using namespace std;
void readfile(const char* filename, float a[], int& n) {
	ifstream input(filename);
	float k;
	n = 0;
	while (input >> k) {
		a[n++] = k;
	}
	input.close();
}
void writefile1d(const char* filename, float a[], int n) {
	ofstream output(filename);
	for (int i = 0; i < n; i++) {
		output << a[i] << " ";
	}
	output << endl;
}
void xuat(float a[], int n) {
	for (int i = 0; i < n; i++) {
		printf("a[%d] = %0.15f ", i, a[i]);
	}
	printf("\n");
}
void relu(float* input, int n, float* out) {
    for (int i = 0; i < n; i++) {
        out[i] = input[i] > 0 ? input[i] : 0;
    }
}
void conv2d(float* input, float* w, int row, int col, int frow, int fcol, int channel_inputs,
    int numofkernel, float* conv_results, float* bias) {
    int out_row = row - frow + 1;
    int out_col = col - fcol + 1;
    int s = 0;

    // Allocate memory for kernel_out
    float**** kernel_out = new float*** [numofkernel];
    for (int k = 0; k < numofkernel; k++) {
        kernel_out[k] = new float** [channel_inputs];
        for (int l = 0; l < channel_inputs; l++) { // fixed typo in loop condition
            kernel_out[k][l] = new float* [frow];
            for (int i = 0; i < frow; i++) {
                kernel_out[k][l][i] = new float[fcol];
                for (int j = 0; j < fcol; j++) {
                    int kernel_idx = k * channel_inputs * frow * fcol + l * frow * fcol + i * fcol + j;
                    kernel_out[k][l][i][j] = w[kernel_idx];
                }
            }
        }
    }

    for (int k = 0; k < numofkernel; k++) {
        for (int i = 0; i < out_row; i++) {
            for (int j = 0; j < out_col; j++) {
                float sum = 0.0;
                for (int l = 0; l < channel_inputs; l++) {
                    for (int m = 0; m < frow; m++) {
                        for (int n = 0; n < fcol; n++) {
                            int input_idx = l * row * col + (i + m) * col + j + n;
                            int kernel_idx = k * channel_inputs * frow * fcol + l * frow * fcol + m * fcol + n;
                             sum += input[input_idx] * kernel_out[k][l][m][n];
                        }
                    }
                }
                conv_results[s++] = sum + bias[k];
            }
        }
    }

    // Free memory for kernel_out
    for (int k = 0; k < numofkernel; k++) {
        for (int l = 0; l < channel_inputs; l++) {
            for (int i = 0; i < frow; i++) {
                delete[] kernel_out[k][l][i];
            }
            delete[] kernel_out[k][l];
        }
        delete[] kernel_out[k];
    }
    delete[] kernel_out;
}

//void conv2d(float* input, float* w, int row, int col, int frow, int fcol, int n_input, int numofkernel, float* conv_results, float* bias, int padding) {
//    int channel_inputs = n_input;
//    int out_row =  row - frow + 1 + 2 * padding;
//    int out_col = col - fcol + 1 + 2 * padding;
//    // Allocate memory for kernel_out
//    float**** kernel_out = new float*** [channel_inputs];
//    for (int l = 0; l < channel_inputs; l++) {
//        kernel_out[l] = new float** [numofkernel];
//        for (int k = 0; k < numofkernel; k++) {
//            kernel_out[l][k] = new float* [frow];
//            for (int i = 0; i < frow; i++) {
//                kernel_out[l][k][i] = new float[fcol];
//                for (int j = 0; j < fcol; j++) {
//                    int kernel_idx = l * numofkernel * frow * fcol + k * frow * fcol + i * fcol + j;
//                    kernel_out[l][k][i][j] = w[kernel_idx];
//                }
//            }
//        }
//    }
//
//    for (int k = 0; k < numofkernel; k++) {
//        for (int i = -padding; i < out_row - padding; i++) {
//            for (int j = -padding; j < out_col - padding; j++) {
//                float sum = 0;
//                for (int l = 0; l < channel_inputs; l++) {
//                    for (int m = 0; m < frow; m++) {
//                        for (int n = 0; n < fcol; n++) {
//                            int input_idx = l * row * col + (i + m) * col + j + n;
//                            int kernel_idx = l * numofkernel * frow * fcol + k * frow * fcol + m * fcol + n;
//                            if (i + m >= 0 && i + m < row && j + n >= 0 && j + n < col) {
//                                sum += input[input_idx] * kernel_out[l][k][m][n];
//                            }
//                        }
//                    }
//                }
//                int h = k * out_row * out_col + (i + padding) * out_col + j + padding;
//                conv_results[k * out_row * out_col + (i + padding) * out_col + j + padding] = sum + bias[k];
//            }
//        }
//    }
//    //for (int i = 0; i < (row - frow + 1) * (col - fcol + 1) * numofkernel; i++) {
//    //  printf("out_conv2d[%d] = %0.10f ", i, conv_results[i]);
//    //}
//    // Free memory for kernel_out
//    for (int l = 0; l < channel_inputs; l++) {
//        for (int k = 0; k < numofkernel; k++) {
//            for (int i = 0; i < frow; i++) {
//                delete[] kernel_out[l][k][i];
//            }
//            delete[] kernel_out[l][k];
//        }
//        delete[] kernel_out[l];
//    }
//    delete[] kernel_out;
//}

//void conv2d(float* input, float* w, int row, int col, int frow, int fcol, int n_input, int numofkernel, float* conv_results, float* bias) {
//    int channel_inputs = n_input;
//    int out_row = row - frow + 1;
//    int out_col = col - fcol + 1;
//    int s = 0;
//    // Allocate memory for kernel_out
//    float**** kernel_out = new float*** [channel_inputs];
//    for (int l = 0; l < channel_inputs; l++) {
//        kernel_out[l] = new float** [numofkernel];
//        for (int k = 0; k < numofkernel; k++) {
//            kernel_out[l][k] = new float* [frow];
//            for (int i = 0; i < frow; i++) {
//                kernel_out[l][k][i] = new float[fcol];
//                for (int j = 0; j < fcol; j++) {
//                    kernel_out[l][k][i][j] = w[s++];
//                }
//            }
//        }
//    }
//
//    for (int k = 0; k < numofkernel; k++) {
//        for (int i = 0; i < out_row; i++) {
//            for (int j = 0; j < out_col; j++) {
//                float sum = 0;
//                for (int l = 0; l < channel_inputs; l++) {
//                    for (int m = 0; m < frow; m++) {
//                        for (int n = 0; n < fcol; n++) {
//                            int input_idx = l * row * col + (i + m) * col + j + n;
//                            int kernel_idx = l * channel_inputs * frow * fcol + k * frow * fcol + m * fcol + n;
//                                 sum += input[input_idx] * kernel_out[l][k][m][n];
//                            //printf("kernel_out[%d][%d][%d][%d] = %0.10f ", l, k, m, n, kernel_out[l][k][m][n]);
//                        }
//                    }
//                }
//                int h = k * out_row * out_col + i * out_col + j;
//                conv_results[k * out_row * out_col + i * out_col + j] = sum + bias[k];
//            }
//        }
//    }
//    //for (int i = 0; i < (row - frow + 1) * (col - fcol + 1) * numofkernel; i++) {
//    //  printf("out_conv2d[%d] = %0.10f ", i, conv_results[i]);
//    //}
//    // Free memory for kernel_out
//    for (int l = 0; l < channel_inputs; l++) {
//        for (int k = 0; k < numofkernel; k++) {
//            for (int i = 0; i < frow; i++) {
//                delete[] kernel_out[l][k][i];
//            }
//            delete[] kernel_out[l][k];
//        }
//        delete[] kernel_out[l];
//    }
//    delete[] kernel_out;
//}

void main() {
    float* input= (float*)malloc((32*32*3) * sizeof(float));
	int n_input;
	readfile("ship.txt", input, n_input);

	//===================Conv2d_0=======================
	float w_0[864];
	float bias_0[32];
	int nw_0;
	int nb_0;
    float* out_0 = (float*)malloc((30*30*32) * sizeof(float));
    float* out_relu_0 = (float*)malloc((30 * 30 * 32) * sizeof(float));

	readfile("W_0.txt", w_0, nw_0);
	readfile("bias_0.txt", bias_0, nb_0);
	conv2d(input, w_0, 32, 32, 3, 3, 3, 32, out_0, bias_0);
	//xuat(out_0, 30 * 30 * 32);
    relu(out_0, (30 * 30 * 32), out_relu_0);
	xuat(out_relu_0, 30 * 30 * 32);
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", out_relu_0, (30 * 30 * 32));
    free(input);
    free(out_0);
    free(out_relu_0);
}