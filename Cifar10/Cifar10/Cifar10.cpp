//#define _CRT_SECURE_NO_WARNINGS
//#define _CRT_SECURE_NO_DEPRECATE  
//#define _CRT_NONSTDC_NO_DEPRECATE
//#include <stdio.h>
//#include <conio.h>
//#include <stdlib.h>
//#include <math.h>
//
//void readfile(const char* filename, float a[], int n) {
//    FILE* file = fopen(filename, "r");
//    if (file == NULL) {
//        fprintf(stderr, "Error: Failed to open file '%s'\n", filename);
//        return;
//    }
//
//    for (int i = 0; i < n; i++) {
//        if (fscanf(file, "%f", &a[i]) != 1) {
//            fprintf(stderr, "Error: Failed to read floating point number from file '%s'\n", filename);
//            break;
//        }
//    }
//
//    fclose(file);
//}
//void writefile(const char* filename, float a[], int n) {
//
//    FILE* fp = fopen(filename, "wb");
//    if (fp == NULL) {
//        printf("Failed to open file for writing.\n");
//        return;
//    }
//
//
//    fwrite(a, sizeof(float), n, fp);
//
//    fclose(fp);
//}
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
        printf("a[%d] = %.15f ", i, a[i]);
    }
    printf("\n");
}
void relu(float* input, int n, float* out) {
    for (int i = 0; i < n; i++) {
        out[i] = input[i] > 0 ? input[i] : 0;
    }
}


//===============================Conv2d============================================================
void conv2d(float* input, float* w, int row, int col, int frow, int fcol, int channel_inputs,
    int numofkernel, float* conv_results, float* bias, int padding) {

    int out_row = row - frow + 2 * padding + 1;
    int out_col = col - fcol + 2 * padding + 1;


    float**** kernel_out = new float*** [numofkernel];
    for (int k = 0; k < numofkernel; k++) {
        kernel_out[k] = new float** [channel_inputs];
        for (int l = 0; l < channel_inputs; l++) {
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


    int padded_row = row + 2 * padding;
    int padded_col = col + 2 * padding;
    float* padded_input = new float[padded_row * padded_col * channel_inputs]();


    for (int l = 0; l < channel_inputs; l++) {
        for (int i = 0; i < padded_row; i++) {
            for (int j = 0; j < padded_col; j++) {
                int padded_idx = l * padded_row * padded_col + i * padded_col + j;
                padded_input[padded_idx] = 0.0;
            }
        }
    }

    for (int l = 0; l < channel_inputs; l++) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                int input_idx = l * row * col + i * col + j;
                int padded_idx = l * padded_row * padded_col + (i + padding) * padded_col + j + padding;
                padded_input[padded_idx] = input[input_idx];
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
                            int input_idx = l * padded_row * padded_col + (i + m) * padded_col + j + n;
                            int kernel_idx = k * channel_inputs * frow * fcol + l * frow * fcol + m * fcol + n;
                            sum += padded_input[input_idx] * kernel_out[k][l][m][n];
                        }
                    }
                }
                int h = k * out_row * out_col + i * out_col + j;
                conv_results[h] = sum + bias[k];
            }
        }
    }


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
    delete[] padded_input;
}

//===============================Code add padding===================================================
/*  void conv2d(float* input, float* w, int row, int col, int frow, int fcol, int channel_inputs,
      int numofkernel, float* padded_results, float* bias, int pad_rows, int pad_cols) {
      int out_row = row - frow + 1;
      int out_col = col - fcol + 1;
      int padded_row = out_row + 2 * pad_rows;
      int padded_col = out_col + 2 * pad_cols;
      int s = 0;

      float**** kernel_out = new float*** [numofkernel];
      for (int k = 0; k < numofkernel; k++) {
          kernel_out[k] = new float** [channel_inputs];
          for (int l = 0; l < channel_inputs; l++) {
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
          for (int i = 0; i < padded_row; i++) {
              for (int j = 0; j < padded_col; j++) {
                  int padded_idx = k * padded_row * padded_col + i * padded_col + j;
                  padded_results[padded_idx] = 0.0;
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
                  int padded_idx = k * padded_row * padded_col + (i + pad_rows) * padded_col + j + pad_cols; //Get index from 33 -->
                  //if (padded_idx == 1024) {
                  //    printf("padd_results: %f ", padded_results[padded_idx]);
                  //}
                  padded_results[padded_idx] = sum + bias[k];
              }
          }
      }

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
  } */

  //=====================================Batch_Normalization_first =========================================
void batch_norm(float* input, float* output, float* mean, float* variance, float* scale, float* shift, int batch_size, int channels, int row, int col) {
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < row * col; i++) {
                int idx = b * channels * row * col + c * row * col + i;
                float normalized_val = (input[idx] - mean[c]) / sqrt(variance[c] + 1e-3);
                output[idx] = scale[c] * normalized_val + shift[c];
            }
        }
    }
}
//===============================Batch_Normalization_2=========================================
//void batch_norm(float* input, float* output, float* mean, float* variance, float* scale, float* shift, int batch_size, int channels, int row, int col) {
//    int num = row * col;
//    for (int c = 0; c < channels; c++) {
//        //float sum = 0.0f;
//        for (int b = 0; b < batch_size; b++) {
//            for (int i = 0; i < num; i++) {
//                int h = b * channels * num + c * num + i;
//                output[h] = (input[h] - mean[c]) / variance[c];
//            }
//        }
//        for (int r = 0; r < row; r++) {
//            for (int c = 0; c < col; c++) {
//                int idx = c * row * col + r * col + c;
//                if (scale != NULL) {
//                    output[idx] = output[idx] * scale[c];
//                }
//                if (shift != NULL) {
//                    output[idx] = output[idx] + shift[c];
//                }
//            }
//        }
//    }
//}


//=========================================================================================================================

//    void batch_norm(float* input, float* output, float* mean, float* variance, float* scale, float* shift, int batch_size, int channels, int row, int col) {
//    int num = row * col;
//    for (int b = 0; b < batch_size; b++) {
//        for (int c = 0; c < channels; c++) {
//            for (int i = 0; i < num; i++) {
//                int h = b * channels * num + c * num + i;
//                output[h] = (input[h] - mean[c]) / sqrtf(variance[c] + 1e-5f);
//                if (scale != NULL) {
//                    output[h] = output[h] * scale[c];
//                }
//                if (shift != NULL) {
//                    output[h] = output[h] + shift[c];
//                }
//            }
//        }
//    }
//}
    //==================================Maxpool2d===========================================
    //Maxpool cut out with padding = 1, stride = 2
void maxpool2d(float* input, int row, int col, int pool_size, int channel_size, int stride, float* output_max) {
    int output_row = ((row - pool_size) / stride) + 1;
    int output_col = ((col - pool_size) / stride) + 1;
    for (int k = 0; k < channel_size; k++) {
        for (int i = 0; i < output_row; i++) {
            for (int j = 0; j < output_col; j++) {
                float max_val = input[k * row * col + i * stride * col + j * stride];
                for (int m = 0; m < pool_size; m++) {
                    for (int n = 0; n < pool_size; n++) {
                        float curr_val = input[k * row * col + (i * stride + m) * col + j * stride + n];
                        if (curr_val > max_val) {
                            max_val = curr_val;
                        }
                    }
                }
                output_max[k * output_row * output_col + i * output_col + j] = max_val;
            }
        }
    }
}
//==================================Dropout===============================================
//void dropout(float* input, float* output, int size, float dropout_prob) {
//    srand(time(NULL));
//    for (int i = 0; i < size; i++) {
//        if ((float)rand() / RAND_MAX > dropout_prob) {
//            output[i] = input[i] / (1 - dropout_prob);
//        }
//        else {
//            output[i] = 0.0f;
//        }
//    }
//}

//====================================Dense================================================
void transposedense(float* input, int height, int width, int channels, float* output) {
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                output[h * width * channels + w * channels + c] = input[c * height * width + h * width + w];
            }
        }
    }
}

void dense(float* input, int n_input, int n_w, float* weights, float* bias, float* output) {

    float* dot = (float*)malloc(n_w * sizeof(float));

    for (int i = 0; i < n_w; i++) {
        dot[i] = 0;
        for (int j = 0; j < n_input; j++) {
            int h = i * n_input + j;
            dot[i] += input[j] * weights[h];
            //if (h == 448) {
            //	printf(" input[%d] = %f ", j, input[j]);
            //}
        }
        //printf("dot[%d] = %f ", i, dot[i]);
        output[i] = dot[i] + bias[i];
    }

    //for (int i = 0; i < n_w; i++) {
    //	printf("output[%i] = %f \n", i, output[i]);
    //}

    free(dot);
}

void softmax(float* input, int n_input, float* output) {
    float max_val = input[0];
    for (int i = 0; i < n_input; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    float exp_sum = 0.0;
    for (int i = 0; i < n_input; i++) {
        float exp_val = expf(input[i] - max_val);
        output[i] = exp_val;
        exp_sum += exp_val;
    }
    for (int i = 0; i < n_input; i++) {
        output[i] /= exp_sum;
    }
}

int argmax(float* arr, int size) {
    float max_val = arr[0];
    int max_idx = 0;
    for (int i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    printf("Gia tri du doan: %d ", max_idx);
    return max_idx;
}
void main() {
    //======================================****FIRST_COVER****===========================================//

    float* input = (float*)malloc((32 * 32 * 3) * sizeof(float));
    int n_input = (32 * 32 * 3);
    readfile("ship_n.txt", input, n_input);
    //xuat(input, n_input);
    //===================Conv2d_0=======================
    float w_0[864];
    float bias_0[32];
    int nw_0 = 864;
    int nb_0 = 32;
    float* out_0 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    float* out_relu_0 = (float*)malloc((32 * 32 * 32) * sizeof(float));

    readfile("W_0.txt", w_0, nw_0);
    //xuat(w_0, 864);
    readfile("bias_0.txt", bias_0, nb_0);
    conv2d(input, w_0, 32, 32, 3, 3, 3, 32, out_0, bias_0, 1);
    //xuat(out_0, 32 * 32 * 32);
    relu(out_0, (32 * 32 * 32), out_relu_0);
    //xuat(out_relu_0, 32 * 32 * 32);
    //writefile("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", out_relu_0, (32 * 32 * 32));
    //===================Batch_Normalization_0================
    float* out_batch_0 = (float*)malloc((32 * 32 * 32) * sizeof(float));

    //float* input_batch_0 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    int n_batch_0 = (32 * 32 * 32);
    float mean_0[32];
    float variance_0[32];
    float scale_0[32];
    float shift_0[32];
    int n_mean_0 = 32;
    int n_variance_0 = 32;
    int n_scale_0 = 32;
    int n_shift_0 = 32;
    //readfile ("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", input_batch_0, n_batch_0);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_0/Mean.txt", mean_0, n_mean_0);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_0/Variance.txt", variance_0, n_variance_0);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_0/Scale.txt", scale_0, n_scale_0);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_0/Shift.txt", shift_0, n_shift_0);
    batch_norm(out_relu_0, out_batch_0, mean_0, variance_0, scale_0, shift_0, 1, 32, 32, 32);

    //writefile1d("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_batch_0.txt", out_batch_0, (32 * 32 * 32));
    //xuat(out_batch_0, (32 * 32 * 32));
    //=====================Conv2d_1===============================
    float* w_1 = (float*)malloc((9216) * sizeof(float));
    float bias_1[32];
    int nw_1 = 9216;
    int nb_1 = 32;
    float* out_1 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    float* out_relu_1 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    readfile("W_1.txt", w_1, nw_1);
    //xuat(w_1, 9216);
    readfile("bias_1.txt", bias_1, nb_1);
    conv2d(out_batch_0, w_1, 32, 32, 3, 3, 32, 32, out_1, bias_1, 1);
    //xuat(out_1, 32 * 32 * 32);
    relu(out_1, (32 * 32 * 32), out_relu_1);
    //xuat(out_relu_1, 32 * 32 * 32);
    //writefile("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_1.txt", out_relu_1, (32 * 32 * 32));

    ////=======================Batch_Normalization_1=================
    float* out_batch_1 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    //float* input_batch_1 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    int n_batch_1 = (32 * 32 * 32);
    float mean_1[32];
    float variance_1[32];
    float scale_1[32];
    float shift_1[32];
    int n_mean_1 = 32;
    int n_variance_1 = 32;
    int n_scale_1 = 32;
    int n_shift_1 = 32;
    //readfile("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", input_batch_0, n_batch_0);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_1/Mean.txt", mean_1, n_mean_1);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_1/Variance.txt", variance_1, n_variance_1);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_1/Scale.txt", scale_1, n_scale_1);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_1/Shift.txt", shift_1, n_shift_1);
    batch_norm(out_relu_1, out_batch_1, mean_1, variance_1, scale_1, shift_1, 1, 32, 32, 32);
    //xuat(out_batch_1, 32 * 32 * 32);
    ////=========================Maxpool_0==============================
    float out_maxpool_0[16 * 16 * 32];
    int l_pool_0 = 16 * 16 * 32;
    maxpool2d(out_batch_1, 32, 32, 2, 32, 2, out_maxpool_0);
    //xuat(out_maxpool_0, 16 * 16 * 32);
    //////=========================Dropout_0================================
    ////int size_0 = 16 * 16 * 32;
    ////float out_drop_0[16 * 16 * 32];
    ////dropout(out_maxpool_0, out_drop_0, size_0, 0.3);

    ////======================================****SECOND_COVER****===========================================//
    ////===================Conv2d_2=======================

    float* w_2 = (float*)malloc((18432) * sizeof(float));
    float bias_2[64];
    int nw_2 = 18432;
    int nb_2 = 64;
    float* out_2 = (float*)malloc((16 * 16 * 64) * sizeof(float));
    float* out_relu_2 = (float*)malloc((16 * 16 * 64) * sizeof(float));

    readfile("W_2.txt", w_2, nw_2);
    //xuat(w_0, 864);
    readfile("bias_2.txt", bias_2, nb_2);
    conv2d(out_maxpool_0, w_2, 16, 16, 3, 3, 32, 64, out_2, bias_2, 1);

    //xuat(out_0, 32 * 32 * 32);
    relu(out_2, (16 * 16 * 64), out_relu_2);
    //xuat(out_relu_2, 16 * 16 * 32);
    //writefile("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_2.txt", out_relu_2, (16 * 16 * 64));
    ////===================Batch_Normalization_2================
    float* out_batch_2 = (float*)malloc((16 * 16 * 64) * sizeof(float));

    //float* input_batch_2 = (float*)malloc((16 * 16 * 64) * sizeof(float));
    int n_batch_2 = (16 * 16 * 64);
    float mean_2[64];
    float variance_2[64];
    float scale_2[64];
    float shift_2[64];
    int n_mean_2 = 64;
    int n_variance_2 = 64;
    int n_scale_2 = 64;
    int n_shift_2 = 64;
    //readfile("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_2.txt", input_batch_2, n_batch_2);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_2/Mean.txt", mean_2, n_mean_2);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_2/Variance.txt", variance_2, n_variance_2);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_2/Scale.txt", scale_2, n_scale_2);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_2/Shift.txt", shift_2, n_shift_2);
    batch_norm(out_relu_2, out_batch_2, mean_2, variance_2, scale_2, shift_2, 1, 64, 16, 16);
    ////=====================Conv2d_3===============================
    float* w_3 = (float*)malloc((36864) * sizeof(float));
    float bias_3[64];
    int nw_3 = 36864;
    int nb_3 = 64;
    float* out_3 = (float*)malloc((16 * 16 * 64) * sizeof(float));
    float* out_relu_3 = (float*)malloc((16 * 16 * 64) * sizeof(float));

    readfile("W_3.txt", w_3, nw_3);
    //xuat(w_0, 864);
    readfile("bias_3.txt", bias_3, nb_3);
    conv2d(out_batch_2, w_3, 16, 16, 3, 3, 64, 64, out_3, bias_3, 1);
    //xuat(out_0, 32 * 32 * 32);
    relu(out_3, (16 * 16 * 64), out_relu_3);
    //xuat(out_relu_0, 32 * 32 * 32);
    //writefile("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", out_relu_3, (16 * 16 * 64));
    ////=======================Batch_Normalization_3===================
    float* out_batch_3 = (float*)malloc((16 * 16 * 64) * sizeof(float));

    //float* input_batch_1 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    int n_batch_3 = (16 * 16 * 64);
    float mean_3[64];
    float variance_3[64];
    float scale_3[64];
    float shift_3[64];
    int n_mean_3 = 64;
    int n_variance_3 = 64;
    int n_scale_3 = 64;
    int n_shift_3 = 64;
    //readfile("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", input_batch_0, n_batch_0);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_3/Mean.txt", mean_3, n_mean_3);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_3/Variance.txt", variance_3, n_variance_3);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_3/Scale.txt", scale_3, n_scale_3);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_3/Shift.txt", shift_3, n_shift_3);
    batch_norm(out_relu_3, out_batch_3, mean_3, variance_3, scale_3, shift_3, 1, 64, 16, 16);
    //=========================Maxpool_1==============================
    float out_maxpool_1[8 * 8 * 64];
    int l_pool_1 = 8 * 8 * 64;
    maxpool2d(out_batch_3, 16, 16, 2, 64, 2, out_maxpool_1);

    ////////=========================Dropout_1================================
    //////int size_1 = 8 * 8 * 64;
    //////float out_drop_1[8 * 8 * 64];
    //////dropout(out_maxpool_1, out_drop_1, size_1, 0.5);

    ////======================================****THIRD_COVER****===========================================//
    ////===================Conv2d_4=======================

    float* w_4 = (float*)malloc((73728) * sizeof(float));
    float bias_4[128];
    int nw_4 = 73728;
    int nb_4 = 128;
    float* out_4 = (float*)malloc((8 * 8 * 128) * sizeof(float));
    float* out_relu_4 = (float*)malloc((8 * 8 * 128) * sizeof(float));

    readfile("W_4.txt", w_4, nw_4);
    //xuat(w_0, 864);
    readfile("bias_4.txt", bias_4, nb_4);
    conv2d(out_maxpool_1, w_4, 8, 8, 3, 3, 64, 128, out_4, bias_4, 1);

    //xuat(out_0, 32 * 32 * 32);
    relu(out_4, (8 * 8 * 128), out_relu_4);
    //xuat(out_relu_0, 32 * 32 * 32);
    //writefile("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_4.txt", out_relu_4, (8 * 8 * 128));
    ////===================Batch_Normalization_4================
    float* out_batch_4 = (float*)malloc((8 * 8 * 128) * sizeof(float));

    //float* input_batch_4 = (float*)malloc((8 * 8 * 128) * sizeof(float));
    int n_batch_4 = (8 * 8 * 128);
    float mean_4[128];
    float variance_4[128];
    float scale_4[128];
    float shift_4[128];
    int n_mean_4 = 128;
    int n_variance_4 = 128;
    int n_scale_4 = 128;
    int n_shift_4 = 128;
    //readfile("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_4.txt", input_batch_4, n_batch_4);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_4/Mean.txt", mean_4, n_mean_4);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_4/Variance.txt", variance_4, n_variance_4);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_4/Scale.txt", scale_4, n_scale_4);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_4/Shift.txt", shift_4, n_shift_4);
    batch_norm(out_relu_4, out_batch_4, mean_4, variance_4, scale_4, shift_4, 1, 128, 8, 8);
    //=====================Conv2d_5===============================
    float* w_5 = (float*)malloc((147456) * sizeof(float));
    float bias_5[128];
    int nw_5 = 147456;
    int nb_5 = 128;
    float* out_5 = (float*)malloc((8 * 8 * 128) * sizeof(float));
    float* out_relu_5 = (float*)malloc((8 * 8 * 128) * sizeof(float));

    readfile("W_5.txt", w_5, nw_5);
    //xuat(w_0, 864);
    readfile("bias_5.txt", bias_5, nb_5);
    conv2d(out_batch_4, w_5, 8, 8, 3, 3, 128, 128, out_5, bias_5, 1);
    //xuat(out_0, 32 * 32 * 32);
    relu(out_5, (8 * 8 * 128), out_relu_5);
    //xuat(out_relu_0, 32 * 32 * 32);
    //writefile("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_5.txt", out_relu_5, (8 * 8 * 128));
    ////=======================Batch_Normalization_5===================
    float* out_batch_5 = (float*)malloc((8 * 8 * 128) * sizeof(float));

    //float* input_batch_1 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    int n_batch_5 = (8 * 8 * 128);
    float mean_5[128];
    float variance_5[128];
    float scale_5[128];
    float shift_5[128];
    int n_mean_5 = 128;
    int n_variance_5 = 128;
    int n_scale_5 = 128;
    int n_shift_5 = 128;
    //readfile("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", input_batch_0, n_batch_0);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_5/Mean.txt", mean_5, n_mean_5);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_5/Variance.txt", variance_5, n_variance_5);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_5/Scale.txt", scale_5, n_scale_5);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_5/Shift.txt", shift_5, n_shift_5);
    batch_norm(out_relu_5, out_batch_5, mean_5, variance_5, scale_5, shift_5, 1, 128, 8, 8);
    ////=========================Maxpool_2==============================
    float out_maxpool_2[8 * 8 * 128];
    int l_pool_2 = 8 * 8 * 128;
    maxpool2d(out_batch_5, 8, 8, 2, 128, 2, out_maxpool_2);

    ////////=========================Dropout_2================================
    //////int size_2 = 4 * 4 * 128;
    //////float out_drop_2[4 * 4 * 128];
    //////dropout(out_maxpool_2, out_drop_2, size_2, 0.5);

    ////======================================****LAST_COVER****===========================================//
    ////===============================Transpose============================
    float out_transpose[4 * 4 * 128];
    transposedense(out_maxpool_2, 4, 4, 128, out_transpose);
    //xuat(out_transpose, 12 * 12 * 32);
    //============================Dense128=============================
    float* w_128 = (float*)malloc(262144 * sizeof(float));
    float bias_128[128];
    int nw_128 = 262144;
    int nb_128 = 128;
    int l_dense_128 = 4 * 4 * 128;
    int l_w_128 = 128;
    float out_dense_128[128];
    float out_relu_128[128];
    readfile("dense_128.txt", w_128, nw_128);
    //xuat(w_128, nw_128);
    readfile("bias_128.txt", bias_128, nb_128);
    dense(out_transpose, l_dense_128, l_w_128, w_128, bias_128, out_dense_128);
    //xuat(out_dense_128, 128);
    relu(out_dense_128, 128, out_relu_128);
    //xuat(out_relu_128, 128);
    //writefile("out_dense_128.txt", out_relu_128, 128);

    ////=============================Batch_Normalization_6================

    float out_batch_6[128];

    //float* input_batch_4 = (float*)malloc((8 * 8 * 128) * sizeof(float));
    int n_batch_6 = 0;
    float mean_6[128];
    float variance_6[128];
    float scale_6[128];
    float shift_6[128];
    int n_mean_6 = 128;
    int n_variance_6 = 128;
    int n_scale_6 = 128;
    int n_shift_6 = 128;
    //readfile("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_4.txt", input_batch_4, n_batch_4);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_6/Mean.txt", mean_6, n_mean_6);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_6/Variance.txt", variance_6, n_variance_6);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_6/Scale.txt", scale_6, n_scale_6);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10/Cifar10/Cifar10/BatchNormalization_6/Shift.txt", shift_6, n_shift_6);
    batch_norm(out_relu_128, out_batch_6, mean_6, variance_6, scale_6, shift_6, 1, 1, 1, 128);
    xuat(out_batch_6, 128);
    //////============================Dropout_3===============================
    ////int size_3 = 1 * 1 * 128;
    ////float out_drop_3[1 * 1 * 128];
    ////dropout(out_batch_6, out_drop_3, size_3, 0.5);
    ////============================Dense_10================================
    float w_10[1280];
    float bias_10[10];
    int nw_10 = 1280;
    int nb_10 = 10;
    int l_dense_10 = 128;
    int l_w_10 = 10;
    float out_dense_10[10];
    float out_softmax_10[10];
    readfile("dense_10.txt", w_10, nw_10);
    readfile("bias_10.txt", bias_10, nb_10);
    dense(out_batch_6, l_dense_10, l_w_10, w_10, bias_10, out_dense_10);
    softmax(out_dense_10, 10, out_softmax_10);
    xuat(out_softmax_10, 10);
    argmax(out_softmax_10, 10);







    //===========================================FREE_MEMORY_ALLOCATED========================================//
    //===========FIRST_COVER===========
    //===========Free_Conv2d_0=========
    free(input);
    free(out_0);
    free(out_relu_0);
    //==========Free_Batch_Norm_0======
    free(out_batch_0);
    //free(input_batch_0);
    //===========Free_Conv2d_1=========
    free(w_1);
    free(out_1);
    free(out_relu_1);
    //==========Free_Batch_Norm_1======
    free(out_batch_1);
    //free(input_batch_1);
    ////==========SECOND_COVER===========

    //==========Free_Conv2d_2==========
    free(w_2);
    free(out_2);
    free(out_relu_2);
    //==========Free_Batch_Norm_2======
    free(out_batch_2);
    //==========Free_Conv2d_3==========
    free(w_3);
    free(out_3);
    free(out_relu_3);
    //==========Free_Batch_Norm_3======
    free(out_batch_3);

    //==========THIRD_COVER============
    //==========Free_Conv2d_4==========
    free(w_4);
    free(out_4);
    free(out_relu_4);
    //===========Free_Batch_Norm_4=====
    free(out_batch_4);
    //===========Free_Conv2d_5=========
    free(w_5);
    free(out_5);
    free(out_relu_5);
    //===========Free_Batch_Norm_5=====
    free(out_batch_5);

    //===========LAST_COVER============
    //===========Dense_128=============
    free(w_128);

}