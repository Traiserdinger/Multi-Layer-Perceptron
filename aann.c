#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define INPUT_NEURON 2
#define HIDDEN_LAYERS 7
#define HIDDEN_NEURON 10
#define OUTPUT_NEURON 1
#define DATASET_SIZE 1200
#define LEARNING_RATE 0.005
#define EPOCHS 5000

double dataset_inputs[DATASET_SIZE][INPUT_NEURON];
double dataset_targets[DATASET_SIZE];

double tanh_activation(double x) {
    return -1.0*tanh(x);
}
double tanh_derivative(double y) {
    return -1.0*(1.0-y*y);
}
double linear(double x) {
    return x;
}
double linear_derivative(double y) {
    return 1.0;
}
double random_weight() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

typedef struct {
    double input[INPUT_NEURON];
    double hidden[HIDDEN_LAYERS][HIDDEN_NEURON];
    double output[OUTPUT_NEURON];
    double weights_input_hidden[INPUT_NEURON][HIDDEN_NEURON];
    double weights_hidden_hidden[HIDDEN_LAYERS - 1][HIDDEN_NEURON][HIDDEN_NEURON];
    double weights_hidden_output[HIDDEN_NEURON];
    double bias_hidden[HIDDEN_LAYERS][HIDDEN_NEURON];
    double bias_output;
    double error[EPOCHS];
} NeuralNetwork;

void load_dataset(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Không thể mở file dữ liệu\n");
        exit(1);
    }
    char line[256];
    int index = 0;
    while (fgets(line, sizeof(line), file) && index < DATASET_SIZE) {
        char *data_split = strtok(line, ",");
        for (int i = 0; i < INPUT_NEURON; i++) {
            dataset_inputs[index][i] = atof(data_split);
            data_split = strtok(NULL, ",");
        }
        dataset_targets[index] = atof(data_split);
        index++;
    }
    fclose(file);
}

void initialize_network(NeuralNetwork *net) {
    for (int i = 0; i < INPUT_NEURON; i++){
        for (int j = 0; j < HIDDEN_NEURON; j++){
            net->weights_input_hidden[i][j] = random_weight();
        }
    }
    for (int l = 0; l < HIDDEN_LAYERS - 1; l++){
        for (int i = 0; i < HIDDEN_NEURON; i++){
            for (int j = 0; j < HIDDEN_NEURON; j++){
                net->weights_hidden_hidden[l][i][j] = random_weight();
            }
        }
    }
    for (int i = 0; i < HIDDEN_NEURON; i++){
        net->weights_hidden_output[i] = random_weight();
    }
    for (int i = 0; i < HIDDEN_LAYERS; i++){
        for (int j = 0; j < HIDDEN_NEURON; j++){
            net->bias_hidden[i][j] = random_weight();
        }
    }
    net->bias_output = random_weight();
}

void forward(NeuralNetwork *net) {
    for (int j = 0; j < HIDDEN_NEURON; j++) {
        double sum = net->bias_hidden[0][j];
        for (int i = 0; i < INPUT_NEURON; i++){
            sum += net->input[i] * net->weights_input_hidden[i][j];
        }
        net->hidden[0][j] = tanh_activation(sum);
    }
    for (int l = 1; l < HIDDEN_LAYERS; l++) {
        for (int j = 0; j < HIDDEN_NEURON; j++) {
            double sum = net->bias_hidden[l][j];
            for (int i = 0; i < HIDDEN_NEURON; i++){
                sum += net->hidden[l-1][i] * net->weights_hidden_hidden[l-1][i][j];
            }
            net->hidden[l][j] = tanh_activation(sum);
        }
    }
    double sum = net->bias_output;
    for (int i = 0; i < HIDDEN_NEURON; i++){
        sum += net->hidden[HIDDEN_LAYERS-1][i] * net->weights_hidden_output[i];
    }
    net->output[0] = linear(sum);
}

// Backpropagation + SGD
void train(NeuralNetwork *net, double target) {
    double delta_output = (net->output[0] - target) * linear_derivative(net->output[0]);
    double delta_hidden[HIDDEN_LAYERS][HIDDEN_NEURON];

    // Last hidden layer
    for (int i = 0; i < HIDDEN_NEURON; i++) {
        delta_hidden[HIDDEN_LAYERS - 1][i] = net->weights_hidden_output[i] * delta_output * tanh_derivative(net->hidden[HIDDEN_LAYERS - 1][i]);
    }

    // Propagate deltas backward
    for (int l = HIDDEN_LAYERS - 2; l >= 0; l--) {
        for (int i = 0; i < HIDDEN_NEURON; i++) {
            double error = 0.0;
            for (int j = 0; j < HIDDEN_NEURON; j++){
                error += net->weights_hidden_hidden[l][i][j] * delta_hidden[l+1][j];
            }
            delta_hidden[l][i] = error * tanh_derivative(net->hidden[l][i]);
        }
    }
    for (int i = 0; i < HIDDEN_NEURON; i++){
        net->weights_hidden_output[i] -= LEARNING_RATE * delta_output * net->hidden[HIDDEN_LAYERS - 1][i];
    }
    net->bias_output -= LEARNING_RATE * delta_output;
    for (int l = HIDDEN_LAYERS - 1; l >= 1; l--) {
        for (int i = 0; i < HIDDEN_NEURON; i++) {
            for (int j = 0; j < HIDDEN_NEURON; j++){
                net->weights_hidden_hidden[l-1][j][i] -= LEARNING_RATE*delta_hidden[l][i]*net->hidden[l-1][j];
            }
            net->bias_hidden[l][i] -= LEARNING_RATE * delta_hidden[l][i];
        }
    }
    for (int i = 0; i < INPUT_NEURON; i++) {
        for (int j = 0; j < HIDDEN_NEURON; j++){
            net->weights_input_hidden[i][j] -= LEARNING_RATE * delta_hidden[0][j] * net->input[i];
        }
    }
    for (int j = 0; j < HIDDEN_NEURON; j++){
        net->bias_hidden[0][j] -= LEARNING_RATE * delta_hidden[0][j];
    }
}
double true_function(double x1, double x2) {
    return sqrt(x1*x1 + log(x1*x2) + x2);
}
void save_outputs(NeuralNetwork *net, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Không thể mở file để ghi output.\n");
        exit(1);
    }

    for (int i = 0; i < DATASET_SIZE; i++) {
        net->input[0] = dataset_inputs[i][0];
        net->input[1] = dataset_inputs[i][1];
        forward(net);
        double x1 = dataset_inputs[i][0];
        double x2 = dataset_inputs[i][1];
        double model_output = net->output[0];
        double true_value = true_function(x1, x2);

        fprintf(file, "%f,%f,%f,%f,%f\n", 
            x1, x2, model_output, dataset_targets[i], true_value);
    }

    fclose(file);
}
int main() {
    srand(time(NULL));
    NeuralNetwork net;
    initialize_network(&net);
    load_dataset("C:\\Users\\tiend\\Downloads\\data4.csv");
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double cal_error = 0.0;
        for (int i = 0; i < DATASET_SIZE; i++) {
            net.input[0] = dataset_inputs[i][0];
            net.input[1] = dataset_inputs[i][1];
            forward(&net);
            cal_error += 0.5*pow(dataset_targets[i] - net.output[0], 2);
            train(&net, dataset_targets[i]);
        }
        net.error[epoch] = cal_error;
        if (epoch % 1000 == 0){
            printf("Epoch %d - Error: %.12f\n", epoch, net.error[epoch]);
        }
    }
    for (int i = 0; i < 10; i++) {
        net.input[0] = dataset_inputs[i][0];
        net.input[1] = dataset_inputs[i][1];
        forward(&net);
        printf("Input: %.4f %.4f -> Output: %.6f | Target: %.6f\n", dataset_inputs[i][0], dataset_inputs[i][1], net.output[0], dataset_targets[i]);
    }
    save_outputs(&net, "C:\\Users\\tiend\\Downloads\\output.csv");
    return 0;
}

