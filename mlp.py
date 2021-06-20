import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import json


## Activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh_derivative(x):
    return 1 - (x * x)

def tanh(x):
    return np.tanh(x)


def init_weights(X, Y, hidden):
    n, x = X.shape # x = Input Dimension
    h = hidden # Hidden Dimension
    y = len(Y.T) # Output Dimension
    
    W1 = np.random.randn(x, h) * np.sqrt(1 / x)
    W2 = np.random.randn(h, y) * np.sqrt(1 / x)
    return W1, W2


def forward_prop(W1, W2, X, Y, activation_function):
    Z1 = X @ W1
    A1 = activation_function(Z1)
    Z2 = A1 @ W2
    A2 = activation_function(Z2)

    error = 0
    if Y is not None:
        error = np.sum(np.abs(A2 - Y)) / len(Y)

    return A1, A2, error


def back_prop(X, Y, A1, A2, W1, W2, activation_function_derivative):
    dl = A2 - Y
    dA2 = dl * activation_function_derivative(A2)
    dW2 = np.dot(A1.T, dA2)
    dl = np.dot(dA2, W2.T)
    dA1 = dl * activation_function_derivative(A1)
    dW1 = np.dot(X.T, dA1)
    return dW1, dW2



def update_weights(W1, W2, dW1, dW2, lr):
    W2 -= lr * dW2
    W1 -= lr * dW1
    return W1, W2


def train(task, X, Y, W1, W2, f, df, epochs, lr, hidden_units, experiment, path, save_log, print_graphs):

    errors = []
    error_to_save = []
    acc = []
    acc_to_save = []
    accuracy = 0
    for epoch in range(epochs):

        A1, A2, error = forward_prop(W1, W2, X, Y, f)
        dW1, dW2 = back_prop(X, Y, A1, A2, W1, W2, df)
        W1, W2 = update_weights(W1, W2, dW1, dW2, lr)

        if task == "XOR":
            accuracy = 100 * (1 - np.mean(np.abs(np.where(A2 < 0.5, 0, 1) - Y)))
            

        accuracy = 100 * (1 - np.mean(np.abs(A2 - Y)))

        if accuracy < 0:
            accuracy = 0

        errors.append(error)
        acc.append(accuracy)

        if epoch % (epochs/10) == 0:
                         error_to_save.append(error)
                         acc_to_save.append(accuracy)
                         print_status(epoch, epochs, error, accuracy)
                         save_logs(path, save_log, task, error_to_save, acc_to_save, int(((epoch/epochs) * 100) + 10), epochs, hidden_units, lr, experiment)
   
    if print_graphs == True:
        plot_result(task, "Error", errors, acc, lr, hidden_units, epochs, experiment)
        plot_result(task, "Accuracy", errors, acc, lr, hidden_units, epochs, experiment)
        



def predict(task, X, Y, W1, W2, f, verbose, return_prediction):
        _, A2, error = forward_prop(W1, W2, X, Y, f)
        prediction = A2
        accuracy = 100 * (1 - np.mean(np.abs(prediction - Y)))
        if accuracy < 0:
            accuracy = 0
            
        if task == "XOR":
            prediction = np.where(prediction < 0.5, 0, 1)
            accuracy = 100 * (1 - np.mean(np.abs(Y - np.where(prediction < 0.5, 0, 1))))
            
        if verbose == True:
            print(f"Prediction : {prediction} \n\n Target : {Y}")
           
        print(f"Error :{error} \t Accuracy :{accuracy}%")

        if return_prediction == True:
            return prediction, error, accuracy


def print_status(epoch, epochs, error, accuracy):
    print(f"{((epoch/epochs) * 100) + 10}% done...")
    print(f"Error {error} after {epoch} epochs")
    print(f"Accuracy {accuracy}% after {epoch} epochs\n")



def save_logs(path, save_log, model_type, error, accuracy, percentage_done, epochs, hidden_units, lr, experiment):
            if save_log == True:
                logs ={
                        "Model": model_type,
                        "Error" : error,
                        "Accuracy" : accuracy,
                        "Hidden Units" : hidden_units,
                        "Learning Rate" : lr,
                        "Epochs" : epochs,
                        "Epoch Percentage Finished": percentage_done
                }
                name = ""
                if experiment == None:
                    name = logs["Model"]
                with open(os.path.join(path,"{}.json".format("{}_percentage_done_".format(name), percentage_done)), "w") as f:
                        json.dump(logs, f)

                name = logs["Model"] + "_experiment_"
                with open(os.path.join(path,"{}_{}.json".format("{}_experiment_{}_percentage_done_".format(name, experiment), percentage_done)), "w") as f:
                        json.dump(logs, f)

def save_prediction(path, save_log, model_type, prediction_error, prediction_accuracy, epochs, hidden_units, lr, experiment):
    if save_log == True:
                logs ={
                        "Model": model_type,
                        "Prediction Error" : prediction_error,
                        "Prediction Accuracy" : prediction_accuracy,
                        "Hidden Units" : hidden_units,
                        "Learning Rate" : lr,
                        "Epochs Training For" : epochs
                       
                 }
                name = ""
                if experiment == None:
                    name = "Prediction_" + logs["Model"]
                with open(os.path.join(path,"{}.json".format(name)), "w") as f:
                        json.dump(logs, f)

                name = "Prediction_" + logs["Model"] + "_experiment_"
                with open(os.path.join(path,"{}_{}.json".format("{}".format(name, experiment), experiment)), "w") as f:
                        json.dump(logs, f)

                        
                        
def plot_result(task, result_type, errors, accuracies, lr, hidden_units, epochs, experiment):

    plt.figure(figsize=(40,20))
    plt.title(f"{task} {result_type} - Learing Rate {lr}, Hidden Units {hidden_units}", fontsize=50)
    plt.xlabel("Epochs", fontsize=50)
    plt.xticks(fontsize=35)
    plt.ylabel(f"Error\n", fontsize=50)
    if result_type == "Accuracy":
        plt.ylabel("Accuracy\n(Percentage)\n", fontsize=50)
    plt.yticks(fontsize=35)

    #Removing scientific notation that appears on the graphs.
    axis = plt.gca()

    axis.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    axis.xaxis.set_major_formatter(mticker.ScalarFormatter())
    axis.xaxis.get_major_formatter().set_scientific(False)
    axis.xaxis.get_major_formatter().set_useOffset(False)

    axis.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    axis.yaxis.set_major_formatter(mticker.ScalarFormatter())
    axis.yaxis.get_major_formatter().set_scientific(False)
    axis.yaxis.get_major_formatter().set_useOffset(False)


    if result_type == "Error":
        plt.plot(np.arange(epochs), errors, linewidth=4)
        plt.savefig(f"{task}_Error_Learing_Rate_{lr}_Hidden_Units_{hidden_units}_Experiment_{experiment}.png")
    elif result_type == "Accuracy":
        plt.plot(np.arange(epochs), accuracies, linewidth=4)
        plt.savefig(f"{task}_Accuracy_Learing_Rate_{lr}_Hidden_Units_{hidden_units}_Experiment_{experiment}.png")



def compare_prediction(prediction, Y, task, experiment):
    plt.figure(figsize=(40,20))
    plt.title("Prediction and Target", fontsize=50)
    y_type = ""
    if task == "Sin":
        plt.xlabel("Test Data Size", fontsize=50)
        plt.ylabel("Sin Values", fontsize=50)
        y_type = "Y_test"
    elif task == "XOR":
        plt.xlabel("Data Size", fontsize=50)
        plt.ylabel("XOR Values", fontsize=50)
        y_type = "Y"
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.plot(prediction, linewidth=3, label = "Prediction")
    plt.plot(Y, linewidth=3, label=y_type)
    plt.legend(loc="upper left", prop={"size": 30})
    plt.savefig(f"Comparing_Prediction_To_Target_Experiment_{experiment}.png")