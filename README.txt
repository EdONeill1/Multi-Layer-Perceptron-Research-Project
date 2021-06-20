* Path must be set if you wish to save logs.

* To use the Multi Layer Perceptron, the following variables must be set :
        - lr (learning rate) => Integer or float (for example 0.001)
	- Input, Output (X,Y) and hidden units
        - epochs => Number of epochs to train to
        - hidden units, input, and output
        - experiment => integer referring to experiment being performed, this can be set to None
        - path 
        - save_logs => True or False.        
        - print_graphs => True or False
        - task => "XOR" or "Sin"
        - verbose => True or False <= Prints out the prediction and target together
        - return_prediction => True or False <= False if the prediction isn't needed otherwise True. 
        
* To initialise weights, define X,Y and the number of hidden units and use the following function :
        
        W1, W2 = mlp.init_weights(X, Y, hidden_units)

* To train:
        
        mlp.train(task, X, Y, W1, W2, ACTIVATION_FUNCTION, ACTIVATION_FUNCTION_DERIVATIVE,
          epochs, lr, hidden_units, experiment, path, save_logs, print_graphs)

* To perform a prediction and save it :

        _, prediction_error, prediction_accuracy = mlp.predict(task, X, Y, W1, W2, ACTIVATION FUNCTION, verbose, return_prediction)
        mlp.save_prediction(path, save_logs, task, prediction_error, prediction_accuracy, epochs, hidden_units, lr, experiment) <= return_prediction == True

* To perform a prediction without saving it
        mlp.predict(task, X, Y, W1, W2, ACTIVATION_FUNCTION, verbose, return_prediction) <= return_prediction == False


* To plot prediction against target/actual values:

        compare_prediction(prediction, Y, task, experiment)
        

* The libraries used in this project were : numpy, random, os, json, matplotlib, and time






