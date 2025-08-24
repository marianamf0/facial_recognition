import time
import numpy as np 
from utils import (tanh, derivate_tanh_from_output, sigmoid, 
    derivate_sigmoid_from_output, function_perceptron_logistic, 
    function_derivate_perceptron_logistic)


def eval_classifier(y_real, y_pred):
    """
    Count correct classifications from one-hot targets and model scores.

    Args:
        y_real (array-like): Ground-truth labels as one-hot (n_samples, n_classes).
        y_pred (array-like): Predicted scores/probabilities/logits (n_samples, n_classes).

    Returns:
        int: Number of samples where argmax(y_pred) == argmax(y_real).
    """
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_real, axis=1)

    return np.sum(pred_labels == true_labels)


def generate_results(percentage_success:list, time:int, name:str): 
    print(f"================ RESULTADOS - {name} ================")
    print(f"Média: {np.mean(percentage_success):6.2f}% | Mediana: {np.median(percentage_success):6.2f}%")
    print(f"Mínimo: {np.min(percentage_success):6.2f}% | Máximo: {np.max(percentage_success):6.2f}%")
    print(f"Desvio Padrão: {np.std(percentage_success):6.2f}")
    print(f"Tempo: {time:.2f} s")


def linear_classifier(x_values, y_values, training_data_rate:float=0.8, 
                      number_simulations:int=50, verbose:bool = False):
    """
    Repeated train/test evaluation of a linear multi-class classifier (least-squares with intercept).

    On each of `number_simulations` rounds, the data are randomly split into train/test.
    A bias (ones) column is prefixed to X, class weights are estimated with the np.linalg.pinv 
    against one-hot targets, and test accuracy (percentage of correct argmax predictions) is 
    recorded. Optionally prints a summary of the accuracy distribution and total runtime.

    Args:
        x_values (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y_values (np.ndarray): One-hot targets of shape (n_samples, n_classes).
        training_data_rate (float): Fraction of samples used for training per split.
        number_simulations (int): Number of random resampling rounds.
        verbose (bool): If True, prints a metrics summary via `generate_results`.

    Returns:
        np.ndarray: Final weight matrix of shape (n_features + 1, n_classes),
                    learned on the last train/test split.
    """
    start_time = time.time()
    quantity_of_data=x_values.shape[0]
    x_values = np.hstack([np.ones((quantity_of_data, 1)), x_values])
    
    quantity_of_train_data = int(np.floor(quantity_of_data*training_data_rate))
    quantity_of_test_data = quantity_of_data - quantity_of_train_data
    
    percentage_success = []
    for _ in range(number_simulations): 
        index_data = np.random.permutation(quantity_of_data)
        x_shuff, y_shuff = x_values[index_data], y_values[index_data]
        
        x_train, x_test = x_shuff[:quantity_of_train_data], x_shuff[quantity_of_train_data:]
        y_train, y_test = y_shuff[:quantity_of_train_data], y_shuff[quantity_of_train_data:]
        
        weight = np.linalg.pinv(x_train) @ y_train
        y_pred_test = x_test @ weight
        
        number_success = eval_classifier(y_real=y_test, y_pred=y_pred_test)
        percentage_success.append(100*(number_success/quantity_of_test_data))
        
    end_time = time.time()
    if verbose: 
        generate_results(percentage_success=percentage_success, time=end_time-start_time, name="MQ")

    return weight


def perceptron_logistic_classifier(x_values, y_values, number_neurons:int, training_data_rate:float=0.8, 
        number_of_rounds:int=50, epoch_numbers:int=1, learning_rate:float=0.01, verbose:bool = False): 
    """
    Multi-class single-layer perceptron, trained by SGD, evaluated via repeated train/test resampling.

    For each round, the dataset is shuffled and split into training and test sets according to
    `training_data_rate`; a bias (ones) column is prepended to `X`; the model is trained with 
    per-sample SGD for `epoch_numbers` epochs using the bipolar sigmoid φ(z)=tanh(z/2) 
    (`function_perceptron_logistic`) and its derivative (`function_derivate_perceptron_logistic`); 
    predictions are computed on the test set, class labels are obtained via argmax, and accuracy is
    measured with `eval_classifier`; if `verbose` is True, an accuracy summary and the elapsed time
    are printed with `generate_results`.

    Args:
        x_values (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y_values (np.ndarray): One-hot targets of shape (n_samples, n_classes).
        number_neurons (int): Number of output neurons/classes (should equal n_classes).
        training_data_rate (float): Fraction of samples used for training per split.
        number_of_rounds (int): Number of random resampling rounds.
        epoch_numbers (int): SGD epochs per round (per-sample updates).
        learning_rate (float): Step size for weight updates.
        verbose (bool): If True, prints a metrics summary via `generate_results`.

    Returns:
        np.ndarray: Final weight matrix of shape (n_features + 1, n_classes),
                    learned on the last train/test split.

    """
    
    start_time = time.time()
    quantity_of_data, number_features = x_values.shape
    quantity_of_train_data = int(np.floor(quantity_of_data*training_data_rate))
    quantity_of_test_data = quantity_of_data - quantity_of_train_data
    
    number_success, percentage_success = [], []
    for _ in range(number_of_rounds): 
        index_data = np.random.permutation(quantity_of_data)
        x_shuff, y_shuff = x_values[index_data], y_values[index_data]
        
        x_train, x_test = x_shuff[:quantity_of_train_data], x_shuff[quantity_of_train_data:]
        y_train, y_test = y_shuff[:quantity_of_train_data], y_shuff[quantity_of_train_data:]
        
        x_train_b = np.concatenate([np.ones((quantity_of_train_data, 1)), x_train], axis=1)
        x_test_b = np.concatenate([np.ones((quantity_of_test_data, 1)), x_test], axis=1)
        
        weight = 0.01 * np.random.randn(number_neurons, number_features + 1)
        
        for _ in range(epoch_numbers): 
            order = np.random.permutation(quantity_of_train_data)
            
            for tt in order:
                x_value = x_train_b[tt]
                y_value = y_train[tt]
                y_pred = function_perceptron_logistic(weight @ x_value)
                
                erro = y_value - y_pred
                DDi = erro * function_derivate_perceptron_logistic(y_pred)
                
                weight += learning_rate*(DDi[:, None] * x_value[None, :]) 
            
        y_pred_test = function_perceptron_logistic(x_test_b @ weight.T)
        
        number_success = eval_classifier(y_real=y_test, y_pred=y_pred_test)
        percentage_success.append(100*(number_success/quantity_of_test_data))
        
    end_time = time.time()
    if verbose: 
        generate_results(percentage_success=percentage_success, time=end_time-start_time, name="PL")

    return weight


def mlp_sigmoid_classifier(x_values, y_values, number_neurons:int,  number_hidden_neurons:int=10,
                   training_data_rate:float=0.8, number_of_rounds:int=50,
                   epoch_numbers:int=20, learning_rate:float=0.01, verbose:bool = False):
    """
    Train/evaluate a one-hidden-layer MLP classifier with sigmoid activations via per-sample SGD.

    On each of `number_of_rounds` iterations, the data are shuffled and split into
    train/test (`training_data_rate`). A bias (ones) column is prepended to X; weights for
    input→hidden (`weight1`) and hidden→output (`weight2`) are initialized. For `epoch_numbers` 
    epochs, the model performs forward passes with `sigmoid` and backpropagates errors using 
    `derivate_sigmoid_from_output`; parameters are updated with learning rate `learning_rate`. 
    After training, test predictions are computed, class labels are obtained by `argmax`, 
    and accuracy (%) is recorded. if `verbose` is True, an accuracy summary and the elapsed time
    are printed with `generate_results`.

    Args:
        x_values (np.ndarray): Feature matrix, shape (n_samples, n_features).
        y_values (np.ndarray): One-hot targets, shape (n_samples, n_classes).
        number_neurons (int): Number of output neurons/classes (should equal n_classes).
        number_hidden_neurons (int): Hidden layer width (default: 10).
        training_data_rate (float): Fraction of samples used for training per round (default: 0.8).
        number_of_rounds (int): Number of random train/test rounds (default: 50).
        epoch_numbers (int): SGD epochs per round (default: 20).
        learning_rate (float): Step size for weight updates (default: 0.01).
        verbose (bool): If True, prints accuracy statistics and runtime.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            weight1: (number_hidden_neurons, n_features + 1)  # input→hidden (with bias)
            weight2: (number_neurons,        number_hidden_neurons + 1)  # hidden→output (with bias)

    """
    
    start_time = time.time()
    quantity_of_data, number_features = x_values.shape
    quantity_of_train_data = int(np.floor(quantity_of_data * training_data_rate))
    quantity_of_test_data = quantity_of_data - quantity_of_train_data

    percentage_success = []
    for _ in range(number_of_rounds):
        index_data = np.random.permutation(quantity_of_data)
        x_shuff, y_shuff = x_values[index_data], y_values[index_data]

        x_train, x_test = x_shuff[:quantity_of_train_data], x_shuff[quantity_of_train_data:]
        y_train, y_test = y_shuff[:quantity_of_train_data], y_shuff[quantity_of_train_data:]
        
        x_train_b = np.hstack([np.ones((quantity_of_train_data, 1)), x_train])     
        x_test_b = np.hstack([np.ones((quantity_of_test_data, 1)),  x_test])      

        w1_scale = 1.0/np.sqrt(number_features + 1)
        w2_scale = 1.0/np.sqrt(number_hidden_neurons + 1)
        weight1 = w1_scale*np.random.randn(number_hidden_neurons, number_features + 1)
        weight2 = w2_scale*np.random.randn(number_neurons, number_hidden_neurons + 1)
        
        for _ in range(epoch_numbers):
            order = np.random.permutation(quantity_of_train_data)
            for tt in order:
                x_value = x_train_b[tt]     
                y_value = y_train[tt]       
                
                y_pred_h_net = weight1 @ x_value                  
                y_pred_h = sigmoid(y_pred_h_net)              
                y_pred_hb = np.hstack([1.0, y_pred_h])         
                y_pred_o_net = weight2 @ y_pred_hb                
                y_pred_o = sigmoid(y_pred_o_net)              
                
                d_o = (y_pred_o - y_value)                        
                d_h = (weight2[:, 1:].T @ d_o)*derivate_sigmoid_from_output(y_pred_h)  

                weight2 -= learning_rate*np.outer(d_o, y_pred_hb)  
                weight1 -= learning_rate*np.outer(d_h, x_value)     

        yi_test = sigmoid(x_test_b @ weight1.T)
        yb_test = np.hstack([np.ones((quantity_of_test_data, 1)), yi_test])
        y_pred_test = sigmoid(yb_test @ weight2.T)               

        number_success = eval_classifier(y_real=y_test, y_pred=y_pred_test)
        percentage_success.append(100.0 * number_success / quantity_of_test_data)

    end_time = time.time()
    if verbose:
        generate_results(percentage_success=percentage_success, time=end_time-start_time, name="MLP-1H(Sigmoid)")
        
    return weight1, weight2


def mlp_tanh_classifier_2h(x_values, y_values, number_neurons:int, number_hidden_neurons:int=64, 
        number_hidden_neurons2:int=32, training_data_rate:float=0.8, number_of_rounds:int=50,
        epoch_numbers:int=100, learning_rate:float=0.1, verbose:bool=False):

    """
    Train/evaluate a two-hidden-layer MLP classifier with tanh activations via per-sample SGD.

    Each of `number_of_rounds` rounds shuffles and splits the data into train/test
    (`training_data_rate`). A bias (ones) column is prepended to X; weights for
    input→H1 (`weight1`), H1→H2 (`weight2`), and H2→output (`weight3`) are initialized
    with N(0, 1/√fan_in). For `epoch_numbers` epochs, the model runs forward passes
    (tanh at H1/H2/output), computes deltas using `derivate_tanh_from_output`, and
    updates parameters with step size `learning_rate`. After training, test predictions
    are produced, class labels are taken by argmax, and accuracy (%) is recorded. If
    `verbose=True`, an accuracy summary and elapsed time are printed.

    Args:
        x_values (np.ndarray): Feature matrix, shape (n_samples, n_features).
        y_values (np.ndarray): One-hot targets, shape (n_samples, n_classes).
        number_neurons (int): Number of output neurons/classes (should equal n_classes).
        number_hidden_neurons (int): Width of the first hidden layer (default: 64).
        number_hidden_neurons2 (int): Width of the second hidden layer (default: 32).
        training_data_rate (float): Fraction of samples used for training per round (default: 0.8).
        number_of_rounds (int): Number of random train/test rounds (default: 50).
        epoch_numbers (int): SGD epochs per round (default: 100).
        learning_rate (float): Step size for weight updates (default: 0.1).
        verbose (bool): If True, prints accuracy statistics and runtime.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            weight1: (number_hidden_neurons,  n_features + 1)            # input → H1 (with bias)
            weight2: (number_hidden_neurons2, number_hidden_neurons + 1) # H1 → H2 (with bias)
            weight3: (number_neurons,        number_hidden_neurons2 + 1) # H2 → output (with bias)

    Notes:
        - Classification uses `argmax` over output activations (no softmax).
        - The returned weights are from the **last** train/test round.
    """
    
    start_time = time.time()
    quantity_of_data, number_features = x_values.shape
    quantity_of_train_data = int(np.floor(quantity_of_data * training_data_rate))
    quantity_of_test_data = quantity_of_data - quantity_of_train_data
    
    percentage_success = []
    for _ in range(number_of_rounds):
        index_data = np.random.permutation(quantity_of_data)
        x_shuff, y_shuff = x_values[index_data], y_values[index_data]

        x_train, x_test = x_shuff[:quantity_of_train_data], x_shuff[quantity_of_train_data:]
        y_train, y_test = y_shuff[:quantity_of_train_data], y_shuff[quantity_of_train_data:]

        x_train_b = np.hstack([np.ones((quantity_of_train_data, 1)), x_train])     
        x_test_b = np.hstack([np.ones((quantity_of_test_data, 1)),  x_test])    

        w1_scale = 1.0/np.sqrt(number_features + 1)
        w2_scale = 1.0/np.sqrt(number_hidden_neurons + 1)
        w3_scale = 1.0/np.sqrt(number_hidden_neurons2 + 1)
        weight1 = w1_scale * np.random.randn(number_hidden_neurons, number_features + 1)
        weight2 = w2_scale * np.random.randn(number_hidden_neurons2, number_hidden_neurons + 1)
        weight3 = w3_scale * np.random.randn(number_neurons, number_hidden_neurons2 + 1)
        
        for _ in range(epoch_numbers):
            order = np.random.permutation(quantity_of_train_data)
            
            for tt in order:
                x_value = x_train_b[tt]     
                y_value = y_train[tt]

                y1 = tanh(weight1 @ x_value)
                y1b = np.hstack([1.0, y1])
                y2 = tanh(weight2 @ y1b)
                y2b = np.hstack([1.0, y2])
                y3 = tanh(weight3 @ y2b)                     

                d3 = derivate_tanh_from_output(y3)*(y_value - y3)
                d2 = derivate_tanh_from_output(y2)*(weight3[:,1:].T @ d3)
                d1 = derivate_tanh_from_output(y1)*(weight2[:,1:].T @ d2)

                weight3 += learning_rate*np.outer(d3, y2b)
                weight2 += learning_rate*np.outer(d2, y1b)
                weight1 += learning_rate*np.outer(d1, x_value)

        y1_te = tanh(x_test_b @ weight1.T)
        y1b_te = np.hstack([np.full((quantity_of_test_data,1), 1.0), y1_te])
        y2_te = tanh(y1b_te @ weight2.T)
        y2b_te = np.hstack([np.full((quantity_of_test_data,1), 1.0), y2_te])
        y_pred_test = tanh(y2b_te @ weight3.T)

        number_success = eval_classifier(y_real=y_test, y_pred=y_pred_test)
        percentage_success.append(100.0 * number_success / quantity_of_test_data)

    end_time = time.time()
    if verbose:
        generate_results(percentage_success=percentage_success, time=end_time-start_time, name="MLP-2H(Tanh)")

    return weight1, weight2, weight3

   