# model performances
- nn_shift_mse_final: architecture is 256 input layer, 256 unit dense layer with ReLu activiation, 0.1 dropout layer, 64 unit dense layer with ReLu activation, 0.1 dropout layer, 8 unit dense layer with ReLu activation, and single output unit with linear activation. Trained for 15 epochs with Adam as the optimizer and using shift_mse loss ((1 + 1/ (1 + K.exp(-(y_pred - y_true)))))))) * K.square(y_pred - y_true); loss = K.mean(loss, axis = 1)). Results:
average number of states explored to find solution:
        for learned model: 204.888
        for manhattan distance: 43739.928
----------------------------------------------------
solution was non-optimal 4.5% of the time
----------------------------------------------------
average length of solution path was:
        for learned model: 28.024
        for manhattan distance: 27.928
- nn_exp_loss_2_final: architecture is 256 input layer, 256 unit dense layer with ReLu activiation, 0.1 dropout layer, 64 unit dense layer with ReLu activation, 0.1 dropout layer, 8 unit dense layer with ReLu activation, and single output unit with linear activation. Trained for 15 epochs with Adam as the optimizer and using exp_loss_2 (loss = K.exp((y_pred - y_true)) / 2;loss = loss + K.square(y_pred - y_true);loss = K.mean(loss, axis = 1)). Results:
average number of states explored to find solution:
        for learned model: 231.298
        for manhattan distance: 43739.928
----------------------------------------------------
solution was non-optimal 4.8% of the time
----------------------------------------------------
average length of solution path was:
        for learned model: 28.03
        for manhattan distance: 27.928
- nn_mse_final: architecture is 256 input layer, 256 unit dense layer with ReLu activiation, 0.1 dropout layer, 64 unit dense layer with ReLu activation, 0.1 dropout layer, 8 unit dense layer with ReLu activation, and single output unit with linear activation. Trained for 15 epochs with Adam as the optimizer and using mse loss. Results:
average number of states explored to find solution:
        for learned model: 210.307
        for manhattan distance: 43739.928
----------------------------------------------------
solution was non-optimal 5.2% of the time
----------------------------------------------------
average length of solution path was:
        for learned model: 28.036
        for manhattan distance: 27.928

- neural_net_exp_loss_2_data_rep_2: architecture is 288 input layer, 256 unit dense layer with ReLu activiation, 0.1 dropout layer, 64 unit dense layer with ReLu activation, 0.1 dropout layer, 8 unit dense layer with ReLu activation, and single output unit with linear activation. Data rep is one hot vectors and x and y distances from correct locations. Trained for 15 epochs with Adam as the optimizer and using exp_loss_2 (loss = K.exp((y_pred - y_true)) / 2;loss = loss + K.square(y_pred - y_true);loss = K.mean(loss, axis = 1)). Results:
average number of states explored to find solution:
        for learned model: 181.999
        for manhattan distance: 43739.928
----------------------------------------------------
solution was non-optimal 4.7% of the time
----------------------------------------------------
average length of solution path was:
        for learned model: 28.026
        for manhattan distance: 27.928
- neural_net_shift_mse_data_rep_2: architecture is 288 input layer, 256 unit dense layer with ReLu activiation, 0.1 dropout layer, 64 unit dense layer with ReLu activation, 0.1 dropout layer, 8 unit dense layer with ReLu activation, and single output unit with linear activation. Data rep is one hot vectors and x and y distances from correct locations. Trained for 15 epochs with Adam as the optimizer and using shift_mse loss ((1 + 1/ (1 + K.exp(-(y_pred - y_true)))))))) * K.square(y_pred - y_true); loss = K.mean(loss, axis = 1)). Results:
average number of states explored to find solution:
        for learned model: 183.922
        for manhattan distance: 43739.928
----------------------------------------------------
solution was non-optimal 3.8% of the time
----------------------------------------------------
average length of solution path was:
        for learned model: 28.008
        for manhattan distance: 27.928