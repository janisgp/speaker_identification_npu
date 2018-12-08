import tensorflow as tf

def LSTM_layer(X, hidden_dim, state_np_init, n_time, return_sequence=False):
    """
    Creates one unidirectional LSTM layer.
    
    args:
        X: input placeholder of shape [batch_size, n_time*features_dim]
        hidden_dim: int, dimension of hidden states of LSTM
        state_np_init: initial state of shape [batch_size, hidden_dim]
        n_time: int, number of time steps
        return_sequence: bool, whether or not to return entire sequence of hidden states
        
    returns:
        outputs: either last hidden state or entire sequence
    
    """
    
    # split operation only support the shape[axis] with integer multiple of 16
    X_in = tf.split(X, n_time, 1)
    
    # define LSTM cell
    lstm_cell = tf.contrib.rnn.LSTMCell(hidden_dim)
    
    # create initial state
    cell_state = tf.convert_to_tensor(state_np_init, dtype=tf.float32)
    hidden_state = tf.convert_to_tensor(state_np_init, dtype=tf.float32)
    state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)
    
    outputs, states = tf.nn.static_rnn(lstm_cell, X_in, initial_state=state, dtype=tf.float32)
    
    if return_sequence:
        return outputs
    else:
        return outputs[-1]
        

def get_simple_LSTM_encoder(encoding_dim: int, n_steps: int, n_feature_dim: int):
    """
    Returns a one layer uni-directional LSTM encoder
    
    args:
        encsoding_dim: int, dimnensionality of encoding
        n_steps: int, number of time steps
        n_feature_dim: int, dim of each feature vector at each time step
        
    returns:
        input_dict: dictionary holding input nodes
        output_dict: dictionary holding output nodes
        label_dict: dictionary holding target input nodes
    """
    
    # adapt to huawei dim limitation of multiples of 16
    n_steps = n_steps + (n_steps % 16 > 0)*(16 - n_steps % 16)
    n_feature_dim = n_feature_dim + (n_feature_dim % 16 > 0)*(16 - n_feature_dim % 16)
    
    # input
    x = tf.placeholder(tf.float32, [None, n_steps * n_feature_dim])
    state_np_init = tf.placeholder(tf.float32, [None, encoding_dim])
    
    # get encoding
    with tf.name_scope("Encoding"):
        encoding = LSTM_layer(x, encoding_dim, state_np_init, n_steps, )
    
    # prepare endpoints of graph
    input_dict = {
        'x': x,
        'state': state_np_init
    }
    output_dict = {
        'encoding': encoding
    }
    label_dict = {
        'encoding': tf.placeholder(tf.float32, [None, encoding_dim])
    }
    
    return input_dict, output_dict, label_dict

def get_simple_LSTM_classification(encoding_dim: int, n_steps: int, n_feature_dim: int, num_classes: int):
    """
    Returns endpoints of classification model.
    
    args:
        encsoding_dim: int, dimnensionality of encoding
        n_steps: int, number of time steps
        n_feature_dim: int, dim of each feature vector at each time step
        num_classes: int, number of classes
        
    returns:
        input_dict: dictionary holding input nodes
        output_dict: dictionary holding output nodes
        label_dict: dictionary holding target input nodes
    """
    
    # get encoding and model inputs
    input_dict_encoder, output_dict_encoder, _ = get_simple_LSTM_encoder(encoding_dim, n_steps, n_feature_dim)
    y = tf.placeholder(tf.float32, [None, num_classes])
    
    # output
    logits =  tf.layers.dense(output_dict_encoder['encoding'], num_classes)
    
    # prepare endpoints of graph
    input_dict = {
        'x': input_dict_encoder['x'],
        'state': input_dict_encoder['state']
    }
    output_dict = {
        'logits': logits
    }
    label_dict = {
        'logits': y
    }
    
    return input_dict, output_dict, label_dict
