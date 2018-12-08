import tensorflow as tf


def softmax_loss(y_true: tf.Tensor, y_pred: tf.Tensor, **kwargs) -> tf.Tensor:
    """
    computes softmax cross entropy loss from logits
    """
    
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))


def batch_cosine_similarity_tf(x1, x2):
    """
    Cosine similarity between x1 and x2
    """
    
    dot = tf.einsum('ij,ij->i', x1, x2)
    return dot


def triplet_loss_tf(y_true: tf.Tensor, y_pred: tf.Tensor, **kwargs) -> tf.Tensor:
    """
    Triplet loss for speaker identification.
    Convention:
    
    y_pred has shape [3*batch_size, ...]
    ->
    fst batch_size elements: encoding of anchor
    snd batch_size elements: encoding of positive sample
    trd batch_size elements: encoding of negative sample
    
    args:
        y_pred: contains encoding of anchor, positive and negative sample in this order. -> shape (3*batch_size, ...)
        y_true: dummy tensor with zeros of shape y_pred.shape
    returns:
        total_loss
    """
    
    # get hyperparameters from kwargs
    alpha = float(kwargs['alpha'])
    batch_size = int(kwargs['batch_size'])
    
    # compute similarties
    anchor = y_pred[0:batch_size]
    positive_ex = y_pred[batch_size:2*batch_size]
    negative_ex = y_pred[2*batch_size:]
    pos_sim = batch_cosine_similarity_tf(anchor, positive_ex)
    neg_sim = batch_cosine_similarity_tf(anchor, negative_ex)
    
    # compute entire loss
    loss = tf.maximum(neg_sim - pos_sim + tf.constant(alpha), 0.0)
    total_loss = tf.reduce_mean(loss)
    return total_loss
