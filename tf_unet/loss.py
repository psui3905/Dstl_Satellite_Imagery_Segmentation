import tensorflow as tf

# TODO: convert every math ops to tensorflow ops
# TODO: check whether the jaccard loss for multi class is correct
# TODO: check again how to handle all zero y labels for jaccard loss

# Small constant to prevent division by zero/ is unlikely to be changed

epsilon = 1e-12

# cross entropy for sigmoid/softmax
def cross_entropy(y_,logits):

    if logits.shape[-1] == 1:
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=logits))
    else:
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits))

# IOU without rounding (soft)
def jaccard_coef(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = tf.reduce_sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + epsilon) / (sum_ - intersection + epsilon)

    return tf.reduce_mean(jac)

# intersection over union (hard)
def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = tf.round(tf.clip_by_value (y_pred, 0, 1))

    intersection = tf.reduce_sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = tf.reduce_sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + epsilon) / (sum_ - intersection + epsilon)

    return tf.reduce_mean(jac)

# loss based on soft jaccard ( -log(jaccard) )
def jaccard_coef_loss(y_true, y_pred):
    
    jaccard = tf.constant(0.0, dtype="float32")
    
    for c in range(y_pred.shape[-1].value):
        jaccard += jaccard_coef(y_true[:,:,:,c], y_pred[:,:,:,c])
    jaccard = tf.divide(jaccard, y_pred.shape[-1].value)

    return -tf.log(jaccard)

# average jaccard_coef_int
def jaccard_coef_int_avg(y_true, y_pred):

    jaccard_int = tf.constant(0.0, dtype="float32")
    # ignore class if y label is all zeros
    cls = y_pred.shape[-1].value
    for c in range(y_pred.shape[-1].value):
        jaccard_int += jaccard_coef_int(y_true[:,:,:,c], y_pred[:,:,:,c])
    jaccard_int = tf.divide(jaccard_int, cls)
    return jaccard_int

# loss based on dice
def dice_loss(y_true, y_pred):
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    squared_sum = tf.reduce_sum(tf.square(y_true)) + tf.reduce_sum(tf.square(y_pred))
    return 1. - 2 * (intersection + epsilon) / (squared_sum + epsilon)

# New jaccard loss
def new_jaccard(y_true, y_pred):

    IOU_by_Class = tf.Variable(tf.zeros([11], tf.float32))
    y_true = tf.argmax(y_true, -1)
    y_pred = tf.argmax(y_pred, -1)

    for cls in range(11):
        true_labels = tf.cast(tf.equal(y_true, cls), tf.float32)
        predicted_labels = tf.cast(tf.equal(y_pred, cls), tf.float32)
        intersection =  tf.reduce_sum(true_labels * predicted_labels, axis=[0, -1, -2])
        sum_ = tf.reduce_sum(true_labels + predicted_labels, axis=[0, -1, -2])
        IOU_by_Class[cls].assign((intersection + epsilon) / (sum_ - intersection + epsilon))
 
    return IOU_by_Class    

def get_loss(self, logits, cost_name, cost_kwargs):
    """
    Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
    Optional arguments are:
    class_weights: weights for the different classes in case of multi-class imbalance
    regularizer: power of the L2 regularizers added to the loss function
    """
    if logits.shape[-1] == 1:
        pred = tf.sigmoid(logits)
    else:
        pred = tf.nn.softmax(logits)        

    with tf.name_scope("cost"):
            
        # logit is needed to pass into cross entropy because it does sigmoid activation internally
        if cost_name == "cross_entropy":
            loss = cross_entropy(self.y, logits)   

        elif cost_name == "dice":
            loss = dice_loss(self.y, pred) * 4

        elif cost_name == "jaccard":
            loss = jaccard_coef_loss(self.y, pred) * 0.25

        elif cost_name == "cross_jac":
            loss = qw(self.y, pred) * 4 + cross_entropy(self.y, logits)

        elif cost_name == "cross_dice":
            loss = ( dice_loss(self.y, pred) * 4 + cross_entropy(self.y, logits) ) / 2.0

        elif cost_name == "L2":
            loss = tf.nn.l2_loss(self.y - logits)
        else:
            raise ValueError("Unknown cost function: " % cost_name)

        # l2 regularisation performed on the variables (Not yet implemented)
        #regularizer = cost_kwargs.pop("regularizer", None)
        #if regularizer == "L2":
        #    regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
        #    loss += (regularizer * regularizers)

        return loss
