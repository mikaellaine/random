
import numpy as np
import tensorflow as tf


# In[14]:


def model_function(features, labels, mode):
    i = tf.reshape(features, [1,1])
    a = tf.layers.dense(inputs=i, units=1, use_bias=False, kernel_initializer=tf.constant_initializer(0.8))
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=a)
    
    l = tf.reshape(labels, [1,1])
    loss = tf.losses.mean_squared_error(labels=l, predictions=a)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


# In[15]:


estimator = tf.estimator.Estimator(model_fn=model_function, model_dir="simplemodel")


# In[6]:


train_inputs = np.asarray( [1.5], dtype=np.float32 )
train_labels = np.asarray( [0.5], dtype=np.float32 )
def train_input():
    return train_inputs, train_labels


# In[12]:


estimator.train(train_input,steps=10)


# In[8]:


print(str(estimator.get_variable_names()))


# In[13]:


print("Weigth value:"+str(estimator.get_variable_value("dense/kernel")))


# In[16]:


test_data = np.asarray([1.5], dtype=np.float32)
test_input_fn = tf.estimator.inputs.numpy_input_fn( x=test_data, num_epochs=1, shuffle=False )
pred_results = estimator.predict(input_fn=test_input_fn) #eval_results is a generator
actual_result = list(pred_results) #actual_result is an array
value = actual_result[0]# of type numpy.ndarray
print("The output is:"+str(value[0]))


# In[ ]: This section is provided simply for deleting the model directory if you wish to start everything from scratch


import shutil
shutil.rmtree("simplemodel")

