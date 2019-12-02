import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

number_of_inputs = 63
number_of_outputs = 1
learning_rate=0.001
training_epochs=100
display_step=5

layer_1_nodes=50
layer_2_nodes=100
layer_3_nodes=50

def readData():
    
    global x_scaled_training, y_scaled_training, x_scaled_testing, y_scaled_testing, x_scaler, y_scaler
    
    dataFrame = pd.read_csv("house_data.csv")
    
    del dataFrame["house_number"]
    del dataFrame['street_name']
    del dataFrame['unit_number']
    del dataFrame['zip_code']

    featuresDataFrame = pd.get_dummies(dataFrame, columns=["city", "garage_type"])
    
    del featuresDataFrame['sale_price']
    
    global x_train, x_test, y_train, y_test
    
    x=featuresDataFrame.as_matrix()
    y=dataFrame[['sale_price']].as_matrix()
    
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    
    x_scaled_training = x_scaler.fit_transform(x_train)
    y_scaled_training = y_scaler.fit_transform(y_train)
    
    x_scaled_testing = x_scaler.transform(x_test)
    y_scaled_testing = y_scaler.transform(y_test)
    
    print(len(x_train[0]), end="\n\n")
#     print(x_scaled_training[:5], end="\n\n")
#     print("The scale on X_data is: \n", x_scaler.scale_, "\nWith adjustments of: \n", x_scaler.min_)
#     print("\nThe scale on Y_data is: \n", y_scaler.scale_, "\nWith adjustments of: \n", y_scaler.min_)
#     print("\nNote: Y values were scaled by multiplying by {:.10f} and adding {:.4f}".format(Y_scaler.scale_[0], y_scaler.min_[0]))



def trainModel():
    global number_of_inputs, number_of_outputs, learning_rate, training_epochs, display_step, layer_1_nodes, layer_2_nodes, layer_3_nodes
    
    with tf.variable_scope('input'):
        x = tf.placeholder(tf.float32, shape=(None, number_of_inputs))
        
    with tf.variable_scope('layer_1'):
        weights = tf.get_variable(name='weights_1', shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name='biases_1', shape=[layer_1_nodes], initializer=tf.zeros_initializer())
        # Using relu and matrix multiplication to define the activation function
        layer_1_output = tf.nn.relu(tf.matmul(x, weights) + biases)
        
    with tf.variable_scope('layer_2'):
        weights = tf.get_variable(name='weights_2', shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name='biases_2', shape=[layer_2_nodes], initializer=tf.zeros_initializer())
        # Using relu and matrix multiplication to define the activation function
        layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)
        
    with tf.variable_scope('layer_3'):
        weights = tf.get_variable(name='weights_3', shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name='biases_3', shape=[layer_3_nodes], initializer=tf.zeros_initializer())
        # Using relu and matrix multiplication to define the activation function
        layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)
        
    with tf.variable_scope('output'):
        weights = tf.get_variable(name='weights_4', shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name='biases_4', shape=[number_of_outputs], initializer=tf.zeros_initializer())
        # Using relu and matrix multiplication to define the activation function
        output = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)
        
    with tf.variable_scope('cost'):
        y=tf.placeholder(tf.float32, shape=(None, 1))
        cost=tf.reduce_mean(tf.squared_difference(output, y))

    with tf.variable_scope('train'):
        optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)
        
    with tf.variable_scope('logging'):
        tf.summary.scalar('current_cost', cost)
        log = tf.summary.merge_all()
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        
        training_writer = tf.summary.FileWriter('./Logs/training', session.graph)
        testing_writer = tf.summary.FileWriter('./Logs/testing', session.graph)
        
        for i in range(training_epochs):
            session.run(optimizer, feed_dict={x : x_scaled_training, y : y_scaled_training})
            training_cost, training_prediction, training_log=session.run([cost, output, log], feed_dict={x: x_scaled_training, y: y_scaled_training})
            testing_cost, testing_prediction, testing_log=session.run([cost, output, log], feed_dict={x: x_scaled_testing, y: y_scaled_testing})
            
            training_writer.add_summary(training_log, i)
            testing_writer.add_summary(testing_log, i)
            
            print("Training Pass: {}".format(i))
            print("Training Cost:", training_cost)
            print("Testing Cost: ", testing_cost)
            #print("Training Prediction:", training_prediction)
        print("Training Complete")
        
readData()
trainModel()
        