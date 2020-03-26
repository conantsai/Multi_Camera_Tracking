import tensorflow as tf

class Layer:
    ## Define convolution layer
    def Conv2D(self, input_op, output_op, kernel_height:int, kernel_width:int, stride_height:int, stride_width:int, norm_init_mean:float, 
               norm_init_stddev:float, const_init:float, padding:str, name:str):
        """[Define convolution layer]
        
        Arguments:
            input_op {tensor} -- [Input tensor]
            output_op {tensor} -- [Output tensor]
            kernel_height {int} -- [Kernel height]
            kernel_width {int} -- [Kernel width]
            stride_height {int} -- [Step height length]
            stride_width {int} -- [Step width length]
            padding {str} -- [Padding algorithm]
            name {str} -- [Layer name]
        
        Returns:
            [tensor] -- [Convolution layer]
        """    
        ## Get number of input channels
        input_channel = input_op.get_shape()[-1].value
        with tf.variable_scope(name) as scope:
            ## Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable(name="weights",
                                                shape=[kernel_height, kernel_width, input_channel, output_op],
                                                dtype=tf.float32,
                                                initializer=tf.random_normal_initializer(mean = norm_init_mean, stddev = norm_init_stddev))
            biases = tf.get_variable(name="biases", 
                                               shape = [output_op], 
                                               dtype=tf.float32, 
                                               initializer = tf.constant_initializer(const_init))

            ## Perform convolution.
            conv = tf.nn.conv2d(input=input_op, 
                                filter=weights, 
                                strides = [1, stride_height, stride_width, 1], 
                                padding=padding,
                                name="conv")

            ## Add the biases.
            pre_activation = tf.nn.bias_add(value=conv, 
                                    bias=biases,
                                    name="pre_activation")
            ## Apply ReLu non linearity.                        
            activation = tf.nn.relu(features=pre_activation,
                                    name="relu")
            # print(activation)
            # print_layer(activation)
            return activation

    ## Define max pool layer
    def MaxPool2D(self, input_op, kernel_height:int, kernel_width:int, stride_height:int, stride_width:int, padding:int, name:str):
        """[Define max pool layer]
        
        Arguments:
            input_op {tensor} -- [input tensor]
            kernel_height {int} -- [kernel height]
            kernel_width {int} -- [kernel width]
            stride_height {int} -- [tep height length]
            stride_width {int} -- [step width length]
            padding {int} -- [padding algorithm]
            name {str} -- [layer name]
        
        Returns:
            [tensor] -- [max_pool layer]
        """    
        with tf.variable_scope(name) as scope:
            maxpool = tf.nn.max_pool(value=input_op,
                                        ksize=[1, kernel_height, kernel_width, 1],
                                        strides=[1, stride_height, stride_width, 1],
                                        padding=padding,
                                        name=name)
        return maxpool

    ## Define dropout layer
    def Dropout(self, input_op, drop_prob:float, name:str):
        """[summary]
        
        Arguments:
            input_op {tensor} -- [input tensor]
            drop_prob {float} -- [Probability of each element being drop]
            name {str} -- [layer name]
        
        Returns:
            [tensor] -- [dropout layer]
        """    
        with tf.variable_scope(name) as scope:
            drop = tf.nn.dropout(x=input_op, 
                            rate=drop_prob,
                            name=name)
        return drop

    ## Define flatten layer
    def Flatten(self, input_op, shape, name:str):
        """[summary]
        
        Arguments:
            input_op {tensor} -- [input tensor]
            shape {tensor} -- [reshape shape]
            name {str} -- [layer name]
        
        Returns:
            [tensor] -- [flatten layer]
        """      
        with tf.variable_scope(name) as scope:
            flatten = tf.reshape(tensor=input_op,
                        shape=shape,
                        name=name)
        return flatten

        
    ## define fullyconnected layer
    def FullyConnected(self, input_op, output_op, norm_init_mean:float, norm_init_stddev:float, const_init:float, activation_function:str, name:str):
        """[Define fullyconnected layer]
        
        Arguments:
            input_op {tensor} -- [input tensor]
            output_op {tensor} -- [output tensor]
            name {str} -- [layer name]
        
        Returns:
            [tensor] -- [activation]
        """    
        input_size = input_op.get_shape()[-1].value
        output_size = output_op
        with tf.variable_scope(name) as scope:
            weight = tf.get_variable(name="weight",
                                            shape=[input_size, output_size],
                                            dtype=tf.float32,
                                            initializer=tf.random_normal_initializer(mean = norm_init_mean, stddev = norm_init_stddev))
            
            bias_initval = tf.get_variable(name="bias", 
                                                    shape = [output_size], 
                                                    dtype=tf.float32,
                                                    initializer = tf.constant_initializer(const_init))

            biases = tf.nn.bias_add(value=tf.einsum('ijkl,lm->ijkm', input_op, weight), 
                                    bias=bias_initval,
                                    name="biases")

            if activation_function =="relu":
                activation = tf.nn.relu(features=biases,
                                        name="relu")
            elif activation_function == "softmax":
                activation = tf.nn.softmax(logits=biases,
                                           name="softmax")
                                    
            # print_layer(activation)
            return activation

    def Conv2D_reuse(self, input_op, output_op, kernel_height:int, kernel_width:int, stride_height:int, stride_width:int, norm_init_mean:float, 
               norm_init_stddev:float, const_init:float, padding:str, name:str):
        """[Define convolution layer]
        
        Arguments:
            input_op {tensor} -- [Input tensor]
            output_op {tensor} -- [Output tensor]
            kernel_height {int} -- [Kernel height]
            kernel_width {int} -- [Kernel width]
            stride_height {int} -- [Step height length]
            stride_width {int} -- [Step width length]
            padding {str} -- [Padding algorithm]
            name {str} -- [Layer name]
        
        Returns:
            [tensor] -- [Convolution layer]
        """    
        ## Get number of input channels
        input_channel = input_op.get_shape()[-1].value
        with tf.variable_scope(name, reuse=True) as scope:
            ## Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable(name="weights",
                                                shape=[kernel_height, kernel_width, input_channel, output_op],
                                                dtype=tf.float32,
                                                initializer=tf.random_normal_initializer(mean = norm_init_mean, stddev = norm_init_stddev))
            biases = tf.get_variable(name="biases", 
                                               shape = [output_op], 
                                               dtype=tf.float32, 
                                               initializer = tf.constant_initializer(const_init))

            ## Perform convolution.
            conv = tf.nn.conv2d(input=input_op, 
                                filter=weights, 
                                strides = [1, stride_height, stride_width, 1], 
                                padding=padding,
                                name="conv")

            ## Add the biases.
            pre_activation = tf.nn.bias_add(value=conv, 
                                    bias=biases,
                                    name="pre_activation")
            ## Apply ReLu non linearity.                        
            activation = tf.nn.relu(features=pre_activation,
                                    name="relu")
            # print(activation)
            # print_layer(activation)
            return activation

    ## Define max pool layer
    def MaxPool2D_reuse(self, input_op, kernel_height:int, kernel_width:int, stride_height:int, stride_width:int, padding:int, name:str):
        """[Define max pool layer]
        
        Arguments:
            input_op {tensor} -- [input tensor]
            kernel_height {int} -- [kernel height]
            kernel_width {int} -- [kernel width]
            stride_height {int} -- [tep height length]
            stride_width {int} -- [step width length]
            padding {int} -- [padding algorithm]
            name {str} -- [layer name]
        
        Returns:
            [tensor] -- [max_pool layer]
        """    
        with tf.variable_scope(name, reuse=True) as scope:
            maxpool = tf.nn.max_pool(value=input_op,
                                        ksize=[1, kernel_height, kernel_width, 1],
                                        strides=[1, stride_height, stride_width, 1],
                                        padding=padding,
                                        name=name)
        return maxpool

    ## Define dropout layer
    def Dropout_reuse(self, input_op, drop_prob:float, name:str):
        """[summary]
        
        Arguments:
            input_op {tensor} -- [input tensor]
            drop_prob {float} -- [Probability of each element being drop]
            name {str} -- [layer name]
        
        Returns:
            [tensor] -- [dropout layer]
        """    
        with tf.variable_scope(name, reuse=True) as scope:
            drop = tf.nn.dropout(x=input_op, 
                            rate=drop_prob,
                            name=name)
        return drop

    ## Define flatten layer
    def Flatten_reuse(self, input_op, shape, name:str):
        """[summary]
        
        Arguments:
            input_op {tensor} -- [input tensor]
            shape {tensor} -- [reshape shape]
            name {str} -- [layer name]
        
        Returns:
            [tensor] -- [flatten layer]
        """      
        with tf.variable_scope(name, reuse=True) as scope:
            flatten = tf.reshape(tensor=input_op,
                        shape=shape,
                        name=name)
        return flatten

        
    ## define fullyconnected layer
    def FullyConnected_reuse(self, input_op, output_op, norm_init_mean:float, norm_init_stddev:float, const_init:float, activation_function:str, name:str):
        """[Define fullyconnected layer]
        
        Arguments:
            input_op {tensor} -- [input tensor]
            output_op {tensor} -- [output tensor]
            name {str} -- [layer name]
        
        Returns:
            [tensor] -- [activation]
        """    
        input_size = input_op.get_shape()[-1].value
        output_size = output_op
        with tf.variable_scope(name, reuse=True) as scope:
            weight = tf.get_variable(name="weight",
                                            shape=[input_size, output_size],
                                            dtype=tf.float32,
                                            initializer=tf.random_normal_initializer(mean = norm_init_mean, stddev = norm_init_stddev))
            
            bias_initval = tf.get_variable(name="bias", 
                                                    shape = [output_size], 
                                                    dtype=tf.float32,
                                                    initializer = tf.constant_initializer(const_init))

            biases = tf.nn.bias_add(value=tf.einsum('ijkl,lm->ijkm', input_op, weight), 
                                    bias=bias_initval,
                                    name="biases")

            if activation_function =="relu":
                activation = tf.nn.relu(features=biases,
                                        name="relu")
            elif activation_function == "softmax":
                activation = tf.nn.softmax(logits=biases,
                                           name="softmax")
                                    
            # print_layer(activation)
            return activation
