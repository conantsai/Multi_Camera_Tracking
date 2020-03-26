import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow

def freeze_graph(input_checkpoint,output_graph):
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "model/flatten1/Flatten/flatten/Reshape"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph() # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    output_name = "" 
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #恢复图并得到数据

        # graph = sess.graph
        # for i, n in enumerate(graph.as_graph_def().node):
        #     if i == 0:
        #         output_name =  n.name
        #     else:
        #         output_name = output_name + "," + n.name 

        # reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
        # var_to_shape_map = reader.get_variable_to_shape_map()
        # for i, key in enumerate(var_to_shape_map):
        #     if i == 0:
        #         output_name =  key
        #     else:
        #         output_name = output_name + "," + key
        # print(output_name)
        
        output_graph_def = graph_util.convert_variables_to_constants(  
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(","))
 
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        # print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点
 
        # for op in graph.get_operations():
        #     print(op.name, op.values())


if __name__ == "__main__":
    freeze_graph(input_checkpoint="model/model.ckpt",output_graph="pb/frozen_model.pb")
    pass
