import tensorflow as tf

gf = tf.GraphDef()
gf.ParseFromString(open('model\model.pb','rb').read())
print([n.name + '=>' +  n.op for n in gf.node if n.op in ('Placeholder', 'Shape')])
