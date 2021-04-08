import tensorflow as tf

class position_loss(tf.keras.losses.Loss):
    def __init__(self):
        super(position_loss, self).__init__()

    def call(self, y_true, y_pred):

        y_pred_index=tf.squeeze(y_pred, axis=3)
        # index_pred = tf.map_fn(lambda x: tf.reduce_mean(tf.cast(tf.where((x > 0.95)), dtype=tf.float32), axis=0), elems=y_pred_index,dtype=tf.float32)

        y_true_index = tf.squeeze(y_true, axis=3)

        # index_true = tf.map_fn(lambda y: tf.reduce_mean(tf.cast(tf.where((y ==1.0)), dtype=tf.float32), axis=0),elems=y_true_index, dtype=tf.float32)

        centriod_pred=tf.map_fn(lambda x:self.cal_position(x),elems=y_pred_index)
        centriod_true=tf.map_fn(lambda y:self.cal_position(y),elems=y_true_index)

        loss0=tf.keras.losses.binary_crossentropy(y_true, y_pred)
        loss1=tf.reduce_mean(tf.square(y_true-y_pred))*100
        loss2=tf.reduce_mean(tf.square(centriod_pred-centriod_true))

        tf.print(loss1,loss2)
        return loss0+loss1+loss2

    def cal_position(self,y_pre):

        x = tf.cast(tf.range(0, tf.shape(y_pre)[0]), dtype=tf.float32)
        x = tf.reshape(x, (1, tf.shape(y_pre)[0]))
        x = tf.tile(x, (tf.shape(y_pre)[0], 1))

        y = tf.cast(tf.range(0, tf.shape(y_pre)[0]), dtype=tf.float32)
        y = tf.reshape(y, (tf.shape(y_pre)[0], 1))
        y = tf.tile(y, (1, tf.shape(y_pre)[0]))

        M00=tf.reduce_sum(y_pre)
        M10=tf.reduce_sum(x*y_pre)
        M01=tf.reduce_sum(y*y_pre)

        position_x=M10/M00
        position_y=M01/M00

        position_x=tf.reshape(position_x,(1,1))
        position_y=tf.reshape(position_y,(1,1))

        position=tf.concat((position_x,position_y),axis=1)

        return position