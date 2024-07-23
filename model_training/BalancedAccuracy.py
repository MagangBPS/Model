import tensorflow as tf

class BalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='balanced_accuracy', **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool)

        self.true_positives.assign_add(tf.reduce_sum(tf.cast(y_true & y_pred, self.dtype)))
        self.true_negatives.assign_add(tf.reduce_sum(tf.cast(~y_true & ~y_pred, self.dtype)))
        self.false_positives.assign_add(tf.reduce_sum(tf.cast(~y_true & y_pred, self.dtype)))
        self.false_negatives.assign_add(tf.reduce_sum(tf.cast(y_true & ~y_pred, self.dtype)))

    def result(self):
        sensitivity = self.true_positives / (self.true_positives + self.false_negatives)
        specificity = self.true_negatives / (self.true_negatives + self.false_positives)
        balanced_accuracy = (sensitivity + specificity) / 2
        return balanced_accuracy

    def reset_state(self):
        self.true_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)
