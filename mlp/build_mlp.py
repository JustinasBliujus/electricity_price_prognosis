import tensorflow as tf
Sequential = tf.keras.models.Sequential
Dense      = tf.keras.layers.Dense
Dropout    = tf.keras.layers.Dropout
Input      = tf.keras.layers.Input

def build_mlp(n_features,n_mlp_layers, dropout_rate, dense_units, learning_rate,activation):
    layers = [Input(shape=(n_features,))]
    for i in range(n_mlp_layers):
        layers.append(Dense(dense_units[i], activation=activation))
        layers.append(Dropout(dropout_rate[i]))
    layers.append(Dense(1))
 
    model = Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
    )
    return model