import tensorflow as tf
Sequential    = tf.keras.models.Sequential
Dense         = tf.keras.layers.Dense
Dropout       = tf.keras.layers.Dropout
Input         = tf.keras.layers.Input
LSTM          = tf.keras.layers.LSTM

def build_lstm(n_features, n_lstm_layers, units_per_layer, dropout_per_layer, dense_units, learning_rate):
    layers = [Input(shape=(1, n_features))]
    for i in range(n_lstm_layers):
        return_seq = (i < n_lstm_layers - 1)
        layers.append(LSTM(units_per_layer[i], return_sequences=return_seq))
        layers.append(Dropout(dropout_per_layer[i]))

    layers.append(Dense(dense_units, activation="relu"))
    layers.append(Dense(1))

    model = Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error"
    )
    return model