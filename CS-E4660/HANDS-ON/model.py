# MLflow start recording training metadata with mlflow.start_run():
# Init ML model
model = keras.Sequential()
# check number of parameter 
n = sys.argv[1] if len(sys.argv) > 1 else 2

node_param = [] 
# Load defaut model configuration if not given
file_name = sys.argv[2] if len(sys.argv) > 2 else "conf.txt"
with open(file_name, 'r') as f:
    content = f.read()
    node_param = content.split(",")

# setup model layer based on loaded configuration
for i in range(int(n)):
    model.add(layers.LSTM(int(node_param[i]), return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(1)))
# Setup model optimizer
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.005))
# Train ML model
fitted_model = model.fit(train_features, train_labels, epochs=2, batch_size=1, verbose=2, validation_data=(test_features, test_labels))
# Create model signature
signature = infer_signature(test_features, model.predict(test_features))
# Let's check out how it looks
# MLflow log model training parameter
mlflow.log_param("number of layer", n)
mlflow.log_param("number of node each layer", node_param)
fit_history = fitted_model.history
# MLflow log training metric
for key in fit_history:
    mlflow.log_metric(key, fit_history[key][-1])

model_dir_path = "./saved_model"

# Create an input example to store in the MLflow model registry
input_example = np.expand_dims(train_features[0], axis=0)

# Let's log the model in the MLflow model registry
model_name = 'LSTM_model'
mlflow.keras.log_model(model,model_name, signature=signature, input_example=input_example)