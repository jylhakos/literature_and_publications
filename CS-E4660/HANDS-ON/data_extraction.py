# data_extraction.py

# Read test dataset from a different file
# We will test the trained ML model with data from station "1161114002" on the same parameter "122"
test_file_name = "./data_grouped/1161114002_122_.csv"
test_grouped_dataset = pd.read_csv(test_file_name)
test_grouped_dataset = test_grouped_dataset.astype({'id':'float','value':'float', 'station_id':'int', 'parameter_id':'int', 'unix_timestamp':'int', 'norm_time':'float'})
test_dataset = test_grouped_dataset.copy()
test_dataset = test_dataset.dropna().drop(['id','station_id','parameter_id','unix_timestamp'], axis=1)
test_dataset_full = test_dataset.sort_values(by=['norm_time'])
# Choose a small part of the data to test the model
start_line = 0
end_line = 100
test_data = test_dataset_full[start_line:end_line]
test_data = test_dataset_full[start_line:end_line]

# Similar to training dataset -> making a time series input for ML prediction
test_serial_data = test_data.drop(['value','norm_time'], axis=1)
test_serial_data['norm_1'] = test_serial_data['norm_value'].shift(1)
test_serial_data['norm_2'] = test_serial_data['norm_value'].shift(2)
test_serial_data['norm_3'] = test_serial_data['norm_value'].shift(3)
test_serial_data['norm_4'] = test_serial_data['norm_value'].shift(4)
test_serial_data['norm_5'] = test_serial_data['norm_value'].shift(5)
test_serial_data['norm_6'] = test_serial_data['norm_value'].shift(6)
test_dataset = test_serial_data[6:]