PARAMS = \
{
	'data_directory': '../Data',
	'mode': 'train',
	'target_model': 'cnn_st_lstm', 
	'target_output': 'length', # length vs angle
	'weight_path': './weights',
	'window_size': 5,
	'stride': 6,
	'filtering': True,
	'hidden_dim': 128,
	'output_dim': 1,
	'device': 'cpu',
	'epochs': 100,
    'batch_size': 64,
	'lr': 0.001,
	'weight_decay': 0.000001,
	'val_ratio': 0.1,
	'test_ratio': 0.1
    
}
