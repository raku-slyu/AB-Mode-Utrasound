import os
from load_data import load_and_process_data_individual
from utils import load_params, data_split

class trainer():
	def __init__(self, params):
		self.params = params

	def define_model(self, ultrasound_data):
		if self.params['target_model'] == 'lstm':
			from models import LSTM_window
			model = LSTM_window(ultrasound_data.shape[-1]*ultrasound_data.shape[-2], int(self.params['hidden_dim']/2), self.params['output_dim'], num_layers=1)
		elif self.params['target_model'] == 'cnnlstm':
			from models import CNN_LSTM
			model = CNN_LSTM(self.params['window_size'], int(self.params['hidden_dim']/2), 
		    	self.params['output_dim'], num_layers=1)
		elif self.params['target_model'] == 'cnn_st_lstm':
			from models import CNN_stLSTM
			model = CNN_stLSTM(self.params['window_size'], int(self.params['hidden_dim']/2), 
		    	self.params['output_dim'], num_layers=1)
		elif self.params['target_model'] == 'cnn_transformer':
			from models import cnn_transformer
			model = cnn_transformer(d_model=self.params['window_size'], nhead=self.params['window_size'], 
				num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=self.params['hidden_dim'], dropout=0.1)
		elif self.params['target_model'] == 'cnn_st_transformer':
			from models import cnn_st_transformer
			model = cnn_st_transformer(d_model_t=self.params['window_size'], nhead_t=self.params['window_size'], 
				num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=self.params['hidden_dim'], dropout=0.1)


		self.model = model.to(self.params['device'])
	

	def train(self, x_train, x_val, y_train, y_val):
		import math
		import torch
		from dataloader import DataLoaderWhole
		from torch.utils.data import DataLoader

		if 'transformer' in self.params['target_model']:
			train_loader = DataLoader(DataLoaderWhole(x_train, \
				y_train[:,:,0], y_train[:,:,1]), \
				batch_size = self.params['batch_size'], shuffle = True, num_workers = 4)

			val_loader = DataLoader(DataLoaderWhole(x_val, \
				y_val[:,:,0], y_val[:,:,1]), \
				batch_size = self.params['batch_size'], shuffle = True, num_workers = 4)
		else:
			train_loader = DataLoader(DataLoaderWhole(x_train, \
				y_train[:,0], y_train[:,1]), \
				batch_size = self.params['batch_size'], shuffle = True, num_workers = 4)

			val_loader = DataLoader(DataLoaderWhole(x_val, \
				y_val[:,0], y_val[:,1]), \
				batch_size = self.params['batch_size'], shuffle = True, num_workers = 4)
		
		self.model.load_state_dict(torch.load('./weights/cnn_st_lstm_whole_length_w5_s6.pth'))

		criterion = torch.nn.MSELoss()
		optimizer = torch.optim.Adam(self.model.parameters(), lr = self.params['lr'], weight_decay = self.params['weight_decay'])

		min_error = math.inf

		for epoch in range(self.params['epochs']):
			total_loss = 0

			self.model.train()

			for i_batch, sample_batched in enumerate(train_loader):
				x_ultr = sample_batched['ultrasound_data'].to(self.params['device']).float()

				y_fasc_len = sample_batched['fascicle_lengths'].to(self.params['device']).float()
				y_penn_ang = sample_batched['pennation_angles'].to(self.params['device']).float()

				if 'transformer' in self.params['target_model']:
					if self.params['target_output'] == 'angle':
						
						y_pred = self.model(x_ultr, y_penn_ang)
						loss = criterion(y_pred.squeeze(1), y_penn_ang[:, -1])
					else:
						y_pred = self.model(x_ultr, y_fasc_len)
						loss = criterion(y_pred.squeeze(1), y_fasc_len[:, -1])
				else:
					y_pred = self.model(x_ultr)
					if self.params['target_output'] == 'angle':
						loss = criterion(y_pred.squeeze(1), y_penn_ang)
					else:
						loss = criterion(y_pred.squeeze(1), y_fasc_len)
						
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				total_loss += loss.item()
			
			self.model.eval()

			with torch.no_grad():
				total_val_loss = 0
				for i_batch_val, sample_batched in enumerate(val_loader):
					x_ultr = sample_batched['ultrasound_data'].to(self.params['device']).float()
					y_fasc_len = sample_batched['fascicle_lengths'].to(self.params['device']).float()
					y_penn_ang = sample_batched['pennation_angles'].to(self.params['device']).float()
						
					if 'transformer' in self.params['target_model']:
						if self.params['target_output'] == 'angle':
							y_pred = self.model(x_ultr, y_penn_ang)
							loss = criterion(y_pred.squeeze(1), y_penn_ang[:,-1])
						else:
							y_pred = self.model(x_ultr, y_fasc_len)
							loss = criterion(y_pred.squeeze(1), y_fasc_len[:,-1])
					else:
						y_pred = self.model(x_ultr)		
						if self.params['target_output'] == 'angle':
							loss = criterion(y_pred.squeeze(1), y_penn_ang)
						else:
							loss = criterion(y_pred.squeeze(1), y_fasc_len)

					total_val_loss += loss.item()

				if min_error > total_val_loss:
					min_error = total_val_loss
					weight_file = os.path.join(self.params['weight_path'], \
						self.params['target_model']+ '_whole_' 
						+ self.params['target_output']+ '_w' + str(self.params['window_size']) 
						+ '_s' + str(self.params['stride']) + '.pth')
					torch.save(self.model.state_dict(), weight_file)
					print('[Epoch %d] Train loss - total_loss: %.4f \t Val loss - total_loss: %.4f, weight updated' \
					% (epoch+1, total_loss/(i_batch+1), \
						total_val_loss/(i_batch_val+1)))
				else:
					print('[Epoch %d] Train loss - total_loss: %.4f \t Val loss - total_loss: %.4f' \
						% (epoch+1, total_loss/(i_batch+1), \
							total_val_loss/(i_batch_val+1)))

	def test(self, x_test, y_test):
		import numpy as np
		import torch
		from dataloader import DataLoaderWhole
		from torch.utils.data import DataLoader

		if 'transformer' in self.params['target_model']:
			test_loader = DataLoader(DataLoaderWhole(x_test, \
				y_test[:,:,0], y_test[:,:,1]), \
				batch_size = 1, shuffle = False, num_workers = 1)		
		else:
			test_loader = DataLoader(DataLoaderWhole(x_test, \
				y_test[:,0], y_test[:,1]), \
				batch_size = 1, shuffle = False, num_workers = 1)
		
		print("testing model")
		
		weight_file = os.path.join(self.params['weight_path'], \
			self.params['target_model']+ '_whole_' 
			+ self.params['target_output']+ '_w' + str(self.params['window_size']) 
			+ '_s' + str(self.params['stride']) + '.pth')
		
		self.model.load_state_dict(torch.load(weight_file))
		
		self.model.eval()
		save_arr = []
		with torch.no_grad():
			total_val_loss = 0
			for i_batch_val, sample_batched in enumerate(test_loader):
				
				x_ultr = sample_batched['ultrasound_data'].to(self.params['device']).float()
				y_fasc_len = sample_batched['fascicle_lengths'].float()#.reshape(-1,1)
				y_penn_ang = sample_batched['pennation_angles'].float()#.reshape(-1,1)
				if 'transformer' in self.params['target_model']:
					if self.params['target_output'] == 'angle':
						ypang = y_penn_ang.clone().to(self.params['device'])
						y_pred = self.model(x_ultr, ypang)
					else:
						yfsl = y_fasc_len.clone().to(self.params['device'])
						y_pred = self.model(x_ultr, yfsl)
					y_fasc_len = y_fasc_len[:, -1]
					y_penn_ang = y_penn_ang[:, -1]
				else:
					y_pred = self.model(x_ultr)
				
				y_pred = y_pred.squeeze(1)
				
				y_pred = y_pred.cpu().numpy().reshape(-1,1)
				y_fasc_len = y_fasc_len.reshape(-1,1)
				y_penn_ang = y_penn_ang.reshape(-1,1)
				if self.params['target_output'] == 'angle':
					save_arr.append(np.concatenate((y_penn_ang, y_pred), axis = 1))
				else:
					save_arr.append(np.concatenate((y_fasc_len, y_pred), axis = 1))
		
		save_arr = np.concatenate(save_arr,0)
		save_file_path = self.params['target_model'] +'_'+ \
	     self.params['target_output']+ '_w' + str(self.params['window_size']) + \
			'_s' + str(self.params['stride']) + '.csv'
		save_file_path = os.path.join('results', save_file_path )
		np.savetxt(save_file_path, save_arr, delimiter=',')
		print("Done")


def main():
	params = load_params()

	ld = load_and_process_data_individual(params['data_directory'], \
				       params['mode'], params['window_size'], params['stride'], \
						params['filtering'], params['target_model'], params['device'])
	ultrasound_data, ydata = ld.loading()

	ds = data_split(params['val_ratio'], params['test_ratio'])
	x_train, x_val, x_test, y_train, y_val, y_test = ds.splitter(ultrasound_data, ydata)

	tr = trainer(params)
	tr.define_model(ultrasound_data)

	if params['mode'] == 'train':
		tr.train(x_train, x_val, y_train, y_val)
	else:
		tr.test(x_test, y_test)
				
		


if __name__ == '__main__':
	main()
