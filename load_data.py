import os
import numpy as np


class load_and_process_data_individual():
        def __init__(self, data_directory, mode, window_size, stride, filtering, target_model, device):
                self.data_directory = data_directory
                self.training_model = mode
                self.window_size = window_size
                self.stride = stride
                self.filtering = filtering
                self.target_model = target_model
                self.device = device

        def loading(self):
                fascicle_length = []
                pennation_angle = []
                ultrasound_data = []
                
                for participant_folder in os.listdir(self.data_directory):
                        # if 'WW174' not in participant_folder:
                        #         continue

                        if '.DS' in participant_folder:
                                continue
                        
                        participant_directory = os.path.join(self.data_directory, participant_folder)


                        for trial_folder in os.listdir(participant_directory):
                                
                                if '.DS' in trial_folder:
                                        continue
                                print(trial_folder)
                                
                                trial_directory = os.path.join(participant_directory, trial_folder)

                                incorporated_ultrasound  = []
                                
                                for idx, csv_file in enumerate(os.listdir(trial_directory)):
                                        if not csv_file.endswith('.csv'):
                                                continue
                                        
                                        csv_file_path = os.path.join(trial_directory, csv_file)
                                        
                                        # time in column
                                        cur_data = np.loadtxt(csv_file_path, delimiter=',', dtype=np.str_)
                                        if self.device == 'mps':
                                                cur_data = cur_data.astype(np.float32)
                                        else:                                       
                                                cur_data = cur_data.astype(np.float64)
                                       
                                        cur_ultrasound = cur_data[2:, :]
                                        cur_ultrasound[np.isnan(cur_ultrasound).any(axis=1)] = 0

                                        incorporated_ultrasound.append(cur_ultrasound)

                                
                                incorporated_ultrasound = np.array(incorporated_ultrasound).T#/256.

                                incorporated_ultrasound = incorporated_ultrasound.reshape(incorporated_ultrasound.shape[0], 1,  incorporated_ultrasound.shape[1], incorporated_ultrasound.shape[2])

                                if self.filtering:
                                        from filtering import filter_series, hampel_filter
                                        import matplotlib.pyplot as plt

                                        cur_data[0, :] = filter_series(cur_data[0, :], 3) # len
                                        cur_data[1, :] = filter_series(cur_data[1, :], 3) # len



                                window_size = self.window_size
                                stride = self.stride


                                idx = 0
                                ultrasound = []
                                f_length = []
                                p_angle = []
                                while True:
                                        if idx+window_size >= incorporated_ultrasound.shape[0]:
                                                break
                                        
                                        ultrasound.append(incorporated_ultrasound[idx:idx+window_size])
                                        if 'transformer' in self.target_model:
                                                f_length.append(cur_data[0, idx:idx+window_size])
                                                p_angle.append(cur_data[1, idx:idx+window_size])
                                        else:
                                                f_length.append(cur_data[0, idx+window_size-1])
                                                p_angle.append(cur_data[1, idx+window_size-1])

                                        idx += stride
                                
                                incorporated_ultrasound = np.array(ultrasound)
                                f_length = np.array(f_length)
                                p_angle = np.array(p_angle)


                                fascicle_length.append(f_length)
                                pennation_angle.append(p_angle)
                                ultrasound_data.append(incorporated_ultrasound)        

                fascicle_length = np.concatenate(fascicle_length, 0)
                pennation_angle = np.concatenate(pennation_angle, 0)
                ultrasound_data = np.concatenate(ultrasound_data, 0)

                if 'transformer' in self.target_model:
                        ydata = np.array([fascicle_length, pennation_angle])
                        ydata = ydata.transpose(1, 2, 0)
                else:
                        ydata = np.array([fascicle_length, pennation_angle]).T



                return ultrasound_data, ydata



