class DataLoaderWhole(object):
    def __init__(self, ultrasound_data, fascicle_lengths, pennation_angles):
        self.ultrasound_data = ultrasound_data
        self.fascicle_lengths = fascicle_lengths
        self.pennation_angles = pennation_angles

    def __getitem__(self, index):
        sample = {
            'ultrasound_data': self.ultrasound_data[index],
            'fascicle_lengths': self.fascicle_lengths[index],
            'pennation_angles': self.pennation_angles[index]
        }
        return sample

    def __len__(self):
        return len(self.ultrasound_data)

