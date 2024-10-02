import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 4

        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.layer1 = self._make_layer(block, 4, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 4, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 4, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])



class LSTM_window(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTM_window, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True, bidirectional = True)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first = True , bidirectional = True)
        self.lstm3 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first = True , bidirectional = True)
        self.lstm4 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first = True , bidirectional = True)

        self.relu = nn.ReLU(inplace=True)

        self.linear = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input):
        input = input.reshape(input.shape[0], input.shape[1], -1)
        lstm_out, _ = self.lstm(input)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out, _ = self.lstm3(lstm_out)
        lstm_out, _ = self.lstm4(lstm_out)

        y_pred = self.linear(lstm_out[:, -1])

        return y_pred



class CNN_LSTM(nn.Module):
    def __init__(self, lstm_input_dim, hidden_dim, output_dim, num_layers):
        super(CNN_LSTM, self).__init__()
        self.cnn_out_shape = 584

        self.resnet_str = ResNet18()
        self.relu = nn.ReLU(inplace=True)

        self.lstm = nn.LSTM(self.cnn_out_shape, hidden_dim, num_layers, batch_first = True, bidirectional = True)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first = True , bidirectional = True)
        self.lstm3 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first = True , bidirectional = True)
        self.lstm4 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first = True , bidirectional = True)

        self.linear = nn.Linear(hidden_dim * 2, output_dim)


    def forward(self, input):
        batch_size, timesteps, c, h, w = input.size()
        conv_in = input.view(batch_size*timesteps, c, h, w)

        conv_out = self.resnet_str(conv_in)

        conv_out = conv_out.view(batch_size, timesteps, self.cnn_out_shape)

        lstm_out, _ = self.lstm(conv_out)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out, _ = self.lstm3(lstm_out)
        lstm_out, _ = self.lstm4(lstm_out)

        y_pred = self.linear(lstm_out[:, -1])

        return y_pred


class CNN_stLSTM(nn.Module):
    def __init__(self, lstm_input_dim, hidden_dim, output_dim, num_layers):
        super(CNN_stLSTM, self).__init__()

        self.resnet_str = ResNet18()
        self.relu = nn.ReLU(inplace=True)
        self.cnn_out_shape = 584

        self.lstm_s1 = nn.LSTM(lstm_input_dim, lstm_input_dim, num_layers, batch_first = True , bidirectional = True)
        self.lstm_s2 = nn.LSTM(lstm_input_dim * 2, lstm_input_dim, num_layers, batch_first = True)

        self.lstm_t1 = nn.LSTM(self.cnn_out_shape, hidden_dim, num_layers, batch_first = True, bidirectional = True)
        self.lstm_t2 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first = True , bidirectional = True)

        self.linear = nn.Linear(hidden_dim * 2, output_dim)


    def forward(self, input):
        batch_size, timesteps, c, h, w = input.size()
        conv_in = input.view(batch_size*timesteps, c, h, w)

        conv_out = self.resnet_str(conv_in)

        conv_out = conv_out.view(batch_size, timesteps, self.cnn_out_shape)
        
        conv_out = conv_out.permute(0, 2, 1)

        lstm_out, _ = self.lstm_s1(conv_out)
        lstm_out, _ = self.lstm_s2(lstm_out)

        lstm_out = lstm_out.permute(0, 2, 1)

        lstm_out, _ = self.lstm_t1(lstm_out)
        lstm_out, _ = self.lstm_t2(lstm_out)

        y_pred = self.linear(lstm_out[:, -1])

        return y_pred


class cnn_transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout=0.1):
        super(cnn_transformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.cnn_out_shape = 584

        self.resnet_str = ResNet18()
        # Define the encoder and decoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout)

        # Define the encoder and decoder stacks
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        # Define the projection layer
        self.proj = nn.Linear(self.cnn_out_shape*d_model, 1)


    def forward(self, src, tgt):
        batch_size, timesteps, c, h, w = src.size()

        conv_in = src.view(batch_size*timesteps, c, h, w)
        conv_out = self.resnet_str(conv_in)

        conv_out = conv_out.view(batch_size, self.cnn_out_shape, timesteps)

        # Encode the input frames
        enc_output = self.encoder(conv_out)

        tgt = tgt.unsqueeze(1)
        

        tgt = torch.tile(tgt, (1, self.cnn_out_shape, 1))

        # Decode the target frames        
        dec_output = self.decoder(tgt, enc_output)
        dec_output = dec_output.view(batch_size, -1)

        output = self.proj(dec_output)


        return output
    
    
class cnn_st_transformer(nn.Module):
    def __init__(self, d_model_t, nhead_t, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout=0.1):
        super(cnn_st_transformer, self).__init__()

        self.cnn_out_shape = 584

        self.resnet_str = ResNet18()
        # Define the encoder and decoder layers
        self.encoder_sp_layer = nn.TransformerEncoderLayer(d_model=self.cnn_out_shape, nhead=8,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout)

        self.encoder_tm_layer = nn.TransformerEncoderLayer(d_model=d_model_t, nhead=d_model_t,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout)

        
        self.decoder_sp_layer = nn.TransformerDecoderLayer(d_model=self.cnn_out_shape, nhead=8,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout)
        self.decoder_tm_layer = nn.TransformerDecoderLayer(d_model=nhead_t, nhead=nhead_t,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout)

        # Define the encoder and decoder stacks
        self.encoder_sp = nn.TransformerEncoder(self.encoder_sp_layer, num_layers=num_encoder_layers)
        self.encoder_tm = nn.TransformerEncoder(self.encoder_tm_layer, num_layers=num_encoder_layers)
        self.decoder_sp = nn.TransformerDecoder(self.decoder_sp_layer, num_layers=num_decoder_layers)
        self.decoder_tm = nn.TransformerDecoder(self.decoder_tm_layer, num_layers=num_decoder_layers)

        # Define the projection layer
        self.proj = nn.Linear(self.cnn_out_shape*d_model_t, 1)
        # self.proj = nn.Linear(self.cnn_out_shape, 1)


    def forward(self, src, tgt):
        batch_size, timesteps, c, h, w = src.size()

        conv_in = src.view(batch_size*timesteps, c, h, w)
        conv_out = self.resnet_str(conv_in)

        conv_out = conv_out.view(batch_size, timesteps, self.cnn_out_shape)

        # Encode the input frames
        # conv_out = conv_out.permute(0, 2, 1)
        enc_output = self.encoder_sp(conv_out)
        enc_output = enc_output.permute(0, 2, 1)
        enc_output = self.encoder_tm(enc_output)

        tgt = tgt.unsqueeze(1)

        tgt = torch.tile(tgt, (1, self.cnn_out_shape, 1))

        # Decode the target frames        
        tgt = tgt.permute(0, 2, 1)
        enc_output = enc_output.permute(0, 2, 1)
        dec_output = self.decoder_sp(tgt, enc_output)
        tgt = tgt.permute(0, 2, 1)
        enc_output = enc_output.permute(0, 2, 1)
        dec_output = self.decoder_tm(tgt, enc_output)
        
        dec_output = dec_output.view(batch_size, -1)

        output = self.proj(dec_output)


        return output

    