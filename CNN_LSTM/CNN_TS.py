from imports import *

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
    
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))
        
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention_weights = torch.bmm(v, energy).squeeze(1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights

class LSTMEncoderDecoderAttention(nn.Module):
    def __init__(self, look_back, forecast_range, n_features, hidden_dim, n_outputs):
        super(LSTMEncoderDecoderAttention, self).__init__()
        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_outputs)
        self.hidden_dim = hidden_dim
        self.forecast_range = forecast_range

    def forward(self, x):
        encoder_outputs, (hidden, cell) = self.encoder(x)
        
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, self.forecast_range, 1)
        decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
        
        outputs = []
        for t in range(self.forecast_range):
            context, _ = self.attention(hidden[-1], encoder_outputs)
            decoder_output_t = decoder_output[:, t, :] + context
            outputs.append(self.fc(decoder_output_t))
        
        return torch.stack(outputs, dim=1)