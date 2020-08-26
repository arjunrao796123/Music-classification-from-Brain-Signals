import torch
import torch.nn as nn

class Conv1dUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=0, batchnorm=True,
                 nonlinearity=nn.LeakyReLU(0.2)):
        super(Conv1dUnit, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel, stride, padding),
                    nn.BatchNorm1d(out_channels), nonlinearity)
        else:
            self.model = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel, stride, padding), nonlinearity)

    def forward(self, x):
        return self.model(x)

class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, batchnorm=True, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features), nonlinearity)
        else:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features), nonlinearity)

    def forward(self, x):
        return self.model(x)


class DisentangledEEG(nn.Module):
    
    def __init__(self, f_dim=256, z_dim=32, in_size=1, channels=64, conv_dim=10, hidden_dim=256, seq_len=20,
                 factorized=True, nonlinearity=True, kernel=320, stride=160, y_size=12):
        super(DisentangledEEG, self).__init__()
        self.f_dim = f_dim
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.conv_dim = conv_dim
        self.hidden_dim = hidden_dim
        self.factorized = factorized
        self.channels = channels
        self.in_size = in_size
        self.kernel = kernel
        self.stride = stride
        self.y_size = y_size

        
        self.z_prior_lstm = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_mean = nn.Linear(self.hidden_dim, z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, z_dim)
        self.f_lstm = nn.LSTM(self.conv_dim, self.hidden_dim, 1 , bidirectional=True, batch_first=True)
        self.f_mean = LinearUnit(self.hidden_dim*2, self.f_dim, False)
        self.f_logvar = LinearUnit(self.hidden_dim*2, self.f_dim, False)

        if self.factorized is True:
            self.z_inter = LinearUnit(self.conv_dim, self.hidden_dim, batchnorm=False)
            self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
            self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        else:
            self.z_lstm = nn.LSTM(self.conv_dim + self.f_dim, self.hidden_dim, 1, bidirectional=True,
                                  batch_first=True)
            self.z_rnn = nn.RNN(self.hidden_dim*2, self.hidden_dim, batch_first=True)
            
            self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
            self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        self.conv = nn.Sequential(Conv1dUnit(channels, conv_dim, kernel=kernel, stride=stride),
                                  nn.BatchNorm1d(conv_dim))
        
        self.y = nn.Sequential(nn.Linear(f_dim, y_size), nn.Tanh())

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def sample_z(self, batch_size, random_sampling = True):
        z_out = None 
        z_means = None
        z_logvars = None

        
        z_t = torch.zeros(batch_size, self.z_dim)
        z_mean_t = torch.zeros(batch_size, self.z_dim)
        z_logvar_t = torch.zeros(batch_size, self.z_dim)
        h_t = torch.zeros(batch_size, self.hidden_dim)
        c_t = torch.zeros(batch_size, self.hidden_dim)

        for _ in range(self.seq_len):
            h_t, c_t = self.z_prior_lstm(z_t, (h_t, c_t))
            z_mean_t = self.z_prior_mean(h_t)
            z_logvar_t = self.z_prior_logvar(h_t)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars  = z_logvar_t.unsqueeze(1)
            else:
                
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
        return z_means, z_logvars, z_out

    def encode_signals(self, x):
        x = x.squeeze(-1)
        x = self.conv(x)
        x = x.view(-1, self.seq_len, self.conv_dim)
        return x

    def reparameterize(self, mean, logvar, random_sampling = False):
        
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.randn_like(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    def encode_f(self, x):
        lstm_out, _ = self.f_lstm(x)
        
        backward = lstm_out[:, 0, self.hidden_dim:2*self.hidden_dim]
        frontal = lstm_out[:, self.seq_len-1, 0:self.hidden_dim]
        lstm_out = torch.cat((frontal, backward), dim=1)
        mean = self.f_mean(lstm_out)
        logvar = self.f_logvar(lstm_out)
        
        return mean, logvar, self.reparameterize(mean, logvar, False)

    def encode_z(self, x, f):
        if self.factorized is True:
            features = self.z_inter(x)
        else:
            
            f_expand = f.unsqueeze(1).expand(-1, self.seq_len, self.f_dim)
            lstm_out, _ = self.z_lstm(torch.cat((x, f_expand), dim = 2))
            features, _ = self.z_rnn(lstm_out)

        mean = self.z_mean(features)
        logvar = self.z_logvar(features)
        return mean, logvar, self.reparameterize(mean, logvar, False)

    def forward(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=False)
        conv_x = self.encode_signals(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x, f)
        f_expand = f.unsqueeze(1).expand(-1, self.seq_len, self.f_dim)
        zf = torch.cat((z, f_expand), dim=2)
        
        yhat = self.y(f)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, yhat