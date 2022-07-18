import torch
from ae_utils_exp import s_init

# kaiming normal initialization
def kai_norm(module, nonlinearity='relu'):
    torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity=nonlinearity)
    return module

class enc_beamform(torch.nn.Module):
    def __init__(self, lat=1):
        super(enc_beamform, self).__init__()
        use_bias = True
        self.w0e = s_init(torch.nn.Linear(1000, 200, bias=use_bias))
        self.w1e = s_init(torch.nn.Linear(200, 80, bias=use_bias))
        self.w2e = s_init(torch.nn.Linear(80, 40, bias=use_bias))
        self.w3e = s_init(torch.nn.Linear(40, lat, bias=use_bias))
        self.act = torch.nn.SELU(True)
        
    def forward(self, x):
        x = self.act(self.w0e(x))
        x = self.act(self.w1e(x))
        x = self.act(self.w2e(x))
        return self.w3e(x)
    
class enc_beamform_vae(torch.nn.Module):
    def __init__(self, lat=1):
        super(enc_beamform_vae, self).__init__()
        use_bias = True
        self.w0e = s_init(torch.nn.Linear(1000, 200, bias=use_bias))
        self.w1e = s_init(torch.nn.Linear(200, 80, bias=use_bias))
        self.w2e = s_init(torch.nn.Linear(80, 40, bias=use_bias))
        self.w3e_mu = s_init(torch.nn.Linear(40, lat, bias=use_bias))
        self.w3e_log_var = s_init(torch.nn.Linear(40, lat, bias=use_bias))
        self.act = torch.nn.SELU(True)
        
    def forward(self, x):
        x = self.act(self.w0e(x))
        x = self.act(self.w1e(x))
        x = self.act(self.w2e(x))
        
        return self.w3e_mu(x), self.w3e_log_var(x)
    
class dec_beamform(torch.nn.Module):
    def __init__(self, lat=1):
        super(dec_beamform, self).__init__()
        use_bias = True
        self.w3d = s_init(torch.nn.Linear(lat, 40, bias=use_bias))
        self.w2d = s_init(torch.nn.Linear(40, 80, bias=use_bias))
        self.w1d = s_init(torch.nn.Linear(80, 200, bias=use_bias))
        self.w0d = s_init(torch.nn.Linear(200, 1000, bias=use_bias))
        self.act = torch.nn.SELU(True)
        
    def forward(self, x):
        x = self.act(self.w3d(x))
        x = self.act(self.w2d(x))
        x = self.act(self.w1d(x))
        return self.w0d(x)
    
class enc_dsprites(torch.nn.Module):
    def __init__(self, lat=1, inp_chan=1):
        super(enc_dsprites, self).__init__()
        use_bias = True
        self.w0e = s_init(torch.nn.Conv2d(inp_chan, 32, 4, padding=1, stride=2))
        self.w1e = s_init(torch.nn.Conv2d(32, 32, 4, padding=1, stride=2))
        self.w2e = s_init(torch.nn.Conv2d(32, 64, 4, padding=1, stride=2))
        self.w3e = s_init(torch.nn.Linear(64*8*8, 256))
        self.w4e = s_init(torch.nn.Linear(256, lat))
        self.act = torch.nn.SELU(True)
        
    def forward(self, x):
        x = self.act(self.w0e(x))
        x = self.act(self.w1e(x))
        x = self.act(self.w2e(x))
        x = x.view(x.shape[0], -1)
        x = self.act(self.w3e(x))
        return self.w4e(x)
    
class dec_dsprites(torch.nn.Module):
    def __init__(self, lat=1, inp_chan=1):
        super(dec_dsprites, self).__init__()
        use_bias = True
        self.w4d = s_init(torch.nn.Linear(lat, 256))
        self.w3d = s_init(torch.nn.Linear(256, 64*8*8))
        self.w2d = s_init(torch.nn.ConvTranspose2d(64, 32, 4, padding=1, stride=2, output_padding=0))
        self.w1d = s_init(torch.nn.ConvTranspose2d(32, 32, 4, padding=1, stride=2, output_padding=0))
        self.w0d = s_init(torch.nn.ConvTranspose2d(32, inp_chan, 4, padding=1, stride=2, output_padding=0))
        self.act = torch.nn.SELU(True)
        
    def forward(self, x):
        x = self.act(self.w4d(x))
        x = self.act(self.w3d(x))
        x = x.view(-1, 64, 8, 8)
        x = self.act(self.w2d(x))
        x = self.act(self.w1d(x))
        x = self.w0d(x)
        return x
    
class enc_dsprites_fc(torch.nn.Module):
    def __init__(self, lat=1):
        super(enc_dsprites_fc, self).__init__()
        use_bias = True
        self.w0e = s_init(torch.nn.Linear(64*64, 1200, bias=use_bias))
        self.w1e = s_init(torch.nn.Linear(1200, 1200, bias=use_bias))
        self.w2e = s_init(torch.nn.Linear(1200, lat, bias=use_bias))
        self.act = torch.nn.ReLU(True)
        
    def forward(self, x):
        x = x.view(-1, 64*64)
        x = self.act(self.w0e(x))
        x = self.act(self.w1e(x))
        return self.w2e(x)
    
class dec_dsprites_fc(torch.nn.Module):
    def __init__(self, lat=1):
        super(dec_dsprites_fc, self).__init__()
        use_bias = True
        self.w2d = s_init(torch.nn.Linear(lat, 1200, bias=use_bias))
        self.w1d = s_init(torch.nn.Linear(1200, 1200, bias=use_bias))
        self.w0d = s_init(torch.nn.Linear(1200, 64*64, bias=use_bias))
        self.act = torch.nn.ReLU(True)
        
    def forward(self, x):
        x = self.act(self.w2d(x))
        x = self.act(self.w1d(x))
        x = self.w0d(x)
        return x.view(-1, 1, 64, 64)
    
class enc_dsprites_vae_fc(torch.nn.Module):
    def __init__(self, lat=1):
        super(enc_dsprites_vae_fc, self).__init__()
        use_bias = True
        self.w0e = kai_norm(torch.nn.Linear(64*64, 1200, bias=use_bias))
        self.w1e = kai_norm(torch.nn.Linear(1200, 1200, bias=use_bias))
        self.w2e_mu = kai_norm(torch.nn.Linear(1200, lat, bias=use_bias), 'linear')
        self.w2e_log_var = kai_norm(torch.nn.Linear(1200, lat, bias=use_bias), 'linear')
        self.act = torch.nn.ReLU(True)
        
    def forward(self, x):
        x = x.view(-1, 64*64)
        x = self.act(self.w0e(x))
        x = self.act(self.w1e(x))
        return self.w2e_mu(x), self.w2e_log_var(x)
    
class dec_dsprites_vae_fc(torch.nn.Module):
    def __init__(self, lat=1):
        super(dec_dsprites_vae_fc, self).__init__()
        self.w3d = kai_norm(torch.nn.Linear(lat, 1200), 'tanh')
        self.w2d = kai_norm(torch.nn.Linear(1200, 1200), 'tanh')
        self.w1d = kai_norm(torch.nn.Linear(1200, 1200), 'tanh')
        self.w0d = kai_norm(torch.nn.Linear(1200, 4096), 'linear')
        self.act = torch.nn.Tanh()
        
    def forward(self, x):
        x = self.act(self.w3d(x))
        x = self.act(self.w2d(x))
        x = self.act(self.w1d(x))
        x = self.w0d(x)
        return x.view(-1, 1, 64, 64)
    
class enc_celeba_small(torch.nn.Module):
    def __init__(self, lat=1, inp_chan=1):
        super(enc_celeba_small, self).__init__()
        self.act = act = torch.nn.SELU(True)
        self.conv_op = torch.nn.Sequential(
            s_init(torch.nn.Conv2d(inp_chan, 32, 4, padding=1, stride=2)), act,# 32
            s_init(torch.nn.Conv2d(32, 32, 4, padding=1, stride=2)), act, # 16
            s_init(torch.nn.Conv2d(32, 64, 4, padding=1, stride=2)), act, # 8
            s_init(torch.nn.Conv2d(64, 64, 4, padding=1, stride=2)), act, # 4
        )
        self.w0e = s_init(torch.nn.Linear(64*4*4, 256))
        self.w1e = s_init(torch.nn.Linear(256, lat))
        
        
    def forward(self, x):
        x = self.conv_op(x)
        x = x.view(x.shape[0], -1)
        x = self.act(self.w0e(x))
        return self.w1e(x)
    
class dec_celeba_small(torch.nn.Module):
    def __init__(self, lat=1, inp_chan=1):
        super(dec_celeba_small, self).__init__()
        self.act = act = torch.nn.SELU(True)
        self.w1d = s_init(torch.nn.Linear(lat, 256))
        self.w0d = s_init(torch.nn.Linear(256, 64*4*4))
        self.deconv_op = torch.nn.Sequential(
            s_init(torch.nn.ConvTranspose2d(64, 64, 4, padding=1, stride=2, output_padding=0)), act, # 8
            s_init(torch.nn.ConvTranspose2d(64, 32, 4, padding=1, stride=2, output_padding=0)), act, # 16
            s_init(torch.nn.ConvTranspose2d(32, 32, 4, padding=1, stride=2, output_padding=0)), act, # 32
            s_init(torch.nn.ConvTranspose2d(32, inp_chan, 4, padding=1, stride=2, output_padding=0))# 64
        )
        
    def forward(self, x):
        x = self.act(self.w1d(x))
        x = self.act(self.w0d(x))
        x = x.view(-1, 64, 4, 4)
        x = self.deconv_op(x)
        return x
    
class enc_celeba_small_vae(torch.nn.Module):
    def __init__(self, lat=1, inp_chan=1):
        super(enc_celeba_small_vae, self).__init__()
        self.act = act = torch.nn.SELU(True)
        self.conv_op = torch.nn.Sequential(
            s_init(torch.nn.Conv2d(inp_chan, 32, 4, padding=1, stride=2)), act,# 32
            s_init(torch.nn.Conv2d(32, 32, 4, padding=1, stride=2)), act, # 16
            s_init(torch.nn.Conv2d(32, 64, 4, padding=1, stride=2)), act, # 8
            s_init(torch.nn.Conv2d(64, 64, 4, padding=1, stride=2)), act, # 4
        )
        self.w0e = s_init(torch.nn.Linear(64*4*4, 256))
        self.w1e_mu = s_init(torch.nn.Linear(256, lat))
        self.w1e_log_var = s_init(torch.nn.Linear(256, lat))
        
        
    def forward(self, x):
        x = self.conv_op(x)
        x = x.view(x.shape[0], -1)
        x = self.act(self.w0e(x))
        return self.w1e_mu(x), self.w1e_log_var(x)
    
class FactorVAE_Discriminator(torch.nn.Module):
    def __init__(self, lat=1, h_size=1024):
        super(FactorVAE_Discriminator, self).__init__()
        self.act = act = torch.nn.SELU(True)
        self.op = torch.nn.Sequential(
            s_init(torch.nn.Linear(lat, h_size)),
            act,
            s_init(torch.nn.Linear(h_size, h_size)),
            act,
            s_init(torch.nn.Linear(h_size, h_size)),
            act,
            s_init(torch.nn.Linear(h_size, 1)),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.op(x).squeeze()