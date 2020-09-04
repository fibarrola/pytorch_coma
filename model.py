import torch
import torch.nn.functional as F
from layers import ChebConv_Coma, Pool

__author__ = ['Priyanka Patel', 'Rodrigo Bonazzola']

class Coma(torch.nn.Module):
    
    ### added tabDims parameter to __init__
    def __init__(self, config, downsample_matrices, upsample_matrices, adjacency_matrices, num_nodes, tabDims):
        super(Coma, self).__init__()

        self.n_layers = config['n_layers']
        self.filters = config['num_conv_filters']
        
        ### Tabular Dims
        self.contXdim = tabDims[0]
        self.catgXdim = tabDims[1]
        self.contCdim = tabDims[2]
        self.catgCdim = tabDims[3]
        Hdim = 128

        
        from helpers import get_cardiac_dataset_len
        num_features = get_cardiac_dataset_len(config)[-1] # number of features per graph node, typically 3 (x, y, z coordinates)
        self.filters.insert(0, num_features)

        self.K = config['polygon_order']
        self.z = config['z']
        self.is_variational = (config['kld_weight'] != 0)
        if self.is_variational:
            self.kld_weight = config['kld_weight']

        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices

        self.A_edge_index, self.A_norm = zip(
            *[ChebConv_Coma.norm(self.adjacency_matrices[i]._indices(), num_nodes[i]) for i in range(len(num_nodes))]
        )

        # Chebyshev convolutions (encoder)
        self.cheb = torch.nn.ModuleList([
            ChebConv_Coma(self.filters[i], self.filters[i+1], self.K[i]) for i in range(len(self.filters)-2)
        ])

        # Chebyshev deconvolutions (decoder)
        self.cheb_dec = torch.nn.ModuleList([
            ChebConv_Coma(self.filters[-i-1], self.filters[-i-2], self.K[i]) for i in range(len(self.filters)-1)
        ])

        self.cheb_dec[-1].bias = None  # No bias for last convolution layer
        self.pool = Pool()

        self.n_features_before_z = self.downsample_matrices[-1].shape[0]*self.filters[-1]

        # Fully connected layer connecting the last pooling layer and the latent space layer.
        ### added Hdim to layer's input dimension
        if self.is_variational:
            self.kld_weight = config['kld_weight']
            self.enc_lin_mu = torch.nn.Linear(self.n_features_before_z+Hdim, self.z)
            self.enc_lin_var = torch.nn.Linear(self.n_features_before_z+Hdim, self.z)
        else:
            self.enc_lin = torch.nn.Linear(self.n_features_before_z+Hdim, self.z)

        ### added self.contCdim+self.catgCdim in layer's input dimension
        self.dec_lin = torch.nn.Linear(self.z+self.contCdim+self.catgCdim, self.filters[-1]*self.upsample_matrices[-1].shape[1])

        ### Tabular enconder layer
        self.tabElayer = torch.nn.Linear(self.contXdim+self.catgXdim,Hdim)
        ### Tabular decoder layer
        self.tabDlayer = torch.nn.Linear(self.filters[-1]*self.upsample_matrices[-1].shape[1], contXdim+catgXdim+contCdim+catgCdim)


        # Why this???
        self.reset_parameters()
        
        
    def encoder(self, x, contX, catgX):
        for i in range(self.n_layers):
            x = F.relu(self.cheb[i](x, self.A_edge_index[i], self.A_norm[i]))
            x = self.pool(x, self.downsample_matrices[i])

        x = x.reshape(x.shape[0], self.n_features_before_z)
        ### Tabular layer and concatenation
        h = F.relu(self.tabElayer(torch.cat([contX, catgX], dim=1)))
        x = torch.cat([x,h],dim =1)
        if self.is_variational:
            mu = self.enc_lin_mu(x)
            log_var = self.enc_lin_var(x)
            return mu, log_var
        else:
            return self.enc_lin(x)


    def decoder(self, z, contC, catgC):
        x = F.relu(self.dec_lin(torch.cat([z,contC,catgC],dim=1)))
        
        ### Tabular part
        h = torch.sigmoid(self.tabDlayer(x))
        contXout = h[:,:self.contXdim]
        catgXout = h[:,self.contXdim:self.contXdim+self.catgXdim]
        contCout = h[:,self.contXdim+self.catgXdim:self.contXdim+self.catgXdim+self.contCdim]
        catgCout = h[:,self.contXdim+self.catgXdim+self.contCdim:]
        
        x = x.reshape(x.shape[0], -1, self.filters[-1])
        for i in range(self.n_layers):
            x = self.pool(x, self.upsample_matrices[-i-1])
            x = F.relu(self.cheb_dec[i](x, self.A_edge_index[self.n_layers-i-1], self.A_norm[self.n_layers-i-1]))
        x = self.cheb_dec[-1](x, self.A_edge_index[-1], self.A_norm[-1])
        return x, contXout, catgXout, contCout, catgCout


    def reset_parameters(self):
        if self.is_variational:
            torch.nn.init.normal_(self.enc_lin_mu.weight, 0, 0.1)
            torch.nn.init.normal_(self.enc_lin_var.weight, 0, 0.1)
        else:
            torch.nn.init.normal_(self.enc_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)


    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) 


    def forward(self, data, contX, catgX, contC, catgC):
        contX = contX.view(-1,self.contXdim)
        catgX = catgX.view(-1,self.catgXdim)
        contC = contC.view(-1,self.contCdim)
        catgC = catgC.view(-1,self.catgCdim)
        x = data #.x
        # batch_size = data.num_graphs
        batch_size = data.shape[0]
        x = x.reshape(batch_size, -1, self.filters[0])
        if self.is_variational:
            self.mu, self.log_var = self.encoder(x, contX, catgX)
            z = self.sampling(self.mu, self.log_var)
        else:
            z = self.encoder(x, contX, catgX)
        x, contXout, catgXout, contCout, catgCout = self.decoder(z, contC, catgC)
        x = x.reshape(-1, self.filters[0])
        return x, contXout, catgXout, contCout, catgCout
