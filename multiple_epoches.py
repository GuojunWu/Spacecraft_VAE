"""
Multiple epochs

"""

#######################
##  Build the model  ##
#######################

class VAE(nn.Module):
    
    # Neural network construction in encoder and decoder
    def _init_(self):
        super()._init_()
        ## Layers in encode ##
        # Hidden layer
        self.fc1 = nn.Linear(5125, 1000)
        # Output layer 
        self.fc21 = nn.Linear(1000,200)
        self.fc21 = nn.Linear(1000,200)
        #? How many layers and how many dimensions?

        ## Layers in decode ##
        # Hidden layer
        sef.fc3 = nn.Linear(200,1000)
        # Output layer
        sef.fc4 = nn.Linear(1000,5125)
        #? Should encoder and decoder be systemetric

    ## Forward propagation
    def encode(self,x):
        h1 = F.relu(self.fc1(x))
        mu = F.relu(self.fc21(h1))
        logvar = F.relu(self.fc22(h1))
        return mu, logvar

    # Bottlenck
    # Sample z from bottleneck N(mu,var): 200 dimensions
    #? How many z are sampled?
    def sample(self,mu,logvar):
        # logvar -> std
        sig = torch.exp(torch.mul(logvar,0.5))
        # or logvar.mul_(0.5).exp_()? use logvar as std

        # Reparameterize: z = mu + sig*eps
        # sample z
        eps = torch.from_numpy(np.random.normal(0, 1, size=std.size())).float()
        #? size of z
        #? eps = Variable(eps,requires_grad=False)
        z = mu+sig*eps
        # ?Sample many times and take average?
        
        return z

    # Decode
    def decode(self,z):
        h3 = F.relu(self.fc1(z))
        x_hat = F.relu(self.fc4(h3))
        return x_hat

    def forward(self,x):
        mu, logvar = self.encode(x)
        z = self.sample(mu,logvar)
        x_hat = self.decode(z)
        return x_hat,mu,logvar

#####################
## Loss Function   ##
#####################
    
def loss_reconstruct(x_hat,x):
    criterion = nn.MSELoss(reduction='sum')
    reconstruct_error = criterion(x_hat,x)
    #? or loss_reconstruct = (x_hat-x).mm(((x_hat-x).transpose)) (check shape of x
    return reconstruct_error

def loss_latent(mu,logvar):
    KLD_element = (torch.exp(logvar),mul_(-1) - torch.pow(mu,2) + logvar).add_(1)
    KLD_loss = torch.sum(KLD_element).mul_(-0.5)
    return KLD_loss


#################
## Train model ##
#################

batch_size = 100
beta = 2

network = VAE()

train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size)
optimizer = optim.Adam(network.parameters(),lr = 0.01)# optimization algorithm

for epoch in range(10): # epoches

    total_loss = 0

    for batch in train_loader: # Get batch
    sensor,_=batch

    x_hat,mu,logvar = network(sensor) # Pass batch

    loss = loss_reconstruct(x_hat,x)+torch.mul(loss_latent(mu, logvar),beta)
    loss_avg = torch.mul(loss,1/batch_size) # Calculate loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss_avg.item()


print(
    "epoch:",epoch,
    "loss:",total_loss
    )
    
