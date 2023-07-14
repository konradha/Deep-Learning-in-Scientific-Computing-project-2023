from Common import NeuralNet
import torch
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import numpy as np

# Set matplotlib st figures don't open in big
plt.switch_backend('Agg')
path = "./"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################################################################################################ Global helper functions
def leg(x, n):
            return torch.special.legendre_polynomial_p(x, n)

# Coefficients for phi
d = torch.tensor([1.0, 1.98398, 1.50823, 0.70075, 0.23489, 0.05133, 0.00760, 0.00048], device = device)

# Quadrature weigths and points
wx = torch.tensor([
    [0.1527533871307258, 	-0.0765265211334973],
    [0.1527533871307258,	 0.0765265211334973],
    [0.1491729864726037,	-0.2277858511416451],
    [0.1491729864726037,	 0.2277858511416451],
    [0.1420961093183820,	-0.3737060887154195],
    [0.1420961093183820,	 0.3737060887154195],
    [0.1316886384491766,	-0.5108670019508271],
    [0.1316886384491766,	 0.5108670019508271],
    [0.1181945319615184,	-0.6360536807265150],
    [0.1181945319615184,	 0.6360536807265150],
    ], device = device)

gl_weights = wx[:, 0]
gl_abcissa = wx[:, 1]

################################################################################################ PINN
class PINN:
    def __init__(self, n_int_, n_sb_,):
        self.n_int = n_int_
        self.n_sb = n_sb_

        # Extrema of the solution domain (mu,x) in [-1,1]x[0, 1.]
        self.domain_extrema = torch.tensor([[-1., 1.],  # mu
                                            [-0., 1.]])  # x space

        # Number of space dimensions
        self.space_dimensions = 1

        # Parameter to balance role of data and PDE ( is equal to the inverse of lambda of the paper)
        self.lambda_u = 10

        # F Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0],
                                    output_dimension=1,
                                    n_hidden_layers=8, 
                                    neurons=20,
                                    regularization_param=0., 
                                    regularization_exp=2.,
                                    retrain_seed=42).to(device)

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        # Training sets S_sb, S_tb, S_int as torch dataloader; saves matrix phi_mat
        self.training_set_sb, self.training_set_int = self.assemble_datasets()

    ################################################################################################ Functions called once
    def convert(self, tens):
        """Function to linearly transform a tensor whose value are between 0 and 1
        to a tensor whose values are between the domain extrema"""
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]
    
    def convert_doublefringe(self,tens):
        """Function to transform the points in order to have more at the boundaries"""

        self.eps_mu0 = 0.1
        domain_eps_x0 = torch.tensor([[-1.0, 1.0],  # mu
                                            [0., self.eps_mu0]])  # x space
        domain_eps_x1 = torch.tensor([[-1.0, 1.0],  # mu
                                            [1.0 - self.eps_mu0, 1.0]])  # x space
        tens[0:tens.shape[0]//2,:] = tens[0:tens.shape[0]//2,:] * (domain_eps_x0[:,1] - domain_eps_x0[:,0]) + domain_eps_x0[:,0]
        tens[tens.shape[0]//2:,:] = tens[tens.shape[0]//2:,:] * (domain_eps_x1[:,1] - domain_eps_x1[:,0]) + domain_eps_x1[:,0]
        """ To add interior nodes around mu = 0:
        domain_eps = torch.tensor([[-self.eps_mu0, self.eps_mu0],  # mu
                                            [0., 1.0]])  # x space
        tens_mu0 = tens * (domain_eps[:,1] - domain_eps[:,0]) + domain_eps[:,0]
        tens = torch.cat([tens,tens_mu0],0)
        """
        return tens

    def add_spatial_boundary_points(self):
        """Function returning the input-output tensor required to assemble the training set S_sb corresponding to the spatial boundary
        Return the tuple of:
            -input spatial boundary points (mu,x) of size (n,2)
            -output values for (u) at those points of size (n,1) for x=0 and then x=1 both at mu not= 0
        Note: we have the discontinuity at y = 0
        """
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        input_sb = self.convert(self.soboleng.draw(self.n_sb))        

        input_sb_0 = torch.clone(input_sb)
        input_sb_0[:, 1] = torch.full(input_sb_0[:, 1].shape, x0)

        input_sb_L = torch.clone(input_sb)
        input_sb_L[:, 1] = torch.full(input_sb_L[:, 1].shape, xL)

        # Enforce the discountinuity
        self.eps = 0.01
        mu_plus = input_sb_0[:, 0] > self.eps
        mu_mins = input_sb_L[:, 0] < -self.eps

        output_sb_0 = torch.zeros((input_sb.shape[0], 1))
        output_sb_L = torch.zeros((input_sb.shape[0], 1))

        # explicitly formulating BC
        output_sb_0[mu_plus] = 1.
        output_sb_L[mu_mins] = 0.
        return torch.cat([input_sb_0[mu_plus], input_sb_L[mu_mins]], 0), torch.cat(
                [output_sb_0[mu_plus], output_sb_L[mu_mins]], 0)

    def add_interior_points(self):
        """Function returning the input-output tensor required to assemble the
            training set S_int corresponding to the interior domain where the PDE is enforced
            With convert_doublefringe() the number of output values doubles
        Returns:
            tuple of tensors: (n_int,2), (n_int * 2,1): input (mu,x) and output (0) at those points
        """
        input_int_normal = self.convert(self.soboleng.draw(self.n_int))
        input_int_eps = self.convert_doublefringe(self.soboleng.draw(self.n_int))
        
        input_int = torch.cat([input_int_normal, input_int_eps], 0)
        output_int = torch.zeros((input_int.shape[0], 1))
        return input_int, output_int

    def assemble_datasets(self):
        """Function returning the training sets S_sb, S_tb, S_int as dataloader"""
        input_sb, output_sb = self.add_spatial_boundary_points()   # S_sb
        input_int, output_int = self.add_interior_points()         # S_int

        # Build and save self.phi_mat for better performance during the training
        self.build_phi(input_int)

        N = 2 # number of interior nodes sets in use
        training_set_sb = DataLoader(torch.utils.data.TensorDataset(input_sb, output_sb), batch_size=2*self.space_dimensions*self.n_sb, shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int*N, shuffle=False)

        return training_set_sb, training_set_int
    
    def build_phi(self,input_int):
        def phi(mu,mu_p):
            """Returns scalar phi(mu,mu_p) of size (n_int, n_weights)."""
            P_mu = torch.tensor([leg(mu, i) for _, i in enumerate(d)],device=device)
            P_mu_p = torch.tensor([leg(mu_p, i) for _, i in enumerate(d)],device=device)
            return d.dot(P_mu * P_mu_p)
        
        # Currently away() simply returns a true array. We left it in to allow the reader to experiment with it.
        mu = input_int[:,0]
        x = input_int[:,1]
        away_from_0 =  self.away(mu,x)
        mu_arr = mu[away_from_0]

        # create phi matrix
        self.phi_mat = torch.stack([torch.tensor([phi(mu_,mu_p) for mu_p in gl_abcissa]
                                                 ,device=device) for mu_ in mu_arr])
        return self.phi_mat

    ################################################################################################### Functions called each epoch
    
    def kernel_mat(self, mu_,x):
        """Returns the integral by quadrature rule: optimized version"""
        u_mat = self.approximate_solution(torch.stack([ torch.stack(
            [torch.tensor((mu_p,x_i),device=device) for mu_p in gl_abcissa]
                                                                            ) for x_i in x])).squeeze()
        return torch.matmul((self.phi_mat * u_mat) , gl_weights)

    def away(self, mu, x):
        """Used to impose additional constraints on the PDE. Currently no constraint used."""
        return torch.full((mu.shape[0],), True)

    def apply_boundary_conditions(self, input_sb):
        """Function to compute the terms required in the definition of the SPATIAL boundary residual"""
        u_pred_sb = self.approximate_solution(input_sb)
        return u_pred_sb

    def compute_pde_residual(self, input_int):
        """compute the pde residual term with n being the number of interior points in use.
        Args:
            input_int
        Returns:
            tensor: pde residual of size (n, 1)
        """
        input_int.requires_grad = True
        mu = input_int[:, 0]
        x  = input_int[:, 1]

        # Currently away() simply returns a full true array of the size of mu. Can be used to apply special conditions on the pde
        away_from_0 = self.away(mu,x) 
        input_int = input_int[away_from_0]
        mu = mu[away_from_0]
        x  = x[away_from_0]

        # Get u and calculate gradient
        u = self.approximate_solution(input_int)
        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]
        grad_u_x = grad_u[:, 1]    

        return (mu * grad_u_x + self.sigma(x) * u[:,0] - .5 * self.sigma(x) * self.kernel_mat(mu, x)).reshape(-1,)

    def sigma(self, x):
        """Simple identity function for cleanliness of the code."""
        return x

    def compute_loss(self, inp_train_sb, u_train_sb, inp_train_int, verbose=True):
        """Function to compute the total loss (weighted sum of spatial boundary loss and interior loss)"""
        # To GPU all sets
        inp_train_sb = inp_train_sb.to(device)
        u_train_sb = u_train_sb.to(device)
        inp_train_int = inp_train_int.to(device)
        # Calculate PDE residual term
        r_int = self.compute_pde_residual(inp_train_int)
        
        # Calculate BC residual term
        u_pred_sb = self.apply_boundary_conditions(inp_train_sb)
        assert (u_pred_sb.shape[1] == u_train_sb.shape[1])
        r_sb = u_train_sb - u_pred_sb

        # Calculate losses
        loss_sb = torch.mean(abs(r_sb) ** 2)
        loss_int = torch.mean(abs(r_int) ** 2)

        loss = torch.log10(self.lambda_u * loss_sb + loss_int) 
        
        if verbose: print("Total loss: ", round(loss.item(), 4), "| BC Loss: ", round(torch.log10(loss_sb).item(), 4), "| Function Loss: ", round(torch.log10(loss_int).item(), 4))

        return loss


    def fit(self, num_epochs, optimizer,scheduler, verbose=True):
        history = list()
        
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            for j, ((inp_train_sb, u_train_sb), (inp_train_int, u_train_int)) in enumerate(zip(self.training_set_sb, self.training_set_int)):
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(inp_train_sb, u_train_sb, inp_train_int, verbose=verbose)
                    loss.backward()

                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)
                
            scheduler.step()

        print('Final Loss: ', history[-1])
        return history

    ################################################################################################ Plotting and results

    def draw(self):
        # Plot values at boundary points
        self.boundary_plot()

        # Plot values at space-boundary points
        self.antiboundary_plot()

        # Plot contour plot over complete (mu,x)-space
        self.contour_plot()

    def contour_plot(self):
        # Scatterplot u on full domain
        inputs = self.soboleng.draw(100000)
        inputs = self.convert(inputs)
        output = self.approximate_solution(inputs.to(device)).reshape(-1, ).cpu()

        # Set colors: Bounds are unevenly spaced
        bounds = np.array([0.000, 0.006, 0.013, 0.021, 0.029, 0.040, 0.047, 0.060, 0.071, 0.099, 0.143, 0.214, 0.286, 0.357, 0.429, 0.500, 0.571, 0.643, 0.714, 0.786, 0.857, 0.929, 1.000])
        norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)

        # create figure
        fig, axs = plt.subplots(nrows=1,ncols=2, figsize=(16,8), dpi=100)
        im = axs[0].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=output.detach(),norm=norm,
                            cmap="jet", s=1, label="u(x,y)")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("mu")
        axs[0].grid(True, which="both", ls=":")
        axs[0].set_title("Approximate solution u(x,y)")
        fig.colorbar(im, ax=axs[0], extend='neither', orientation='vertical')

        xl, xr       = axs[0].get_xlim()
        y_down, y_up = axs[0].get_ylim()
        ratio = 2.
        axs[0].set_aspect(abs((xr-xl)/(y_up-y_down)) * ratio)

        axs[1].scatter(inputs[:, 1].detach(), np.arccos(inputs[:, 0].detach()),
                                c=output.detach(), cmap="jet", s=1, label="u(x,theta)")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("theta")
        axs[1].set_title("Approximate solution u(x,theta=arccos(mu))")
        plt.colorbar(axs[1].collections[0], ax=axs[1])

        xl, xr       = axs[1].get_xlim()
        y_down, y_up = axs[1].get_ylim()
        ratio = 2.
        axs[1].grid(True, which="both",)
        axs[1].set_aspect(abs((xr-xl)/(y_up-y_down)) * ratio)

        plt.tight_layout()
        plt.show()
        plt.savefig(str(path + "figures/scatter_u"))


    def antiboundary_plot(self):
        # Scatterplot boundary points
        len = 200
        inputs = self.soboleng.draw(len)
        domain_0 = torch.tensor([[-1.0, 1.0],  # mu
                                            [0., 0.]])  # x space
        domain_1 = torch.tensor([[-1.0, 1.0],  # mu
                                            [1., 1.]])  # x space
        sb_0 = inputs * (domain_0[:, 1] - domain_0[:, 0]) + domain_0[:, 0]
        sb_1 = inputs * (domain_1[:, 1] - domain_1[:, 0]) + domain_1[:, 0]
        sb = torch.cat([sb_0, sb_1], 0)
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

        pred_sb = self.approximate_solution(sb.to(device)).reshape(-1,).cpu()

        # Scatterplot at x=0
        axs[0].set_title("Values of u at x = 0")
        axs[0].scatter(sb[:len, 0].detach().numpy(),  pred_sb[:len].detach().numpy(),
                    label="approximate solution at boundaries", color="blue")

        axs[0].set_xlabel("mu")
        axs[0].set_ylabel("u")

        # Scatterplot at x=1
        axs[1].set_title("Values of u at x = 1")
        axs[1].scatter(sb[len:, 0].detach().numpy(),  pred_sb[len:].detach().numpy(),  color="blue")
        axs[1].set_xlabel("mu")
        axs[1].set_ylabel("u")

        # display legend
        handles, labels = [], []
        for ax in axs:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        fig.legend(handles, labels)

        plt.show()
        plt.savefig(str(path + "figures/antiboundary_values"))

    def boundary_plot(self):
        # Scatterplot boundary points ie. (x=0, mu>0) and (x=1,mu<0)
        for (sb, u_sb) in self.training_set_sb:
            fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

            pred_sb = self.approximate_solution(sb.to(device)).reshape(-1,).cpu()
            mu_plus = sb[:, 0] > 0.
            mu_mins = sb[:, 0] < 0.

            # Scatterplot at x=0
            axs[0].set_title("Values of u at boundary points")
            axs[0].scatter(sb[:, 0][mu_plus].detach().numpy(),  u_sb[mu_plus].detach().numpy(), label="imposed BC",
                        color="red")
            axs[0].scatter(sb[:, 0][mu_plus].detach().numpy(),  pred_sb[mu_plus].detach().numpy(),
                        label="approximate solution at boundaries", color="blue")

            axs[0].set_xlabel("mu")
            axs[0].set_ylabel("u")

            # Scatterplot at x=1
            axs[1].set_title("Values of u at boundary points")
            axs[1].scatter(sb[:, 0][mu_mins].detach().numpy(),  u_sb[mu_mins].detach().numpy(), color="red")
            axs[1].scatter(sb[:, 0][mu_mins].detach().numpy(),  pred_sb[mu_mins].detach().numpy(),  color="blue")
            axs[1].set_xlabel("mu")
            axs[1].set_ylabel("u")

            # display legend
            handles, labels = [], []
            for ax in axs:
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)
            fig.legend(handles, labels)

            plt.show()
            plt.savefig(str(path + "figures/boundary_values"))

################################################################################################ Main function
# Initialize Pinn
n_int, n_sb = 8192, 2048
p = PINN(n_int, n_sb)

# We use the ADAM optimizer
optimizer_LBFGS = optim.LBFGS(p.approximate_solution.parameters(),
                                lr=.5,
                                max_iter=5,
                                max_eval=5,
                                history_size=150,
                                line_search_fn="strong_wolfe",
                                tolerance_change=1.0 * np.finfo(float).eps)
optimizer_Adam = optim.Adam(p.approximate_solution.parameters(),
                                lr=9e-4,)
optimizer = optimizer_Adam
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

# Training loop
n_epochs = 10000
losses = p.fit(n_epochs, optimizer, scheduler, verbose=True) 

# Plot loss values
plt.figure()
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Log10 of loss")
plt.legend()
plt.savefig(str(path + "figures/loss" + str(n_epochs)))
plt.show()

# Plot the values at the boundary points, at the space-boundaries and the contour plot over the whole domain
p.draw()
