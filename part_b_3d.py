from Common import NeuralNet
import gc
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
torch.autograd.set_detect_anomaly(True)
from matplotlib import colors
from tqdm import tqdm
import numpy as np

pi = np.pi

class PINN:
    def __init__(self, n_int_, n_sb_, device_):
        self.n_int = n_int_
        self.n_sb = n_sb_
        self.device = device_

        
        self.domain_extrema = torch.tensor([[-0., 1.], # X
                                            [-0., 1.], # Y
                                            [-0., 1.], # Z
                                            [-0., pi], # phi
                                            [-0., 2. * pi]  # theta
                                            ])

        # Number of space dimensions
        self.space_dimensions = 3

        # Parameter to balance role of data and PDE
        self.lambda_u = 10

        # F Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0],
                                    output_dimension=1,
                                    n_hidden_layers=4,
                                    neurons=20,
                                    regularization_param=0.1,
                                    regularization_exp=2.,
                                    retrain_seed=42)

        self.approximate_solution.to(self.device)

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_int = self.assemble_datasets()

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    # Function returning the input-output tensor required to assemble the training set S_sb corresponding to the spatial boundary
    def add_spatial_boundary_points(self):
        # spatial: [:3]
        # angle:   [3:] (defined on unit sphere surface)
        # we don't do \nu for now...
        x0 = y0 = z0 = self.domain_extrema[1, 0]
        xL = yL = zL = self.domain_extrema[1, 1]
        phi0 = self.domain_extrema[3, 0]
        theta0 = self.domain_extrema[4, 0]
        phiL = self.domain_extrema[3, 1]
        thetaL = self.domain_extrema[4, 1]

        n_per_side = self.n_sb // self.space_dimensions
        n_per_face = n_per_side // 2

        input_sb = self.convert(self.soboleng.draw(n_per_face))

        xyz = input_sb[:, :3]
        omega = input_sb[:, 3:]

        x_0, x_L = torch.clone(xyz), torch.clone(xyz)
        y_0, y_L = torch.clone(xyz), torch.clone(xyz)
        z_0, z_L = torch.clone(xyz), torch.clone(xyz)

        x_0 = torch.cat((x_0, omega), 1)
        y_0 = torch.cat((y_0, omega), 1)
        z_0 = torch.cat((z_0, omega), 1)

        x_L = torch.cat((x_L, omega), 1)
        y_L = torch.cat((y_L, omega), 1)
        z_L = torch.cat((z_L, omega), 1)

        x_0[:, 0] = 0.
        y_0[:, 1] = 0.
        z_0[:, 2] = 0.

        x_L[:, 0] = 1.
        y_L[:, 1] = 1.
        z_L[:, 2] = 1.

        # zero Dirichlet BC

        output_sb_0_x = torch.zeros(input_sb.shape[0])
        output_sb_0_y = torch.zeros(input_sb.shape[0])
        output_sb_0_z = torch.zeros(input_sb.shape[0])

        output_sb_1_x = torch.zeros(input_sb.shape[0])
        output_sb_1_y = torch.zeros(input_sb.shape[0])
        output_sb_1_z = torch.zeros(input_sb.shape[0])

        return torch.cat([x_0, x_L, y_0, y_L, z_0, z_L], 0), torch.cat(
            [output_sb_0_x, output_sb_1_x, output_sb_0_y, output_sb_1_y,
             output_sb_0_z, output_sb_1_z, ], 0).reshape(-1, 1)

    def draw_boundary(self):
        inp, _ = self.add_spatial_boundary_points()

        def u(x):
            u = torch.ones(inp.shape[0])
            b00 = torch.where(x[:, 0] == 0.)
            b01 = torch.where(x[:, 1] == 0.)
            b02 = torch.where(x[:, 2] == 0.)

            b10 = torch.where(x[:, 0] == 1.)
            b11 = torch.where(x[:, 1] == 1.)
            b12 = torch.where(x[:, 2] == 1.)

            for i, b in enumerate([b00, b01, b02, b10, b11, b12]):
                u[b] = i * .1
            return u

        u_values = u(inp)
        x = inp[:, 0].detach().numpy()
        y = inp[:, 1].detach().numpy()
        z = inp[:, 2].detach().numpy()

        def angle(a1, a2, a3):
            return 1.1 * torch.ones(inp.shape[0])

        phi = inp[:, 3]
        theta = inp[:, 4]
        om1 = torch.cos(phi) * torch.sin(theta)
        om2 = torch.sin(phi) * torch.sin(theta)
        om3 = torch.cos(theta)

        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(221, projection='3d')

        ax1.scatter(x, y, z, c=u_values, label="boundary")
        ax2.scatter(om1, om2, om3, c=angle(om1, om2, om3), label="angle")
        fig.subplots_adjust(hspace=2)
        plt.tight_layout()
        plt.show()

    #  Function returning the input-output tensor required to assemble the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))
        return input_int, output_int

    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        input_sb, output_sb = self.add_spatial_boundary_points()  # S_sb
        input_int, output_int = self.add_interior_points()  # S_int

        training_set_sb = DataLoader(torch.utils.data.TensorDataset(input_sb, output_sb),
                                     batch_size=(self.space_dimensions + 2) * self.n_sb, shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int,
                                      shuffle=False)

        return training_set_sb, training_set_int

    # Function to compute the terms required in the definition of the SPATIAL boundary residual
    def apply_boundary_conditions(self, input_sb):
        u_pred_sb = self.approximate_solution(input_sb)
        return u_pred_sb

    def sig(self, x, y, z):
        return torch.ones(x.shape[0], device=self.device)

    def I_b(self, x, y, z):
        r0 = .5
        c = .5
        cc = torch.ones((3)) * c  # center
        r = torch.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2)
        r0 = torch.clone(r)
        mask = r0 >= .5

        r0[mask] = 0.
        r0[~mask] = abs(r0[~mask] - .5)

        return r0

        # Function to compute the PDE residuals

    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True

        x = input_int[:, 0]
        y = input_int[:, 1]
        z = input_int[:, 2]

        phi = input_int[:, 3]
        theta = input_int[:, 4]

        u = self.approximate_solution(input_int)

        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]
        grad_u_x = grad_u[:, 0]
        grad_u_y = grad_u[:, 1]
        grad_u_z = grad_u[:, 2]


        om1 = torch.cos(phi) * torch.sin(theta)
        om2 = torch.sin(phi) * torch.sin(theta)
        om3 = torch.cos(theta)

        # LHS 
        I_b = self.I_b(x, y, z)

        t1 = (I_b + self.sig(x, y, z)) * u 
        t2 = om1 * grad_u_x + om2 * grad_u_y + om3 * grad_u_z
        t3 = self.kernel(input_int) * 0.5 * (1/4*pi)

        # RHS 
        t4 = - (I_b ** 2)
        return (t1 + t2 + t3 + t4).reshape(-1,)



    def kernel(self, x):
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
            [0.1019301198172404,	-0.7463319064601508],
            [0.1019301198172404,	 0.7463319064601508],
            [0.0832767415767048,	-0.8391169718222188],
            [0.0832767415767048,	 0.8391169718222188],
            [0.0626720483341091,	-0.9122344282513259],
            [0.0626720483341091,	 0.9122344282513259],
            [0.0406014298003869,	-0.9639719272779138],
            [0.0406014298003869,	 0.9639719272779138],
            [0.0176140071391521,	-0.9931285991850949],
            [0.0176140071391521,	 0.9931285991850949],
            ], device=self.device)

        gl_weights = wx[:, 0]
        gl_abcissa = wx[:, 1]

        def leg(x, n):
            return torch.special.legendre_polynomial_p(x, n)

        """
        self.domain_extrema = torch.tensor([[-0., 1.], # X
                                            [-0., 1.], # Y
                                            [-0., 1.], # Z
                                            [-0., pi], # phi
                                            [-0., 2. * pi]  # theta
                                            ])
        """
        abcissa_phi = (torch.clone(gl_abcissa) + 1) * self.domain_extrema[3, 1]
        abcissa_theta = (torch.clone(gl_abcissa) + 1) * self.domain_extrema[4, 1]

        weights_phi = torch.clone(gl_weights)
        weights_theta = torch.clone(gl_weights)
        sin_phi = torch.sin(abcissa_phi)
        x_shape = x.shape[0]
        abc_cardinality = abcissa_phi.shape[0]
        cart_prod_cardinality = abcissa_phi.shape[0] * abcissa_theta.shape[0]


        dummy = torch.linspace(0, x_shape-1, x_shape, dtype=torch.float32,device=self.device)
        # so evaluate u at (gl_abcissa.shape[0] ** 2) * x.shape[0] times
        eval_points = torch.cartesian_prod(dummy, abcissa_phi, abcissa_theta)
        cart_prod_cardinality = abcissa_phi.shape[0] * abcissa_theta.shape[0] 
        # repeat the x values enough make it easy to do dot product
        x_repeat = x.repeat_interleave(cart_prod_cardinality, 0)
        eval_points = torch.cat((x_repeat[:, :3], eval_points[:, 1:]), 1)

        u_eval = self.approximate_solution(eval_points).reshape(-1,)
        weights_phi = weights_phi * sin_phi
        u_eval = u_eval.reshape([x_shape,abc_cardinality,abc_cardinality])

        # 1st contraction over theta
        kernel2 = torch.einsum('ijk, k',u_eval,weights_phi)
        # print(f'{kernel2.shape=}')
        # 2nd contraction over phi
        kernel2 = torch.einsum('ij, j',kernel2,weights_theta)
        # print(f'{kernel2.shape=}')

        """
        # for every point (x, y, z) we integrate the entire abcissa ie.
        # integrate over the unit sphere
        kernel = u_int @ weights
        """
        return  kernel2 * pi**2 * 0.5



    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_sb, u_train_sb, inp_train_int, verbose=True):
        u_pred_sb = self.apply_boundary_conditions(inp_train_sb)
        assert (u_pred_sb.shape[1] == u_train_sb.shape[1])

        r_sb = u_pred_sb - u_train_sb
        r_int = self.compute_pde_residual(inp_train_int)

        loss_int = torch.mean(abs(r_int) ** 2)
        loss_sb = torch.mean(abs(r_sb) ** 2)
        loss_u = loss_sb
        loss = torch.log10(self.lambda_u * (loss_sb) + loss_int)
        if verbose: print("Total loss: ", round(loss.item(), 4), "| PDE Loss: ", round(torch.log10(loss_u).item(), 4),
                          "| Function Loss: ", round(torch.log10(loss_int).item(), 4))

        return loss

    def fit(self, num_epochs, optimizer, verbose=True):
        history = list()


        for epoch in tqdm(range(num_epochs)):
            
            for j, ((inp_train_sb, u_train_sb), (inp_train_int, u_train_int)) in enumerate(zip(self.training_set_sb, self.training_set_int)):
                inp_train_sb = inp_train_sb.to(self.device)
                u_train_sb = u_train_sb.to(self.device)
                inp_train_int = inp_train_int.to(self.device)
                u_train_int = u_train_int.to(self.device)

                if epoch == 0:
                    print(f'{inp_train_sb.device=}')
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(inp_train_sb, u_train_sb, inp_train_int, verbose=verbose)
                    loss.backward()

                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])
        return history

    def draw(self, Ib=False):

        inputs = self.convert(self.soboleng.draw(30000))
        inputs = inputs.to(self.device)

        output = None
        if Ib:
            output = self.I_b(inputs[:, 0], inputs[:, 1], inputs[:, 2]) ** 2
        else:
            output = self.kernel(inputs) * 4 * pi

        inputs = inputs.cpu()
        output = output.cpu()
       
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = inputs[:, 0], inputs[:, 1], inputs[:, 2]
        x, y, z = x.detach().numpy(), y.detach().numpy(), z.detach().numpy()
        u = output.detach().numpy()

        spherical = False
        cap = True
        mask = None
        if spherical:
            data_points = np.column_stack((x, y, z))
            distances = np.linalg.norm(data_points, axis=1)
            mask = distances > .5
        elif cap:
            cut_center = np.array([0.5, 0.5, 0.5])  # center of the cut cone
            phi_min = 0  # minimum azimuthal angle of the cut cone
            phi_max = pi / 2  # maximum azimuthal angle of the cut cone
            theta_min = 0  # minimum polar angle of the cut cone
            theta_max = pi / 2  # maximum polar angle of the cut cone

            # Filter the data points to only include those that are outside the cut cone
            data_points = np.column_stack((x, y, z))
            data_vectors = data_points - cut_center
            distances = np.linalg.norm(data_vectors, axis=1)
            phi = np.arctan2(data_vectors[:, 1], data_vectors[:, 0])
            theta = np.arccos(data_vectors[:, 2] / distances)
            mask = (phi < phi_min) | (phi > phi_max) | (theta < theta_min) | (theta > theta_max)
        else:
            cut = np.array([.5] * 3)
            cut_normal = np.array([1.] * 3)
            data = np.column_stack((x, y, z))
            data_vectors = data - cut
            dot_products = np.dot(data_vectors, cut_normal)
            mask = dot_products > .0

        x, y, z = x[mask], y[mask], z[mask]
        u = u[mask]
        scatter = ax.scatter(x, y, z, c=u,)
               
        ax.grid(True, which="both", ls=":")
        ax.set_title("Approximate Solution")

        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
        ax.set_xlim(-.5, 1.5)
        ax.set_ylim(-.5, 1.5)
        ax.set_zlim(-.5, 1.5)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis.line.set_color('none')
        ax.yaxis.line.set_color('none')
        ax.zaxis.line.set_color('none')


        ax.azim = 45 - 5

        fig.colorbar(scatter)
        plt.tight_layout()
        plt.show()


n_int, n_sb = 256, 256
no_epochs = 2500
device = torch.device("cpu")

p = PINN(n_int, n_sb,device)
#p.draw_boundary()
optimizer = optim.Adam(p.approximate_solution.parameters(),
                                lr=5e-4,)
losses = p.fit(no_epochs, optimizer, verbose=True)
p.draw(Ib=False)
