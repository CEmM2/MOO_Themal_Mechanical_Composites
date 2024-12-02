#%%
import numpy as np
import torch
from App.torch_model import create_model,run_model
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from matplotlib.colors import Normalize,LinearSegmentedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{textgreek}"
})
from plots import create_custom_colorscale
import scienceplots
#%%
'''
   N_data[:,:,0] = data[:,:,0]/matrix_vals['rho']
    N_data[:,:,1] = data[:,:,1]/matrix_vals['E']
    N_data[:,:,2] = data[:,:,2]/matrix_vals['E']
    N_data[:,:,3] = data[:,:,3]/matrix_vals['G']
    N_data[:,:,4] = data[:,:,4]/matrix_vals['K']
    N_data[:,:,5] = data[:,:,5]/matrix_vals['K']
    return N_data
'''
class NNOptimizationProblem(Problem):
    def __init__(self, model,run_model):
        # Initialize the problem with 10 variables, 3 objectives
        super().__init__(
            n_var=10,          # number of input variables
            n_obj=3,           # number of objectives
            n_constr=0,        # number of constraints
            xl=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25]),    # lower bounds
            xu=np.array([3.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 0.75])  # upper bounds
        )
        self.model = model
        self.run_model=run_model

    def _evaluate(self, x, out, *args, **kwargs):
        # Convert numpy array to torch tensor (ensuring double precision)
        
        
        # Evaluate the model (no need for torch.no_grad() since we're not training)
        y = self.run_model(self.model, x)
        
        
        # Extract objectives
        # - minimize output[0]
        # - maximize output[2] (minimize -output[2])
        # - minimize output[5]
        out["F"] = np.column_stack([
            y[:, 0],     
            -y[:, 3],
            y[:,5],
        ])

def optimize_neural_network(model, run_model,n_generations=100, pop_size=100):
    """
    Optimize the custom neural network using NSGA-II
    
    Parameters:
    -----------
    model : NeuralNetwork
        The custom neural network to optimize
    n_generations : int
        Number of generations for NSGA-II
    pop_size : int
        Population size for NSGA-II 
    
    Returns:
    --------
    res : OptimizeResult
        The optimization result containing the Pareto front
    """
    # Create the problem
    problem = NNOptimizationProblem(model,run_model)
    
    # Configure the algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=30),
        eliminate_duplicates=True
    )
    
    # Run the optimization
    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_generations),
        seed=1,
        verbose=True
    )
    
    return res
#%%
#%%
matrix_vals = {'rho':1.42,'E':2.5,'G':2.5/(2*(1+0.34)),'K':0.12}
model = create_model()
    
# Run optimization
res = optimize_neural_network(model,run_model, n_generations=2000, pop_size=300)
    


#%%