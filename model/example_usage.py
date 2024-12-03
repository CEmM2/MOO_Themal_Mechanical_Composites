#%%
import numpy as np
import torch
from model import create_model,run_model
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

#%%
class RVE_Opt(Problem):
    PROPERTY_MAP = {
        'density': 0,
        'E11': 1,
        'E22': 2,
        'G12': 3,
        'K1': 4,
        'K2': 5
    }
    def __init__(self, model,run_model,n_obj,obj_names):
        # Initialize the problem with 10 variables, n_obj objectives
        if n_obj != len(obj_names):
            raise ValueError("Number of objectives must match length of objective names")
        if not all(name in self.PROPERTY_MAP for name in obj_names):
            invalid_names = [name for name in obj_names if name not in self.PROPERTY_MAP]
            raise ValueError(f"Invalid property names: {invalid_names}")
        super().__init__(
            n_var=10,          # number of input variables
            n_obj=n_obj,           # number of objectives
            n_constr=0,        # number of constraints
            obj_names =obj_names,#names of the objectives e.g.['density','G12','K2']
            xl=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25]),    # lower bounds
            xu=np.array([3.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 0.75])  # upper bounds
        )
        self.model = model
        self.run_model=run_model
        self.obj_names = obj_names

    def _evaluate(self, x, out, *args, **kwargs):
        # Convert numpy array to torch tensor (ensuring double precision)
        
        
        # Evaluate the model (no need for torch.no_grad() since we're not training)
        y = self.run_model(self.model, x)
        # Extract objectives
        objectives = []
        for prop_name in self.obj_names:
            idx = self.PROPERTY_MAP[prop_name]
            value = y[:, idx]
            if prop_name in ['E11','E22','G12']:
                value = -value # we aim to maximize these objectives so we add a minus sign
            objectives.append(value)
        out["F"] = np.column_stack(objectives)
        

def optimize_RVE(model, run_model,n_obj,obj_names,n_generations=100, pop_size=100):
    """
    Optimize the RVE using NSGA-II
    
    Parameters:
    -----------
    model : NeuralNetwork
        The custom neural network to be used as surrogate model for optimizing properties
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
    problem = RVE_Opt(model,run_model,n_obj,obj_names)
    
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
n_obj = 3
obj_names = ['density','G12','K2']
# Run optimization
res = optimize_RVE(model,run_model,n_obj,obj_names, n_generations=2000, pop_size=300)
    


#%%
