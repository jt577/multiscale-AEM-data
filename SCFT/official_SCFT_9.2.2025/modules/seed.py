import numpy as np
import torch

def make_grid(gridShape):
    """
    Build real-space grid r with shape (N, N, N, 3) over [0,L)^3.
    """
    x = np.linspace(0, 1, gridShape[0], endpoint=False)
    y = np.linspace(0, 1, gridShape[1], endpoint=False)
    z = np.linspace(0, 1, gridShape[2], endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return np.stack((X, Y, Z), axis=-1)  # (*gridShape,3)

def lamella(r, period, phase=0.0):
    """
    Single-mode lamella along x with spacing `period`.
    Cubic space group.
    """
    x = r[...,0]
    return np.cos(2*np.pi*x/period + phase)

def cylinders(r, period, phase=0.0):
    """
    Cylinders in the xy-plane with nearest-neighbor spacing `period`.
    Note that the lattice determines the space group of the cylinders.
    """
    x, y = r[...,0], r[...,1]
    field = np.zeros(r.shape[:-1])
    field = np.cos(2*np.pi/period * x + phase) + np.cos(2*np.pi/period * y + phase)
    return field

def spheres(r, period, phase=0.0):
    """
    Body-centered cubic spheres (6 modes) with spacing `period`.
    Cubic space group.
    """
    G = 2*np.pi/period
    dirs = [
        ( 1,  1,  0),
        ( 1, -1,  0),
        ( 1,  0,  1),
        ( 1,  0, -1),
        ( 0,  1,  1),
        ( 0,  1, -1),
    ]
    field = np.zeros(r.shape[:-1])
    for ux, uy, uz in dirs:
        field += np.cos(G*(ux*r[...,0] + uy*r[...,1] + uz*r[...,2]) + phase)
    return field


def gyroid(r, period):
    """
    Gyroid analytic form.
    f = sin(2πx/a)cos(2πy/a) + sin(2πy/a)cos(2πz/a) + sin(2πz/a)cos(2πx/a)
    Cubic space group.
    """
    a = period
    x, y, z = r[...,0], r[...,1], r[...,2]
    return (
        np.sin(2*np.pi*x/a)*np.cos(2*np.pi*y/a)
        + np.sin(2*np.pi*y/a)*np.cos(2*np.pi*z/a)
        + np.sin(2*np.pi*z/a)*np.cos(2*np.pi*x/a)
    )

def diamond(r, period):
    """
    Schwarz diamond structure analytic form.
    Cubic space group.
    """
    x, y, z = r[...,0], r[...,1], r[...,2]
    u = 2*np.pi*x/period
    v = 2*np.pi*y/period
    w = 2*np.pi*z/period
    return np.cos(u)*np.cos(v)*np.cos(w) - np.sin(u)*np.sin(v)*np.sin(w)

def O70(r, period):
    """
    O70 structure analytic form.
    Orthorhombic space group with side lengths: (1, 2, 2*sqrt(3)).
    """
    x, y, z = r[...,0], r[...,1], r[...,2]
    Glist = [
        ( 0,  0,  4),
        ( 1,  1,  1), ( 1,  1, -1),
        ( 1, -1,  1), (-1,  1,  1),
        ( 0,  2,  2), ( 0,  2, -2),
    ]
    return sum(np.cos(2*np.pi*(gx*x + gy*y + gz*z)/period) for gx,gy,gz in Glist)

def primitive(r, period):
    """
    Schwarz primitive analytic form.
    Cubic space group.
    """
    x, y, z = r[...,0], r[...,1], r[...,2]
    return np.cos(2*np.pi*x/period) + np.cos(2*np.pi*y/period) + np.cos(2*np.pi*z/period)


def make_seed(morphology, num_phases, gridShape, amplitude=1e-2, period=None, noise_amp=1e-3):
    """
    Build flattened torch.Tensor initial guess for an ABC triblock SCFT.
    
    morphology: 'lamella', 'cylinders', 'spheres', or 'gyroid'
    gridShape : number of grid points per dim
    amplitude : seed amplitude
    period    : periodicity of structure
    noise_amp : amplitude of small random noise
    """
    if period is None:
        period = 1
    
    # build grid
    r = make_grid(gridShape)
    
    # choose field generator
    if   morphology == 'lamella':
        f = lamella(r, period)
    elif morphology == 'cylinders':
        f = cylinders(r, period)
    elif morphology == 'spheres':
        f = spheres(r, period)
    elif morphology == 'gyroid':
        f = gyroid(r, period)
    elif morphology == 'diamond':
        f = diamond(r, period)
    elif morphology == 'O70':
        f = O70(r, period)
    elif morphology == 'primitive':
        f = primitive(r, period)
    else:
        raise ValueError(f"Unknown morphology '{morphology}'")
    
    x0 = amplitude * f[None, ...] + noise_amp * np.random.uniform(-1, 1, size=(num_phases, gridShape[0], gridShape[1], gridShape[2]))

    x0 = x0.reshape(-1)
    return torch.tensor(x0, dtype=torch.float64)