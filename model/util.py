MIT License

Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

import torch
import numpy as np

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep) ** 2
    elif schedule == "cosine":
        timesteps = torch.linspace(0, 1, n_timestep + 1) + cosine_s
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas.clone().detach()).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = torch.clamp(betas, min=0, max=0.999)
    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas

def extract_into_tensor(a, t, x_shape):
    b = t.size(0)
    t = t.to(a.device)
    t = t.view(-1)
    out = a.gather(0, t)
    return out.view(b, *((1,) * (len(x_shape) - 1)))
