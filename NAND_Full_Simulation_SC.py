#%%

import torch
from tqdm.cli import tqdm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, ceil, floor

# Constants
meters = 1
centimeters = 1e-2 * meters
millimeters = 1e-3 * meters
micrometers = 1e-6 * meters
nanometers = 1e-9 * meters
inches = 2.54 * centimeters
feet = 12 * inches

millivolts = 1e-3
volts = 1
kilovolts = 1e3

seconds = 1
nanoseconds = 1e-9 * seconds
picoseconds = 1e-12 * seconds
femtoseconds = 1e-15 * seconds

hertz = 1 / seconds
kilohertz = 1e3 * hertz
megahertz = 1e6 * hertz
gigahertz = 1e9 * hertz

eps0 = 8.85418782e-12 * 1 / meters
miu0 = 4 * np.pi * 1e-7 * 1 / meters
c0 = 1 / np.sqrt(miu0 * eps0) * meters / seconds

# Simulation parameters
USE_TMZ_MODE = False
USE_TEZ_MODE = True

USE_DSRC = False
USE_ESRC = True
USE_HSRC = True

USE_ADVANCED_VISUALISATION = True
DO_DIELECTRIC_SMOOTHING = True
APPLY_NONLINEARITY = True
USE_SINGLE_PRECISION = False

DISPLAY_LABELS = True
AUTO_FPS_ADJUSTMENT = True
SHUTDOWN_ON_EXIT = False
PLAY_ON_FINISH = False
DISPLAY_TRANSFER_WINDOW = True

DO_FOURIER_TRANSFORM = False
DO_TRANSFER_CALCULATION = False

USE_EMPTY_GRID = True
USE_FULL_TFSF_BOX = False
USE_TFSF_SOURCE = True
USE_HARD_SOURCE = False
USE_DIRICHLET_BC = True
USE_PERIODIC_BC = False

#%%

cases_upper_AND = torch.tensor([-55, -55, -55, -55]).int()
cases_middle_AND = torch.tensor([105, 105, 105, 105]).int()
cases_lower_AND = torch.tensor([-55, -55, -55, -55]).int()

cases_upper_NOT = torch.tensor([0, 0, 0, 0]).int()
cases_middle_NOT = torch.tensor([0, 0, 0, 0]).int()
cases_lower_NOT = torch.tensor([0, 0, 0, 0]).int()

cases_brd_dim = torch.tensor([0.1000]).float()

totalCases = 1

# Grid parameters
NBUFF = torch.tensor([5, 5, 5, 5]).int()
NLAM = 25
NDIM = 3

# Device parameters
NCELL = torch.tensor([18, 18]).int()
CELLS = torch.tensor([19, 88]).int()
scale = torch.tensor([(3**.5) / 2, 1])
NDEVICE = torch.ceil(CELLS * NCELL * scale).int()
cylinder_diameters = NCELL * 0.32
bridge_cylinder_diamters = NCELL * cases_brd_dim
bending_cylinder_diameters = NCELL * 0.125
coupling_cylinder_diameters = NCELL * 0.25
NORMALIZED_FREQ = 0.232
bridge_length = 1

# Material parameters
n_filler = 2.95
n_cyl = 1
n_coupl = 1
chi3 = 1e-4
ermax = 2
urmax = 1
nmax = np.sqrt(ermax * urmax)
NPML = torch.tensor([12, 12, 12, 12])
PMLEXP = 3

# Source parameters
input_recording_files = 'alternating_80k_55amp.mat'
REUSE_SOURCE_RECORDING = False
FMAX = 5.0 * gigahertz
NFREQ = 1000
FREQ = torch.linspace(0, FMAX, NFREQ)
LAMD = 1.55 * micrometers
STEPS = 200000
BEAM_WIDTH = torch.round(NCELL[1] * np.sqrt(3) / 2 / 2)
SRC_START_WIDTH = 1200
SRC_START = SRC_START_WIDTH * 4
SRC_END = float('inf')
PULSE_LENGTH = 7000

# AND params
beam_main_amp_AND = cases_middle_AND[0]
beam_upper_amp_AND = cases_upper_AND[0]
beam_lower_amp_AND = cases_lower_AND[0]
UPPER_FIRST_PULSE_AT_AND = 40000
UPPER_NEXT_PULSE_AFTER_AND = 160000
LOWER_FIRST_PULSE_AT_AND = 40000
LOWER_NEXT_PULSE_AFTER_AND = 80000

# NOT params
beam_main_amp_NOT = cases_middle_NOT[0]
beam_upper_amp_NOT = cases_upper_NOT[0]
beam_lower_amp_NOT = cases_lower_NOT[0]
UPPER_FIRST_PULSE_AT_NOT = 40000
UPPER_NEXT_PULSE_AFTER_NOT = 160000
LOWER_FIRST_PULSE_AT_NOT = 40000
LOWER_NEXT_PULSE_AFTER_NOT = 80000

PHASE_DEVIATION = 0
FREQ_DEVIATION = 1

gen_name_prefix = 'xl_and_arrow_'

# %%

# Plot parameters
FULLSCREEN = False
RECORD_ANIMATION = False
FPS = 1000
movie_duration = 25 * seconds
movie_name = f"{gen_name_prefix}{round(cases_brd_dim.item() * 10000)}_"
REC_QUALITY = 100
FRAME_RATE = 30
plotting_amp = 80

#%%

# Recording parameters
RECORD_FIELDS = True
DO_FULL_RECORDING = True
steps_per_recording = 100
file_name = f"{gen_name_prefix}{round(cases_brd_dim.item() * 1000)}_brd.mat"

if DO_FULL_RECORDING:
    observation_points_upper = torch.arange(1618)
    observation_points_middle = torch.arange(1618)
elif RECORD_FIELDS:
    torch.save({}, file_name)

outlets = {
    'AND': {'up': False, 'lo': False},
    'NOT': {'md': True, 'lo': True}
}

JUST_VISUALIZE_STRUCTURE = False

# Open a figure window
if FULLSCREEN:
    plt.figure(figsize=(14, 14))
else:
    plt.figure()

# Compute grid
lam0 = c0 / FMAX
dx = LAMD / NCELL[0]
dy = LAMD / NCELL[0]

BUFFS = NBUFF + NPML
Nx = (torch.sum(BUFFS[:2]) + NDEVICE[0]).item()
Ny = (torch.sum(BUFFS[2:]) + NDEVICE[1]).item()

xa = torch.arange(0, Nx) * dx
ya = torch.arange(0, Ny) * dy

# Build device on grid
Nx2 = 2 * Nx
Ny2 = 2 * Ny

ER2x = torch.ones(Nx2, Ny2)
UR2x = torch.ones(Nx2, Ny2)
CHI2x = torch.zeros(Nx2, Ny2)
PEC2x = torch.ones(Nx2, Ny2)

DeviceNx2x = 2 * NDEVICE[0]
DeviceNy2x = 2 * NDEVICE[1]
BUFFS2x = 2 * BUFFS

eps_geometry_matrix = torch.ones(DeviceNx2x, DeviceNy2x)
chi_geometry_matrix = torch.ones(DeviceNx2x, DeviceNy2x)
pec_geometry_matrix = torch.ones(DeviceNx2x, DeviceNy2x)

CellNx = CELLS[0].item()
CellNy = CELLS[1].item()

eps_cell_matrix = torch.ones(CellNx, CellNy)
chi_cell_matrix = torch.zeros(CellNx, CellNy)
diameter_cell_matrix = torch.ones(CellNx, CellNy)

# Draw in cell matrix
eps_cell_matrix *= n_cyl ** 2
half_bridge_len = round(bridge_length / 2) + 2

# Smaller cylinders in the middle
diameter_cell_matrix[:, torch.round(torch.linspace(CellNy / 2, 3, 1)).long()] = 1 / cylinder_diameters[0] * bridge_cylinder_diamters[0]

# AND part
# diameter_cell_matrix[round(CellNx / 2),    :round(CellNy / 2 - half_bridge_len)] = -2j
# diameter_cell_matrix[round(CellNx / 2 - 2) :round(CellNy / 2 - half_bridge_len)] = -2j
# diameter_cell_matrix[round(CellNx / 2 + 2),:round(CellNy / 2 - half_bridge_len)] = -2j

# # NOT part
# diameter_cell_matrix[round(CellNx / 2 + 2), torch.round(CellNy / 2 + half_bridge_len):] = -2j
# diameter_cell_matrix[round(CellNx / 2 - 2 + 2), round(CellNy / 2 + half_bridge_len):] = -2j
# diameter_cell_matrix[round(CellNx / 2 + 2 + 2), round(CellNy / 2 + half_bridge_len):] = -2j

# # The bridge section
# diameter_cell_matrix[round(CellNx / 2), round(torch.linspace(CellNy / 2, 2, 1))] = -2j

# # Cutting the arrowhead at the end of AND part
# diameter_cell_matrix[round(CellNx / 2 - 1), round(CellNy / 2 - 3)] = -2j
# diameter_cell_matrix[round(CellNx / 2 + 1), round(CellNy / 2 - 3)] = -2j

# PECs
eps_geometry_matrix *= n_filler ** 2
chi_geometry_matrix *= chi3
diameter_cell_matrix *= cylinder_diameters[0]

ER2x *= n_filler ** 2

# Redraw in auxiliary matrix
if not USE_EMPTY_GRID:
    tempx = DeviceNx2x / CellNx
    tempy = DeviceNy2x / CellNy
    for cell_x in range(1, CellNx, 2):
        for cell_y in range(CellNy):
            end_x = int(cell_x * tempx)
            end_y = int(cell_y * tempy)
            start_x = int(end_x - tempx)
            start_y = int(end_y - tempy)
            mid_x = int((start_x + end_x) / 2)
            mid_y = int((start_y + end_y) / 2)
            for x in range(start_x, end_x):
                for y in range(start_y, end_y):
                    if (x - mid_x) ** 2 + (y - mid_y) ** 2 < (2 * diameter_cell_matrix[cell_x, cell_y]) ** 2:
                        if x > 0 and y > 0:
                            eps_geometry_matrix[x, y] = eps_cell_matrix[cell_x, cell_y]
                            chi_geometry_matrix[x, y] = chi_cell_matrix[cell_x, cell_y]

    for cell_x in range(2, CellNx, 2):
        for cell_y in range(2, CellNy):
            end_x = int(cell_x * tempx)
            end_y = int(cell_y * tempy)
            start_x = int(end_x - tempx)
            start_y = int(end_y - tempy)
            mid_x = int((start_x + end_x) / 2)
            mid_y = int((start_y + end_y) / 2)
            for x in range(start_x, end_x):
                for y in range(start_y, end_y):
                    if (x - mid_x) ** 2 + (y - mid_y) ** 2 < (2 * diameter_cell_matrix[cell_x, cell_y]) ** 2:
                        if x > 0 and y > 0:
                            eps_geometry_matrix[x, y - int(tempy / 2)] = eps_cell_matrix[cell_x, cell_y]
                            chi_geometry_matrix[x, y - int(tempy / 2)] = chi_cell_matrix[cell_x, cell_y]

# Redraw in 2x grid
ER2x[BUFFS2x[0]:Nx2 - BUFFS2x[1], BUFFS2x[2]:Ny2 - BUFFS2x[3]] = eps_geometry_matrix
CHI2x[BUFFS2x[0]:Nx2 - BUFFS2x[1], BUFFS2x[2]:Ny2 - BUFFS2x[3]] = chi_geometry_matrix
PEC2x[BUFFS2x[0]:Nx2 - BUFFS2x[1], BUFFS2x[2]:Ny2 - BUFFS2x[3]] = pec_geometry_matrix

# Perform dielectric smoothing
if DO_DIELECTRIC_SMOOTHING:
    ID = torch.ones(2, 2) / 4
    ER2x[:Nx2 - 1, :Ny2 - 1] = torch.nn.functional.conv2d(ER2x.unsqueeze(0).unsqueeze(0), ID.unsqueeze(0).unsqueeze(0), padding=0).squeeze()
    UR2x[:Nx2 - 1, :Ny2 - 1] = torch.nn.functional.conv2d(UR2x.unsqueeze(0).unsqueeze(0), ID.unsqueeze(0).unsqueeze(0), padding=0).squeeze()
    CHI2x[:Nx2 - 1, :Ny2 - 1] = torch.nn.functional.conv2d(CHI2x.unsqueeze(0).unsqueeze(0), ID.unsqueeze(0).unsqueeze(0), padding=0).squeeze()

# Parse back onto 1x grid
if USE_TMZ_MODE:
    ERz = ER2x[::2, ::2]
    URx = UR2x[::2, 1::2]
    URy = UR2x[1::2, ::2]
    if APPLY_NONLINEARITY:
        CHI = CHI2x[::2, ::2]

if USE_TEZ_MODE:
    URz = UR2x[::2, ::2]
    ERx = ER2x[1::2, ::2]
    ERy = ER2x[::2, 1::2]
    PECx = PEC2x[::2, 1::2]
    PECy = PEC2x[1::2, ::2]
    if APPLY_NONLINEARITY:
        CHIx = CHI2x[1::2, ::2]
        CHIy = CHI2x[::2, 1::2]

# Clear used memory
del ER2x, UR2x, CHI2x, eps_geometry_matrix, chi_geometry_matrix, pec_geometry_matrix

# Calculate port positions
PORT_1 = torch.round(torch.tensor([BUFFS[1] + ((CELLS[0] / 2 - np.sqrt(3) / 2) * NCELL[0]), BUFFS[2] + (1 / 2) * NCELL[0]]))
PORT_2 = torch.round(torch.tensor([BUFFS[3] + ((CELLS[0] / 2 + np.sqrt(3) / 2) * NCELL[0]), Ny - (1 / 2) * NCELL[0] - BUFFS[3]]))

ports = {
    'p1': {'x': torch.round(torch.arange(PORT_1[0] - NCELL[0] * np.sqrt(3) / 2, PORT_1[0] + NCELL[0] * np.sqrt(3) / 2)).int(),
           'y': torch.round(PORT_1[1] + NCELL[0] / 2).int()},
    'p2': {'x': torch.round(torch.arange(PORT_2[0] - NCELL[0] * np.sqrt(3) / 2, PORT_2[0] + NCELL[0] * np.sqrt(3) / 2)).int(),
           'y': torch.round(PORT_2[1] - NCELL[0] / 2).int()}
}

# Compute time step
dt = 0.95 / (np.sqrt(1 / dx ** 2 + 1 / dy ** 2) * c0)

# Compute source
tau = PULSE_LENGTH * dt
omega = 2 * np.pi * c0 / LAMD * NORMALIZED_FREQ
t_axis = torch.arange(0, STEPS) * dt
tprop = nmax * np.sqrt(Nx ** 2 + Ny ** 2) * np.sqrt(dx ** 2 + dy ** 2) / c0
sigma = xa[torch.round(BEAM_WIDTH).long()]
smooth_window = 1 / (1 + torch.exp(-(t_axis - SRC_START * dt) / (SRC_START_WIDTH * dt))) * (1 - 1 / (1 + torch.exp(-(t_axis - SRC_END * dt) / (SRC_START_WIDTH * dt))))

# AND gate source params
t0_lower_AND = LOWER_FIRST_PULSE_AT_AND * dt
t1_lower_AND = LOWER_NEXT_PULSE_AFTER_AND * dt
t0_upper_AND = UPPER_FIRST_PULSE_AT_AND * dt
t1_upper_AND = UPPER_NEXT_PULSE_AFTER_AND * dt

Ny_src_lo_AND = torch.round(BUFFS[2] + 1 * NCELL[1])
Ny_src_hi_AND = Ny
Nx_src_lo_AND = 0
Nx_src_hi_AND = Nx

Nx_src_upper_AND = torch.round(BUFFS[1] + (CELLS[0] / 2 - 2) * NCELL[1] * np.sqrt(3) / 2).int()
Nx_src_middle_AND = torch.round(BUFFS[1] + (CELLS[0] / 2) * NCELL[1] * np.sqrt(3) / 2).int()
Nx_src_lower_AND = torch.round(BUFFS[1] + (CELLS[0] / 2 + 2) * NCELL[1] * np.sqrt(3) / 2).int()

y0_AND = ya[torch.round(BUFFS[2] + 4 * NCELL[1]).long()]
    

x0_upper_AND = xa[Nx_src_upper_AND];
x0_middle_AND = xa[Nx_src_middle_AND];
x0_lower_AND = xa[Nx_src_lower_AND];


t0_lower_NOT = LOWER_FIRST_PULSE_AT_NOT * dt
t1_lower_NOT = LOWER_NEXT_PULSE_AFTER_NOT * dt
t0_upper_NOT = UPPER_FIRST_PULSE_AT_NOT * dt
t1_upper_NOT = UPPER_NEXT_PULSE_AFTER_NOT * dt

Ny_src_lo_NOT = torch.round(BUFFS[2] + 32 * NCELL[1] + bridge_length * NCELL[1])
Ny_src_hi_NOT = Ny
Nx_src_lo_NOT = 0
Nx_src_hi_NOT = Nx

Nx_src_upper_NOT = torch.round(BUFFS[1] + (CELLS[0] / 2 - 2 + 2) * NCELL[1] * (3 ** 0.5) / 2).int()
Nx_src_middle_NOT = torch.round(BUFFS[1] + (CELLS[0] / 2 + 2) * NCELL[1] * (3 ** 0.5) / 2).int()
Nx_src_lower_NOT = torch.round(BUFFS[1] + (CELLS[0] / 2 + 2 + 2) * NCELL[1] * (3 ** 0.5) / 2).int()

x0_upper_NOT = xa[Nx_src_upper_NOT]
x0_middle_NOT = xa[Nx_src_middle_NOT]
x0_lower_NOT = xa[Nx_src_lower_NOT]

# %% Initialize the source
A = -n_filler; # generally sqrt(eps_r / miu_r)
half_dt = dt / 2
half_cell= n_filler * (dx / (2*c0))

if not REUSE_SOURCE_RECORDING:
    #
    ## AND PART
    ##
    # UPPER BEAM
    Esrc_upper_beam_AND = np.exp(-((t_axis - t0_upper_AND) / tau) ** 2) * np.sin(t_axis * omega * FREQ_DEVIATION)
    Hsrc_upper_beam_AND = np.exp(-((t_axis - half_dt - half_cell - t0_upper_AND) / tau) ** 2) * np.sin((t_axis - half_dt - half_cell) * omega * FREQ_DEVIATION)

    num_pulses_upper_AND = int((STEPS - UPPER_FIRST_PULSE_AT_AND) / UPPER_NEXT_PULSE_AFTER_AND + 1)
    for pulse in range(1, num_pulses_upper_AND + 1):
        Esrc_upper_beam_AND += np.exp(-((t_axis - t0_upper_AND - t1_upper_AND * pulse) / tau) ** 2) * np.sin(t_axis * omega * FREQ_DEVIATION)
        Hsrc_upper_beam_AND += np.exp(-((t_axis - half_dt - half_cell - t0_upper_AND - t1_upper_AND * pulse) / tau) ** 2) * np.sin((t_axis - half_dt - half_cell) * omega * FREQ_DEVIATION)

    # LOWER BEAM
    Esrc_lower_beam_AND = np.exp(-((t_axis - t0_lower_AND) / tau) ** 2) * np.sin(t_axis * omega * FREQ_DEVIATION)
    Hsrc_lower_beam_AND = np.exp(-((t_axis - half_dt - half_cell - t0_lower_AND) / tau) ** 2) * np.sin((t_axis - half_dt - half_cell) * omega * FREQ_DEVIATION)

    num_pulses_lower_AND = int((STEPS - LOWER_FIRST_PULSE_AT_AND) / LOWER_NEXT_PULSE_AFTER_AND + 1)
    for pulse in range(1, num_pulses_lower_AND + 1):
        Esrc_lower_beam_AND += np.exp(-((t_axis - t0_lower_AND - t1_lower_AND * pulse) / tau) ** 2) * np.sin(t_axis * omega * FREQ_DEVIATION)
        Hsrc_lower_beam_AND += np.exp(-((t_axis - half_dt - half_cell - t0_lower_AND - t1_lower_AND * pulse) / tau) ** 2) * np.sin((t_axis - half_dt - half_cell) * omega * FREQ_DEVIATION)

    # COMPLETE
    Esrc_main_beam_AND = smooth_window * np.sin(t_axis * omega + PHASE_DEVIATION * np.pi / 180)
    Esrc_full_AND = (1 / A) * (beam_upper_amp_AND * np.exp(-((xa - x0_upper_AND) / sigma) ** 2).reshape(-1, 1) @ Esrc_upper_beam_AND.reshape(1, -1) +
                               beam_lower_amp_AND * np.exp(-((xa - x0_lower_AND) / sigma) ** 2).reshape(-1, 1) @ Esrc_lower_beam_AND.reshape(1, -1) +
                               beam_main_amp_AND * np.exp(-((xa - x0_middle_AND) / sigma) ** 2).reshape(-1, 1) @ Esrc_main_beam_AND.reshape(1, -1))

    Hsrc_main_beam_AND = smooth_window * np.sin((t_axis - half_dt - half_cell) * omega + PHASE_DEVIATION * np.pi / 180)
    Hsrc_full_AND = (beam_upper_amp_AND * np.exp(-((xa - x0_upper_AND) / sigma) ** 2).reshape(-1, 1) @ Hsrc_upper_beam_AND.reshape(1, -1) +
                     beam_lower_amp_AND * np.exp(-((xa - x0_lower_AND) / sigma) ** 2).reshape(-1, 1) @ Hsrc_lower_beam_AND.reshape(1, -1) +
                     beam_main_amp_AND * np.exp(-((xa - x0_middle_AND) / sigma) ** 2).reshape(-1, 1) @ Hsrc_main_beam_AND.reshape(1, -1))

    #
    ## NOT PART
    ##
    # UPPER BEAM
    Esrc_upper_beam_NOT = np.exp(-((t_axis - t0_upper_NOT) / tau) ** 2) * np.sin(t_axis * omega * FREQ_DEVIATION)
    Hsrc_upper_beam_NOT = np.exp(-((t_axis - half_dt - half_cell - t0_upper_NOT) / tau) ** 2) * np.sin((t_axis - half_dt - half_cell) * omega * FREQ_DEVIATION)

    num_pulses_upper_NOT = int((STEPS - UPPER_FIRST_PULSE_AT_NOT) / UPPER_NEXT_PULSE_AFTER_NOT + 1)
    for pulse in range(1, num_pulses_upper_NOT + 1):
        Esrc_upper_beam_NOT += np.exp(-((t_axis - t0_upper_NOT - t1_upper_NOT * pulse) / tau) ** 2) * np.sin(t_axis * omega * FREQ_DEVIATION)
        Hsrc_upper_beam_NOT += np.exp(-((t_axis - half_dt - half_cell - t0_upper_NOT - t1_upper_NOT * pulse) / tau) ** 2) * np.sin((t_axis - half_dt - half_cell) * omega * FREQ_DEVIATION)

    # LOWER BEAM
    Esrc_lower_beam_NOT = np.exp(-((t_axis - t0_lower_NOT) / tau) ** 2) * np.sin(t_axis * omega * FREQ_DEVIATION)
    Hsrc_lower_beam_NOT = np.exp(-((t_axis - half_dt - half_cell - t0_lower_NOT) / tau) ** 2) * np.sin((t_axis - half_dt - half_cell) * omega * FREQ_DEVIATION)

    num_pulses_lower_NOT = int((STEPS - LOWER_FIRST_PULSE_AT_NOT) / LOWER_NEXT_PULSE_AFTER_NOT + 1)
    for pulse in range(1, num_pulses_lower_NOT + 1):
        Esrc_lower_beam_NOT += np.exp(-((t_axis - t0_lower_NOT - t1_lower_NOT * pulse) / tau) ** 2) * np.sin(t_axis * omega * FREQ_DEVIATION)
        Hsrc_lower_beam_NOT += np.exp(-((t_axis - half_dt - half_cell - t0_lower_NOT - t1_lower_NOT * pulse) / tau) ** 2) * np.sin((t_axis - half_dt - half_cell) * omega * FREQ_DEVIATION)

    # COMPLETE
    Esrc_main_beam_NOT = smooth_window * np.sin(t_axis * omega + PHASE_DEVIATION * np.pi / 180)
    Esrc_full_NOT = (1 / A) * (beam_upper_amp_NOT * np.exp(-((xa - x0_upper_NOT) / sigma) ** 2).reshape(-1, 1) @ Esrc_upper_beam_NOT.reshape(1, -1) +
                               beam_lower_amp_NOT * np.exp(-((xa - x0_lower_NOT) / sigma) ** 2).reshape(-1, 1) @ Esrc_lower_beam_NOT.reshape(1, -1) +
                               beam_main_amp_NOT * np.exp(-((xa - x0_middle_NOT) / sigma) ** 2).reshape(-1, 1) @ Esrc_main_beam_NOT.reshape(1, -1))

    Hsrc_main_beam_NOT = smooth_window * np.sin((t_axis - half_dt - half_cell) * omega + PHASE_DEVIATION * np.pi / 180)
    Hsrc_full_NOT = (beam_upper_amp_NOT * np.exp(-((xa - x0_upper_NOT) / sigma) ** 2).reshape(-1, 1) @ Hsrc_upper_beam_NOT.reshape(1, -1) +
                     beam_lower_amp_NOT * np.exp(-((xa - x0_lower_NOT) / sigma) ** 2).reshape(-1, 1) @ Hsrc_lower_beam_NOT.reshape(1, -1) +
                     beam_main_amp_NOT * np.exp(-((xa - x0_middle_NOT) / sigma) ** 2).reshape(-1, 1) @ Hsrc_main_beam_NOT.reshape(1, -1))
    
# plt.imshow(Hsrc_full_NOT[:])
# plt.plot(Hsrc_full_AND.sum(0))
# Hsrc_full_NOT.shape


src_max = Hsrc_full_NOT.max()

# Compute PML parameters
Nx2 = 2 * Nx
Ny2 = 2 * Ny

# TODO: Vectorize this
# Boundaries
sigx = torch.zeros((Nx2, Ny2))
for nx in range(1, 2 * NPML[0] + 1):
    nx1 = 2 * NPML[0] - nx + 1
    sigx[nx1, :] = (0.5 * eps0 / dt) * (nx / (2 * NPML[0])) ** PMLEXP

for nx in range(1, 2 * NPML[1] + 1):
    nx1 = Nx2 - 2 * NPML[1] + nx - 1
    sigx[nx1, :] = (0.5 * eps0 / dt) * (nx / (2 * NPML[1])) ** PMLEXP

sigy = torch.zeros((Nx2, Ny2))
for ny in range(1, 2 * NPML[2] + 1):
    ny1 = 2 * NPML[2] - ny + 1
    sigy[:, ny1] = (0.5 * eps0 / dt) * (ny / (2 * NPML[2])) ** PMLEXP

for ny in range(1, 2 * NPML[3] + 1):
    ny1 = Ny2 - 2 * NPML[3] + ny - 1
    sigy[:, ny1] = (0.5 * eps0 / dt) * (ny / (2 * NPML[3])) ** PMLEXP

# Bridge Dampener (commented out in original code)
# for ny in range(1, 3 * NCELL[1] + 1):
#     ny1 = Ny2 // 2 + 4 * NCELL[1] - ny
#     sigy[:, ny1] = (0.5 * eps0 / dt) * (ny / (2 * NCELL[1])) ** PMLEXP
# for ny in range(1, 3 * NCELL[1] + 1):
#     ny1 = Ny2 // 2 - 2 * NCELL[1] + ny - 1
#     sigy[:, ny1] = (0.5 * eps0 / dt) * (ny / (2 * NCELL[1])) ** PMLEXP

# For opening intersection with the waveguide (commented out in original code)
# sigy[linspRound((CellNx / 2) * NCELL[0] - 5, sqrt(3) / 2 * NCELL[0] + 3, 2),
#      linspRound((CellNy / 2) * NCELL[1], 3 * bridge_length * NCELL[1], 2)] = 0

#%%
assert USE_TEZ_MODE
# if USE_TEZ_MODE:
# Dx COMPONENT
sigDx = sigx[0:Nx2:2, 1:Ny2:2]
sigDy = sigy[0:Nx2:2, 1:Ny2:2]
mDx0 = 1 / dt + sigDy / (2 * eps0)
mDx1 = 1 / mDx0 * (1 / dt - sigDy / (2 * eps0))
mDx2 = c0 / mDx0
mDx3 = 1 / mDx0 * c0 * dt * sigDx / eps0

# Dy COMPONENT
sigDx = sigx[1:Nx2:2, 0:Ny2:2]
sigDy = sigy[1:Nx2:2, 0:Ny2:2]
mDy0 = 1 / dt + sigDx / (2 * eps0)
mDy1 = 1 / mDy0 * (1 / dt - sigDx / (2 * eps0))
mDy2 = c0 / mDy0
mDy3 = 1 / mDy0 * c0 * dt * sigDy / eps0

# Hz COMPONENT FIELD
sigHx = sigx[0:Nx2:2, 0:Ny2:2]
sigHy = sigy[0:Nx2:2, 0:Ny2:2]
mHz0 = 1 / dt + (sigHx + sigHy) / (2 * eps0) + sigHx * sigHy * dt / (4 * eps0 ** 2)
mHz1 = 1 / mHz0 * (1 / dt - (sigHx + sigHy) / (2 * eps0) - sigHx * sigHy * dt / (4 * eps0 ** 2))
mHz2 = -1 / mHz0 * c0 / URz
mHz4 = -1 / mHz0 * dt / eps0 ** 2 * (sigHx * sigHy)

# Ex and Ey COMPONENTS
mEx1 = 1 / ERx * PECx  # without PML
mEy1 = 1 / ERy * PECy

# INITIALIZE FIELDS
Ex = torch.zeros((Nx, Ny))
Ey = torch.zeros((Nx, Ny))
Dx = torch.zeros((Nx, Ny))
Dy = torch.zeros((Nx, Ny))
Hz = torch.zeros((Nx, Ny))

# INITIALIZE CURL TERMS
CEz = torch.zeros((Nx, Ny))
CHy = torch.zeros((Nx, Ny))
CHx = torch.zeros((Nx, Ny))

# INITIALIZE INTEGRATION TERMS
ICHx = torch.zeros((Nx, Ny))
ICHy = torch.zeros((Nx, Ny))
IHz  = torch.zeros((Nx, Ny))

# INITIALIZE NONLINEAR TERMS
if APPLY_NONLINEARITY:
    mAbsDx1 = CHIx * (mEx1 ** 3) * 2
    mAbsDx2 = CHIx * (mEx1 ** 3) * 3
    mAbsDy1 = CHIy * (mEy1 ** 3) * 2
    mAbsDy2 = CHIy * (mEy1 ** 3) * 3
    AbsD2 = torch.zeros((Nx, Ny))

if APPLY_NONLINEARITY:
    nonlinear_term = torch.zeros((Nx, Ny))

# TODO: 
assert not DO_TRANSFER_CALCULATION

if DO_FULL_RECORDING:
    upper_guide_recording = torch.zeros((len(observation_points_upper), int(np.ceil(STEPS / steps_per_recording))))
    middle_guide_recording = torch.zeros((len(observation_points_middle), int(np.ceil(STEPS / steps_per_recording))))
    

# if JUST_VISUALIZE_STRUCTURE:
ERx[ports['p1']['x'], ports['p1']['y']] = 15
ERy[ports['p2']['x'], ports['p2']['y']] = 15

ERx[Nx_src_lo_AND:Nx_src_hi_AND, Ny_src_lo_AND-1:Ny_src_lo_AND+1] = -10
ERy[Nx_src_lo_NOT:Nx_src_hi_NOT, Ny_src_lo_NOT-1:Ny_src_lo_NOT+1] = -10

if DO_TRANSFER_CALCULATION:
    ERx[rec_line_AND['x'], rec_line_AND['y']] = 0
    ERx[rec_line_NOT['x'], rec_line_NOT['y']] = 0

plt.imshow(np.log(1 + sigx[0:Nx2:2, 1:Ny2:2] + sigy[0:Nx2:2, 1:Ny2:2]) + ERx + ERy + np.log(1e-4 + PECx + PECy) + 
            0.5 * (ERy + ERx + 2 * PECx + 2 * PECy), aspect='auto')
plt.axis('image')
plt.show()

# %%

for T in tqdm(range(1, STEPS + 1)):
    assert USE_TEZ_MODE
    ## COMPUTE CURL OF E
    # Hz Ex Ey CEz
    if USE_DIRICHLET_BC:
        CEz[0:Nx-1, 0:Ny-1] = (Ey[1:Nx, 0:Ny-1] - Ey[0:Nx-1, 0:Ny-1]) / dx - \
                                (Ex[0:Nx-1, 1:Ny] - Ex[0:Nx-1, 0:Ny-1]) / dy
        CEz[0:Nx-1, Ny-1] = (Ey[1:Nx, Ny-1] - Ey[0:Nx-1, Ny-1]) / dx - \
                            (0 - Ex[0:Nx-1, Ny-1]) / dy
        CEz[Nx-1, 0:Ny-1] = (0 - Ey[Nx-1, 0:Ny-1]) / dx - \
                            (Ex[Nx-1, 1:Ny] - Ex[Nx-1, 0:Ny-1]) / dy
        CEz[Nx-1, Ny-1] = (0 - Ey[Nx-1, Ny-1]) / dx - \
                            (0 - Ex[Nx-1, Ny-1]) / dy

    if USE_PERIODIC_BC:
        CEz[0:Nx-1, 0:Ny-1] = (Ey[1:Nx, 0:Ny-1] - Ey[0:Nx-1, 0:Ny-1]) / dx - \
                                (Ex[0:Nx-1, 1:Ny] - Ex[0:Nx-1, 0:Ny-1]) / dy
        CEz[0:Nx-1, Ny-1] = (Ey[1:Nx, Ny-1] - Ey[0:Nx-1, Ny-1]) / dx - \
                            (Ex[0:Nx-1, 0] - Ex[0:Nx-1, Ny-1]) / dy
        CEz[Nx-1, 0:Ny-1] = (Ey[0, 0:Ny-1] - Ey[Nx-1, 0:Ny-1]) / dx - \
                            (Ex[Nx-1, 1:Ny] - Ex[Nx-1, 0:Ny-1]) / dy
        CEz[Nx-1, Ny-1] = (Ey[0, Ny-1] - Ey[Nx-1, Ny-1]) / dx - \
                            (Ex[Nx-1, 0] - Ex[Nx-1, Ny-1]) / dy

    ## INJECT TF/SF SOURCE INTO E

    if USE_TFSF_SOURCE:
        CEz[Nx_src_lo_AND:Nx_src_hi_AND, Ny_src_lo_AND-1] -= Esrc_full_AND[:, T-1] / dy
        CEz[Nx_src_lo_NOT:Nx_src_hi_NOT, Ny_src_lo_NOT-1] -= Esrc_full_NOT[:, T-1] / dy

    ## UPDATE H INTEGRATIONS

    IHz += Hz

    ## UPDATE H FIELD

    Hz = mHz1 * Hz + mHz2 * CEz + mHz4 * IHz

    ## HARD SOURCE *OPTIONAL*

    if USE_HARD_SOURCE:
        Hz[Nx_src_lo:Nx_src_hi, Ny_src_lo] = Hsrc_full[:, T-1]

    ## COMPUTE CURL OF H

    # Hz Ex Ey CHx CHy
    if USE_DIRICHLET_BC:
        CHx[0:Nx, 0] = (Hz[0:Nx, 0] - 0) / dy
        CHx[0:Nx, 1:Ny] = (Hz[0:Nx, 1:Ny] - Hz[0:Nx, 0:Ny-1]) / dy
        CHy[0, 0:Ny] = - (Hz[0, 0:Ny] - 0) / dx
        CHy[1:Nx, 0:Ny] = (Hz[0:Nx-1, 0:Ny] - Hz[1:Nx, 0:Ny]) / dx

    if USE_PERIODIC_BC:
        CHx[0:Nx, 0] = (Hz[0:Nx, 0] - Hz[0:Nx, Ny-1]) / dy
        CHx[0:Nx, 1:Ny] = (Hz[0:Nx, 1:Ny] - Hz[0:Nx, 0:Ny-1]) / dy
        CHy[0, 0:Ny] = - (Hz[0, 0:Ny] - Hz[Nx-1, 0:Ny]) / dx
        CHy[1:Nx, 0:Ny] = - (Hz[1:Nx, 0:Ny] - Hz[0:Nx-1, 0:Ny]) / dx

    ## INJECT TF/SF SOURCE INTO H

    if USE_TFSF_SOURCE:
        CHx[Nx_src_lo_AND:Nx_src_hi_AND, Ny_src_lo_AND] += Hsrc_full_AND[:, T-1] / dy
        CHx[Nx_src_lo_NOT:Nx_src_hi_NOT, Ny_src_lo_NOT] += Hsrc_full_NOT[:, T-1] / dy

    ## UPDATE D INTEGRATIONS

    ICHx += CHx
    ICHy += CHy

    ## UPDATE D FIELD

    Dx = mDx1 * Dx + mDx2 * CHx + mDx3 * ICHx
    Dy = mDy1 * Dy + mDy2 * CHy + mDy3 * ICHy

    ## UPDATE E FIELD

    if APPLY_NONLINEARITY:
        AbsD2 = (Dx ** 2 + Dy ** 2)
        Ex = mEx1 * Dx * (1 + mAbsDx1 * AbsD2) / (1 + mAbsDx2 * AbsD2)
        Ey = mEy1 * Dy * (1 + mAbsDy1 * AbsD2) / (1 + mAbsDy2 * AbsD2)
    else:
        Ex = mEx1 * Dx
        Ey = mEy1 * Dy

    ## CALCULATE LEARNING PARAMETERS

    if DO_TRANSFER_CALCULATION:
        # AND PART
        curr_trn_hz_AND[1::2, T-1] = Hz[rec_line_AND['x'], rec_line_AND['y'] - NCELL[1] // 2]
        curr_trn_hz_AND[0::2, T-1] = Hz[rec_line_AND['x'], rec_line_AND['y']]

        # NOT PART
        curr_trn_hz_NOT[1::2, T-1] = Hz[rec_line_NOT['x'], rec_line_AND['y'] - NCELL[1] // 2]
        curr_trn_hz_NOT[0::2, T-1] = Hz[rec_line_NOT['x'], rec_line_NOT['y']]

    if DO_FULL_RECORDING:
        if T % steps_per_recording == 0:
            upper_guide_recording[:, T // steps_per_recording - 1] = \
                Hz[(BUFFS[1] + (CELLS[0] / 2 - 2) * NCELL[1] * np.sqrt(3) / 2).int(), 
                   observation_points_upper]
            middle_guide_recording[:, T // steps_per_recording - 1] = \
                Hz[Nx // 2, observation_points_middle]

    ## VISUALISE THE FIELD

    if T % FPS == 0:
        if DISPLAY_TRANSFER_WINDOW:
            # plt.subplot(2, 2, [1, 2])
            pass
        if USE_TEZ_MODE:
            if USE_ADVANCED_VISUALISATION:
                # draw2d(ya, xa, ERx.T, Hz.T, NPML)
                plt.imshow(ERx)
                assert False
            else:
                plt.imshow(Hz)
            if DISPLAY_LABELS:
                plt.title(f'Time step : {T}, Hz Mode, ω = {NORMALIZED_FREQ}, A_upper = {beam_upper_amp_AND}, A_lower = {beam_lower_amp_AND}')
        plt.axis('image')
        if DISPLAY_LABELS:
            plt.gca()
            c = plt.colorbar()
            c.set_label('Hz Field Amplitude')
            plt.xlabel('$\\mathbf{X}$', fontsize=14)
            plt.ylabel('$\\mathbf{Y}$', fontsize=14)
        plt.gca().get_yaxis().set_tick_params(direction='in', labelleft=False)
        plt.gca().get_xaxis().set_tick_params(direction='in', labelbottom=False)
        if DISPLAY_TRANSFER_WINDOW:
            plt.subplot(2, 2, [3, 4])
            # plt.plot(,'b-','LineWidth', 2)
            # plt.hold(True)
            plt.plot(upper_guide_recording[863, :], 'r-', linewidth=2)
            plt.hold(False)
            plt.title('Output amplitude')
            plt.ylim([-1, 1] * plotting_amp)
        if RECORD_ANIMATION:
            F = plt.gcf().canvas.draw()
            # writeVideo(vidObj, F)  # Commented out as it requires video writer setup
        plt.draw()
        plt.pause(0.001)

## AFTER LOOP STEPS

if RECORD_ANIMATION:
    # close(vidObj)  # Commented out as it requires video writer setup
    pass

if DO_FULL_RECORDING:
    np.savez(f'full_rec_{ncase}_{file_name}', upper_guide_recording=upper_guide_recording, middle_guide_recording=middle_guide_recording)
elif RECORD_FIELDS:
    np.savez(file_name, curr_trn_hz_AND=curr_trn_hz_AND, curr_trn_hz_NOT=curr_trn_hz_NOT, ERx=ERx, dt=dt, dx=dx, dy=dy,
             Hsrc_upper_beam_AND=Hsrc_upper_beam_AND, Hsrc_lower_beam_AND=Hsrc_lower_beam_AND, Hsrc_main_beam_AND=Hsrc_main_beam_AND,
             Hsrc_upper_beam_NOT=Hsrc_upper_beam_NOT, Hsrc_lower_beam_NOT=Hsrc_lower_beam_NOT, Hsrc_main_beam_NOT=Hsrc_main_beam_NOT,
             Hsrc_full_AND=Hsrc_full_AND, Hsrc_full_NOT=Hsrc_full_NOT, CHIx=CHIx, append=True)
# %%
