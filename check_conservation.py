from netCDF4 import Dataset
import numpy as np

files = [
"netcdf_2026_04_30/mpi_conservation_1_node_set.nc",
"netcdf_2026_04_30/mpi_conservation_4_node_set.nc"
]

datasets: list[Dataset] = []

for file in files:
    datasets.append(Dataset(file, mode='r'))

etas = []
hus = []
hvs = []
for nc in datasets:
    simulation = nc.groups['forecasts'].groups['CDKLM16']
    etas.append(simulation.variables['eta'])

    moments = simulation.groups['moments']
    hus.append(moments.groups['u'].variables['hu'])
    hvs.append(moments.groups['v'].variables['hv'])

tests = {'eta': etas, 'hu': hus, 'hv': hvs}

print("Checking if initial conditions are the same:")
for key, value in tests.items():
    values = []
    for array in value:
        values.append(array[0, :, :])

    exact = np.array_equal(*tuple(values))
    similar = np.allclose(*tuple(values))
    print(f"    {key}: {exact}/{similar}")


print("Checking 1 are the same")
for key, value in tests.items():
    values = []
    for array in value:
        values.append(array[1, :, :])

    exact = np.array_equal(*tuple(values))
    similar = np.allclose(*tuple(values), atol=1e-02)
    print(f"    {key}: {exact}/{similar}")


print("Checking if the final results are the same")
for key, value in tests.items():
    values = []
    for array in value:
        values.append(array[-1, :, :])

    exact = np.array_equal(*tuple(values))
    similar = np.allclose(*tuple(values), atol=1e-04)
    print(f"    {key}: {exact}/{similar}")
