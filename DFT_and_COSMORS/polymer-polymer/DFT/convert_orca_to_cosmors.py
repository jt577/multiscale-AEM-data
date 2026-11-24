import os
import re
import numpy as np

out_files = ['1a.out', '1b.out', '1c.out', '1d.out', '1e.out', '1f.out', '1g.out', '2a.out', '2b.out', '2c.out', '2d.out', '2e.out', '3a.out', '3b.out', '4a.out', '4b.out', '4c.out', '4d.out']
xyz_files = ['1a.xyz', '1b.xyz', '1c.xyz', '1d.xyz', '1e.xyz', '2a.xyz', '2b.xyz', '2c.xyz', '2d.xyz', '2e.xyz', '3a.xyz', '3b.xyz', '4a.xyz', '4b.xyz', '4c.xyz', '4d.xyz']
cpcm_files = ['1a.cpcm', '1b.cpcm', '1c.cpcm', '1d.cpcm', '1e.cpcm', '2a.cpcm', '2b.cpcm', '2c.cpcm', '2d.cpcm', '2e.cpcm', '3a.cpcm', '3b.cpcm', '4a.cpcm', '4b.cpcm', '4c.cpcm', '4d.cpcm', '4e.out']
cosmors_files = ['1a.cosmors', '1b.cosmors', '1c.cosmors', '1d.cosmors', '1e.cosmors', '1f.cosmors', '1g.cosmors', '2a.cosmors', '2b.cosmors', '2c.cosmors', '2d.cosmors', '2e.cosmors', '3a.cosmors', '3b.cosmors', '4a.cosmors', '4b.cosmors', '4c.cosmors', '4d.cosmors', '4e.cosmors']

def get_number_of_atoms(filepath):
    number_of_atoms = None
    with open(filepath, "r") as file:
        content = file.read()
        # Regular expression to match lines with atomic coordinates
        pattern = re.compile(r"^(\s*\w+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+)", re.MULTILINE)
        matches = pattern.findall(content)
        number_of_atoms = len(matches)
    return number_of_atoms

def get_number_of_surface_points(filepath):
    number_of_surface_points = None
    with open(filepath, "r") as file:
        content = file.read()
        # Regular expression to match the line with number of surface points
        pattern = re.compile(r"GEPOL surface points\s*\.\.\.\s*(\d+)", re.MULTILINE)
        match = pattern.search(content)
        if match:
            number_of_surface_points = int(match.group(1))
    return number_of_surface_points

def find_volume(filepath):
    volume = None
    with open(filepath, 'r') as file:
        for line in file:
            if "GEPOL Volume" in line:
                # Split the line into words and get the last word which should be the volume
                volume = float(line.split()[-1])
    return volume

def find_area(filepath):
    area = None
    with open(filepath, 'r') as file:
        for line in file:
            if "GEPOL Surface-area" in line:
                # Split the line into words and get the last word which should be the volume
                area = float(line.split()[-1])
    return area

def find_cpcm_dielectric_energy(filepath):
    cpcm_dielectric_energy = None
    with open(filepath, 'r') as file:
        for line in file:
            if "CPCM Dielectric" in line:
                # should be the energy in Eh
                cpcm_dielectric_energy = float(line.split()[-4])
    return cpcm_dielectric_energy

def find_energy(filepath):
    energy = None
    with open(filepath, 'r') as file:
        for line in file:
            if "FINAL SINGLE POINT ENERGY" in line:
                energy = float(line.split()[-1])
    return energy

def append_file_to_output(output_file_name, sub_file_name):
    with open(sub_file_name, 'r') as sub_file:  # 'a' mode for appending
        for line in sub_file:
            output_file.write(line)

def extract_coordinates_with_radii(filename):
    cpcm_radii = {
        'H': 1.300, # H
        'B': 2.048, # B
        'C': 2.000, # C
        'N': 1.830, # N
        'O': 1.720, # O
        'F': 1.720, # F
        'Si': 2.480, # Si
        'P': 2.130, # P
        'S': 2.160, # S
        'Cl': 2.050, # Cl
        'Br': 2.160, # Br
        'I': 2.320, # I
    }

    coordinates_with_radii = []
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Find the last Cartesian coordinates block
    start_index = None
    for i in range(len(lines)):
        if "CARTESIAN COORDINATES (A.U.)" in lines[i]:
            start_index = i

    if start_index is not None:
        for j in range(start_index+3, len(lines)):  # Start at start_index+3 to skip the header lines
            line = lines[j]
            if line.strip() == '':
                break

            # Extract the atomic symbol and coordinates
            parts = line.split()
            atom_symbol = parts[1]
            x, y, z = map(float, parts[5:8])
            radius = cpcm_radii.get(atom_symbol, "?")

            # Append to the list
            coordinates_with_radii.append((x, y, z, radius))

    return coordinates_with_radii


for outfilestring, filez, filexyz, filecpcm in zip(cosmors_files, out_files, xyz_files, cpcm_files):
    with open(outfilestring, 'w') as output_file:

        output_file.write(f'TestName : DFT_COSMO_BP86_def2-TZVP+def2-TZVPD_SP\n')

        output_file.write('\n'+'#'*50+'\n')
        output_file.write('#ENERGY\n')
        energy = find_energy(filez)
        output_file.write(f"FINAL SINGLE POINT ENERGY       {energy}\n")
        output_file.write('\n'+'#'*50+'\n')
        #output_file.write('#DIPOLE MOMENT (Debye)\n')
        #output_file.write(f'{dipole_moment}\n')
        output_file.write('#XYZ_FILE\n')
        append_file_to_output(output_file, filexyz)
        output_file.write('\n'+'#'*50+'\n')
        output_file.write("#COSMO\n")
        append_file_to_output(output_file, filecpcm)
        output_file.write("\n")




