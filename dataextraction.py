from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import os
import csv
from Bio.PDB import PDBParser
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.Polypeptide import PPBuilder
from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import random

# open the dataset

datasetname = 'demo'
disscoeff_file = 'INDEX_demo_PL_data.2021'

folders = os.listdir(datasetname)

# create a csv file name
ligand_file_path = 'data_ligand_less.csv'
protein_file_path = 'data_protein_less.csv'

AMINO_ACIDS = ['GLY', 
                'ALA', 
                'VAL', 
                'ILE', 
                'LEU', 
                'MET', 
                'PHE', 
                'CYS', 
                'ASP', 
                'GLU',
                'HIS',
                'LYS',
                'ASN',
                'PRO',
                'GLN',
                'ARG',
                'SER',
                'THR',
                'TRP',
                'TYR',
                ]

def getFingerprints(protein_names): 
    fingerprints = []
    names = []   
    for ix in protein_names:
        if (ix == "readme" or ix == "index"):
            continue
        path = os.path.join(datasetname, ix)
        path_ligand = os.path.join(path, ix + "_ligand.mol2")
        mol = Chem.MolFromMol2File(path_ligand)
        if mol:
            fpgen = AllChem.GetMorganGenerator(radius = 2)
            fp = fpgen.GetFingerprint(mol)
            fingerprints.append(fp)
            names.append(ix)
        else:
            continue
    fingerprints = np.vstack(fingerprints)
    pca = PCA(n_components = 5)
    pca.fit(fingerprints)
    fp_transform = pca.transform(fingerprints)
    for i, comp in enumerate(pca.explained_variance_ratio_):
        print("The variance explained by principal component {} is {:>5.2f}%".format(i + 1, comp*100))
    print("Total variance explained by this subset is {:>5.2f}%".format(100*np.sum(pca.explained_variance_ratio_)))
    end = dict(zip(names, fp_transform))
    return end

def getLabels(filepath):
    # store all of the protein names and the dissociation coefficients
    protein_names = []
    diss_coeffs = []

    # store prefix values for milli, micro, nano, and pico
    prefixes = {'m': pow(10, -3), 'u': pow(10, -6), 'n': pow(10, -9), 'p' : pow(10, -12)}

    # open the data file
    f = open(filepath, "r")
    for line in f:
        # for each line, store the protein name and store the dissociation coefficient
        split_line = line.split("  ")
        name = split_line[0]
        protein_names.append(name)
        # coeff = float(split_line[3])

        coeff_text = split_line[3]
        coeff = format(float(coeff_text[3:-2]) * prefixes[coeff_text[-2]], '.3e')
        diss_coeffs.append(coeff)
        

    # join the dissociation coefficients and the protein names to form a dict
    binding_affinities = dict(zip(protein_names, diss_coeffs))
    return binding_affinities

def getLigandProperties(filepath, ix):
    # create the ligand mol
    mol = Chem.MolFromMol2File(filepath)
    
    # collect physicochemical properties of the molecule
    data = []
    if mol:
        data.append(Descriptors.MolWt(mol))
        data.append(Descriptors.HeavyAtomMolWt(mol))
        data.append(Descriptors.MolLogP(mol))
        data.append(Descriptors.TPSA(mol))
        data.append(Descriptors.NumRotatableBonds(mol))
        data.append(Descriptors.NumHDonors(mol))
        data.append(Descriptors.NumHAcceptors(mol))
        data.append(Descriptors.MaxPartialCharge(mol))
        data.append(Descriptors.MinPartialCharge(mol))
        data.append(AllChem.ComputeMolVolume(mol))
        # data.extend(fps[ix])
        # data.append(AllChem.ComputeMolShape(mol))
    
    return data


def getPocketProperties(ix, filepath):
    parser = PDBParser()
    
    # parse through the protein pocket pdb
    pocket = parser.get_structure(ix, filepath)
    
    # create a dict to find frequency of each amino acid
    amino_freq = dict(zip(AMINO_ACIDS, [0] * len(AMINO_ACIDS)))
    
    # store the number of water molecules
    numOfWater = 0
    
    numOfChains = 0
    numOfCations = 0
    data = []
    # iterate through each residue and count number of each residue and number of waters
    for model in pocket:
        for chain in model:
            numOfChains += 1
            for residue in chain:
                if residue.resname in AMINO_ACIDS:
                    amino_freq[residue.resname] += 1
                elif residue.resname == "HOH":
                    numOfWater += 1
                else:
                    numOfCations += 1
    for val in amino_freq.values():
        data.append(val)
    data.append(numOfWater)
    data.append(numOfChains)
    data.append(numOfCations)
    
    full_sequence = ""
    ppb = PPBuilder()
    for pp in ppb.build_peptides(pocket):
        full_sequence += str(pp.get_sequence())
        
    analysis = ProteinAnalysis(full_sequence)
    data.append(analysis.isoelectric_point())
    ext_coeff = analysis.molar_extinction_coefficient()
    data.append(ext_coeff[0])
    data.append(ext_coeff[1])
    data.append(analysis.gravy()) 
    data.append(analysis.aromaticity())
    data.append(analysis.instability_index())
    
    atoms = list(pocket.get_atoms())
    pocket_coords = [(atom.coord).tolist() for atom in atoms]
    hull = ConvexHull(np.array(pocket_coords))
    data.append(hull.area)
    data.append(hull.volume)
    # pocket_string = ""
    # pocket_string += str(pocket_coords[0][0]) + ','+ str(pocket_coords[0][1])+','+str(pocket_coords[0][2])
    # for i in range(1, len(pocket_coords)):
    #     pocket_string += '|'
    #     pocket_string += str(pocket_coords[i][0]) + ','+str(pocket_coords[i][1])+','+str(pocket_coords[i][2])
    # data.append(pocket_string)
    return data

# open a new csv file and begin writing into it
ligand_file = open(ligand_file_path, mode='w', newline='')
ligand_writer = csv.writer(ligand_file)

protein_file = open(protein_file_path, mode='w', newline='')
protein_writer = csv.writer(protein_file)

# make the column title names and add to the csv file
ligand_columns = ["MolWt", 
            "HeavyAtomMolWt", 
            "LogP", 
            "TPSA", 
            "NumRotatableBonds", 
            "NumHDonors", 
            "NumHAcceptors", 
            "MaxPartialCharge", 
            "MinPartialCharge", 
            "MolVolume",
            # "Fp1",
            # "Fp2",
            # "Fp3",
            # "Fp4",
            # "Fp5",
            # "3DShape",
            'DissociationCoeff'
            ]
protein_columns = AMINO_ACIDS.copy()
protein_columns.extend(['NumOfH2O',
                        'NumOfChains',
                        'NumOfCations',
                        'IsoelectricPt',
                        'RedExtinctionCoeff',
                        'OxExtinctionCoeff',
                        'AvgHydrophobicity',
                        'Aromaticity',
                        'InstabilityIdx',
                        'ConvexHullSurfArea',
                        'ConvexHullVol',
                        # 'PocketCoords',
                        'DissociationCoeff'])
print(ligand_columns)
ligand_writer.writerow(ligand_columns)
protein_writer.writerow(protein_columns)

# first gather all of the binding affinities

# open the index file with all of the binding data
path = os.path.join(datasetname, "index")
path_affinities = os.path.join(path, disscoeff_file)

binding_affinities = getLabels(path_affinities)    

fps = getFingerprints(folders)

random.seed(0)
random.shuffle(folders)

count = 0
# iterate through each folder
for ix in folders:
    # if readme then skip it
    if (ix == "readme" or ix == "index"):
        continue
    
    if (count == 3000):
        break
    count += 1
    
    # open the folder and make a file path for the ligand mol2 file
    path = os.path.join(datasetname, ix)
    path_ligand = os.path.join(path, ix + "_ligand.mol2")
    
    ligand_data = getLigandProperties(path_ligand, ix)
    if (len(ligand_data) <= 0):
        continue
    ligand_data.append(binding_affinities[ix])
    ligand_writer.writerow(ligand_data)
    
    # create a file path to the protein pocket pdb
    path_pocket = os.path.join(path, ix + "_pocket.pdb")
    
    protein_data = getPocketProperties(ix, path_pocket)    
    protein_data.append(binding_affinities[ix])
    protein_writer.writerow(protein_data)
    
    
ligand_file.close()
protein_file.close()
            
print(f'\nCSV file {ligand_file_path} has been created successfully.')
print(f'CSV file {protein_file_path} has been created successfully.')




# don't have to standardize the data, do a comparison between the data