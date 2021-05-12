
from pyrosetta import *
from rosetta import *
import numpy as onp
import glob
from rosetta.core.scoring import *
from rosetta.protocols.docking import *

init()
sf = ScoreFunction()
sf = get_fa_scorefxn()

def PairwiseEnergyMapForPdbFile(curpose, resi, fa_energy):

  sf_pose = sf(curpose)
  dict_ = {}
  columns=['residue_i','residue_j','score_type','score_value']
  for key in columns:
    dict_[key] = []
  
  score_types = sf.get_nonzero_weighted_scoretypes()
  mylist = [fa_energy]
  
  for st in score_types:
    strst = str(st)
    strst_clean = strst.split('.')[1]
    if strst_clean in mylist:
      score_tuple = pyrosetta.toolbox.atom_pair_energy._reisude_pair_energies(resi,curpose,sf,st,0.25)
      for entry in score_tuple:
        dict_['score_type'].append(strst_clean)
        dict_['score_value'].append(entry[1])
        dict_['residue_j'].append(entry[0])
        dict_['residue_i'].append(resi)
        
  return dict_

def get_pairwise_energy(pose, energy):
  fa_energy = onp.zeros((6,6))

  for res in range(1, 7):
    res_dict = PairwiseEnergyMapForPdbFile(pose, res, energy)
    residue_j = res_dict['residue_j']
    score_value = res_dict['score_value']
    for j, energy_j in zip(residue_j, score_value):
      if (energy_j < 2.5):
        fa_energy[res - 1][j - 1] = energy_j

  return fa_energy

def get_feature_matrix(pdb_files):
  weights = [1, .55, .9375, .875]
  fa_atr, fa_rep, fa_sol, fa_elec, enr = [], [], [], [], []
  for pdb_file in pdb_files:
    pose = pose_from_pdb(pdb_file)
    pose_fa_atr = get_pairwise_energy(pose, 'fa_atr')
    pose_fa_rep = get_pairwise_energy(pose, 'fa_rep')
    pose_fa_sol = get_pairwise_energy(pose, 'fa_sol')
    pose_fa_elec = get_pairwise_energy(pose, 'fa_elec')
    pose_enr = weights[0] * pose_fa_atr + weights[1] * pose_fa_rep + weights[2] * pose_fa_sol + weights[3] * pose_fa_elec

    fa_atr.append(pose_fa_atr)
    fa_rep.append(pose_fa_rep)
    fa_sol.append(pose_fa_sol)
    fa_elec.append(pose_fa_elec)
    enr.append(pose_enr)

  return fa_atr, fa_rep, fa_sol, fa_elec, enr

pdb_files= glob.glob('PDBs/*.pdb')
peptides = [i.split('/')[-1].split('_')[0] for i in pdb_files]
fa_atr, fa_rep, fa_sol, fa_elec, enr = get_feature_matrix(pdb_files)

onp.save('Feature_fa_atr.npy', fa_atr)
onp.save('Feature_fa_rep.npy', fa_rep)
onp.save('Feature_fa_sol.npy', fa_sol)
onp.save('Feature_fa_elec.npy', fa_elec)
onp.save('Feature_enr.npy', enr)
onp.save('Index.npy', peptides)