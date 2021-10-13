import argparse
from argparse import RawTextHelpFormatter
from matplotlib import pyplot as plt
import numpy as np
import os

from prospr.prediction import predict
from prospr.dataloader import get_label


def make_plots(pred, label, base_save_path):
    """Create plots comparing distogram, secondary structure, phi/psi, and accessible surface area predictions to labels."""
    dist_pred = pred['dist']
    dist_pred = np.argmax(dist_pred, axis=0)
    dist_label = label['bin_mat']
    joint_dist = dist_label
    triu_indices = np.triu_indices(n=dist_label.shape[0], k=1)
    joint_dist[triu_indices[0], triu_indices[1]] = dist_pred[triu_indices[0], triu_indices[1]]
    plt.figure()
    plt.imshow(joint_dist, cmap='viridis_r')
    plt.xticks([])
    plt.yticks([])
    # plt.title("Distance Prediction - prediction above diagonal, label below diagonal")
    plt.savefig(os.path.join(base_save_path, 'dist_pred_label.pdf'), bbox_inches='tight', dpi=300)
    pred['ss'] = pred['ss'][..., :len(label['ss'])]
    pred['phi'] = pred['phi'][..., :len(label['ss'])]
    pred['psi'] = pred['psi'][..., :len(label['ss'])]
    pred['asa'] = pred['asa'][..., :len(label['ss'])]
    ss_pred = pred['ss']
    ss_pred = np.argmax(ss_pred, axis=0)
    ss_pred = np.tile(ss_pred, (15, 1))
    ss_label = label['ss']
    ss_label = np.tile(ss_label, (15, 1))
    joint_ss = np.concatenate((ss_pred, ss_label), axis=0)
    plt.figure()
    plt.imshow(joint_ss)
    # plt.title("Secondary Structure")
    # plt.xlabel("Residue")
    # plt.ylabel("Prediction\n\n\nTarget", rotation=0, y=.2, x=0, ha="right")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(base_save_path, 'ss_pred_label.pdf'), bbox_inches='tight', dpi=300)

    phi_pred = pred['phi']
    phi_pred = np.argmax(phi_pred, axis=0)
    phi_pred = np.tile(phi_pred, (15, 1))
    phi_label = label['phi']
    phi_label = np.tile(phi_label, (15, 1))
    joint_phi = np.concatenate((phi_pred, phi_label), axis=0)
    plt.figure()
    plt.imshow(joint_phi)
    # plt.title("Phi Angle")
    # plt.xlabel("Residue")
    # plt.ylabel("Prediction\n\n\nTarget", rotation=0, y=.2, x=0, ha="right")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(base_save_path, 'phi_pred_label.pdf'), bbox_inches='tight', dpi=300)

    psi_pred = pred['psi']
    psi_pred = np.argmax(psi_pred, axis=0)
    psi_pred = np.tile(psi_pred, (15, 1))
    psi_label = label['psi']
    psi_label = np.tile(psi_label, (15, 1))
    joint_psi = np.concatenate((psi_pred, psi_label), axis=0)
    plt.figure()
    plt.imshow(joint_psi)
    # plt.title("Psi Angle")
    # plt.xlabel("Residue")
    # plt.ylabel("Prediction\n\n\nTarget", rotation=0, y=.2, x=0, ha="right")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(base_save_path, 'psi_pred_label.pdf'), bbox_inches='tight', dpi=300)

    asa_pred = pred['asa']
    asa_pred = np.argmax(asa_pred, axis=0)
    asa_pred = np.tile(asa_pred, (15, 1))
    asa_label = label['asa']
    asa_label = np.tile(asa_label, (15, 1))
    joint_asa = np.concatenate((asa_pred, asa_label), axis=0)
    plt.figure()
    plt.imshow(joint_asa)
    # plt.title("Accessible Surface Area")
    # plt.xlabel("Residue")
    # plt.ylabel("Prediction\n\n\nTarget", rotation=0, y=.2, x=0, ha="right")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(base_save_path, 'asa_pred_label.pdf'), bbox_inches='tight', dpi=300)


def evaluate(args):
    """Make plots comparing predictions to true labels"""
    name = os.path.basename(args.a3m.split('.a3m')[0])
    base_save_path = os.path.join(args.output_dir, name)
    os.makedirs(base_save_path, exist_ok=True)

    args.save = False
    pred = predict(args)
    label = get_label(args.pdb)
    make_plots(pred, label, base_save_path=base_save_path)