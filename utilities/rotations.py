# ------------------------------------------------
# Copyright 2021 Kanupriya Pande
# Contact kpande@lbl.gov
# ------------------------------------------------

import numpy as np


def rot_z(angle):
    rot = np.array([(np.cos(angle), -np.sin(angle), 0.),
                    (np.sin(angle), np.cos(angle), 0.),
                    (0., 0., 1.)])
    return rot


def der_rot_z(angle):
    d_rot = np.array([(-np.sin(angle), -np.cos(angle), 0.),
                      (np.cos(angle), -np.sin(angle), 0.),
                      (0., 0., 0.)])
    return d_rot


def rot_x(angle):
    rot = np.array([(1., 0., 0.),
                    (0., np.cos(angle), -np.sin(angle)),
                    (0., np.sin(angle), np.cos(angle))])
    return rot


def der_rot_x(angle):
    d_rot = np.array([(0.0, 0.0, 0.0),
                      (0.0, -np.sin(angle), -np.cos(angle)),
                      (0.0, np.cos(angle), -np.sin(angle))])
    return d_rot


def rot_y(angle):
    rot = np.array([(np.cos(angle), 0., np.sin(angle)),
                    (0., 1., 0.),
                    (-np.sin(angle), 0., np.cos(angle))])
    return rot


def der_rot_y(angle):
    d_rot = np.array([(-np.sin(angle), 0.0, np.cos(angle)),
                      (0.0, 0.0, 0.0),
                      (-np.cos(angle), 0.0, -np.sin(angle))])
    return d_rot
