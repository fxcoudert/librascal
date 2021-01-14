from rascal.representations import (
    SortedCoulombMatrix,
    SphericalExpansion,
    SphericalInvariants,
)
from rascal.utils import from_dict, to_dict
from test_utils import load_json_frame, BoxList, Box, dot
import unittest
import numpy as np
import sys
import os
import json
from copy import copy, deepcopy
from scipy.stats import ortho_group

rascal_reference_path = "reference_data"
inputs_path = os.path.join(rascal_reference_path, "inputs")
dump_path = os.path.join(rascal_reference_path, "tests_only")


class TestSortedCoulombRepresentation(unittest.TestCase):
    def setUp(self):
        """
        builds the test case. Test the order=1 structure manager implementation
        against a triclinic crystal.
        """

        fn = os.path.join(inputs_path, "CaCrP2O7_mvc-11955_symmetrized.json")
        self.frame = load_json_frame(fn)

        self.hypers = dict(
            cutoff=3.0,
            sorting_algorithm="row_norm",
            size=50,
            central_decay=0.5,
            interaction_cutoff=3,
            interaction_decay=-1,
        )

    def test_representation_transform(self):

        rep = SortedCoulombMatrix(**self.hypers)

        features = rep.transform([self.frame])

        test = features.get_features(rep)

    def test_serialization(self):
        rep = SortedCoulombMatrix(**self.hypers)

        rep_dict = to_dict(rep)

        rep_copy = from_dict(rep_dict)

        rep_copy_dict = to_dict(rep_copy)

        self.assertTrue(rep_dict == rep_copy_dict)


class TestSphericalExpansionRepresentation(unittest.TestCase):
    def setUp(self):
        """
        builds the test case. Test the order=1 structure manager implementation
        against a triclinic crystal.
        """
        self.seed = 18012021

        fns = [
            os.path.join(inputs_path, "CaCrP2O7_mvc-11955_symmetrized.json"),
            os.path.join(inputs_path, "SiC_moissanite_supercell.json"),
            os.path.join(inputs_path, "methane.json"),
        ]
        self.species = [1, 6, 8, 14, 15, 20, 24]
        self.frames = [load_json_frame(fn) for fn in fns]

        self.max_radial = 6
        self.max_angular = 4

        self.hypers = {
            "interaction_cutoff": 6.0,
            "cutoff_smooth_width": 1.0,
            "max_radial": self.max_radial,
            "max_angular": self.max_angular,
            "gaussian_sigma_type": "Constant",
            "gaussian_sigma_constant": 0.5,
            "expansion_by_species_method": "user defined",
            "global_species": self.species,
        }

    def test_representation_transform(self):
        rep = SphericalExpansion(**self.hypers)

        features = rep.transform(self.frames)

        ref = features.get_features(rep)

    def test_serialization(self):
        rep = SphericalExpansion(**self.hypers)

        rep_dict = to_dict(rep)

        rep_copy = from_dict(rep_dict)

        rep_copy_dict = to_dict(rep_copy)

        self.assertTrue(rep_dict == rep_copy_dict)

    def test_radial_dimension_reduction_test(self):
        rep = SphericalExpansion(**self.hypers)
        features_ref = rep.transform(self.frames).get_features(rep)
        K_ref = features_ref.dot(features_ref.T)

        # identity test
        hypers = deepcopy(self.hypers)
        projection_matrices = [
            np.eye(self.max_radial).tolist() for _ in range(self.max_angular + 1)
        ]
        hypers["optimization"] = {
            "RadialDimReduction": {
                "projection_matrices": {sp: projection_matrices for sp in self.species}
            }
        }
        rep = SphericalExpansion(**hypers)
        features_test = rep.transform(self.frames).get_features(rep)
        self.assertTrue(np.allclose(features_ref, features_test))

        # random orthogonal matrix test, orthogonal matrix should be recoverable by linear regression
        hypers = deepcopy(self.hypers)
        np.random.seed(self.seed)
        projection_matrices = [
            ortho_group.rvs(self.max_radial).tolist()
            for _ in range(self.max_angular + 1)
        ]
        hypers["optimization"] = {
            "RadialDimReduction": {
                "projection_matrices": {sp: projection_matrices for sp in self.species}
            }
        }
        rep = SphericalExpansion(**hypers)
        features_test = rep.transform(self.frames).get_features(rep)
        K_test = features_test.dot(features_test.T)
        self.assertTrue(np.allclose(K_ref, K_test))


class TestSphericalInvariantsRepresentation(unittest.TestCase):
    def setUp(self):
        """
        builds the test case. Test the order=1 structure manager implementation
        against a triclinic crystal.
        """

        fns = [
            os.path.join(inputs_path, "CaCrP2O7_mvc-11955_symmetrized.json"),
            os.path.join(inputs_path, "SiC_moissanite_supercell.json"),
            os.path.join(inputs_path, "methane.json"),
        ]
        self.frames = [load_json_frame(fn) for fn in fns]

        global_species = []
        for frame in self.frames:
            global_species.extend(frame["atom_types"])
        self.global_species = list(np.unique(global_species))

        self.hypers = dict(
            soap_type="PowerSpectrum",
            interaction_cutoff=3.5,
            max_radial=6,
            max_angular=6,
            gaussian_sigma_constant=0.4,
            gaussian_sigma_type="Constant",
            cutoff_smooth_width=0.5,
        )

    def test_representation_transform(self):

        rep = SphericalInvariants(**self.hypers)

        features = rep.transform(self.frames)

        test = features.get_features(rep)
        kk_ref = np.dot(test, test.T)

        # test that the feature matrix exported to python in various ways
        # are equivalent
        X_t = features.get_features(rep, self.global_species)
        kk = np.dot(X_t, X_t.T)
        self.assertTrue(np.allclose(kk, kk_ref))

        X_t = features.get_features(rep, self.global_species + [70])
        kk = np.dot(X_t, X_t.T)
        self.assertTrue(np.allclose(kk, kk_ref))

        species = copy(self.global_species)
        species.pop()
        X_t = features.get_features(rep, species)
        kk = np.dot(X_t, X_t.T)
        self.assertFalse(np.allclose(kk, kk_ref))

        X_t = features.get_features_by_species(rep)
        kk = dot(X_t, X_t)
        self.assertTrue(np.allclose(kk, kk_ref))

    def test_serialization(self):
        rep = SphericalInvariants(**self.hypers)

        rep_dict = to_dict(rep)

        rep_copy = from_dict(rep_dict)

        rep_copy_dict = to_dict(rep_copy)

        self.assertTrue(rep_dict == rep_copy_dict)
