/**
 * file   spherical_invariants_example.cc
 *
 * @author Max Veit <max.veit@epfl.ch>
 *
 * @date   26 June 2019
 *
 * @brief  Example for computing the spherical invariants (SOAP)
 *
 * Copyright © 2018 Max Veit, Felix Musil, COSMO (EPFL), LAMMM (EPFL)
 *
 * librascal is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * librascal is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this software; see the file LICENSE. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "structure_managers/structure_manager_centers.hh"
#include "structure_managers/adaptor_strict.hh"
#include "structure_managers/adaptor_neighbour_list.hh"
#include "structure_managers/make_structure_manager.hh"
#include "rascal_utility.hh"
#include "representations/representation_manager_sorted_coulomb.hh"
#include "representations/representation_manager_spherical_expansion.hh"
#include "representations/representation_manager_spherical_invariants.hh"
#include "representations/feature_manager_dense.hh"
#include "basic_types.hh"
#include "atomic_structure.hh"

#include <iostream>
#include <basic_types.hh>
#include <cmath>
#include <list>
#include <functional>
#include <string>
#include <initializer_list>
#include <chrono>

using namespace rascal;  // NOLINT

using Representation_t = RepresentationManagerSphericalInvariants<
    AdaptorStrict<AdaptorNeighbourList<StructureManagerCenters>>>;

int main(int argc, char * argv[]) {
  if (argc < 2) {
    std::cerr << "Must provide atomic structure json filename as argument";
    std::cerr << std::endl;
    return -1;
  }

  std::string filename{argv[1]};

  double cutoff{4.};
  json hypers{{"max_radial", 3},
              {"max_angular", 2},
              {"compute_gradients", true},
              {"soap_type", "PowerSpectrum"},
              {"normalize", true}};

  json fc_hypers{{"type", "Cosine"},
                 {"cutoff", {{"value", cutoff}, {"unit", "AA"}}},
                 {"smooth_width", {{"value", 0.5}, {"unit", "AA"}}}};
  json sigma_hypers{{"type", "Constant"},
                    {"gaussian_sigma", {{"value", 0.4}, {"unit", "AA"}}}};

  hypers["cutoff_function"] = fc_hypers;
  hypers["gaussian_density"] = sigma_hypers;
  hypers["radial_contribution"] = {{"type", "GTO"}};

  json structure{{"filename", filename}};
  json adaptors;
  json ad1{{"name", "AdaptorNeighbourList"},
           {"initialization_arguments",
            {{"cutoff", cutoff}, {"consider_ghost_neighbours", false}}}};
  json ad2{{"name", "AdaptorStrict"},
           {"initialization_arguments", {{"cutoff", cutoff}}}};
  adaptors.emplace_back(ad1);
  adaptors.emplace_back(ad2);
  auto manager =
      make_structure_manager_stack<StructureManagerCenters,
                                   AdaptorNeighbourList, AdaptorStrict>(
          structure, adaptors);

  Representation_t representation{manager, hypers};
  representation.compute();

  constexpr size_t n_centers_print{4};
  constexpr size_t n_neigh_print{1};

  // auto soap = representation.get_representation_full();
  // std::cout << "Sample SOAP elements \n"
  //<< soap(0, 0) << " " << soap(0, 1) << " " << soap(0, 2) << "\n"
  //<< soap(1, 0) << " " << soap(1, 1) << " " << soap(1, 2) << "\n"
  //<< soap(2, 0) << " " << soap(2, 1) << " " << soap(2, 2) << "\n";
  // Print the first few elements and gradients, so we know we're getting
  // something
  std::cout << "Expansion of first " << n_centers_print << " centers:";
  std::cout << std::endl;
  std::cout << "Note that the coefficients are printed with species pairs along"
               " the columns and n-n'-l along the rows."
            << std::endl;
  std::cout << "Gradients are printed with: First Cartesian component, "
               "then species pairs, along the columns; n-n'-l along the rows.";
  std::cout << std::endl;
  size_t center_count{0};
  for (auto center : manager) {
    if (center_count >= n_centers_print) {
      break;
    }
    size_t n_species_center{
        representation.soap_vectors.get_keys(center).size()};
    std::cout << "============================" << std::endl;
    std::cout << "Center " << center.get_index();
    std::cout << " of type " << center.get_atom_type() << std::endl;
    std::cout << representation.soap_vectors.get_dense_row(center);
    std::cout << std::endl;
    auto keys_center = representation.soap_vectors[center].get_keys();
    std::cout << "Center data keys: ";
    for (auto key : keys_center) {
      std::cout << "(";
      for (auto key_sp : key) {
        std::cout << key_sp << ", ";
      }
      std::cout << "\b\b) ";
    }
    std::cout << std::endl;
    auto keys_grad_center =
        representation.soap_vector_gradients[center].get_keys();
    std::cout << "Center gradient keys: ";
    for (auto key : keys_grad_center) {
      std::cout << "(";
      for (auto key_sp : key) {
        std::cout << key_sp << ", ";
      }
      std::cout << "\b\b) ";
    }
    std::cout << std::endl;
    std::cout << "Gradient of this expansion wrt center pos: " << std::endl;
    // clang-format off
    // makes an absolute mess of the below
    std::cout << Eigen::Map<Eigen::MatrixXd>(
           representation.soap_vector_gradients.get_dense_row(center).data(),
           3 * n_species_center,
           representation.soap_vector_gradients.get_nb_comp())
      .transpose();
    // clang-format on
    std::cout << std::endl;
    size_t neigh_count{0};
    for (auto neigh : center) {
      if (neigh_count >= n_neigh_print) {
        break;
      }
      auto keys_neigh = representation.soap_vector_gradients[neigh].get_keys();
      std::cout << "Neighbour keys: ";
      for (auto key : keys_neigh) {
        std::cout << "(";
        for (auto key_sp : key) {
          std::cout << key_sp << ", ";
        }
        std::cout << "\b\b) ";
      }
      std::cout << std::endl;
      std::cout << "Gradient of the above wrt atom " << neigh.back();
      std::cout << " of type " << neigh.get_atom_type() << std::endl;
      // clang-format off
      std::cout << Eigen::Map<Eigen::MatrixXd>(
          representation.soap_vector_gradients.get_dense_row(neigh).data(),
          3 * n_species_center,
          representation.soap_vector_gradients.get_nb_comp())
        .transpose();
      // clang-format on
      std::cout << std::endl;
      ++neigh_count;
    }
    ++center_count;
  }
}