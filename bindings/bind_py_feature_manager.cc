/**
 * @file   bind_py_feature_manager.cc
 *
 * @author Felix Musil <felix.musil@epfl.ch>
 *
 * @date   30 Oct 2018
 *
 * @brief  File for binding the Feature Managers
 *
 * Copyright  2018  Felix Musil, COSMO (EPFL), LAMMM (EPFL)
 *
 * Rascal is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * Rascal is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "bind_include.hh"

using namespace rascal;  // NOLINT

template<typename T>
void bind_feature_manager_base(py::module & m_garbage) {
  using Base_t = FeatureManagerBase<T>;

  std::string featurebase_name =
      internal::GetBindingTypeName<Base_t>();
  py::class_<Base_t>(m_garbage, featurebase_name.c_str());
}

/**
 * Bind a feature manager
 */
template <template <class> class FeatureManager_t, typename T>
decltype(auto) bind_feature_manager(py::module & mod, py::module &) {
  using Feature_t = FeatureManager_t<T>;
  using Base_t = FeatureManagerBase<T>;

  std::string feature_name = internal::GetBindingTypeName<Feature_t>();

  py::class_<Feature_t, Base_t> feature(mod, feature_name.c_str());
  // use custom constructor to pass json formated string as initializer
  // an alternative would be to convert python dict to json internally
  // but needs some workon in the pybind machinery
  feature.def(py::init([](int & n_feature, std::string & hyper_str) {
    // convert to json
    json hypers = json::parse(hyper_str);
    return std::make_unique<Feature_t>(n_feature, hypers);
  }));
  feature.def("reserve", &Feature_t::reserve);
  feature.def("append", (void (Feature_t::*)(RepresentationManagerBase &)) &
                            Feature_t::push_back);
  feature.def_property_readonly("size", &Feature_t::size,
                                py::return_value_policy::copy);
  feature.def_property_readonly("shape", &Feature_t::shape,
                                py::return_value_policy::copy);
  feature.def("get_feature_matrix", &Feature_t::get_feature_matrix,
              py::return_value_policy::reference_internal,
              py::keep_alive<1, 0>());

  return feature;
}

/**
 * Bind a feature manager
 */
template <template <typename, typename> class FeatureManager, typename T, typename key_t>
decltype(auto) bind_sparse_feature_manager(py::module & mod, py::module &) {
  using Feature_t = FeatureManager<T, key_t>;
  using Base_t = FeatureManagerBase<T>;
  std::string feature_name = internal::GetBindingTypeName<Feature_t>();
  py::class_<Feature_t, Base_t> feature(mod, feature_name.c_str());
  // use custom constructor to pass json formated string as initializer
  // an alternative would be to convert python dict to json internally
  // but needs some workon in the pybind machinery
  feature.def(py::init([](int & inner_size, std::string & hyper_str) {
    // convert to json
    json hypers = json::parse(hyper_str);
    return std::make_unique<Feature_t>(inner_size, hypers);
  }));
  feature.def("reserve", &Feature_t::reserve);
  feature.def("append", (void (Feature_t::*)(RepresentationManagerBase &)) &
                            Feature_t::push_back);
  feature.def_property_readonly("size", &Feature_t::size,
                                py::return_value_policy::copy);
  feature.def_property_readonly("shape", &Feature_t::shape,
                                py::return_value_policy::copy);
  feature.def("get_feature_matrix", &Feature_t::get_feature_matrix_dense,
              py::return_value_policy::reference_internal,
              py::keep_alive<1, 0>());

  return feature;
}

//! Feature aggregator python binding
void add_feature_managers(py::module & mod, py::module & m_garbage) {
  bind_feature_manager_base<double>(m_garbage);
  bind_feature_manager_base<float>(m_garbage);

  auto feature_double =
      bind_feature_manager<FeatureManagerDense, double>(mod, m_garbage);

  auto feature_float =
      bind_feature_manager<FeatureManagerDense, float>(mod, m_garbage);

  bind_sparse_feature_manager<FeatureManagerBlockSparse, double, int>(mod, m_garbage);

}
