/**
 * file   symmetry_functions.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 * @author Markus Stricker <markus.stricker@epfl.ch>
 *
 * @date   17 Dec 2018
 *
 * @brief implementation of symmetry functions for neural nets (G-functions in
 * Behler-Parinello-speak)
 *
 * Copyright © 2018 Till Junge, Markus Stricker COSMO (EPFL), LAMMM (EPFL)
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
 * along with librascal; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef SRC_RASCAL_REPRESENTATIONS_SYMMETRY_FUNCTIONS_HH_
#define SRC_RASCAL_REPRESENTATIONS_SYMMETRY_FUNCTIONS_HH_

#include "rascal/json_io.hh"
#include "rascal/structure_managers/property_typed.hh"
#include "rascal/units.hh"
#include "rascal/utils/utils.hh"
#include "rascal/utils/tuple_standardisation.hh"

#include <sstream>
#include <string>

#include "Eigen/Dense"

namespace rascal {

  using units::UnitStyle;

  enum class SymmetryFunType { One, Gaussian, Cosine, Angular1, Angular2 };

  /* ---------------------------------------------------------------------- */
  inline std::string get_name(SymmetryFunType fun_type) {
    switch (fun_type) {
    case SymmetryFunType::One: {
      return "One";
      break;
    }
    case SymmetryFunType::Gaussian: {
      return "Gaussian";
      break;
    }
    case SymmetryFunType::Cosine: {
      return "Cosine";
      break;
    }
    case SymmetryFunType::Angular1: {
      return "Angular1";
      break;
    }
    case SymmetryFunType::Angular2: {
      return "Angular2";
      break;
    }
    default:
      throw std::runtime_error("undefined symmetry function type");
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <SymmetryFunType FunType>
  struct SymmetryFun {};

  template <>
  struct SymmetryFun<SymmetryFunType::Gaussian> {
    static constexpr size_t Order{2};
    static constexpr size_t NbParams{2};

    using Return_t = std::tuple<double, Eigen::Matrix<double, ThreeD, 1>>;
    /**
     * usually, derivatives are aligned with the distance vector, in which case
     * a scalar return type is sufficient. (important for triplet-related
     * functions)
     */
    static constexpr bool DerivativeIsCollinear{true};

    template <class Derived>
    static double eval_function(const Eigen::MatrixBase<Derived> & params,
                                const double & r_ij) {
      static_assert(Derived::RowsAtCompileTime == NbParams, "size mismatch");
      static_assert(Derived::ColsAtCompileTime == 1, "Needs a column vector");
      auto && eta{params(0)};
      auto && r_s{params(1)};
      auto && delta_r = r_ij - r_s;
      return exp(-eta * delta_r * delta_r);
    }

    template <class Derived1, class Derived2>
    static Return_t eval_derivative(const Eigen::MatrixBase<Derived1> & params,
                                    const double & r_ij,
                                    const Eigen::MatrixBase<Derived2> & n_ij) {
      static_assert(Derived1::RowsAtCompileTime == NbParams, "size mismatch");
      static_assert(Derived1::ColsAtCompileTime == 1, "Needs a column vector");
      static_assert(Derived2::RowsAtCompileTime == ThreeD,
                    "dimension mismatch");
      static_assert(Derived2::ColsAtCompileTime == 1, "Needs a column vector");
      auto && eta{params(0)};
      auto && r_s{params(1)};
      auto && delta_r{r_ij - r_s};
      auto && fun_val{eval_function(params, r_ij)};
      return Return_t(fun_val, n_ij * (-2. * eta * delta_r * fun_val));
    }

    static Eigen::Matrix<double, NbParams, 1> read(const json & params,
                                                   const UnitStyle & units) {
      Eigen::Matrix<double, NbParams, 1> retval{};
      retval(0) = json_io::check_units(units.distance(-1, 2), params.at("eta"));
      retval(1) = json_io::check_units(units.distance(), params.at("r_s"));
      return retval;
    }
  };

  // template <>
  // struct SymmetryFun<SymmetryFunType::Angular1> {
  //   static constexpr size_t Order{3};
  //   static constexpr size_t NbParams{4};
  //   using ParamShape = Eigen::Matrix<double, NbParams, 1>;
  //   /**
  //    * usually, derivatives are aligned with the distance vector, in which
  //    case
  //    * a scalar return type is sufficient. (important for triplet-related
  //    * functions)
  //    */
  //   static constexpr bool DerivativeIsCollinear{false};

  //   static double eval_function(const Eigen::Map<ParamS> & params,
  //                               const double & r_ij, const double & r_jk,
  //                               const double & r_ik, const double cos_theta)
  //                               {
  //     auto && zeta{params(0)};
  //     auto && eta{params(1)};
  //     auto && lambda{params(2)};
  //     auto && r_s{params(2)};
  //     auto && delta_r = r_ij - r_s;
  //     return exp(-eta * delta_r * delta_r);
  //   }

  //   template <class Derived>
  //   static auto eval_derivative(const Eigen::MatrixBase<ParamShape> & params,
  //                               const double & r_ij,
  //                               const Eigen::MatrixBase<Derived> & n_ij)
  //       -> decltype(auto) {
  //     auto && eta{params(0)};
  //     auto && r_s{params(1)};
  //     auto && delta_r{r_ij - r_s};
  //     return n_ij * (-2. * eta * delta_r * exp(-eta * delta_r * delta_r));
  //   }

  //   static Eigen::Matrix<double, NbParams, 1> read(const json & params,
  //                                                  const UnitStyle & units) {
  //     Eigen::Matrix<double, NbParams, 1> retval{};
  //     retval(0) = json_io::check_units(units.distance(-1, 2),
  //     param.at("eta")); retval(1) = json_io::check_units(units.distance(),
  //     param.at("r_s")); return retval;
  //   }

  //  protected:
  // };

}  // namespace rascal

#endif  // SRC_RASCAL_REPRESENTATIONS_SYMMETRY_FUNCTIONS_HH_
