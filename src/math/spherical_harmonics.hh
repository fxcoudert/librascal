/**
 * file   spherical_harmonics.hh
 *
 * @author  Felix Musil <felix.musil@epfl.ch>
 * @author  Max Veit <max.veit@epfl.ch>
 *
 * @date   14 October 2018
 *
 * @brief implementation of the spherical harmonics
 *
 * Copyright  2018  Felix Musil, Max Veit, COSMO (EPFL), LAMMM (EPFL)
 *
 * Rascal is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * Rascal is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this software; see the file LICENSE. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef SRC_MATH_SPHERICAL_HARMONICS_HH_
#define SRC_MATH_SPHERICAL_HARMONICS_HH_

#include "math_utils.hh"
#include <vector>

namespace rascal {
  namespace math {
    /**
     * Compute a set of normalized associated Legendre polynomials
     *
     * These are normalized for use in computing the real spherical harmonics;
     * see math::compute_spherical_harmonics() for details.  The m==0 harmonics
     * require an extra factor of 1/√2 (see below).  Only positive-m functions
     * are computed.
     *
     * @param cos_theta (aka x) Where to evaluate the polynomial
     *
     * @param max_angular Compute up to this angular momentum number (l_max)
     *
     * @return assoc_legendre_polynom
     *        An (Eigen)matrix containing the evaluated polynomials.
     *        Sized l_max by (2*lmax + 1); the row is indexed by l and the
     *        column by m >= 0.
     */
    Matrix_t compute_assoc_legendre_polynom(double cos_theta,
                                            size_t max_angular);

    /**
     * Compute cos(mφ) and sin(mφ) from the recurrence relations
     *
     * The relations are (these are the same recurrence relations used to
     * calculate the Chebyshev polynomials):
     *
     * cos(mφ) = 2cos(φ)cos((m-1)φ) - cos((m-2)φ)
     * sin(mφ) = 2cos(φ)sin((m-1)φ) - sin((m-2)φ)
     *
     * and they require only cos(φ) and sin(φ) to start.
     *
     * @param cos_phi Value of cos(φ) to start the relation
     *
     * @param sin_phi Value of sin(φ) to start the relation
     *
     * @param max_m Compute up to a maximum value of max_m (inclusive)
     *
     * @return cos_sin_m_phi
     *        (Eigen)matrix containing the results.
     *        Sized max_m by 2 with the cos(mφ) stored in the first column
     *        and sin(mφ) in the second column, m being the row index
     */
    MatrixX2_t compute_cos_sin_angle_multiples(double cos_phi, double sin_phi,
                                               size_t max_m);

    /**
     * Compute a full set of spherical harmonics given a direction vector
     *
     * Follows the algorithm described in https://arxiv.org/abs/1410.1748
     *
     * In brief, this computes the real spherical harmonics where the imaginary
     * components of the usual complex functions are instead stored in the
     * negative-m indices:
     *
     *               ╭ √((2l+1)/(2*π) * (l+m)!/(l-m)!) P_l^-m(cos(θ)) sin(-mφ)
     *               |                                                  for m<0
     * Y_l^m(θ, φ) = ┤ √((2l+1)/(4*π)) P_l(cos(θ)) for m==0
     *               |
     *               ╰ √((2l+1)/(2*π) * (l-m)!/(l+m)!) P_l^m(cos(θ)) cos(mφ)
     *                                                                  for m>0
     *
     * In case you're wondering why it's 1/2π on the m=/=0 components (instead
     * of 1/4π), there's an extra factor of 1/2 that comes from integrating cos²
     * or sin² over the full circle of φ, so these are indeed normalized in the
     * same sense as the complex spherical harmonics:
     *
     * ∫∫_(Sphere) dΩ Y_l^m(θ, φ) Y_l'^m'(θ, φ) = δ_(ll')δ_(mm')
     *
     * (this extra factor of 1/√2 is not included in the normalization of the
     * associated Legendre polynomials defined above; however, all other
     * normalization factors are.)
     *
     * @param direction Unit vector giving the angles (arguments for the Y_l^m)
     *
     * @param max_angular Compute up to this angular momentum number (l_max)
     *
     * @return  (Eigen)vector containing the results.
     *          Sized (l_max+1)**2, contains the l,m components in compressed
     *          format, i.e. (00)(1-1)(10)(11)(2-2)....
     */
    Vector_t compute_spherical_harmonics(
        const Eigen::Ref<const Eigen::Vector3d> & direction,
        size_t max_angular);

    /**
     * Compute a full set of spherical harmonics and their Cartesian gradients
     *
     * Spherical harmonics are defined as described in
     * math::compute_spherical_harmonics().  Gradients are defined with respect
     * to motion of the central atom, which is the opposite sign of the usual
     * definition with respect to the _arguments_ of the Y_l^m.  The actual
     * Cartesian gradients include an extra factor of 1/r that is not included
     * here; the rest is independent of radius.
     *
     * @param direction Unit vector giving the angles (arguments for the Y_l^m)
     *
     * @param max_angular Compute up to this angular momentum number (l_max)
     *
     * @return  (Eigen)array containing the results.
     *          Sized 4 by (l_max+1)^2; the first index collects the
     *          value of the harmonic and the x, y, and z gradient components.
     *          The second index collects l and m quantum numbers, stored in
     *          compact format (m varies fastest, from -l to l, and l from 0 to
     *          l_max).
     *
     * @warning This function will access the associated Legendre polynomials
     *          for m=l+1 and assume they are equal to zero.  The implementation
     *          of the polynomials in this file respects this convention.
     *
     * @todo Add an option to switch off the computation of gradients, so this
     *       function becomes equivalent to math::compute_spherical_harmonics()
     */
    Matrix_t compute_spherical_harmonics_derivatives(
        const Eigen::Ref<const Eigen::Vector3d> & direction,
        size_t max_angular);

    // New class which contains the above functions to precompute as much as
    // possible.
    class SphericalHarmonics {
     protected:
      using Matrix_Ref = typename Eigen::Ref<const Matrix_t>;
      using MatrixX2_Ref = typename Eigen::Ref<const MatrixX2_t>;
      using Vector_Ref = typename Eigen::Ref<const Vector_t>;
      // using Vector_Ref = typename Eigen::Ref<const Eigen::VectorXd>;

      size_t max_angular{0};
      Vector_t angular_coeffs1{};
      Vector_t angular_coeffs2{};
      Vector_t harmonics{};
      Matrix_t assoc_legendre_polynom{};
      MatrixX2_t cos_sin_m_phi{};
      Matrix_t coeff_a{};
      Matrix_t coeff_b{};

     public:
      SphericalHarmonics() {}

      // Precomputes as many parameters as possible
      void precompute(size_t max_angular) {
        using math::pow;
        using std::sqrt;
        this->max_angular = max_angular;
        /// Technically abs(sin(θ)), but θ only goes from [0, π)
        // double sin_theta = sqrt(1.0 - pow(cos_theta, 2));
        // Matrix_t assoc_legendre_polynom(max_angular + 1, max_angular +
        // 1);
        this->assoc_legendre_polynom =
            Matrix_t::Zero(this->max_angular + 1, this->max_angular + 1);
        this->coeff_a =
            Matrix_t::Zero(this->max_angular + 1, 2 * this->max_angular + 1);
        this->coeff_b =
            Matrix_t::Zero(this->max_angular + 1, 2 * this->max_angular + 1);
        this->harmonics =
            Vector_t::Zero((this->max_angular + 1) * (this->max_angular + 1));
        this->cos_sin_m_phi = MatrixX2_t::Zero(this->max_angular + 1, 2);

        for (size_t angular_l{0}; angular_l < this->max_angular + 1;
             angular_l++) {
          double lsq = angular_l * angular_l;
          double lm1sq = (angular_l - 1) * (angular_l - 1);
          for (size_t m_count{0}; m_count < angular_l + 1; m_count++) {
            double msq = m_count * m_count;
            this->coeff_a(angular_l, m_count) =
                sqrt((4 * lsq - 1.0) / (lsq - msq));
            this->coeff_b(angular_l, m_count) =
                -1.0 * sqrt((lm1sq - msq) / (4 * lm1sq - 1.0));
          }
        }
        // this->angular_coeffs1.reserve(this->max_angular);
        this->angular_coeffs1 = Vector_t::Zero(this->max_angular + 1);
        this->angular_coeffs2 = Vector_t::Zero(this->max_angular + 1);
        for (size_t angular_l{0}; angular_l < this->max_angular + 1;
             angular_l++) {
          angular_coeffs1(angular_l) = sqrt(2 * angular_l + 1);
          angular_coeffs2(angular_l) = -sqrt(1.0 + 0.5 / angular_l);
        }
      }

      // computation from here
      void compute_assoc_legendre_polynom(double cos_theta) {
        using math::pow;
        using std::sqrt;
        double sin_theta = sqrt(1.0 - pow(cos_theta, 2));
        const double SQRT_INV_2PI = sqrt(0.5 / PI);
        const double SQRT_THREE_HALF = sqrt(3.0/2.0);
        // Compute the associated Legendre polynomials: l < 2 are special cases
        // These include the normalization factors usually needed in the
        // spherical harmonics
        double l_accum{SQRT_INV_2PI};
        this->assoc_legendre_polynom(0, 0) = SQRT_INV_2PI;
        if (this->max_angular > 0) {
          this->assoc_legendre_polynom(1, 0) =
              cos_theta * SQRT_THREE * SQRT_INV_2PI;
          l_accum = l_accum * -SQRT_THREE_HALF * sin_theta;
          this->assoc_legendre_polynom(1, 1) = l_accum;
        }
        for (size_t angular_l{2}; angular_l < this->max_angular + 1;
             angular_l++) {
          // for l > 1 : Use the recurrence relation
          // TODO(max-veit) don't bother calculating m =/= 0 if sin(theta) == 0
          //                (z-axis)
          this->assoc_legendre_polynom.row(angular_l)
              .head(angular_l - 1).array() =
              cos_theta * this->assoc_legendre_polynom.row(angular_l - 1)
                                .head(angular_l - 1).array();
          this->assoc_legendre_polynom.row(angular_l).
                head(angular_l - 1).array() +=
                this->coeff_b.row(angular_l).head(angular_l - 1).array() *
                    this->assoc_legendre_polynom.row(angular_l - 2)
                        .head(angular_l - 1).array();
          this->assoc_legendre_polynom.row(angular_l)
               .head(angular_l - 1).array() *=
                 this->coeff_a.row(angular_l).head(angular_l - 1).array();
          this->assoc_legendre_polynom(angular_l, angular_l - 1) =
              // cos_theta * sqrt(2 * angular_l + 1) * l_accum;
              l_accum * cos_theta * this->angular_coeffs1(angular_l);
          l_accum = l_accum * sin_theta * this->angular_coeffs2(angular_l);
          this->assoc_legendre_polynom(angular_l, angular_l) = l_accum;
        }
      }

      /**
       * Compute a full set of spherical harmonics given a direction vector
       *
       * Follows the algorithm described in https://arxiv.org/abs/1410.1748
       *
       * In brief, this computes the real spherical harmonics where the
       * imaginary components of the usual complex functions are instead stored
       * in the negative-m indices:
       *
       *               ╭ √((2l+1)/(2*π) * (l+m)!/(l-m)!) P_l^-m(cos(θ)) sin(-mφ)
       *               |                                              for m<0
       *               |
       * Y_l^m(θ, φ) = ┤ √((2l+1)/(4*π)) P_l(cos(θ)) for m==0
       *               |
       *               | √((2l+1)/(2*π) * (l-m)!/(l+m)!) P_l^m(cos(θ)) cos(mφ)
       *               ╰                                              for m>0
       *
       * In case you're wondering why it's 1/2π on the m=/=0 components (instead
       * of 1/4π), there's an extra factor of 1/2 that comes from integrating
       * cos² or sin² over the full circle of φ, so these are indeed normalized
       * in the same sense as the complex spherical harmonics:
       *
       * ∫∫_(Sphere) dΩ Y_l^m(θ, φ) Y_l'^m'(θ, φ) = δ_(ll')δ_(mm')
       *
       * (this extra factor of 1/√2 is not included in the normalization of the
       * associated Legendre polynomials defined above; however, all other
       * normalization factors are.)
       *
       * @param direction Unit vector giving the angles (arguments for the
       * Y_l^m)
       *
       * @param max_angular Compute up to this angular momentum number (l_max)
       *
       * @return  (Eigen)vector containing the results.
       *          Sized (l_max+1)**2, contains the l,m components in compressed
       *          format, i.e. (00)(1-1)(10)(11)(2-2)....
       */
      void calc(const Eigen::Ref<const Eigen::Vector3d> & direction) {
        using math::pow;
        using std::sqrt;

        // The cosine against the z-axis is just the z-component of the
        // direction vector
        double cos_theta = direction[2];
        // The less efficient, but more intuitive implementation:
        // double phi = std::atan2(direction[1], direction[0]);
        double sqrt_xy = std::hypot(direction[0], direction[1]);
        // For a vector along the z-axis, define phi=0
        double cos_phi{1.0}, sin_phi{0.0};
        if (sqrt_xy >= math::dbl_ftol) {
          cos_phi = direction[0] / sqrt_xy;
          sin_phi = direction[1] / sqrt_xy;
        }
        // change cos_sin_m_phi to this->cos_sin_m_phi
        this->compute_assoc_legendre_polynom(cos_theta);
        this->compute_cos_sin_angle_multiples(cos_phi, sin_phi);

        size_t lm_base{0};  // starting point for storage
        for (size_t angular_l{0}; angular_l < this->max_angular + 1;
             angular_l++) {
          // std::cout << "angular_l: "<< angular_l << std::endl;
          // uses symmetry of spherical harmonics,
          // careful with the storage order

          auto legendre_l = this->assoc_legendre_polynom.row(angular_l);

          this->harmonics.segment(lm_base, angular_l) =
              (legendre_l.segment(1,angular_l).array() *
               cos_sin_m_phi.col(1).segment(1,angular_l).transpose().array())
                  .reverse();
          this->harmonics(lm_base + angular_l) = legendre_l(0) * INV_SQRT_TWO;
          this->harmonics.segment(lm_base + angular_l +1, angular_l) =
              legendre_l.segment(1, angular_l + 1).array() *
              cos_sin_m_phi.col(0).segment(1, angular_l).transpose().array();

          lm_base += 2 * angular_l + 1;
        }  // for (l in [0, lmax])
      }

      /**
       * Compute cos(mφ) and sin(mφ) from the recurrence relations
       *
       * The relations are (these are the same recurrence relations used to
       * calculate the Chebyshev polynomials):
       *
       * cos(mφ) = 2cos(φ)cos((m-1)φ) - cos((m-2)φ)
       * sin(mφ) = 2cos(φ)sin((m-1)φ) - sin((m-2)φ)
       *
       * and they require only cos(φ) and sin(φ) to start.
       *
       * @param cos_phi Value of cos(φ) to start the relation
       *
       * @param sin_phi Value of sin(φ) to start the relation
       *
       * @param max_m Compute up to a maximum value of max_m (inclusive)
       *
       * @return cos_sin_m_phi
       *        (Eigen)matrix containing the results.
       *        Sized max_m by 2 with the cos(mφ) stored in the first column
       *        and sin(mφ) in the second column, m being the row index
       */
      void compute_cos_sin_angle_multiples(const double & cos_phi,
                                           const double & sin_phi) {
        this->cos_sin_m_phi.row(0) << 1.0, 0.0;
        if (this->max_angular>0) {
          this->cos_sin_m_phi.row(1) << cos_phi, sin_phi;
        }
        auto two_cos_phi = cos_phi + cos_phi;
        for (size_t m_count{2}; m_count < this->max_angular + 1; m_count++) {
          this->cos_sin_m_phi.row(m_count) =
                two_cos_phi * this->cos_sin_m_phi.row(m_count - 1) -
                this->cos_sin_m_phi.row(m_count - 2);
        }
      }

      inline MatrixX2_Ref get_cos_sin_m_phi() {
        return MatrixX2_Ref(this->cos_sin_m_phi);
      }

      inline Matrix_Ref get_assoc_legendre_polynom() {
        return Matrix_Ref(this->assoc_legendre_polynom);
      }

      inline Vector_Ref get_harmonics() { return Vector_Ref(this->harmonics); }
    };

  }  // namespace math
}  // namespace rascal

#endif  // SRC_MATH_SPHERICAL_HARMONICS_HH_