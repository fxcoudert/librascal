/**
 * file   structure_manager_centers.hh
 *
 * @author Felix Musil <felix.musil@epfl.ch>
 * @author Markus Stricker <markus.stricker@epfl.ch>
 *
 * @date   06 August 2018
 *
 * @brief basic manager implementation with atoms and centers with ability to
 *        read from json file and be constructed from existing data
 *
 * Copyright © 2018 Felix Musil, Markus Stricker, COSMO (EPFL), LAMMM (EPFL)
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

#ifndef STRUCTURE_MANAGER_CENTERS_H
#define STRUCTURE_MANAGER_CENTERS_H

// inclusion of librascal data structure
#include "structure_managers/structure_manager.hh"
#include "lattice.hh"
#include "atomic_structure.hh"
#include "basic_types.hh"
#include "json_io.hh"

// data types and operations are based on the Eigen library
#include <Eigen/Dense>

// standard header inclusion
#include <stdexcept>
#include <vector>
#include <array>

/**
 * All functions and classes are in the namespace <code>rascal</code>, which
 * ensures that they don't clash with other libraries one might use in
 * conjunction.
 */
namespace rascal {
  //! forward declaration for traits
  class StructureManagerCenters;

  /**
   * traits specialisation for ManagerCenters: traits are used for vector
   * allocation and further down the processing chain to determine what
   * functionality the given StructureManager already contains to avoid
   * recomputation. See also the implementation of adaptors.
   */
  template <>
  struct StructureManager_traits<StructureManagerCenters> {
    // this manager only works with 3D data (positions, cell)
    constexpr static int Dim{3};
    // MaxOrder is set to 1 because it has just atoms
    constexpr static size_t MaxOrder{1};
    constexpr static AdaptorTraits::Strict Strict{AdaptorTraits::Strict::no};
    constexpr static bool HasDirectionVectors{false};
    constexpr static bool HasDistances{false};
    using LayerByOrder = std::index_sequence<0>;
  };

  /**
   * Definition of the new StructureManager class. To add your own, please stick
   * to the convention of using 'StructureManagerYours', where 'Yours' will give
   * a hint of what it is about.
   */
  class StructureManagerCenters:
    // public inheritance the base class
    public StructureManager<StructureManagerCenters> {
     // Publicly accessible variables and function of the class are given
     // here. These provide the interface to access the neighbourhood.
   public:
    // for convenience, the names are shortened
    using traits = StructureManager_traits<StructureManagerCenters>;
    using Parent = StructureManager<StructureManagerCenters>;
    // here you see why -- definition of used function return types
    using Vector_ref = typename Parent::Vector_ref;
    using AtomRef_t = typename Parent::AtomRef;

    /**
     * Eigen::Map is a convenient way to access data in the 'Eigen-way', if it
     * is already stored in a contiguous array.  The positions of the JSON file
     * and the cell vectors are put into contiguous arrays, which are member
     * variables of this class. Access is provided via the Eigen::Maps
     *
     * The following types are defined globally in atomic_structure.hh as common
     * return types for accessing atom and cell related data.
     */
    using Cell_t = AtomicStructure<traits::Dim>::Cell_t;
    using Cell_ref = AtomicStructure<traits::Dim>::Cell_ref;

    using AtomTypes_t = AtomicStructure<traits::Dim>::AtomTypes_t;
    using AtomTypes_ref = AtomicStructure<traits::Dim>::AtomTypes_ref;
    using ConstAtomTypes_ref = AtomicStructure<traits::Dim>::ConstAtomTypes_ref;

    using PBC_t = AtomicStructure<traits::Dim>::PBC_t;
    using PBC_ref = AtomicStructure<traits::Dim>::PBC_ref;

    using Positions_t = AtomicStructure<traits::Dim>::Positions_t;
    using Positions_ref = AtomicStructure<traits::Dim>::Positions_ref;

    /*
     * Here, the types for internal data structures are defined, based on
     * standard types.  In general, we try to use as many standard types, where
     * possible. It reduces the dependance on external libraries. If you want to
     * use e.g. the <code>Eigen</code> library, the access to the data can be
     * given via the above mentioned Eigen::Maps, which wrap arround contiguous
     * arrays of the internally saved data. It should be straight forward to
     * even include native <code>Eigen</code> types, since the compilation
     * checks for the library by default.
     */

    /**
     * A ClusterRef_t is a return type for iterators. It gives a light-weight
     * reference to an atom, a pair, a triplet,... to the AtomRefs of all
     * implicated atoms.  The template parameters Order and MaxOrder give the
     * pair/triplet/ and the maximum body order, e.g. up to pair level.  To
     * increase the MaxOrder, use an <code>adaptor</code>.
     */
    template <size_t Order>
    using ClusterRef_t = typename Parent::template ClusterRef<Order>;

    //! Default constructor, values are set during .update() function
    StructureManagerCenters()
      : atoms_object{}, atoms_index{}, lattice{}, offsets{}, natoms{}
    {}

    //! Copy constructor
    StructureManagerCenters(const StructureManagerCenters & other) = delete;

    //! Move constructor
    StructureManagerCenters(StructureManagerCenters && other) = default;

    //! Destructor
    virtual ~StructureManagerCenters() = default;

    //! Copy assignment operator
    StructureManagerCenters &
    operator=(const StructureManagerCenters & other) = delete;

    //! Move assignment operator
    StructureManagerCenters &
    operator=(StructureManagerCenters && other) = default;

    /**
     * invokes the reinitialisation based on existing data. E.g. when the
     * atom positions are provided by a simulation method, which evolves in
     * time, this function updates the data.
     */
    void update(const Eigen::Ref<const Eigen::MatrixXd,
                0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> positions,
                const Eigen::Ref<const Eigen::VectorXi> atom_types,
                const Eigen::Ref<const Eigen::MatrixXd> cell,
                const Eigen::Ref<const PBC_t> pbc);

    /**
     * invokes an update from a file, which holds a structure in the format of
     * the ASE atoms object
     */
    void update(const std::string filename);

    // TODO(felix): build/update ambiguity
    //! makes atom index lists and offsets
    void build();

    //! required for the construction of vectors, etc
    constexpr static int dim() {return traits::Dim;}

    /**
     * Returns a traits::Dim by traits::Dim matrix with the cell vectors of the
     * structure.
     */
    inline Cell_ref get_cell() {
      return Cell_ref(this->atoms_object.cell.data());
    }

    //! Returns the type of a given atom, given an AtomRef
    inline int & get_atom_type(const int & atom_index) {
      return this->atoms_object.atom_types(atom_index);
    }

    //! Returns the type of a given atom, given an AtomRef
    inline const int & get_atom_type(const int & atom_index) const {
      return this->atoms_object.atom_types(atom_index);
    }

    //! Returns an a map with all atom types.
    inline AtomTypes_ref get_atom_types() {
      AtomTypes_ref val(this->atoms_object.atom_types.data(),
                        1, this->natoms);
      return val;
    }

    //! Returns an a map with all atom types.
    inline ConstAtomTypes_ref get_atom_types() const {
      ConstAtomTypes_ref val(this->atoms_object.atom_types.data(),
                        1, this->natoms);
      return val;
    }

    //! Returns a map of size traits::Dim with 0/1 for periodicity
    inline PBC_ref get_periodic_boundary_conditions() {
      return Eigen::Map<PBC_t>(this->atoms_object.pbc.data());
    }

    //! Returns the position of an atom, given an AtomRef
    inline Vector_ref get_position(const AtomRef_t & atom) {
      auto index{atom.get_index()};
      auto p = this->get_positions();
      auto * xval{p.col(index).data()};
      return Vector_ref(xval);
    }

    //! Returns the position of an atom, given an atom index
    inline Vector_ref get_position(const size_t & atom_index) {
      auto p = this->get_positions();
      auto * xval{p.col(atom_index).data()};
      return Vector_ref(xval);
    }

    //! returns a map to all atomic positions.
    inline Positions_ref get_positions() {
      return Positions_ref(this->atoms_object.positions.data(), traits::Dim,
                           this->atoms_object.positions.size() / traits::Dim);
    }

    //! returns number of I atoms in the list
    inline size_t get_size() const {return this->natoms;}

    //! returns the number of neighbours of a given i atom
    template<size_t Order, size_t Layer>
    inline size_t get_cluster_size(const ClusterRefKey<Order, Layer>
                                   & /*cluster*/) const {
      static_assert(Order <= traits::MaxOrder,
                    "this implementation only handles atoms.");
      return 1;
    }

    //! dummy function, since no neighbours are present her
    inline int get_cluster_neighbour(const Parent& /*cluster*/,
                                     size_t index) const {
      // dummy argument is the atom itself
      return this->atoms_index[0][index];
    }

    //! Dummy function, since neighbours are not present at this Order
    template<size_t Order, size_t Layer>
    inline int get_cluster_neighbour(const ClusterRefKey<Order, Layer>
                                     & /*cluster*/, size_t) const {
      static_assert(Order <= traits::MaxOrder,
                    "this implementation only handles atoms.");
      return 0;
    }

    /**
     * Return the linear index of cluster (i.e., the count at which
     * this cluster appears in an iteration
     */
    template<size_t Order>
    inline size_t
    get_offset_impl(const std::array<size_t, Order> & counters) const;

    //! Function for returning the number of atoms
    size_t get_nb_clusters(size_t cluster_size) const;

    /**
     * Function for reading data from a JSON file in the ASE format. See the
     * definition of <code>AtomicStructure</code> and adapt the fields, which
     * should be read to your case.
     */
    void read_structure_from_json(const std::string filename);

    /**
     * Function for reading data from an XYZ file format. See defition
     * [here](https://en.wikipedia.org/wiki/XYZ_file_format).
     */
    void read_structure_from_xyz(const std::string filename);

   protected:
    /**
     * Object which can interface to the json header to read and write atom
     * related data in the ASE format: positions, cell, periodicity, atom types
     * (corresponding to element numbers)
     */
    AtomicStructure<traits::Dim> atoms_object{};

    /**
     * store atoms index per order,i.e.
     *   - atoms_index[0] lists all i-atoms
     * the structure here is only used for conformance, could just be a vector.
     */
    std::array<std::vector<int>, traits::MaxOrder> atoms_index;

    //! Lattice type for storing the cell and querying cell-related data
    Lattice<traits::Dim> lattice;

    /**
     * A vector which stores the absolute offsets for each atom to access the
     * correct variables in the neighbourlist. Here they are just the 1, 2, 3,
     * etc. corresponding to the sequence in which the atoms are provided during
     * construction.
     */
    std::vector<size_t> offsets{};

    //! Total number of atoms in structure
    size_t natoms{};
  };

  /* ---------------------------------------------------------------------- */
  //! used for construction (not for iteration)
  template<size_t Order>
  inline size_t StructureManagerCenters::
  get_offset_impl(const std::array<size_t, Order> & /*counters*/) const {
    static_assert(Order == 1,
                   "this manager only handles atoms.");
    return 0;
  }
}  // rascal

#endif /* STRUCTURE_MANAGER_CENTERS_H */
