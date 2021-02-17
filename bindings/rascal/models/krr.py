from ..utils import BaseIO, is_notebook
from ..lib import compute_sparse_kernel_gradients, compute_sparse_kernel_neg_stress

import numpy as np
import ase

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class KRR(BaseIO):
    """Kernel Ridge Regression model. Only supports sparse GPR
    training for the moment.

    Parameters
    ----------
    weights : np.array
        weights of the model

    kernel : Kernel
        kernel class used to train the model

    X_train : SparsePoints
        reference samples used for the training

    self_contributions : dictionary
        map atomic number to the property baseline, e.g. isolated atoms
        energies when the model has been trained on total energies.
    """

    def __init__(self, weights, kernel, X_train, self_contributions):
        # Weights of the krr model
        self.weights = weights
        self.kernel = kernel
        self.X_train = X_train
        self.self_contributions = self_contributions
        self.target_type = kernel.target_type

    def _get_property_baseline(self, managers):
        """build total baseline contribution for each prediction"""
        if self.target_type == "Structure":
            Y0 = np.zeros(len(managers))
            for i_manager, manager in enumerate(managers):
                if isinstance(manager, ase.Atoms):
                    numbers = manager.get_atomic_numbers()
                    for sp in numbers:
                        Y0[i_manager] += self.self_contributions[sp]
                else:
                    for at in manager:
                        Y0[i_manager] += self.self_contributions[at.atom_type]
        elif self.target_type == "Atom":
            n_centers = 0
            for manager in managers:
                n_centers += len(manager)
            Y0 = np.zeros(n_centers)
            i_center = 0
            for manager in managers:
                for center in manager:
                    Y0[i_center] = self.self_contributions[center.atom_type]
                    i_center += 1
        return Y0

    def predict(self, managers, KNM=None):
        """Predict properties associated with the atomic structures in managers.

        Parameters
        ----------
        managers : AtomsList
            list of atomic structures with already computed features compatible
            with representation in kernel
        KNM : np.array, optional
            precomputed sparse kernel matrix

        Returns
        -------
        np.array
            predictions
        """
        if KNM is None:
            kernel = self.kernel(managers, self.X_train, (False, False))
        else:
            if len(managers) != KNM.shape[0]:
                raise ValueError(
                    "KNM size mismatch {}!={}".format(len(managers), KNM.shape[0])
                )
            elif self.X_train.size() != KNM.shape[1]:
                raise ValueError(
                    "KNM size mismatch {}!={}".format(self.X_train.size(), KNM.shape[1])
                )
            kernel = KNM
        Y0 = self._get_property_baseline(managers)
        return Y0 + np.dot(kernel, self.weights).reshape((-1))

    def predict_forces(self, managers, KNM=None):
        """Predict negative gradients w.r.t atomic positions, e.g. forces, associated with the atomic structures in managers.

        Parameters
        ----------
        managers : AtomsList
            list of atomic structures with already computed features compatible
            with representation in kernel
        KNM : np.array, optional
            precomputed sparse kernel matrix

        Returns
        -------
        np.array
            predictions
        """
        if self.kernel.kernel_type != "Sparse":
            raise NotImplementedError(
                "force prediction only implemented for kernels with kernel_type=='Sparse'"
            )
        if KNM is None:
            rep = self.kernel._representation
            gradients = compute_sparse_kernel_gradients(
                rep,
                self.kernel._kernel,
                managers.managers,
                self.X_train._sparse_points,
                self.weights.reshape((1, -1)),
            )
        else:
            n_atoms = 0
            for manager in managers:
                n_atoms += len(manager)
            if 3 * n_atoms != KNM.shape[0]:
                raise ValueError(
                    "KNM size mismatch {}!={}".format(3 * n_atoms, KNM.shape[0])
                )
            elif self.X_train.size() != KNM.shape[1]:
                raise ValueError(
                    "KNM size mismatch {}!={}".format(self.X_train.size(), KNM.shape[1])
                )
            gradients = np.dot(KNM, self.weights).reshape((-1, 3))

        return -gradients

    def predict_stress(self, managers, KNM=None):
        """Predict gradients w.r.t cell parameters, e.g. stress, associated with the atomic structures in managers.
        The stress is returned using the Voigt order: xx, yy, zz, yz, xz, xy.

        Parameters
        ----------
        managers : AtomsList
            list of atomic structures with already computed features compatible
            with representation in kernel
        KNM : np.array, optional
            precomputed sparse kernel matrix

        Returns
        -------
        np.array
            predictions
        """
        if self.kernel.kernel_type != "Sparse":
            raise NotImplementedError(
                "stress prediction only implemented for kernels with kernel_type=='Sparse'"
            )

        if KNM is None:
            rep = self.kernel._representation
            neg_stress = compute_sparse_kernel_neg_stress(
                rep,
                self.kernel._kernel,
                managers.managers,
                self.X_train._sparse_points,
                self.weights.reshape((1, -1)),
            )
        else:
            if 6 * len(managers) != KNM.shape[0]:
                raise ValueError(
                    "KNM size mismatch {}!={}".format(6 * len(managers), KNM.shape[0])
                )
            elif self.X_train.size() != KNM.shape[1]:
                raise ValueError(
                    "KNM size mismatch {}!={}".format(self.X_train.size(), KNM.shape[1])
                )
            neg_stress = np.dot(KNM, self.weights).reshape((len(managers), 6))

        return -neg_stress

    def get_weights(self):
        return self.weights

    def _get_init_params(self):
        init_params = dict(
            weights=self.weights,
            kernel=self.kernel,
            X_train=self.X_train,
            self_contributions=self.self_contributions,
        )
        return init_params

    def _set_data(self, data):
        super()._set_data(data)

    def _get_data(self):
        return super()._get_data()

    def get_representation_calculator(self):
        return self.kernel._rep


def get_strides(frames):
    Nstructures = len(frames)
    Ngrad_stride = [0]
    Ngrads = 0
    for frame in frames:
        n_at = len(frame)
        Ngrad_stride.append(n_at * 3)
        Ngrads += n_at * 3
    Ngrad_stride = np.cumsum(Ngrad_stride) + Nstructures
    return Nstructures, Ngrads, Ngrad_stride


def compute(i_frame, frame, representation, X_sparse, kernel):
    feat = representation.transform([frame])
    en_row = kernel(feat, X_sparse)
    grad_rows = kernel(feat, X_sparse, grad=(True, False))
    return en_row, grad_rows


def compute_KNM(frames, X_sparse, kernel, soap):
    Nstructures, Ngrads, Ngrad_stride = get_strides(frames)
    KNM = np.zeros((Nstructures + Ngrads, X_sparse.size()))
    pbar = tqdm(frames, desc="compute KNM", leave=False)
    for i_frame, frame in enumerate(frames):
        en_row, grad_rows = compute(i_frame, frame, soap, X_sparse, kernel)
        KNM[Ngrad_stride[i_frame] : Ngrad_stride[i_frame + 1]] = grad_rows
        KNM[i_frame] = en_row
        pbar.update()
    pbar.close()
    return KNM


def train_gap_model(
    kernel,
    frames,
    KNM_,
    X_sparse,
    y_train,
    self_contributions,
    grad_train=None,
    lambdas=None,
    jitter=1e-8,
):
    """
    Defines the procedure to train a SOAP-GAP model [1]:
    .. math::
        Y(A) = \sum_{i \in A} y_{a_i}(X_i),
    where :math:`Y(A)` is the predicted property function associated with the
    atomic structure :math:`A$, :math:`i` and :math:`a_i` are the index and
    species of the atoms in structure :math:`X` and :math:`y_a(A_i)` is the
    atom centered model that depends on the central atomic species.
    The individual predictions are given by:
    .. math::
        y_{a_i(A_i) = \sum_m^{M} \alpha_m \delta_{b_m a_i} k(A_i,T_m),
    where :math:`k(\cdot,\cdot)` is a kernel function, :math:`\alpha_m` are the
    weights of the model and :math:`b_m is the atom type associated with the
    sparse point :math:`T_m`.
    Hence a kernel element for the target property :math:`Y(A)` is given by:
    .. math::
        KNM_{Am} = \sum_{i \in A} \delta_{b_m a_i} k(A_i,T_m)
    and for :math:`\vec{\nabla}_iY(A)`:
    .. math::
       KNM_{A_{i}m} = \delta_{b_m a_i} \sum_{j \in A_i} \vec{\nabla}_i k(A_j,T_m)
    The training is given by:
    .. math::
        \bm{\alpha} =  K^{-1} \bm{Y},
    where :math:`K` is given by:
    .. math::
        K = K_{MM} + K_{MN} \Lambda^{-2} K_{NM},
    :math:`\bm{Y}=K_{MN} \Lambda^{-2} \bm{y}$, :math:`\bm{y}` the training
    targets and :math:`\Lambda` the regularization matrix.
    The regularization matrix is chosen to be diagonal:
    .. math::
        \Lambda^{-1}_{nn} = \delta_{nn} * lambdas[0] / \sigma_{\bm{y}} * np.sqrt(Natoms)
    for the targets and
    .. math::
        \Lambda^{-1}_{nn} = \delta_{nn} * lambdas[1] / \sigma_{\bm{y}},
    for the derivatives of the targets w.r.t. the atomic positions and
    :math:`\sigma_{\bm{y}}` is the standard deviation of the target property
    (not derivatives).

    Parameters
    ----------
    kernel : Kernel
        SparseKernel to compute KMM and initialize the model. It was used to
        build KNM_.
    frames : list(ase.Atoms)
        Training structures
    KNM_ : np.array
        kernel matrix to use in the training, typically computed with:
        KNM = kernel(managers, X_sparse)
        KNM_down = kernel(managers, X_sparse, grad=(True, False))
        KNM = np.vstack([KNM, KNM_down])
        when training with derivatives.
    X_sparse : SparsePoints
        basis samples to use in the model's interpolation
    y_train : np.array
        reference property
    self_contributions : dictionary
        map atomic number to the property baseline, e.g. training on
        total energies is not very recommended so one would provide
        the corresponding isolated atom energies so that the model
        can be trained on the corresponding formation energies and
        still predict total energies.
    grad_train : np.array, optional
        derivatives of y_train w.r.t. to the atomic motion, e.g.
        minus interatomic forces, by default None
    lambdas : list/tuple, optional
        regularisation parameter for the training, i.e. lambdas[0] -> property
        and lambdas[1] -> gradients of the property, by default None
    jitter : double, optional
        small jitter for the numerical stability of solving the linear system,
        by default 1e-8

    Returns
    -------
    KRR
        a trained model that can predict the property and its gradients

    .. [1] Ceriotti, M., Willatt, M. J., & Csányi, G. (2018).
        Machine Learning of Atomic-Scale Properties Based on Physical Principles.
        In Handbook of Materials Modeling (pp. 1–27). Springer, Cham.
        https://doi.org/10.1007/978-3-319-42913-7_68-1
    """
    KMM = kernel(X_sparse)
    Y = y_train.reshape((-1, 1)).copy()
    KNM = KNM_.copy()
    n_centers = Y.shape[0]
    Natoms = np.zeros(n_centers)
    Y0 = np.zeros((n_centers, 1))
    for iframe, frame in enumerate(frames):
        Natoms[iframe] = len(frame)
        for sp in frame.get_atomic_numbers():
            Y0[iframe] += self_contributions[sp]
    Y = Y - Y0
    delta = np.std(Y)
    # lambdas[0] is provided per atom hence the '* np.sqrt(Natoms)'
    # the first n_centers rows of KNM are expected to refer to the
    #  property
    KNM[:n_centers] /= lambdas[0] / delta * np.sqrt(Natoms)[:, None]
    Y /= lambdas[0] / delta * np.sqrt(Natoms)[:, None]

    if grad_train is not None:
        KNM[n_centers:] /= lambdas[1] / delta
        F = grad_train.reshape((-1, 1)).copy()
        F /= lambdas[1] / delta
        Y = np.vstack([Y, F])

    KMM[np.diag_indices_from(KMM)] += jitter

    K = KMM + np.dot(KNM.T, KNM)
    Y = np.dot(KNM.T, Y)
    weights = np.linalg.lstsq(K, Y, rcond=None)[0]
    model = KRR(weights, kernel, X_sparse, self_contributions)

    # avoid memory clogging
    del K, KMM
    K, KMM = [], []

    return model
