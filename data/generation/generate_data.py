import os
import argparse
import csv
import numpy as np
import dag_generator as gen
from sklearn.preprocessing import StandardScaler


class dataset_generator:
    """ Generate datasets using dag_generator.py. `nb_dag` dags are sampled and
    then data are generated accordingly to the chosen parameters (e.g.
    mechanisms). Can generate dataset with 'hard stochastic' interventions """

    def __init__(self, mechanism, cause, intervention_type, struct_interv_distr, noise, noise_coeff, nb_nodes, expected_degree,
                 nb_points, suffix, rescale, obs_data=True, nb_interventions=10,
                 min_nb_target=1, max_nb_target=3, conservative=False,
                 cover=False, verbose=True):
        """
        Generate a dataset containing interventions. The setting is similar to the
        one in the GIES paper (Characterization and greedy learning of interventional Markov
        equivalence classes of directed acyclic graphs). Save the lists of targets
        in a separate file.

        Args:
            mechanism (str): Type of mechanism use to generate the data
            (linear|polynomial|sigmoid_add|sigmoid_mix|gp_add|gp_mix|
            anm|nn|nn_add|pnl_gp_mechanism|pnl_mult_mechanism|post_nonlinear|x)
            cause (str): Distribution of initial causes
            (gmm_cause|gaussian|variable_gaussian|uniform|uniform_positive)
            noise (str): Distribution of noises
            (gaussian|variable_gaussian|uniform|laplace|absolute_gaussian|nn)
            noise_coeff (float): Noise coefficient

            nb_nodes (int): Number of nodes in each DAG
            expected_degree (int): Expected number of edges per node
            nb_points (int): Number of points per interventions (thus the total =
                            nb_interventions * nb_points)
            rescale (bool): if True, rescale each variables
            suffix (str): Suffix that will be added at the end of the folder name

            nb_interventions (int): number of interventional settings
            obs_data (bool): if True, the first setting is generated without any interventions
            min_nb_target (int): minimal number of targets per setting
            max_nb_target (int): maximal number of targets per setting. For a fixed
                                 number of target, one can make min_nb_target==max_nb_target
            conservative (bool): if True, make sure that the intervention family is
                                 conservative: i.e. that all nodes have not been
                                 intervened in at least one setting.
            cover (bool): if True, make sure that all nodes have been
                                 intervened on at least in one setting.
            verbose (bool): if True, print messages to inform users
        """

        self.mechanism = mechanism
        self.cause = cause
        self.noise = noise
        self.noise_coeff = noise_coeff
        self.nb_nodes = nb_nodes
        self.expected_degree = expected_degree
        self.i_dataset = 0
        self.nb_points = nb_points
        self.suffix = suffix
        self.rescale = rescale
        self.folder = f'data_p{nb_nodes}_e{nb_nodes * expected_degree}_n{nb_points}_{suffix}'
        self.verbose = verbose
        self.generator = None
        self.struct_interv_distr = struct_interv_distr
        self.intervention_type = intervention_type

        # attributes related to interventional data
        self.obs_data = obs_data
        self.nb_interventions = nb_interventions
        self.min_nb_target = min_nb_target
        self.max_nb_target = max_nb_target
        self.conservative = conservative
        self.cover = cover

        # assert that the parameters
        self._checkup()
        self._create_folder()

    def _checkup(self):
        possible_mechanisms = ["linear","polynomial","sigmoid_add","sigmoid_mix","gp_add","gp_mix",
                               "anm","nn","nn_add","pnl_gp_mechanism","pnl_mult_mechanism","post_nonlinear","x","circle","adn"]
        possible_causes = ["gmm_cause","gaussian","variable_gaussian","uniform","uniform_positive"]
        possible_noises = ["gaussian","variable_gaussian","uniform","laplace","absolute_gaussian","nn"]

        assert self.mechanism in possible_mechanisms, \
                f"mechanism doesn't exist. It has to be in {possible_mechanisms}"
        assert self.cause in possible_causes, \
                f"initial cause doesn't exist. It has to be in {possible_causes}"
        assert self.noise in possible_noises, \
                f"noise doesn't exist. It has to be in {possible_noises}"

        assert self.nb_interventions <= self.nb_points, \
                "nb_interventions should be smaller or equal to nb_points"
        assert self.min_nb_target <= self.max_nb_target, \
                "min_nb_target should be smaller or equal to max_nb_target"
        assert self.max_nb_target <= self.nb_nodes, \
                "max_nb_target should be smaller or equal to the total number of nodes (nb_nodes)"
        if self.cover:
            assert self.max_nb_target * self.nb_interventions >= self.nb_nodes, \
                    "In order to cover, there should be more interventions or a higher min_nb_target"
        if self.conservative and not self.obs_data:
            assert (self.nb_nodes - self.max_nb_target) * self.nb_interventions >= self.nb_nodes

    def _create_folder(self):
        """Create folders

        fname(str): path """
        try:
            os.mkdir(self.folder)
        except OSError:
            print(f"Cannot create the folder: {self.folder}")

    def _initialize_dag(self):
        if self.verbose:
            print(f'Sampling the DAG #{self.i_dataset}')
        self.generator = gen.DagGenerator(self.mechanism,
                                         noise=self.noise,
                                         noise_coeff=self.noise_coeff,
                                         cause=self.cause,
                                         npoints=self.nb_points,
                                         nodes=self.nb_nodes,
                                         expected_density=self.expected_degree,
                                         rescale=self.rescale)

        self.generator.init_variables()
        self.generator.save_dag_cpdag(self.folder, i+1)

    def _pick_targets(self, nb_max_iteration=100000):
        nodes = np.arange(self.nb_nodes)
        not_correct = True
        i = 0

        if(self.max_nb_target == 1):
            # if  self.nb_interventions == self.nb_nodes:
            #     intervention = np.random.choice(self.nb_nodes, self.nb_interventions - 1, replace=False)
            # else:
            intervention = np.random.choice(self.nb_nodes, self.nb_interventions, replace=False)
            targets = [[i] for i in intervention]

        else:
            while(not_correct and i < nb_max_iteration):
                targets = []
                not_correct = False
                i += 1

                # pick targets randomly
                for _ in range(self.nb_interventions):
                    nb_targets = np.random.randint(self.min_nb_target, self.max_nb_target+1, 1)
                    intervention = np.random.choice(self.nb_nodes, nb_targets, replace=False)
                    targets.append(intervention)

                # apply rejection sampling
                if self.cover and not self._is_covering(nodes, targets):
                    not_correct = True
                if self.conservative and not self.obs_data and not self._is_conservative(nodes, targets):
                    not_correct = True

            if i == nb_max_iteration:
                raise ValueError("Could generate appropriate targets. \
                                 Exceeded the maximal number of iterations")

            for i, t in enumerate(targets):
                targets[i] = np.sort(t)

        return targets

    def _is_conservative(self, elements, lists):
        for e in elements:
            conservative = False

            for l in lists:
                if e not in l:
                    conservative = True
                    break
            if not conservative:
                return False
        return True

    def _is_covering(self, elements, lists):
        return set(elements) == self._union(lists)

    def _union(self, lists):
        union_set = set()

        for l in lists:
            union_set = union_set.union(set(l))
        return union_set

    def _save_data(self, i, data, mask=None):
        if mask is None:
            data_path = os.path.join(self.folder, f'data{i+1}.npy')
            np.save(data_path, data)
        else:
            data_path = os.path.join(self.folder, f'data_interv{i+1}.npy')
            np.save(data_path, data)

            data_path = os.path.join(self.folder, f'intervention{i+1}.csv')
            with open(data_path, 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerows(mask)

    def generate(self, intervention=False):
        if self.generator is None:
            self._initialize_dag()

        if intervention:
            data = np.zeros((self.nb_points, self.nb_nodes))

            # one-liner taken from https://stackoverflow.com/questions/20348717/algo-for-dividing-a-number-into-almost-equal-whole-numbers/20348992
            num = self.nb_points
            if self.obs_data:
                div = self.nb_interventions + 1
            else:
                div = self.nb_interventions
            points_per_interv = [num // div + (1 if x < num % div else 0)  for x in range (div)]
            # points_per_interv = int(self.nb_points / self.nb_interventions)
            mask_intervention = []
            nb_env = self.nb_interventions

            # randomly pick targets
            target_list = self._pick_targets()

            # perform interventions
            for j in range(nb_env):
                targets = target_list[j]

                # generate the datasets with the given interventions
                dataset, _ = self.generator.intervene(self.intervention_type,
                                                      targets,
                                                      points_per_interv[j],
                                                      self.struct_interv_distr)
                self.generator.reinitialize()

                # put dataset and targets in arrays
                if j == 0:
                    start = 0
                else:
                    start = np.cumsum(points_per_interv[:j])[-1]
                end = start + points_per_interv[j]
                data[start:end, :] = dataset
                mask_intervention.extend([targets for i in range(points_per_interv[j])])

            # setting without interventions
            if self.obs_data:
                j = self.nb_interventions
                self.generator.change_npoints(points_per_interv[j])
                dataset, _ = self.generator.generate()
                targets = []

                # put dataset and targets in arrays
                if j == 0:
                    start = 0
                else:
                    start = np.cumsum(points_per_interv)[-2]
                data[start:, :] = dataset
                mask_intervention.extend([targets for i in range(points_per_interv[j])])

            if self.rescale:
                scaler = StandardScaler()
                scaler.fit_transform(data)
            # data = rotate(data, 45)
            self._save_data(self.i_dataset, data, mask_intervention)

        else:
            self.generator.change_npoints(self.nb_points)
            data, _ = self.generator.generate()

            if self.rescale:
                scaler = StandardScaler()
                scaler.fit_transform(data)
            self._save_data(self.i_dataset, data)


def rotate(point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in degree.
    """

    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    rotated_point = np.dot(point, R)

    return rotated_point


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mechanism', type=str, default='anm',
                        help='Type of mechanism use to generate the data ')
    parser.add_argument('--intervention-type', type=str, default='structural',
                        help='Type of intervention (structural or parametric)')
    parser.add_argument('--initial-cause', type=str, default='uniform',
                        help='Distribution of initial causes')
    parser.add_argument('--struct-interv-distr', type=str, default='uniform',
                        help='Distribution of intervened node')
    parser.add_argument('--noise', type=str, default='variable_gaussian',
                        help='Distribution of noises')
    parser.add_argument('--noise-coeff', type=float, default=0.4,
                        help='Noise coefficient')
    parser.add_argument('--nb-nodes', type=int, default=10,
                        help='Number of nodes in the DAGs')
    parser.add_argument('--expected-degree', type=float, default=1,
                        help='Expected number of edges per node')
    parser.add_argument('--nb-dag', type=int, default=10,
                        help='Number of DAGs to generate dataset from')
    parser.add_argument('--nb-points', type=int, default=1000,
                        help='Number of points per dataset')
    parser.add_argument('--rescale', action='store_true',
                        help='Rescale the variables')
    parser.add_argument('--suffix', type=str, default='GP',
                        help='Suffix that will be added at the \
                        end of the folder name')

    # Arguments related to interventions
    parser.add_argument('--intervention', action='store_true',
                        help='if True, generate data with interventions')
    parser.add_argument('--nb-interventions', type=int, default=3,
                        help='number of interventional settings')
    parser.add_argument('--obs-data', action='store_true',
                        help='if True, the first setting is generated without any interventions')
    parser.add_argument('--min-nb-target', type=int, default=1,
                        help='minimal number of targets per setting')
    parser.add_argument('--max-nb-target', type=int, default=3,
                        help='maximal number of targets per setting')
    parser.add_argument('--conservative', action='store_true',
                        help='if True, make sure that the intervention family is conservative: i.e. that all nodes have not been intervened in at least one setting.')
    parser.add_argument('--cover', action='store_true',
                        help='if True, make sure that all nodes have been intervened on at least in one setting.')

    arg = parser.parse_args()
    generator = dataset_generator(arg.mechanism, arg.initial_cause,
                                  arg.intervention_type, arg.struct_interv_distr, arg.noise, arg.noise_coeff, arg.nb_nodes,
                                  arg.expected_degree, arg.nb_points, arg.suffix, arg.rescale,
                                  arg.obs_data, arg.nb_interventions, arg.min_nb_target, arg.max_nb_target,
                                  arg.conservative, arg.cover)

    for i in range(arg.nb_dag):
        generator.i_dataset = i
        generator.generator = None

        # first, generate the observational data
        print("Generating the observational data...")
        generator.generate(False)

        # then generate interventional data
        if arg.intervention:
            print("Generating the interventional data...")
            generator.generate(True)
