from data_utils import locate_dataset, write_to_json
from tqdm import tqdm
from typing import Dict, List, Optional
import os
import random


class Sampler:

    """
    A class to sample a subset of registration problems from the registration graph
    and store them in a JSON file. 

    The problems can be either curated based on worm posture similarity or randomly
    selected depending on the `diy_registration_problems` flag.
    """

    def __init__(
        self,
        dataset_dict: Dict[str, List[str]],
        problem_dict: Optional[Dict[str, Dict[str, List[str]]]] = None,
        diy_registration_problems: bool = False
    ):
        """
        Initialize the Sampler object.

        Args:
            dataset_dict (Dict[str, List[str]]): Dictionary containing datasets for 
                training, validation, and testing. Problems are sampled from the 
                `registration_problems.txt` file of each dataset, which contains 
                problems curated from the registration graph. The dictionary format is:
                {
                    "train": ["YYYY-MM-DD-X", ...],
                    "valid": ["YYYY-MM-DD-X", ...],
                    "test": ["YYYY-MM-DD-X", ...]
                }

            problem_dict (Optional[Dict[str, Dict[str, List[str]]]]): Optional dictionary 
                of specific problems to be sampled for each dataset. This allows for 
                subsetting problems within each dataset, formatted as:
                {
                    "train": {
                        "2022-01-09-01": ["102to675", "104to288", ...],
                        ...
                    },
                    "valid": {
                        "2022-02-16-04": ["1022to1437", "1029to1372", ...],
                        ...
                    },
                    "test": {
                        "2022-04-14-04": ["1013to1212", "1021to1049", ...],
                        ...
                    }
                }

            diy_registration_problems (bool): If False, registration problems are sampled 
                based on worm posture similarity from the registration graph. If True, 
                problems are sampled randomly without these constraints.
        """
        if problem_dict == None:
            self.dataset_dict = dataset_dict
            self.problem_dict = None
        elif dataset_dict == None:
            self.problem_dict = problem_dict
            self.dataset_dict = None

        if diy_registration_problems:
            self.neuron_roi_path = "/data1/prj_register/deepreg_labels"
        else:
            self.neuron_roi_path = None

        self.output_dict = {
                "train": dict(),
                "valid": dict(),
                "test": dict()
        }

    def __call__(self, output_file_name: str, num_problems: int = -1):
        """
        Create a JSON file that contains all the regsitration problems.

        Args:
            output_file_name (str): The name of the output file to be created.
            num_problems (int, optional): The number of problems to include. If set to -1, 
                all available problems are used. Default is -1.

        Example:
            >>> dataset_dict = {
            >>>     'train': ['2023-08-07-01'],
            >>>     'valid': ['2023-08-07-16'],
            >>>     'test': ['2022-04-14-04']
            >>> }
            >>> sampler = Sampler(dataset_dict)
            >>> output_file_name = 'registration_problems.json'
            >>> sampler(output_file_name)
        """
        if self.problem_dict == None:

            if self.neuron_roi_path is not None:
                self.create_problems_to_register(output_file_name, num_problems)

            else:
                self.sample_from_datasets(output_file_name, num_problems)

        elif self.dataset_dict == None:
            self.sample_from_problems(output_file_name, num_problems)

    def sample_from_problems(self, output_file_name: str, num_problems: int):
        """
        Samples a subset of registration problems from the provided problem dictionary 
        and saves them in a JSON file.

        Args:
            output_file_name (str): The name of the output file where the sampled problems 
                will be saved.
            num_problems (int): The number of problems to sample per dataset. 

        This method iterates through the problem dictionary, randomly sampling the specified
        number of problems from each dataset type (train, valid, test), and stores the 
        sampled problems in a dictionary. The result is saved as a JSON file.

        Example:
            >>> problem_dict = {
            >>>     'train': {
            >>>         '2022-01-09-01': ['102to675', '104to288', ...], 
            >>>         ...
            >>>     },
            >>>     'valid': {
            >>>         '2022-02-16-04': ['1022to1437', '1029to1372', ...], 
            >>>         ...
            >>>     },
            >>>     'test': {
            >>>         '2022-04-14-04': ['1013to1212', '1021to1049', ...], 
            >>>         ...
            >>>     }
            >>> }
            >>> sampler = Sampler(dataset_dict=None, problem_dict=problem_dict)
            >>> output_file_name = 'sampled_problems.json'
            >>> sampler(output_file_name, num_problems=100)
        """
        for dataset_type, dataset_names in self.problem_dict.items():

            for dataset_name in tqdm(dataset_names):

                sampled_problems = random.sample(
                        self.problem_dict[dataset_type][dataset_name],
                        num_problems)
                self.output_dict[dataset_type][dataset_name] = sampled_problems

        write_to_json(self.output_dict, output_file_name)

    def create_problems_to_register(self, output_file_name: str, num_problems: int):

        """
        Randomly samples pairs of distinct time points to register for each dataset 
        and saves them to a JSON file.

        This method selects `num_problems` pairs of time points (a fixed and a moving 
        time point) from neuron ROI data files for each dataset. The registration problem 
        is represented as a string in the format 'moving_time_point_to_fixed_time_point'. 

        Args:
            output_file_name (str): The name of the output file where the sampled 
                registration problems will be saved.
            num_problems (int): The number of registration problems (time point pairs) 
                to sample for each dataset.

        Raises:
            AssertionError: If the `diy_registration_problems` flag is not set to True, 
                i.e., `self.neuron_roi_path` is not defined.

        Example:
            >>> dataset_dict = {
            >>>     'train': ['2023-08-07-01'],
            >>>     'valid': ['2023-08-07-16'],
            >>>     'test': ['2022-04-14-04']
            >>> }
            >>> sampler = Sampler(dataset_dict, diy_registration_problems=True)
            >>> output_file_name = 'random_registration_problems.json'
            >>> sampler(output_file_name, num_problems=100)
        """

        assert self.neuron_roi_path is not None, \
            "Need to set `diy_registration_problems = True`"

        for dataset_type, dataset_names in self.dataset_dict.items():

            for dataset_name in tqdm(dataset_names):

                nrrd_files = os.listdir(
                        f"{self.neuron_roi_path}/{dataset_name}/img_roi_neurons")
                time_points = [file.split(".")[0] for file in nrrd_files]
                fixed_time_points = random.choices(time_points, k = num_problems)
                moving_time_points = random.choices(
                        list(filter(lambda item: item not in fixed_time_points,
                            time_points)), k = num_problems
                )
                self.output_dict[dataset_type][dataset_name] = [
                        f"{moving_t}to{fixed_t}" for moving_t, fixed_t in
                        list(zip(moving_time_points, fixed_time_points))
                ]
        write_to_json(self.output_dict, output_file_name)

    def sample_from_datasets(self, output_file_name: str, num_problems: int):

        """
        Samples registration problems from datasets and saves them in a JSON file.

        This method retrieves all available registration problems for each dataset 
        and samples a subset according to the specified number of problems. If 
        `num_problems` is set to -1, a custom sampling scheme is applied.

        Args:
            output_file_name (str): The name of the output file where the sampled problems 
                will be saved.
            num_problems (int): The number of problems to sample per dataset. If set to -1, 
                a custom sampling scheme defined by `_sample_registration_problems` is used 
                to determine the number of samples.

        Example:
            >>> dataset_dict = {
            >>>     'train': ['2023-08-07-01'],
            >>>     'valid': ['2023-08-07-16'],
            >>>     'test': ['2022-04-14-04']
            >>> }
            >>> sampler = Sampler(dataset_dict)
            >>> output_file_name = 'sampled_registration_problems.json'
            >>> sampler(output_file_name, num_problems=100)
        """
        for dataset_type, dataset_names in self.dataset_dict.items():

            for dataset_name in tqdm(dataset_names):
                problems = self._get_all_problems(dataset_name)
                # sample accordings to the defined scheme if the number of
                # samples per dataset is not specified
                if num_problems == -1:
                    sampled_problems = self._sample_registration_problems(problems)
                else:
                    sampled_problems = random.sample(problems, num_problems)
                self.output_dict[dataset_type][dataset_name] = sampled_problems

        write_to_json(self.output_dict, output_file_name)

    def _get_all_problems(self, dataset_name: str) -> List[str]:

        """ Retrieve all registration problems from a dataset. """

        dataset_path = locate_dataset(dataset_name)
        if os.path.exists(f"{dataset_path}/registration_problems.txt"):
            lines = open(
                f"{dataset_path}/registration_problems.txt", "r").readlines()
            problems = [line.strip().replace(" ", "to") for line in lines]
        else:
            raise FileNotFoundError(
                f"Can't find {dataset_path}/registration_problems.txt")
        return problems

    def _sample_registration_problems(
        self,
        problems: List[str],
        cutoff: int = 600
    ) -> List[str]:

        """ Sample a subset of registration problems based on time interval lengths. """

        # sort problems by the length of moving and fixed time interval
        interval_to_problems_dict = dict()
        for problem in problems:

            interval = abs(
                        int(problem.split("to")[0]) -
                        int(problem.split("to")[1])
                    )
            if interval not in interval_to_problems_dict.keys():
                interval_to_problems_dict[interval] = [problem]
            else:
                interval_to_problems_dict[interval].append(problem)

        sampled_problems = []
        for interval, problems in interval_to_problems_dict.items():
            if interval > cutoff:
                sampled_problems += random.sample(
                        problems, int(0.8 * len(problems)))
            else:
                sampled_problems += random.sample(
                        problems, int(0.5 * len(problems)))

        return sampled_problems


def main():

    train_datasets = ['2022-01-06-01']
    valid_datasets = ['2022-01-06-01']

    # train_datasets = ['2022-01-17-01']
    # valid_datasets = ['2023-07-28-04']
    test_datasets = []
    dataset_dict = {
        "train": train_datasets,
        "valid": valid_datasets,
        "test": test_datasets}

    sampler = Sampler(dataset_dict, diy_registration_problems=False)
    sampler("registration_problems_gfp", num_problems=5)

    # sampler("registration_problems_demo", num_problems=5)


if __name__ == "__main__":
    main()
