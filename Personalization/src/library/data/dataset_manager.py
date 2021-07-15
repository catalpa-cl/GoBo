import json
from sys import argv
from colorama import init
from termcolor import colored

from library.data.dataset_algebra import compile_expression


def request_datasets(datasets):
    """
    Asks the user to set some datasets.

    :param datasets: The datasets to ask for.
    :return: A dict containing the requested datasets.
    """
    # Print the hierarchy of all datasets.
    data_manager = DatasetManager('../data/datasets.json')
    print(str(data_manager), '\n')

    # Load all file paths.
    files = data_manager.get_files()
    output = dict()

    for idx, dataset in enumerate(datasets):
        if idx + 1 < len(argv):
            reply = argv[idx + 1]
        else:
            # Request a dataset.
            print(colored('The {} set: '.format(dataset), 'red'), end='')
            reply = input()

        parse_tree = compile_expression(reply, files)
        output[dataset] = parse_tree.eval()

    return output


def load_datasets(train, valid, test):
    data_manager = DatasetManager('../data/datasets.json')

    # Load all file paths.
    files = data_manager.get_files()

    parse_tree = compile_expression(train, files)
    train = parse_tree.eval()

    parse_tree = compile_expression(valid, files)
    valid = parse_tree.eval()

    parse_tree = compile_expression(test, files)
    test = parse_tree.eval()

    return train, valid, test


def load_dataset(dataset):
    data_manager = DatasetManager('../data/datasets.json')

    files = data_manager.get_files()
    parse_tree = compile_expression(dataset, files)
    dataset = parse_tree.eval()

    return dataset


def load_file(path):
    with open(path, 'r') as f:
        return f.read()


class File:
    def __init__(self, name, path):
        """
        :param name: The name of the dataset.
        :param path: The path of the dataset.
        """
        self.name = name
        self.path = path

    def __str__(self):
        """
        :return: Returns the name of the dataset.
        """
        return self.name


class Dir:
    def __init__(self, path, subdirs, files):
        """
        :param path: The path of this directory.
        :param subdirs: The list of subdirectories of this directory.
        :param files: The files in this directory.
        """
        self.path = path
        self.subdirs = subdirs
        self.files = files

    def get_files(self):
        """
        :return: Return all files in this directory.
        """
        files = {f.name: ('', f.path) for f in self.files}

        for subdir in self.subdirs:
            files = {**files, **subdir.get_files()}

        for name, (path, file_name) in files.items():
            files[name] = (self.path + '/' + path, file_name)

        return files

    def __str__(self):
        return 'Directory "{}"'.format(self.path)


class DatasetManager:
    def __init__(self, path):
        """
        :param path: The path of the json file that defines the hierarchy.
        """
        json_hierarchy = json.loads(load_file(path))
        self.root_dir = self.set_up_hierarchy(json_hierarchy)

    def set_up_hierarchy(self, dir):
        """
        Recursively set up the hierarchy of the datasets.

        :param dir: The hierarchy as a json file.
        :return: Returns the root directory.
        """
        subdirs = list()
        files = list()

        if 'dirs' in dir:
            for subdir in dir['dirs']:
                subdirs.append(self.set_up_hierarchy(subdir))

        if 'files' in dir:
            for file in dir['files']:
                files.append(File(file['name'], file['path']))

        new_dir = Dir(dir['path'], subdirs, files)

        return new_dir

    def get_files(self):
        return self.root_dir.get_files()

    def __str__(self):
        """
        Return the hierarchy as a nicely formatted string.

        :return: The hierarchy as a string.
        """
        init()

        stack = [(self.root_dir, 0)]
        s = ''

        while len(stack) != 0:
            dir, depth = stack.pop()
            s += ('  ' * depth) + '--> ' + str(dir) + ':\n'

            for file in dir.files:
                s += colored(('  ' * (depth + 3)) + str(file) + '\n', 'green')

            for subdir in dir.subdirs:
                stack.append((subdir, depth + 1))

        return s


def main():
    manager = DatasetManager('../../data/datasets.json')
    init()
    print(manager)
    print(manager.get_files())


if __name__ == '__main__':
    main()
