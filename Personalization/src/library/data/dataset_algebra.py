import re

from library.data.loader import DataLoader


class LoadOperator:
    def __init__(self, path, name):
        self.path = path
        self.name = name

    def eval(self):
        return DataLoader(self.path, self.name).samples


class ResizeOperator:
    def __init__(self, dataset, num_samples):
        self.dataset = dataset
        self.num_samples = num_samples

    def eval(self):
        return self.dataset.eval()[:self.num_samples]


class CombineOperator:
    def __init__(self, datasets):
        self.datasets = datasets

    def eval(self):
        result = self.datasets[0].eval()

        for dataset in self.datasets[1:]:
            result += dataset.eval()

        return result


def compile_expression(expression, files):
    unary_matcher = '([0-9a-zA-Z-_]+)(\[([0-9]+)\])?'
    unaries = list()

    for match in re.finditer(unary_matcher, expression):
        dataset = match.group(1)
        path, name = files[dataset]

        operator = LoadOperator('../data/' + path, name)

        if match.group(2) is not None:
            operator = ResizeOperator(operator, int(match.group(2)[1:-1]))

        unaries.append(operator)

    return CombineOperator(unaries)



