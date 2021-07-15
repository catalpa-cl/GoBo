import sys

from library.data.dataset_manager import load_datasets


def compare(set_A, set_B):
    for a, b in zip(set_A, set_B):
        if a.gt_text != b.gt_text:
            print(a.gt_text, ' vs ', b.gt_text)
            return False

    return True


def main():
    texts = ['brown', 'cedar', 'domain_A_train', 'domain_B_train', 'domain_A_test', 'domain_B_test', 'nonwords']

    train_sets = list()

    for id in sys.argv[1:]:
        # The requests for the datasets for writer A.
        train_request = '+'.join([id + '-' + text for text in texts])
        test_A_request = id + '-' + 'domain_A_test'
        test_B_request = id + '-' + 'domain_B_test'

        # Load all necessary datasets.
        train_all, test_A, test_B = load_datasets(train_request, test_A_request, test_B_request)

        train_all = sorted(train_all, key=lambda x: x.gt_text)

        train_sets.append((id, train_all))

    correct_set = train_sets[0][1]

    for id, train_set in train_sets[1:]:
        if not compare(correct_set, train_set):
            print("ID", id, "is incorrect!")
            print([s.gt_text for s in train_set])
            print([s.gt_text for s in correct_set])
        else:
            print("ID", id, "is correct!")

        hist = dict()

        for word in [s.gt_text for s in train_set]:
            if word not in hist:
                hist[word] = 0

            hist[word] += 1

        print(set([s.gt_text for s in correct_set]) - set([s.gt_text for s in train_set]))
        print(set([s.gt_text for s in train_set]) - set([s.gt_text for s in correct_set]))


if __name__ == '__main__':
    main()