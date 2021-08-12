import sys
import re
import colorama
from termcolor import colored
from pandas import DataFrame
from editdistance import eval

from library.utils import *
from library.const import *

from library.retrain import retrain
from library.validate import validate
from library.evaluation.evaluation import evaluate
from library.data.dataset_manager import load_datasets


def compute_results(model, test_samples, log_path):
    predictions = validate(model, test_samples)
    results = list()

    for sample, (label, pred, prob) in zip(test_samples, predictions):
        err = eval(pred, label)

        results.append([sample.img_path, label, pred, prob, err])

    DataFrame(data=np.asarray(results)).to_csv(log_path)

    return evaluate(predictions)


def compute_correlation(log_path, model, train_all, test_A, test_B):
    test_A_stats = ([], [])
    test_B_stats = ([], [])
    test_AB_stats = ([], [])
    words = []
    paths = []
    texts = []

    # A regex matcher to extract the dataset name from a given image path.
    set_matcher = re.compile('.*?/\d+-(.*?)-\d+-\d+-\d+\.png')

    total = len(train_all)
    num_runs = 10
    step_size = 10

    train_all = sorted(train_all, key=lambda x: x.gt_text)
    random.seed(420)

    # Compute the initial wer and cer on the three test sets.
    base_A_wer, base_A_cer = evaluate(validate(model, test_A))
    base_B_wer, base_B_cer = evaluate(validate(model, test_B))
    base_AB_wer, base_AB_cer = evaluate(validate(model, test_A + test_B))

    print(colored('Test A: {:.2f}% WER {:.2f}% CER'.format(base_A_wer, base_A_cer), 'green'))
    print(colored('Test B: {:.2f}% WER {:.2f}% CER'.format(base_B_wer, base_B_cer), 'green'))
    print(colored('Test AB: {:.2f}% WER {:.2f}% CER'.format(base_AB_wer, base_AB_cer), 'green'))

    for run in range(num_runs):
        random.shuffle(train_all)

        test_A_stats[0].append([base_A_wer])
        test_A_stats[1].append([base_A_cer])

        test_B_stats[0].append([base_B_wer])
        test_B_stats[1].append([base_B_cer])

        test_AB_stats[0].append([base_AB_wer])
        test_AB_stats[1].append([base_AB_cer])

        words.append([])
        paths.append([])
        texts.append([])

        for step in range(total // step_size + 1):
            # Take all samples from step 1 to step 2.
            train_samples = train_all[:(step+1)*step_size]

            print(colored('Retraining: {} to {}'.format(0, (step+1)*step_size), 'green'))
            retrain(model, train_samples, test_A, verbose=True)

            word_log_path = log_path + 'results_{}_{}_{}.csv'

            print(colored('Domain A:', 'red'))
            test_A_wer, test_A_cer = compute_results(model, test_A, log_path=word_log_path.format(run, step, 'A'))

            print(colored('Domain B:', 'red'))
            test_B_wer, test_B_cer = compute_results(model, test_B, log_path=word_log_path.format(run, step, 'B'))

            print(colored('Domain AB:', 'red'))
            test_AB_wer, test_AB_cer = compute_results(model, test_A + test_B, log_path=word_log_path.format(run, step, 'AB'))

            test_A_stats[0][-1].append(test_A_wer)
            test_A_stats[1][-1].append(test_A_cer)

            test_B_stats[0][-1].append(test_B_wer)
            test_B_stats[1][-1].append(test_B_cer)

            test_AB_stats[0][-1].append(test_AB_wer)
            test_AB_stats[1][-1].append(test_AB_cer)

            words[-1] += [s.gt_text for s in train_samples]
            paths[-1] += [s.img_path for s in train_samples]
            texts[-1] += [set_matcher.match(s.img_path).group(1) for s in train_samples]

            # Load the weights of the baseline model.
            print(colored('Loading weights', 'green'))
            model.load_checkpoint(Const.baseline)

    DataFrame(data=np.asarray(test_A_stats[0])).to_csv(log_path + '/' + 'test_A_wer.csv')
    DataFrame(data=np.asarray(test_A_stats[1])).to_csv(log_path + '/' + 'test_A_cer.csv')

    DataFrame(data=np.asarray(test_B_stats[0])).to_csv(log_path + '/' + 'test_B_wer.csv')
    DataFrame(data=np.asarray(test_B_stats[1])).to_csv(log_path + '/' + 'test_B_cer.csv')

    DataFrame(data=np.asarray(test_AB_stats[0])).to_csv(log_path + '/' + 'test_AB_wer.csv')
    DataFrame(data=np.asarray(test_AB_stats[1])).to_csv(log_path + '/' + 'test_AB_cer.csv')

    DataFrame(data=np.asarray(words)).to_csv(log_path + '/' + 'words.csv')
    DataFrame(data=np.asarray(paths)).to_csv(log_path + '/' + 'paths.csv')
    DataFrame(data=np.asarray(texts)).to_csv(log_path + '/' + 'texts.csv')


def main():
    colorama.init()
    set_seeds()

    ids = sys.argv[1:]
    texts = ['brown', 'cedar', 'domain_A_train', 'domain_B_train', 'nonwords']

    model = load_model(Const.baseline)

    if not os.path.isdir('../logs/experiment0/'):
        os.mkdir('../logs/experiment0/')

    for id in ids:
        # The requests for the datasets for writer A.
        train_request = '+'.join([id + '-' + text for text in texts])
        test_A_request = id + '-' + 'domain_A_test'
        test_B_request = id + '-' + 'domain_B_test'

        # Load all necessary datasets.
        train_all, test_A, test_B = load_datasets(train_request, test_A_request, test_B_request)

        if not os.path.isdir('../logs/experiment0/' + id):
            os.mkdir('../logs/experiment0/' + id)

        try:
            compute_correlation('../logs/experiment0/' + id + '/', model, train_all, test_A, test_B)
        except Exception as e:
            with open('../logs/experiment0/' + id + '/error.txt', 'w') as f:
                f.write(str(e))


if __name__ == '__main__':
    main()
