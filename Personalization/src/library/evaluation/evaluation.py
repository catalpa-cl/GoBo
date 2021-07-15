from editdistance import eval
from library.evaluation.logger import set_up_logger


def evaluate(results, log_path=None, verbose=False):
    word_errors = 0
    char_errors = 0
    total_chars = 0

    if log_path:
        log_path = log_path + 'evaluation.txt'

    logger = set_up_logger(log_path=log_path)

    for label, pred, prob in results:
        # Compute the Levenshtein Distance between prediction and label.
        dist = eval(pred, label)

        if dist != 0:
            word_errors += 1

        char_errors += dist
        total_chars += len(label)

        if verbose:
            # Print the result to the console.
            result = '[OK]' if dist == 0 else '[ERR {}]'.format(dist)
            logger.log(result, label, '->', pred, '({:.2f}%).'.format(prob * 100))

    wer = word_errors / len(results) * 100
    cer = char_errors / total_chars * 100

    logger.log('{:.2f}% WER {:.2f}% CER'.format(wer, cer))
    logger.close()

    return wer, cer
