from sys import stdout
from termcolor import colored


def set_up_logger(log_path=None, verbose=True):
    logger = Logger()

    try:
        if verbose:
            logger.add_stream(stdout)

        if log_path:
            logger.add_stream(open(log_path, 'w'))
    except Exception as e:
        print(colored('Logger error: ' + str(e), 'yellow'))

    return logger


class Logger:
    def __init__(self):
        self.streams = list()
        self.files = list()

    def add_file(self, file):
        self.streams.append(file)
        self.files.append(file)

    def add_stream(self, stream):
        self.streams.append(stream)

    def log(self, *args, sep=' ', end='\n'):
        for stream in self.streams:
            stream.write(sep.join(str(arg) for arg in args) + end)

    def close(self):
        for file in self.files:
            file.close()
