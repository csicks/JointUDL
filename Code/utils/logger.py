import sys
import os


class Logger(object):
    '''
    Class for writing logs for the program
    '''
    def __init__(self, filename, stream=sys.stdout, print_label=True):
        '''
        Initialization.

        Args:
            filename: name/path of log file
            stream: other output stream (to print on screen)
        '''
        if os.path.exists(filename):
            os.remove(filename)
        self.filename = filename
        self.stream = stream
        self.print_label = print_label

    def write(self, info):
        '''
        Write string into log file.

        Args:
            info: string to write into log file
        '''
        file = open(self.filename, 'a')
        file.write(info)
        if self.print_label:
            self.stream.write(info)
        file.close()
