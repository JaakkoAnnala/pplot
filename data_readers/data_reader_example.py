import numpy as np
import subprocess
import io


def data_reader_example(pp, fname):
    # reads data from file fname in some way implemented here

    # Example:
    # the lines in file 'fname' looks something like:
    # [some text] x = 123.123, [some more text] y = 456.456, [and text]
    dat = subprocess.run(['sed', '-n', '-e', 's/.*x\s*=\s*\([^\s]*\),.*y\s*=\s*\([^\s]*\),.*/\\1 \\2/p',fname ],
          stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    dat  = dat.stdout.decode('utf-8')
    datf = io.StringIO(dat)
    dat = np.genfromtxt(datf)

    labels = ["x","y"]
    # returns the data and optionally labels
    return (dat, labels)

    # now this can be used as: $pplot.py data.txt 1 -x0 -rf data_reader_example