# Tool for quickly plotting things from commandline.

## Quick start
```console
$pplot.py data.txt 0 1 2
```
Plots columns 0 1 2 from file `data.txt`, assumes the file contains data separated by white spaces.

```console
$pplot.py data*.txt -c 0 1 2
```
Plots columns 0 1 2 from all files matched by `data*.txt`.

One file can be piped in through stdin (use `-p` to select which columns to plot from piped data):
```console
$ cat data.txt | pplot.py -p 1 2 3
```

For help:
```console
$pplot.py -h
```

## Plot Expressions:
Plot expressions from the read in data. The `i`th column of `n`th file is denoted as `f[n]_[i]` in the expressions. All numpy array functions and constants should be usable. E.g.
```console
$pplot.py file0.txt file1.txt -expr "( f0_0 - mean(f0_0) )/pi"  "sin(f1_1) - cos(f0_1)"
```
If expressions contain mixed files are assumed to have the same length.

## Using in scripts
Can be used in scripts to edit a figure further. See `pp_script_example.py`.

## Custom data reader:
It is possible to write a custom data reader method. The method should be put into `data_readers/` folder with the following format: The file containing the code for the reader `[reader_name].py` should contain a function with the same name and the signature `def [reader_name](pp: pplot, fname: str)`. The function is expected to return a tuple `( data: numpy.ndarray , labels: list\[str\] )`. The data is expected to be 2d array with shape `(num_rows, num_cols)`. Labels can be an empty list [], if not it is expected to be a list of strings with length num_cols containing the labels. You can do what ever in the method even read a custom binary format.

Then to use a specific reader use the command `-rf [reader_name]`.

See `data_readers/data_reader_example.py` for example:
```console
$pplot.py data.txt 1 -x0 -rf data_reader_example
```

## Dependencies:
### Python3:
- numpy      (tested on v1.22.3)
- sympy      (tested on v1.9)
- matplotlib (tested on v3.5.1)

### Optional:
#### GNU utils:
- `grep`
- `wc`

Should be installed and in your $PATH.
For now, used in default data reader for finding `tag_data_start` `tag_data_start` and `tag_header` tags.
Can be disabled by setting defaults as `tag_data_start=None`, `tag_data_start=None`, `tag_header=None`.


## Warning:
The script does not do much of error handling. For example, if you pass a column that does not exit the script will just crash for now and print not that useful python stack trace.

There are bound to be bugs.