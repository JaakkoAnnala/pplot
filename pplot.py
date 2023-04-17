#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm
import matplotlib.ticker as mtick
import argparse
import glob
import shlex

import re
import subprocess

from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr

from data_readers import *

def genfromtxt(*args, **kwargs):
    res = np.genfromtxt(*args, **kwargs)
    # annoying hack to get same shape for the read in array even with only 1 row.
    if len(res.shape) == 1:
        tmp=[]
        for i in range(0,len(res)):
            tmp.append( [ res[i] ] )
        res = np.array(tmp)
        res=np.transpose(res)
    return res

def merge_ranges(ranges):
    ranges.sort(key=lambda x: x[0])
    idx = 0
    for i in range(1, len(ranges)):
        if (ranges[idx][1] >= ranges[i][0]):
            ranges[idx][1] = max(ranges[idx][1], ranges[i][1])
        else:
            idx += 1
            ranges[idx] = ranges[i]
    return ranges

def is_in_ranges(i, ranges):
    for r in ranges:
        if i >= r[0] and i <= r[1]: return True
    return False

def txt_find_data(pp, fname):
    starts = []
    ends   = []
    # just use grep for now.. should be faster than anything in python.
    if pp.tag_data_start:
        starts = subprocess.run(['grep', '-n', pp.tag_data_start,fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
        starts = starts.stdout.splitlines()
    if pp.tag_data_end:
        ends   = subprocess.run(['grep', '-n', pp.tag_data_end  ,fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    starts = [int(s.decode("utf-8").split(":")[0]) for s in starts]
    if ends:
        ends   = ends.stdout.splitlines()
        ends   = [int(s.decode("utf-8").split(":")[0]) for s in ends]
    if not starts and not ends: return []
    
    # arrange the ranges
    # [a,-1] means from a to end of file
    ranges = []
    ls = len(starts)
    le = len(ends)
    if ls != le:
        print("WARNING: found " +("more `" if ls > le else "less `") + str(pp.tag_data_start) + "`'s than `"+ str(pp.tag_data_end) + "`'s in file "+fname,file=sys.stderr)
        print("         Attempting to merge ranges...",file=sys.stderr)
    if le <= 0: return [[starts[0], -1]]
    elif ls <= 0: return [[0,ends[0]]]
    else :      ranges = [[starts[0],ends[0]]]
    si=1
    ei=1
    for i in range(1, max(ls,le)):
        s = -1
        e = -1
        
        while si < ls and is_in_ranges(starts[si], ranges):
            si+=1
        if si < ls:
            s = starts[si]

        while ei < le and (is_in_ranges(ends[ei], ranges) or ends[ei] < s):
            ei+=1
        if ei < le:
            e = ends[ei]

        if s != -1: ranges.append([s,e])
        si+=1
        ei+=1
    return ranges

# default data reader
def get_data_txt(pp, fname):
    labels = []
    if pp.tag_header or pp.args.head: labels = get_data_labels(pp,fname)
    ranges = txt_find_data(pp, fname)
    #print(ranges)
    if not ranges:
        return ( genfromtxt(fname, skip_header=0, invalid_raise=False, 
                                   comments=pp.line_comment, delimiter=pp.data_separator) 
                , labels )


    # use wc for now
    n_lines = subprocess.run(['wc', '-l',fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    n_lines = int(n_lines.stdout.decode('utf-8').split(' ')[0])
    footer = 0 if ranges[0][1] == -1 else n_lines - ranges[0][1]
    
    dat = genfromtxt(fname, skip_header=ranges[0][0], skip_footer=footer, invalid_raise=False,
                            comments=pp.line_comment, delimiter=pp.data_separator)
    
    for i in range(1,len(ranges)):
        footer = 0 if ranges[i][1] == -1 else n_lines - ranges[i][1]
        dat2 = genfromtxt(fname, skip_header=ranges[i][0], skip_footer=footer, invalid_raise=False,
                                 comments=pp.line_comment, delimiter=pp.data_separator)
        dat = np.concatenate( (dat,dat2) )
    return ( dat , labels )

def get_data_labels(pp, fname):
    header = ""
    tag_h = pp.tag_header
    if pp.args.head:
        try:
            row = int(pp.args.head)
            header = subprocess.run(['sed', '-n', f'{row}p',fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
            header = header.stdout.decode('utf-8')
            header = header.strip().split(' ')
            if header == ['']:
                header = []
                print(f"WARNING: the given line for -head was empty: {fname}:{row}",file=sys.stderr)
            return header
        except:
            tag_h = pp.args.head
    header = subprocess.run(['sed', '-n', '-e', f's/^.*{tag_h}\s*//p',fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    header = header.stdout.decode('utf-8')
    header = header.strip().split(' ')
    if header == ['']:
        header = []
        print(f"WARNING: did not find header tag `{tag_h}` in file: {fname}")
    return header

class pplot:
    # defaults
    _data_reader_f_=get_data_txt   # default data reader func
    plt_fmt        = '-'           # matplotlib plot format
    # defaults for get_data_txt
    tag_data_start = "#data_start" # if set to None should disable searching for the data and just read everything in the file
    tag_data_end   = "#data_end"   # 
    tag_header     = "#header"     # if set to None does not try to get header as a default, the option -head can still be used
    line_comment   = "#"           # line comment when reading data
    data_separator = None          # None defaults to any number of white space

    # Internals:
    parser = None
    args = None
    #plot
    fig=None 
    axs=None # all axi
    ax =None # axis currently being plotted to
    n_subx=0 # number of subplots in x dir
    n_suby=0 # number of subplots in y dir
    subplt_x=0 # x-coord of subplot for the next plotting
    subplt_y=0 # y-coord of subplot for the next plotting
    last_plot=None # ref to last plot, for some color selection stuff
    # aa plot
    aa_data=None
    aa_d=None
    weights=None
    labels=None
    current_fi=0 # current file index being read

    data       = [] # contains all the data read from any type of files in order they appear in the command line arguments
    fname_data = [] # filenames corresponding to the data above, in same order
    data_cols  = [] # columns to plot for each file, in same order as above
    data_labels= [] #
    piped      = [] # data piped from stdin
    piped_cols = [] # columns to plot from piped data
    def __init__(self, arg_str=None):
        self.parser = argparse.ArgumentParser(description='Plot data from files with matplotlib. Data can also be piped through stdin.') 
        self.parser.usage = f"{sys.argv[0]} FILE1 [columns to plot] FILE2 [columns to plot] ... [options]"
        p = self.parser.add_argument
        p('files',type=str,nargs='*' ,help="Text file paths. Can be interleaved with numbers signifying which columns to plot for each file. E.g. file1.dat 1 2 file2.dat 3 1. The option `-c` overrides these selections to enable the use of wildcards. E.g. '*.dat -c 1 2' , plots columns 1 & 2 from all files found matching *.dat.")
        p('-c'   ,type=int,nargs='+' ,help="Column      : List of columns to plot for all given files. Overrides columns given in the file list.")
        p('-x'   ,type=int           ,help="X-axis      : Which column to use as the x axis. Default is to use row number as x coord.")
        p('-e'   ,type=int           ,help="Error bar   : Plot error bars from given column.")
        p('-p'   ,type=int,nargs='+' ,help="Piped Column: List of columns to plot for piped in data.")
        p('-s'   ,type=str           ,help='Separator   : Option for default data reader. Default is any number of whitespaces.')
        p('-head',type=str,nargs='?' ,const=self.tag_header
                                     ,help=f"Header      : Option for default data reader. If no argument given try to get header from first line that contains `{self.tag_header}`. If given input is convertible to integer read the header from that line. Otherwise find the line with the given string and use the preceding string to get the labels.")
        p('-logy',action='store_true',help="Log y axis.")
        p('-logx',action='store_true',help="Log x axis.")
        p('-lny' ,action='store_true',help="Base e Log y axis.")
        p('-lnx' ,action='store_true',help="Base e Log x axis.")
        p("-hist",type=int,nargs='?' ,const=-1
                                     ,help="Plot data as a histogram. Optionally give number of bins. Default use sqrt(data_length) bins.")
        p("-norm",action="store_true",help="Normalize histogram such that the integral equals 1.")
        p("-fold",action="store_true",help="Fold histogram to range [0,1].")
        p("-we"  ,type=int           ,help="Weight the histogram with the given data idx giving weights exp(-x[i]+x[0]).")
        p('-subf',action='store_true',help="Plot each file to separate sub plot")
        p('-subc',action='store_true',help="Plot each column to separate sub plot")
        p("-expr",type=str,nargs='+' ,help="Plot evaluated expression for columns. Data from n:th file and i:th row is expressed as 'f[n]_[i]', i:th row of piped in input is expressed as 'p[i]'. All numpy array functions should be usable. If '-x' is set assumes that the X-axis is the same between files and files have the same number of rows and uses the X-axis data from the first file. Example: -expr 'f0_1 - mean(f0_1) + f1_2*0.5' ")
        p('-b'   ,type=int           ,help="Plot data binned in bins of the given size.")
        p('-be'  ,action='store_true',help="Show standard deviation/sqrt(bin_size) error bars for -b Plot.")
#TODO        p('-i'   ,action='store_true',help="Return to interactive python shell.")
        p('-int' ,type=float,nargs=2 ,help="Integrate. Get the area under the folded histogram between a range.")
        p('-rf'  ,type=str           ,help="Read Function: Name of custom read data function to be used. Custom read functions are put into the 'data_readers' folder that is in the same folder as this script. Custom data reader file and method should have the same name. E.g. data_readers/custom_reader.py should contain a function named 'custom_reader' with the signature 'def custom_reader(pp :pplot, file_path: str)' the function is expected to return a tuple ( data: numpy.ndarray , labels: list[str] ). The data is expected to be 2d array with shape (num_rows, num_cols). Labels can be an empty list [], if not it is expected to be a list of strings with length num_cols containing the labels.")
        p('-surf',type=str,nargs='?' ,const='grid_data' #TODO: flattened array
                                     ,help="Surface: plots surface from file assuming the file contains an X-Y grid of Z-values.")
        p('-cmap',type=str           ,help="Set matplotlib.cm colormap for `-surf`. https://matplotlib.org/stable/gallery/color/colormap_reference.html")
        
        p('-title',type=str          ,help="Set the figure title to given string. Supports LaTeX, note that dollar sign must be escaped '\$'.")
        p('-xlab'  ,type=str         ,help="Set the x label. Supports LaTeX, note that dollar sign must be escaped '\$'.")
        p('-ylab'  ,type=str         ,help="Set the y label. Supports LaTeX, note that dollar sign must be escaped '\$'.")
        p('-ps'   ,type=str          ,help="Set the matplotlib plot format style string e.g. 'o-'. Note '-' has to be last if present.")
#TODO        p('-db'  ,type=int           ,help="Begin at given row.")
#TODO        p('-de'  ,type=int           ,help="End at given row.")
        p("-ts",type=str             ,help="Tag Start : Option for default data reader.")
        p("-te",type=str             ,help="Tag End   : Option for default data reader.")


        p=None
        if arg_str is not None: self.args = self.parser.parse_args(shlex.split(arg_str))
        else: self.args = self.parser.parse_args()
        #print(f"args: {self.args}")
        

        # set up the data reader func
        if self.args.rf:
            try: 
                mod = globals()[self.args.rf]
            except:
                print(f"ERROR: did not find {self.args.rf} in globals(). Check that data_readers has a file with that name. Check help for `-rf` command.",file=sys.stderr)
                exit(1)
            try:
                reader_f = getattr(mod, self.args.rf)
            except:
                print(f"ERROR: cound not find method {self.args.rf}() in module {self.args.rf}. The file name and method name of data reader implementation should match. Check help for `-rf` command.",file=sys.stderr)
                exit(1)
            self._data_reader_f_ = reader_f

        if self.args.s is not None: self.data_separator = self.args.s
        if self.args.ts is not None: self.tag_data_start= self.args.ts
        if self.args.te is not None: self.tag_data_end  = self.args.te

        self.subplt_x=0
        self.subplt_y=0

        self.get_piped()
        self.read_all_data()
        
        #nothing to do
        if len(self.piped)==0 and len(self.data)==0:
            self.parser.print_usage()
            exit(0)

        self.plot_all()

        if arg_str is None:
            plt.legend()
            plt.show()


    def read_all_data(self):
        if not self.args.files: return
        # parse filenames and possible indices
        #  fname1.dat 1 2 3 fname2.dat 4 3 2 etc..
        first = True
        cols = []
        for s in self.args.files:
            try: 
                col_i = int(s)
                cols.append(col_i)
            except:
                # new file reset cols arr
                if not first: self.data_cols.append( cols )
                else: first = False
                cols = []
                fname = s
                self.fname_data.append( fname )
        self.data_cols.append( cols )
        
        for i in range(0, len(self.fname_data)):
            f = self.fname_data[i]
            self.current_fi = i #useful in some data readers
            if self.args.rf: dh = self._data_reader_f_(self,  f )
            else: dh = self._data_reader_f_( f ) #python is weird
            self.data.append(dh[0])
            self.data_labels.append(dh[1])


    def get_label(self, file_i, col_i):
        #print(self.data_labels[file_i])
        #print(f"col_i {col_i}")
        dname = "" if (len(self.data_labels[file_i])==0 or len(self.data_labels[file_i])<col_i) else self.data_labels[file_i][col_i]
        return f"{self.fname_data[file_i]}:{col_i} {dname}"

    def plot_binned_stdev(self, bin_size, data_x, data_y, label=None):
        nbins = int(len(data_x)/bin_size)
        n, _ = np.histogram(data_x, bins=nbins)
        sy, _ = np.histogram(data_x, bins=nbins, weights=data_y)
        sy2, edges = np.histogram(data_x, bins=nbins, weights=data_y*data_y)
        mean = sy / n
        std = np.sqrt(sy2/n - mean*mean)
        #axis.plot(data_x, data_y, 'o')
        if self.args.be:
            self.ax.errorbar((edges[1:] + edges[:-1])/2, mean, yerr=std/np.sqrt(bin_size), fmt='-', capsize=5, label=label)
        else:
            self.ax.plot((edges[1:] + edges[:-1])/2, mean,   label=label)

    def plot_hist(self, *data,label=None, weights=None):
        #print(weights)
        c = next(plt.gca()._get_lines.prop_cycler)['color'] # get the color cyclers next color
        c = matplotlib.colors.to_rgb(c)
        h = self.args.hist
        bins = h if (h and h>0) else int(np.sqrt(len(data[0])))
        if self.args.fold:
            self.last_plot=self.ax.hist(data[0]%1.0,label=label, bins=bins,
                              color=c, histtype='stepfilled',
                              edgecolor=(*c,1), facecolor=(*c,0.5),
                              weights=weights, density=self.args.norm)
            lp = self.last_plot
            if self.args.int:
                i1=0
                i2=0
                for i in range(0,len(lp[1])):
                    if i1==0 and lp[1][i] > self.args.int[0]: i1 = i
                    if i2==0 and lp[1][i] > self.args.int[1]:
                        i2 = i
                        break
                if i2 == 0: i2 = len(lp[1])
                area = np.sum( np.diff(lp[1][i1:i2]) * lp[0][i1:(i2-1)] )
                print(f"{label} [{i1},{i2}] integral [{self.args.int[0]}:{self.args.int[1]}] = {area}")
        else:
            self.last_plot=self.ax.hist(data[0],label=label, bins=bins,
                              color=c, histtype='stepfilled',
                              edgecolor=(*c,1), facecolor=(*c,0.5),
                              weights=weights, density=self.args.norm)

    def plot_expr(self,expr):
        f = parse_expr(expr)
        f_syms = list(f.free_symbols)
        arg_arr = []
        for sym in f_syms:
            sym_str = str(sym)
            if sym_str[0] == 'f':
                idxs = sym_str[1:].split("_")
                if len(idxs) != 2: continue # TODO: can there be some name collision here. something of form f(.*)_(.*) not intended to be data name f[int]_[int]...
                fi = int(idxs[0])
                di = int(idxs[1])
                arg_arr.append( self.data[fi][:,di] )
            elif sym_str[0] == 'p':
                try:
                    di = int(sym_str[1:])
                    arg_arr.append( self.piped[:,di] )
                except:
                    pass

        #idxs_x = [ int(str(f_syms[i])[1]) for i in range(0,len(f_syms)) ]
        np_f = lambdify(f_syms, f, 'numpy')
        expr_data = np_f( *arg_arr )#*[data[:,i] for i in idxs]  )
        #if self.args.x is not None:
        self.plot_one(0,0 ,expr_data=expr_data, label=expr)
        #else:
        #    self.plot_one(expr_data, label=expr)
        #print(expr_data)

    def get_piped(self):
        if sys.stdin.isatty(): return
        self.piped = genfromtxt(sys.stdin, dtype=np.double, comments=self.line_comment)
        print(f"piped in data shape: {self.piped.shape}")
        if self.args.p:
            self.piped_cols = self.args.p

    def plot_one(self, file_i, col_i, expr_data=None, piped_data=False, label=None):
        #assert( (expr_data is not None) and (piped_data is not None), "plot_one should not get both expr_data and piped_data at the same time..")
        if self.args.subc: self.set_axis_to_plot()

        if self.args.surf:
            # assuming column = x, row = y
            dat = self.data[file_i] if not piped_data else self.piped
            shape = dat.shape
            x = np.arange(0,shape[1])
            y = np.arange(0,shape[0])
            x,y = np.meshgrid(x,y)
            cmap = cm.viridis if not self.args.cmap else getattr(cm, self.args.cmap)
            tmp = self.ax.plot_surface(x,y,dat, cmap=cmap, label=label)
            # https://github.com/matplotlib/matplotlib/issues/4067 ... annoying
            tmp._edgecolors2d = tmp._edgecolor3d
            tmp._facecolors2d = tmp._facecolor3d
            return

        # x-axis
        x = None 
        if (self.args.x is not None): 
            x = self.data[file_i][:,self.args.x] if not piped_data else self.piped[:,self.args.x]
        # y-axis
        y = None
        if expr_data is not None: y = expr_data
        elif piped_data: y = self.piped[:,col_i]
        else: y = self.data[file_i][:,col_i]

        # set error bar column
        err = None 
        if (self.args.e is not None): 
            err = self.data[file_i][:,self.args.e] if not piped_data else self.piped[:,self.args.e]
        # weights
        weights = None
        if self.args.we is not None and self.args.hist: # dont bother if not hist plot
            #print(f"computing we using: {self.args.we}")
            weights = self.data[file_i][:,self.args.we] if not piped_data else self.piped[:,slef.args.we]
            w0 = weights[0]
            weights = np.exp(-weights+w0)


        # axes settings
        if self.args.logx: self.ax.set_xscale('log')
        if self.args.logy: self.ax.set_yscale('log')
        if self.args.lnx:
            self.ax.set_xscale('log',base=np.e)
            self.ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,pos: f"{np.log(x)}"))
        if self.args.lny:
            self.ax.set_yscale('log',base=np.e)
            self.ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,pos: f"{np.log(x)}"))
        if self.args.xlab is not None: self.ax.set_xlabel(self.args.xlab)
        if self.args.ylab is not None: self.ax.set_ylabel(self.args.ylab)

        
        # plot histogram
        if self.args.hist:
            self.plot_hist(y, label=label, weights=weights)
            return
        
        # plot binned data
        if self.args.b and self.args.b > 0:
            if not x: x = np.arange(0,len(y))
            self.plot_binned_stdev(self.args.b, x, y, label=label)
            return
        
        # plot normal
        fmt = self.plt_fmt if not self.args.ps else self.args.ps
        if err is not None: 
            if x is None: x = np.arange(0,len(y))
            self.ax.errorbar(x, y, fmt=fmt,yerr=err,capsize=5)
        else:
            if x is not None: self.ax.plot(x, y,fmt,label=label)
            else: self.ax.plot(y,fmt,label=label)
        
    def set_axis_to_plot(self):
        #if self.ax: self.ax.legend()
        # figure out sub plot stuff
        if (not self.args.subf) and (not self.args.subc):
            self.ax = self.axs
            return
        #print(f"subxy {self.n_subx},{self.n_suby}    px,py = {self.subplt_x},{self.subplt_y}")
        self.ax = self.axs[self.subplt_x] if (self.n_suby==1) else self.axs[self.subplt_y, self.subplt_x]
        self.subplt_x+=1
        if self.subplt_x==self.n_subx:
            self.subplt_x=0
            self.subplt_y+=1
        #self.ax.text(0,0.05,f"{label}",transform=self.ax.transAxes)

    def calc_n_sub_plots(self):
        n_expr = 0 if not self.args.expr else len(self.args.expr)
        n_piped = len(self.piped_cols)
        
        if self.args.subf:
            ret = len(self.data)
            if len(self.piped) > 0: ret += 1
            return ret
        
        if self.args.c:
            mult = len(self.fname_data) + n_piped
            return sum(1 for i in self.args.c if i>=0)*mult + n_expr
        nsub=0
        for c in self.data_cols:
            nsub += len(c)
        return nsub + n_piped + n_expr

    def plot_all(self):

        # set up sub plot stuff
        sp_kw = {}
        if self.args.surf: sp_kw["projection"]="3d"
        len_data = len(self.data)
        n_subs = self.calc_n_sub_plots()
        #print(f"n_sub_plots = {n_subs}")
        if self.args.subf or self.args.subc:
            if n_subs < 3:
                self.fig, self.axs = plt.subplots(2,1, subplot_kw=sp_kw)
                self.n_subx=2
                self.n_suby=1
            else:
                n = int(np.ceil(np.sqrt(n_subs)))
                self.fig, self.axs = plt.subplots(n,n, subplot_kw=sp_kw)
                self.n_subx=n
                self.n_suby=n
        else:
            self.fig, self.axs = plt.subplots(1,1, subplot_kw=sp_kw)
            self.ax = self.axs

        # set figure subtitle
        if self.args.title: self.fig.suptitle(self.args.title, usetex=True, fontsize=16)

        # if plotting surface just make all cols arrays = [0] 
        # so we plot one surface from each file only once and at least once
        if self.args.surf:
            if len(self.piped) > 0: self.piped_cols = [0]
            self.data_cols = [ [0] for i in self.fname_data]
            if self.args.c is not None: self.args.c = [0]


        # do the plotting
        for i in range(len(self.data)):
            # -c option overrides single file column selections:
            # to enable something like: $pp *.dat -c 1 2 3
            if self.args.subf: self.set_axis_to_plot()
            cols = self.data_cols[i] if not self.args.c else self.args.c
            for col in cols:
                    self.plot_one(i,col, label=self.get_label(i, col))
                    self.ax.legend()


        
        if len(self.piped_cols) > 0:
            if self.args.subf: self.set_axis_to_plot()
            for col in self.piped_cols:
                if col == -1: continue
                #err = None if not self.args.e else self.data[i][:,self.args.e]
                self.plot_one(0,col,piped_data=True, label=f"pipe:{col}")

        if self.args.expr:
            for expr in self.args.expr:
                self.plot_expr(expr)
        
        # TODO: make an option
        # print data labels for reference
        for fi in range(0,len(self.data_labels)):
            print(f"{self.fname_data[fi]}:")
            for di in range(0,len(self.data_labels[fi])):
                print(f"    {di}: {self.data_labels[fi][di]}")




if __name__ == "__main__":
    pp = pplot()



