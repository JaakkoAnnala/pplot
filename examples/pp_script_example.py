import numpy as np
import matplotlib.pyplot as plt
# use some other matplotlib config file
# plt.style.use(absolute_path_to_matplotlibrc)

# have the py_plot script somewhere that python can find it 
# or just do this:
# import sys
# sys.path.append(your_absolute_path_to_pplot)
import pplot

# wildcards (i.e. something like *.dat) have to be expanded manually here, since it is done by the shell...
pp = pplot.pplot("test/fs.dat 1 5 2 test/fs_tags.dat 0  -expr 'log(f0_0)' 'sin(f1_0*20)' ")
gca = pp.fig.gca()

gca.set_xlabel("time",usetex=True,fontsize=16)
gca.set_ylabel("$\\sin(x),\\cos(x)$, etc.",usetex=True,fontsize=16)
gca.set_title("Some functions",usetex=True,fontsize=16)

gca.figure.set_size_inches(w=8,h=7)
gca.set_ylim(-2,2)

gca.get_lines()[0].set_label('$\\sin(x)$')      
gca.get_lines()[1].set_label('$\\cos(x)+$noise')
gca.get_lines()[2].set_label('$\\tan(x)$')      
gca.get_lines()[3].set_label('$x$')             
gca.get_lines()[4].set_label('$\\log(x)$')      
gca.get_lines()[5].set_label('$\\sin(20x)$')    

gca.legend()

plt.subplots_adjust(
top=0.898,
bottom=0.111,
left=0.108,
right=0.929,
hspace=0.2,
wspace=0.2
)
plt.show()