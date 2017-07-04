import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, show

file = open('../data/paren-train.txt','r').read()
string = file[:100]
# data = pd.read_csv('paran-data-df.csv', index_col=0)
# data.index.name = 'chars'
# data.columns.name = 'cell'

data = np.loadtxt('paren-data.csv',delimiter=',')

output_file("patch.html")




p = figure(plot_width=1100, plot_height=400)

p.multi_line([range(100), range(100)], [list(data[:,2]), list(data[:,1])],
             color=["firebrick", "navy"], alpha=[0.8, 0.3], line_width=4)

show(p)