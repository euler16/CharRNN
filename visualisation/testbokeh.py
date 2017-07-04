from math import pi
import pandas as pd

from bokeh.io import show
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, BasicTicker, PrintfTickFormatter,ColorBar
from bokeh.models import FuncTickFormatter
from bokeh.plotting import figure

file = open('../data/paren-train.txt','r').read()
string = file[:100]
data = pd.read_csv('paran-data-df.csv', index_col=0)
data.index.name = 'chars'
data.columns.name = 'cell'

index = {i:string[i] for i in range(len(data.index))}

seq = [str(i) for i in data.index]
cell = list(data.columns)

df = pd.DataFrame(data.stack(), columns=['value']).reset_index()
colors = ["#ffffd9","#edf8b1","#c7e9b4","#7fcdbb","#41b6c4","#1d91c0","#225ea8","#253494","#081d58"]

colors.reverse()
mapper = LinearColorMapper(palette=colors, low=2, high=-2)# df.value.min(), high=df.value.max())
source = ColumnDataSource(df)
TOOLS = "hover,pan,reset,wheel_zoom"

p = figure(title="GRU Hidden State Activations",  x_range=seq, y_range=list(reversed(cell)), x_axis_location="above",
			plot_width=1100, plot_height=400,
			tools=TOOLS, toolbar_location='below')

p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "8pt"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = pi / 3
p.xaxis.formatter = FuncTickFormatter(code="""
												var labels = %s;
												return labels[tick];
											"""%index)

p.rect(x="chars", y="cell", width=1, height=1, source=source, fill_color={'field': 'value', 'transform': mapper},
									line_color=None)

p.select_one(HoverTool).tooltips = [('value', '@value')]
show(p)      # show the plot