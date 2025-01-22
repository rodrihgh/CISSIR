import seaborn as sns
from cycler import cycler

palette = ["#5c8acf", "#88CDA2", "#842e5c", "#cb3154", "#8cd0f3"]
def_cycler = (cycler(color=palette[:4]) + cycler(linestyle=['-', '--', ':', '-.']))


def paper_style(line_cycler=True):
    sns.set_palette(palette)
    rc = {'figure.figsize': (3.5, 2.65), #'backend': 'pgf',
          'text.usetex': True, 'pgf.texsystem': 'pdflatex',
          'font.family': 'serif', 'font.serif': []}
    if line_cycler:
        rc['axes.prop_cycle'] = def_cycler
    sns.set_theme(context='paper', style='whitegrid', palette=palette, rc=rc)


def talk_style(line_cycler=True):
    sns.set_palette(palette)
    rc = {'figure.figsize': (3.5, 2.65)}
    if line_cycler:
        rc['axes.prop_cycle'] = def_cycler
    sns.set_theme(context='paper', style='whitegrid', palette=palette, font="Calibri", rc=rc)
