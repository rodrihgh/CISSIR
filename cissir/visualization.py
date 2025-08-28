import seaborn as sns
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from cissir.utils import Path, plot_path

# Color shortcuts
b = "#5c8acf"   # Blue
g = "#88CDA2"   # Green
p = "#842e5c"   # Purple
r = "#cb3154"   # Red
c = "#8cd0f3"   # Cyan

palette = [b, g, p, r, c]
def_cycler = (cycler(color=palette[:4]) + cycler(linestyle=['-', '--', ':', '-.']))
def_figsize = (3.5, 2.65)
tex_preamble = r'\usepackage{amsfonts}'


def paper_style(line_cycler=True, figsize=def_figsize):
    sns.set_palette(palette)
    rc = {'figure.figsize': figsize,  # 'backend': 'pgf',
          'text.usetex': True, 'pgf.texsystem': 'pdflatex',
          'pgf.preamble': tex_preamble,
          'text.latex.preamble': tex_preamble,
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


def inline_annotation(x, y, text, x_label=None, y_label=None, rotation="curve",
                      background_color="white", background_alpha=0.5,
                      horizontal_alignment='center', vertical_alignment='center',
                      axis=None, **kwargs):

    if x_label is not None:
        text_arg = np.argmin(np.abs(x_label - x))
    elif y_label is not None:
        text_arg = np.argmin(np.abs(y_label - y))
    else:
        raise ValueError("Either x_label or y_label must be specified.")

    if rotation == "curve":
        y_diff = y[text_arg + 1] - y[text_arg - 1]
        x_diff = x[text_arg + 1] - x[text_arg - 1]
        rotation_deg = np.rad2deg(np.arctan(y_diff / x_diff))
    else:
        rotation_deg = rotation

    if axis is None:
        axis = plt.gca()
    axis.text(x[text_arg], y[text_arg], text, ha=horizontal_alignment, va=vertical_alignment,
              bbox=dict(facecolor=background_color, alpha=background_alpha,
                        edgecolor=None, boxstyle="Round,pad=0.0"),
              rotation=rotation_deg, **kwargs)


def _myLogFormat(y, pos):
    mant_ticks = [1, 2, 3, 5, 7]
    # Find the number of decimal places required
    log_y = np.log10(y)
    ord_mag = np.floor(log_y).astype(int)
    mantissa = np.round(np.power(10, log_y - ord_mag), 1)
    if mantissa in mant_ticks:
        # Insert that number into a format string
        decimalplaces = np.maximum(-ord_mag, 0)  # =0 for numbers >=1
        formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
        # Return the formatted tick label
        return formatstring.format(y)
    else:
        return ''


def format_log_ticks(axis, which='both', minor=True):
    if which == 'both':
        axis.xaxis.set_major_formatter(ticker.FuncFormatter(_myLogFormat))
        axis.yaxis.set_major_formatter(ticker.FuncFormatter(_myLogFormat))
        if minor:
            axis.xaxis.set_minor_formatter(ticker.FuncFormatter(_myLogFormat))
            axis.yaxis.set_minor_formatter(ticker.FuncFormatter(_myLogFormat))
    elif which == 'x':
        axis.xaxis.set_major_formatter(ticker.FuncFormatter(_myLogFormat))
        if minor:
            axis.xaxis.set_minor_formatter(ticker.FuncFormatter(_myLogFormat))
    elif which == 'y':
        axis.yaxis.set_major_formatter(ticker.FuncFormatter(_myLogFormat))
        if minor:
            axis.yaxis.set_minor_formatter(ticker.FuncFormatter(_myLogFormat))
    else:
        raise ValueError("Invalid value for 'which': {}".format(which))

    return


def save(fig, fname, save_dir=None, save_format="pgf", bbox_inches='tight', **kwargs):
    if isinstance(fname, Path) or "/" in fname:
        save_path = fname
    elif save_dir is not None:
        save_path = save_dir / Path(fname)
    else:
        save_path = plot_path/fname
    if "." not in save_format:
        save_format = "." + save_format

    fig.savefig(save_path.with_suffix(save_format), bbox_inches=bbox_inches, **kwargs)
