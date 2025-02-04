"""
Text, debug and I/O utility functions.
"""

import sys
from pathlib import Path

base_path = Path(__file__).parents[1].absolute()
if str(base_path) not in sys.path:
    sys.path.append(str(base_path))
plot_path = base_path/"plots"
res_path = base_path/"results"


class PrintBuffer:
    def __init__(self, print_input=True, sep="\n"):
        self.print_input = print_input
        self._buffer = []
        self.sep = sep

    def print(self, string=None, clear=True):
        if string is None:
            print(*self._buffer, sep=self.sep)
            if clear:
                self.clear()
        else:
            self._buffer.append(string)
            if self.print_input:
                print(string)

    def clear(self):
        self._buffer.clear()


class TextTable:
    def __init__(self, header, align=None, pad=10, row_sep="\n", col_sep="|"):
        n_header = len(header)
        self._num_cols = n_header

        if not isinstance(pad, list):
            pad = [pad] * n_header

        pad = [max(p - 2, len(h)) for p, h in zip(pad, header)]
        self._pad = pad
        self._buffer = []
        self._row_sep = row_sep
        self._col_sep = col_sep

        if align is None:
            align = ['left'] * n_header
        self._align = align

        self.row(header)
        self._align_row(align)

    def row(self, columns):
        num_cols = len(columns)
        if num_cols < self._num_cols:
            columns += [''] * (self._num_cols - num_cols)
        elif num_cols > self._num_cols:
            raise ValueError("Too many columns")
        row = []
        align = self._align
        pad = self._pad
        col_sep = self._col_sep
        for a, c, p in zip(align, columns, pad):
            if a == 'left':
                row.append(f"{c:<{p}}")
            elif a == 'right':
                row.append(f"{c:>{p}}")
            elif a == 'centered':
                row.append(f"{c:^{p}}")
            else:
                raise ValueError(f"Unknown alignment '{a}'")
        self._buffer.append(col_sep+" "+f" {col_sep} ".join(row)+" "+col_sep)

    def _align_row(self, align):
        align_row = []
        pad = self._pad
        col_sep = self._col_sep
        for a, p in zip(align, pad):
            if a == 'left':
                align_row.append(f"{':':-<{p+2}}")
            elif a == 'right':
                align_row.append(f"{':':->{p+2}}")
            elif a == 'centered':
                align_row.append(f"{':':-<{p+1}}:")
            else:
                raise ValueError(f"Unknown alignment '{a}'")
        self._buffer.append(col_sep.join([""]+align_row+[""]))

    def output(self):
        return self._row_sep.join(self._buffer)

    def print(self):
        print(self.output())
