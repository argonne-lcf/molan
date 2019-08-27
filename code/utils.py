import os
import json
from tqdm.autonotebook import tqdm
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit import DataStructs
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
import numpy as np
import numpy.matlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import SVG, display, HTML
import cairosvg

LATEX_TABLE = r'''\documentclass{{standalone}}
\usepackage{{booktabs}}
\usepackage{{multirow}}
\usepackage{{graphicx}}
\usepackage{{xcolor,colortbl}}

\begin{{document}}

{}

\end{{document}}
'''


def get_curve_elbow(values):
    """https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve """
    n = len(values)
    allCoord = np.vstack((range(n), values)).T

    first_v = allCoord[0]
    line_vec = allCoord[-1] - allCoord[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))

    vec_from_first = allCoord - first_v

    scalar_product = np.sum(
        vec_from_first * np.matlib.repmat(line_vec_norm, n, 1), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel

    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))

    elbow_index = np.argmax(dist_to_line)
    return elbow_index


def display_svg(svg):
    return display(SVG(svg))


def save_latextable(df, adir, name):
    """ddataframe to latex"""
    filename = os.path.join(adir, '{}.tex'.format(name))
    a_str = df.to_latex(multicolumn=True, multirow=True, escape=False)
    with open(filename, 'w') as a_file:
        a_file.write(LATEX_TABLE.format(a_str))


def save_molgrid(svg, adir, name):
    filename = os.path.join(adir, '{}.{}'.format(name, 'svg'))
    with open(filename, 'w') as afile:
        afile.write(svg)
    filename = os.path.join(adir, '{}.{}'.format(name, 'png'))
    cairosvg.svg2png(bytestring=svg, write_to=filename)

def discrete_colormap(colors):
    return mpl.colors.ListedColormap(colors)

def save_figure(adir, name):
    for ext in ['png', 'svg']:
        filename = os.path.join(adir, '{}.{}'.format(name, ext))
        plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)


def header_str(a_str, n=80):
    """Returns a string formatted as a header."""
    return '{{:=^{:d}}}'.format(n).format(' ' + a_str + ' ')


def header_html(astr, level=1):
    html_code = '<h{}>{}</h{}>'.format(level, astr, level)
    return display(HTML(html_code))


def header_legend(label, label_space=2, **legend_kwargs):
    fmt = '{{:<{:d}s}}'.format(label_space) + ' ({:>5s}, {:>5s}, {:>5s})'
    header = fmt.format(label, 'n', 'mean', 'std')
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ex_labels = [header]
    ex_handles = [mpl.patches.Rectangle(
        (0, 0), 0, 0, alpha=0.0) for i in ex_labels]
    if 'prop' not in legend_kwargs.keys():
        legend_kwargs['prop'] = {'family': 'monospace', 'size': 'x-small'}
    return plt.legend(handles=ex_handles + handles, labels=ex_labels + labels, **legend_kwargs)


def legend_stats_label(label, y, label_space=2):
    fmt = '{{:<{:d}s}}'.format(label_space) + ' ({:>5s}, {:>5s}, {:>5s})'
    n = str(len(y))
    mu = '{:.1f}'.format(np.mean(y))
    std = '{:.1f}'.format(np.std(y))
    return fmt.format(label, n, mu, std)


def plot_settings():
    """Plot settings"""

    sns.set_style("white")
    sns.set_style('ticks')

    sns.set_context("paper", font_scale=2.25)
    sns.set_palette(sns.color_palette('bright'))
    # matplotlib stuff
    plt_params = {
        'figure.figsize': (10, 8),
        'lines.linewidth': 3,
        'axes.linewidth': 2.5,
        'savefig.dpi': 300,
        'xtick.major.width': 2.5,
        'ytick.major.width': 2.5,
        'xtick.minor.width': 1,
        'ytick.minor.width': 1,
        'legend.framealpha': 1.0,
        'figure.dpi': 86
    }
    mpl.rcParams.update(plt_params)
    return


def isosmiles(smiles):
    """Convert smiles to isosmiles."""
    mol = AllChem.MolFromSmiles(smiles)
    new_smi = AllChem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return new_smi


def get_all_jsons(path, suffixes=None):
    """Retrieve jsons from a given path and suffixes."""
    files = []
    json_files = [f for f in os.listdir(path) if f.endswith('.json')]
    if suffixes:
        for filename in json_files:
            suffix = filename.split('_')[-1].replace('.json', '')
            if suffix in suffixes:
                files.append(filename)
    else:
        files = json_files
    return files


def read_json_data(files, path, data_fn, ):
    """Read jsons with a function and return list of information."""

    data = []
    for file in tqdm(files):
        filename = os.path.join(path, file)
        with open(filename) as afile:
            sample_dict = json.load(afile)
            sample_dict['filename'] = file
            data.append(data_fn(sample_dict))
    return data


def diverse_mols_indexes(mol_list, n_pick, radius=4, seed=42):
    fps = [GetMorganFingerprint(mol, radius) for mol in mol_list]
    picker = MaxMinPicker()
    n_fps = len(fps)

    def fp_distance(i, j): return 1 - \
        DataStructs.DiceSimilarity(fps[i], fps[j])
    indexes = picker.LazyPick(fp_distance, n_fps, n_pick, seed=seed)
    return indexes
