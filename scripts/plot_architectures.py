"""Architecture figures of the learned models, drawn with PlotNeuralNet (LaTeX / TikZ).

    python -m scripts.plot_architectures                      # every figure
    python -m scripts.plot_architectures --model viewgraph    # one figure
    python -m scripts.plot_architectures --no-compile         # only write the .tex
    CUDA_VISIBLE_DEVICES=0 python -m scripts.plot_architectures --sinograms   # re-render the sinogram images first

- arch_unet: SinogramUNet, shared by UNet2D and UNet2dHLCC.
- arch_unet_hlcc_p4: UNet2dHLCC followed by the HLCC moment regression P4.
- arch_unet_ushape, arch_unet_hlcc_p4_ushape: the same two figures with the U-Net drawn as a U (larger).
- arch_viewgraph: ViewGraphNet, shared by the GCN and the SNN (only the fixed transport differs).

The U-Net figures show the input and outputs of test sample --sample, rendered by --sinograms into
results/figures/architectures/sinograms/ (needs ASTRA, a GPU and the checkpoints); without them, plain boxes are
drawn. Channel counts and sizes come from DEFAULT_MODEL_CONFIG and GeometryConfig. PlotNeuralNet is not vendored:

    git clone --depth 1 https://github.com/HarisIqbal88/PlotNeuralNet tools/PlotNeuralNet

The .tex sources go to results/figures/architectures/ and are compiled with tectonic or pdflatex (--latex); the
PDFs go to results/figures/.
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from src.config import GeometryConfig
from src.models.factory import DEFAULT_MODEL_CONFIG
from src.paths import PROJECT_ROOT, RESULTS_DIR

PLOTNN_ROOT = PROJECT_ROOT / "tools" / "PlotNeuralNet"
PDF_DIR = RESULTS_DIR / "figures"
TEX_DIR = PDF_DIR / "architectures"
SINO_DIR = TEX_DIR / "sinograms"

if not (PLOTNN_ROOT / "pycore/tikzeng.py").exists():
    sys.exit(f"PlotNeuralNet not found in {PLOTNN_ROOT}: clone it first (see the docstring of this script).")
sys.path.insert(0, str(PLOTNN_ROOT))
from pycore.tikzeng import to_begin, to_connection, to_end, to_head, to_skip  # noqa: E402

# The figures of the thesis (the _ushape ones and arch_viewgraph) have a natural width of at most 15.4 cm, the text
# width, so that no text prints below 7 pt when they are included at that width: labels in \footnotesize (8 pt) or
# \small (9 pt), and the sub/superscripts of 8 and 9 pt formulas at 7 pt instead of the default 6 / 5 pt. The
# geometry of PlotNeuralNet (1 unit = 1 cm) is shrunk by a per-figure unit, which leaves the text at its size (see
# document()).
COLORS = r"""
\DeclareMathSizes{8}{8}{7}{7}
\DeclareMathSizes{9}{9}{7}{7}
\newcommand{\LabelFont}{\footnotesize}
\usetikzlibrary{calc,decorations.pathreplacing,fit}
\def\IOColor{rgb:white,4;black,1}
\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ActColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\UpColor{rgb:blue,2;green,1;black,0.3}
\def\CopyColor{rgb:white,1;black,3}
\def\OutColor{rgb:magenta,5;black,7}
\def\GraphColor{rgb:blue,5;green,2.5;white,5}
\def\GraphActColor{rgb:blue,5;green,5;white,3}
\def\SumColor{rgb:blue,5;green,15}
\def\DCColor{rgb:red,2;blue,5;white,2}
\def\HLCCColor{rgb:green,3;blue,1;white,8}
"""

# "bus": path from the input to the residual / data-consistency nodes, below the network. "op": flowchart boxes.
# Thinner lines and arrows than PlotNeuralNet's, in proportion to the smaller drawing.
BUS_STYLE = r"""
\tikzstyle{connection}=[very thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
\tikzstyle{copyconnection}=[very thick,every node/.style={sloped,allow upside down},
    draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]
\renewcommand{\midarrow}{\tikz \draw[-{Stealth[length=5pt,width=4.5pt]},line width=1.2pt,draw=\edgecolor]
    (-1mm,0) -- (0,0);}
\renewcommand{\copymidarrow}{\tikz \draw[-{Stealth[length=5pt,width=4.5pt]},line width=1.2pt,
    draw={rgb:blue,4;red,1;green,1;black,3}] (-1mm,0) -- (0,0);}
\tikzstyle{bus}=[very thick,-Stealth,draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7,rounded corners=3pt]
\tikzstyle{op}=[draw=black!60,fill=\HLCCColor,rounded corners=3pt,align=flush center,font=\footnotesize,
    inner sep=3pt]
"""


# TikZ helpers: the pics of pycore.tikzeng, with free colours and labels.

def labels(*values) -> str:
    """xlabel / width list of a PlotNeuralNet pic: {{"32","32"}}."""
    return "{{" + ",".join(f'"{v}"' for v in values) + ",}}"


def pic(kind: str, name: str, at: str, offset: Tuple[float, float, float] = (0, 0, 0), **opts) -> str:
    body = ",\n        ".join(f"{key}={value}" for key, value in opts.items())
    dx, dy, dz = offset
    return f"\n\\pic[shift={{({dx},{dy},{dz})}}] at ({at})\n    {{{kind}={{name={name},\n        {body}}}}};\n"


def box(name, at, channels="", fill=r"\ConvColor", width=2.0, height=40.0, depth=40.0, size="",
        caption="", offset=(0, 0, 0), opacity=0.6) -> str:
    return pic("Box", name, at, offset, caption=f"{{{caption}}}", xlabel=labels(channels), zlabel=f"{{{size}}}",
               fill=fill, opacity=opacity, height=height, width=width, depth=depth)


def banded_box(name, at, channels: Sequence, fill=r"\ConvColor", bandfill=r"\ActColor", width=2.0,
               height=40.0, depth=40.0, size="", caption="", offset=(0, 0, 0), opacity=0.4) -> str:
    """Conv(s) followed by a normalisation / activation band, one sub-box per entry of ``channels``."""
    widths = "{" + ",".join(str(width) for _ in channels) + "}"
    return pic("RightBandedBox", name, at, offset, caption=f"{{{caption}}}", xlabel=labels(*channels),
               zlabel=f"{{{size}}}", fill=fill, bandfill=bandfill, opacity=opacity, height=height, width=widths,
               depth=depth)


def ball(name, at, logo, fill=r"\SumColor", radius=2.5, offset=(0, 0, 0), caption="") -> str:
    # The pic scales the logo by 4 * 0.2 * radius: undone, so that the logo keeps its font size.
    logo = rf"\scalebox{{{1 / (0.8 * radius):.4f}}}{{{logo}}}"
    return pic("Ball", name, at, offset, fill=fill, opacity=0.6, radius=radius, logo=f"{{{logo}}}",
               caption=f"{{{caption}}}")


SUM_LOGO = r"\large$+$"
DC_LOGO = r"\footnotesize\textsf{DC}"


def io_label(name: str, title: str, size: str, sep: str = "4pt", font: str = r"\small", above: bool = False) -> str:
    """Caption of an input / output box (the built-in caption wraps on thin boxes), centred under it, ``sep`` below
    its front bottom edge, or above its back top edge."""
    side, corner = ("above", "farnortheast") if above else ("below", "nearsoutheast")
    return rf"""
\node[{side}={sep},align=center,font={font},inner sep=0pt] ({name}-io) at ({name}-{corner} -| {name}-anchor)
    {{\textbf{{{title}}}\\{size}}};
"""


def note(name: str, text: str, sep: str = "11pt", font: str = r"\LabelFont", bold: bool = False) -> str:
    """Text centred under the front bottom edge of block ``name``, below its channel labels (``sep``)."""
    text = rf"\textbf{{{text}}}" if bold else text
    return rf"""
\node[below={sep},align=center,font={font},inner sep=0pt] ({name}-note)
    at ($({name}-nearsouthwest)!0.5!({name}-nearsoutheast)$) {{{text}}};
"""


def image_slab(name: str, at: str, png: Optional[Path], height: float, depth: float,
               offset=(0, 0, 0)) -> str:
    """Thin box whose east face shows a sinogram (views along the height, detector along the depth)."""
    return slab_box(name, at, height, depth, offset) + slab_image(name, png, height, depth)


def slab_box(name: str, at: str, height: float, depth: float, offset=(0, 0, 0)) -> str:
    return box(name, at, fill=r"\IOColor", width=0.4, height=height, depth=depth, offset=offset)


def slab_image(name: str, png: Optional[Path], height: float, depth: float) -> str:
    """The sinogram on the east face of the slab ``name``; nothing if the PNG is missing."""
    if png is None or not png.exists():
        return ""
    # 0.2: PlotNeuralNet box units -> units of the drawing (the zy canvas scales them with the unit vectors)
    return rf"""
\begin{{scope}}[shift={{({name}-east)}}]
    \node[canvas is zy plane at x=0,transform shape,inner sep=0pt] at (0,0)
        {{\includegraphics[width={depth * 0.2:.3f}cm,height={height * 0.2:.3f}cm]{{{png.relative_to(TEX_DIR).as_posix()}}}}};
\end{{scope}}
"""


def sinogram(name: str) -> Optional[Path]:
    path = SINO_DIR / f"{name}.png"
    return path if path.exists() else None


def legend(anchor: str, entries: Sequence[Tuple[str, str, str]], columns: int = 4, col_width: float = 6.0,
           row_height: float = 0.5) -> str:
    """Legend below the figure, in cm (not shrunk with the drawing); ``anchor`` is its top left corner.
    entries: (kind, colour, text), kind in box / band / ball / arrow (colour "fill|band" for a band)."""
    lines = [f"\n\\begin{{scope}}[shift={{({anchor})}}]"]
    for idx, (kind, color, text) in enumerate(entries):
        x, y = (idx % columns) * col_width, -(idx // columns) * row_height - 0.3
        if kind == "box":
            lines.append(rf"\filldraw[fill={color},fill opacity=0.7,draw=black!60] ({x}cm,{y}cm) rectangle ++(0.5cm,0.3cm);")
        elif kind == "band":
            fill, band = color.split("|")
            lines.append(rf"\filldraw[fill={fill},fill opacity=0.7,draw=black!60] ({x}cm,{y}cm) rectangle ++(0.37cm,0.3cm);")
            lines.append(rf"\filldraw[fill={band},fill opacity=0.7,draw=black!60] ({x + 0.37}cm,{y}cm) rectangle ++(0.13cm,0.3cm);")
        elif kind == "ball":
            lines.append(rf"\shade[ball color={color}] ({x + 0.25}cm,{y + 0.15}cm) circle (0.17cm);")
        elif kind == "arrow":
            lines.append(rf"\draw[bus] ({x}cm,{y + 0.15}cm) -- ++(0.5cm,0);")
        lines.append(rf"\node[anchor=west,font=\small] at ({x + 0.6}cm,{y + 0.15}cm) {{{text}}};")
    lines.append(r"\end{scope}")
    return "\n".join(lines) + "\n"


def down_connection(source: str, target: str) -> str:
    """Encoder arm of the U: right, down to the next level, then into ``target``."""
    return rf"""
\path ({source}-east) -- ({target}-west |- {source}-east) coordinate[pos=0.5] ({source}-down);
\draw[connection] ({source}-east) -- ({source}-down) -- node {{\midarrow}} ({source}-down |- {target}-west)
    -- ({target}-west);
"""


def up_connection(source: str, target: str) -> str:
    """Decoder arm of the U: right, then up into the bottom front edge of ``target``."""
    return rf"""
\coordinate ({target}-bottom) at ($({target}-nearsouthwest)!0.5!({target}-nearsoutheast)$);
\draw[connection] ({source}-east) -- ({source}-east -| {target}-bottom) -- node {{\midarrow}} ({target}-bottom);
"""


def skip_connection(source: str, target: str) -> str:
    """Skip connection across the U, at the height of its level."""
    return rf"""
\draw[copyconnection] ({source}-east) -- node {{\copymidarrow}} ({target}-west);
"""


def channel_width(channels: int, scale: float) -> float:
    """Box thickness grows with log2(channels), so that wide layers stay readable."""
    return round(max(0.6, scale * math.log2(max(channels, 1))), 2)


def times(rows: int, cols: int) -> str:
    return rf"${rows}\times{cols}$"


def document(body: List[str], tex_dir: Path, unit: float = 1.0) -> List[str]:
    """The figure, drawn at ``unit`` cm per PlotNeuralNet unit (1 by default): the x, y and z vectors are shrunk, so
    that the boxes, offsets and images scale while the text and the explicit lengths (in pt or cm) do not.

    The edge labels of the pics (channel counts, and the empty captions and depth labels, which are wide nodes)
    are left out of the bounding box: the channel counts lie inside the figure anyway."""
    layers_root = os.path.relpath(PLOTNN_ROOT, tex_dir)
    head = to_head(layers_root).replace("border=8pt", "border=2pt")
    z = -0.385 * unit  # TikZ default z vector: (-3.85mm, -3.85mm)
    scope = (rf"\begin{{scope}}[x={unit}cm,y={unit}cm,z={{({z:.4f}cm,{z:.4f}cm)}},"
             r"every edge quotes/.append style={font=\LabelFont,overlay}]" "\n")
    return [head, COLORS, to_begin(), BUS_STYLE, scope, *body, "\\end{scope}\n", to_end()]


UNET_LEGEND = [
    ("band", r"\ConvColor|\ActColor", r"conv $3\times3$ + InstanceNorm + LeakyReLU"),
    ("box", r"\PoolColor", r"max pooling $2\times2$"),
    ("box", r"\UpColor", r"bilinear $\times2$ + conv $3\times3$"),
    ("box", r"\CopyColor", r"skip connection (concatenation)"),
    ("box", r"\OutColor", r"conv $1\times1$"),
    ("ball", r"\SumColor", r"residual: $y + f_\theta(y)$"),
    ("ball", r"\DCColor", r"data consistency (measured views kept)"),
    ("arrow", "", r"copy of the input"),
]


def unet_layers_row(chans, size, height, depth, width) -> List[str]:
    """Encoder and decoder of the SinogramUNet in one row, the skip connections passing over the top."""
    arch = []
    # Encoder: ConvBlock (2 x [conv 3x3, InstanceNorm, LeakyReLU]) then max pooling
    prev = "in"
    for lvl in range(4):
        name = f"enc{lvl}"
        arch.append(banded_box(name, f"{prev}-east", (chans[lvl], chans[lvl]), width=width[lvl], height=height[lvl],
                               depth=depth[lvl], size=size[lvl], offset=(2.2 if lvl == 0 else 1.2, 0, 0),
                               caption="Bottleneck" if lvl == 3 else ""))
        arch.append(to_connection(prev, name))
        prev = name
        if lvl < 3:
            arch.append(box(f"pool{lvl}", f"{name}-east", fill=r"\PoolColor", width=1, height=height[lvl + 1],
                            depth=depth[lvl + 1], opacity=0.5))
            prev = f"pool{lvl}"

    # Decoder: bilinear x2 + conv 3x3, concatenation with the skip, ConvBlock
    for lvl in (2, 1, 0):
        arch += [
            box(f"up{lvl}", f"{prev}-east", chans[lvl], fill=r"\UpColor", width=width[lvl], height=height[lvl],
                depth=depth[lvl], offset=(1.2, 0, 0), opacity=0.5),
            box(f"copy{lvl}", f"up{lvl}-east", chans[lvl], fill=r"\CopyColor", width=width[lvl], height=height[lvl],
                depth=depth[lvl], opacity=0.25),
            banded_box(f"dec{lvl}", f"copy{lvl}-east", (chans[lvl], chans[lvl]), width=width[lvl],
                       height=height[lvl], depth=depth[lvl]),
            to_connection(prev, f"up{lvl}"),
            to_skip(of=f"enc{lvl}", to=f"copy{lvl}", pos=1.25),
        ]
        prev = f"dec{lvl}"
    return arch


def unet_layers_u(chans, size, height, depth, width, first: float = 2.6, gap: float = 1.4,
                  drop: Sequence[float] = (0.0, 9.5, 8.0, 6.5), dec_gap: Optional[float] = None) -> List[str]:
    """The same layers drawn as a U: the encoder goes down one level per pooling, the decoder comes back up, and
    the skip connections cross the U at the height of their level. ``first``: offset of enc0 from the input,
    ``gap`` (``dec_gap``): offset between the levels of the encoder (decoder), ``drop[l]``: vertical step from
    level l - 1 down to level l. The feature map size of each encoder level is written under its channel counts."""
    dec_gap = gap if dec_gap is None else dec_gap
    arch = [banded_box("enc0", "in-east", (chans[0], chans[0]), width=width[0], height=height[0], depth=depth[0],
                       offset=(first, 0, 0)),
            note("enc0", size[0]),
            to_connection("in", "enc0")]

    # Encoder: max pooling opens the next level, then ConvBlock (2 x [conv 3x3, InstanceNorm, LeakyReLU])
    for lvl in range(1, 4):
        arch += [
            box(f"pool{lvl}", f"enc{lvl - 1}-east", fill=r"\PoolColor", width=1, height=height[lvl], depth=depth[lvl],
                offset=(gap, -drop[lvl], 0), opacity=0.5),
            banded_box(f"enc{lvl}", f"pool{lvl}-east", (chans[lvl], chans[lvl]), width=width[lvl], height=height[lvl],
                       depth=depth[lvl]),
            note(f"enc{lvl}", size[lvl] + (r"\\[1pt]\textbf{Bottleneck}" if lvl == 3 else "")),
            down_connection(f"enc{lvl - 1}", f"pool{lvl}"),
        ]

    # Decoder: skip features concatenated with the bilinear x2 + conv 3x3 branch (torch.cat([skip, x])), ConvBlock
    prev = "enc3"
    for lvl in (2, 1, 0):
        arch += [
            box(f"copy{lvl}", f"{prev}-east", chans[lvl], fill=r"\CopyColor", width=width[lvl], height=height[lvl],
                depth=depth[lvl], offset=(dec_gap, drop[lvl + 1], 0), opacity=0.25),
            box(f"up{lvl}", f"copy{lvl}-east", fill=r"\UpColor", width=width[lvl], height=height[lvl],
                depth=depth[lvl], opacity=0.5),
            banded_box(f"dec{lvl}", f"up{lvl}-east", (chans[lvl], chans[lvl]), width=width[lvl], height=height[lvl],
                       depth=depth[lvl]),
            up_connection(prev, f"up{lvl}"),
            skip_connection(f"enc{lvl}", f"copy{lvl}"),
        ]
        prev = f"dec{lvl}"
    return arch


def unet_figure(filters: int, views: int, detectors: int, u_shape: bool = False, unit: float = 1.0) -> List[str]:
    """SinogramUNet: three poolings, bilinear + conv decoder, residual output, data consistency.

    Compact row by default; ``u_shape`` draws the classic U, which takes more room, at ``unit`` cm per unit."""
    padded = math.ceil(views / 8) * 8  # the backbone pads the views to a multiple of 8
    levels = range(4)
    chans = [filters * 2**lvl for lvl in levels]
    size = [times(padded // 2**lvl, detectors // 2**lvl) for lvl in levels]
    height = [round(40 * 0.75**lvl, 1) for lvl in levels]
    depth = [round(h * detectors / padded, 1) for h in height]
    if u_shape:  # 0.37 cm for 32 channels, +0.07 cm per doubling: room for the channel counts in \LabelFont
        width = [round((0.37 + 0.07 * math.log2(c / 32)) / (0.2 * unit), 2) for c in chans]
    else:
        width = [channel_width(c, 0.75) for c in chans]

    arch = [image_slab("in", "0,0,0", sinogram("incomplete"), height[0], depth[0])]
    if u_shape:
        # input label below the size of enc0, which it would overlap
        arch += [io_label("in", "Incomplete $y$", times(views, detectors), sep="21pt", font=r"\footnotesize"),
                 *unet_layers_u(chans, size, height, depth, width, first=2.0, gap=1.0, dec_gap=0.6,
                                drop=(0.0, 10.4, 9.0, 7.6))]
        step, radius = 1.12, 0.24 / (0.2 * unit)  # 0.28 cm between the output blocks; ball radius 0.24 cm
    else:
        arch += [io_label("in", "Incomplete $y$", times(views, detectors)),
                 *unet_layers_row(chans, size, height, depth, width)]
        step, radius = 1.1, 2.5

    # The copy of the input feeds the residual and the data consistency, above a U or below a row.
    if u_shape:
        bus = r"""
\draw[bus] ($(in-east)!0.25!(enc0-west)$) -- ++(0,7.0) coordinate (bus) -- (bus -| sum-north)
    node[pos=0.5,above,font=\small,opacity=1] {measured sinogram $y$: residual learning and data consistency}
    -- (sum-north);
\draw[bus] (bus -| sum-north) -- (bus -| dc-north) -- (dc-north);
"""
        legend_anchor = "$(in-io.west |- enc3-note.south)+(0,-0.4cm)$"
    else:
        bus = r"""
\draw[bus] ($(in-east)+(0.9,0)$) -- ++(0,-7.6) coordinate (bus) -- (bus -| sum-south)
    node[pos=0.5,below,font=\small,opacity=1] {measured sinogram $y$: residual learning and data consistency}
    -- (sum-south);
\draw[bus] (bus -| sum-south) -- (bus -| dc-south) -- (dc-south);
"""
        legend_anchor = "$(in-west)+(0,-9.2)$"

    # Output: conv 1x1, residual on the input, data consistency (measured views copied back)
    arch += [
        box("head", "dec0-east", 1, fill=r"\OutColor", width=0.8 if not u_shape else 0.15 / (0.2 * unit),
            height=height[0], depth=depth[0], offset=(step if u_shape else 1.2, 0, 0)),
        ball("sum", "head-east", SUM_LOGO, radius=radius, offset=(step + 0.2 * radius, 0, 0)),
        ball("dc", "sum-east", DC_LOGO, fill=r"\DCColor", radius=radius, offset=(step + 0.2 * radius, 0, 0)),
        image_slab("out", "dc-east", sinogram("unet2d"), height[0], depth[0], offset=(step, 0, 0)),
        io_label("out", r"Completed $\hat y$", times(views, detectors),
                 font=r"\footnotesize" if u_shape else r"\small"),
        to_connection("dec0", "head"),
        to_connection("head", "sum"),
        to_connection("sum", "dc"),
        to_connection("dc", "out"),
        bus,
        legend(legend_anchor, UNET_LEGEND, columns=2, col_width=7.4) if u_shape
        else legend(legend_anchor, UNET_LEGEND, columns=4, col_width=7.2),
    ]
    return arch


def mini_unet_row(prev: str, views: int, detectors: int, height0: float) -> List[str]:
    """Schematic SinogramUNet in one row, without labels (detailed in arch_unet): menc0..menc3, mdec2..mdec0."""
    padded = math.ceil(views / 8) * 8
    height = [round(height0 * 0.75**lvl, 1) for lvl in range(4)]
    depth = [round(h * detectors / padded, 1) for h in height]
    arch = []
    for lvl in range(4):
        name = f"menc{lvl}"
        arch += [banded_box(name, f"{prev}-east", ("", ""), width=0.6 + 0.3 * lvl, height=height[lvl],
                            depth=depth[lvl], offset=(2.0 if lvl == 0 else 0.7, 0, 0)),
                 to_connection(prev, name)]
        prev = name
    for lvl in (2, 1, 0):
        name = f"mdec{lvl}"
        arch += [banded_box(name, f"{prev}-east", ("", ""), width=0.6 + 0.3 * lvl, height=height[lvl],
                            depth=depth[lvl], offset=(0.7, 0, 0)),
                 to_connection(prev, name),
                 to_skip(of=f"menc{lvl}", to=name, pos=1.25)]
        prev = name
    return arch


def mini_unet_u(prev: str, views: int, detectors: int, height0: float, unit: float = 1.0,
                gap_cm: float = 0.6) -> List[str]:
    """The same schematic U-Net drawn as a U, at ``unit`` cm per unit (offsets and thicknesses given in cm)."""
    padded = math.ceil(views / 8) * 8
    height = [round(height0 * 0.75**lvl, 1) for lvl in range(4)]
    depth = [round(h * detectors / padded, 1) for h in height]
    drop = [0.0, 1.0 / unit, 0.85 / unit, 0.7 / unit]  # vertical step from level l - 1 down to level l
    gap = gap_cm / unit
    width = [round((0.12 + 0.04 * lvl) / (0.2 * unit), 2) for lvl in range(4)]  # thickness of each conv
    arch = [banded_box("menc0", f"{prev}-east", ("", ""), width=width[0], height=height[0], depth=depth[0],
                       offset=(0.8 / unit, 0, 0)),
            to_connection(prev, "menc0")]
    for lvl in range(1, 4):
        arch += [banded_box(f"menc{lvl}", f"menc{lvl - 1}-east", ("", ""), width=width[lvl],
                            height=height[lvl], depth=depth[lvl], offset=(gap, -drop[lvl], 0)),
                 down_connection(f"menc{lvl - 1}", f"menc{lvl}")]
    prev = "menc3"
    for lvl in (2, 1, 0):
        arch += [banded_box(f"mdec{lvl}", f"{prev}-east", ("", ""), width=width[lvl], height=height[lvl],
                            depth=depth[lvl], offset=(gap, drop[lvl + 1], 0)),
                 up_connection(prev, f"mdec{lvl}"),
                 skip_connection(f"menc{lvl}", f"mdec{lvl}")]
        prev = f"mdec{lvl}"
    return arch


def unet_hlcc_p4_figure(views: int, detectors: int, max_order: int, u_shape: bool = False,
                        unit: float = 1.0) -> List[str]:
    """UNet2dHLCC_P<K>: the U-Net trained with MSE + HLCC penalty, then the moment regression of orders 0..K.

    The U-Net runs left to right at the top and the regression below it: its first row ends under the output
    of the U-Net, the second runs right to left, so that the final output comes back under the input. ``u_shape``
    draws the U-Net as a U instead of a row, at ``unit`` cm per unit; the flowchart is laid out in cm."""
    height = 24.0
    depth = round(height * detectors / views, 1)
    correction = sinogram(f"p{max_order}_correction_2d")
    # The update: its text, and beside it the correction on the test sample (if rendered)
    update = (r"$\hat y_{P_<K>}=\hat y+M\odot\delta$\\ $M$: missing views inside the\\ shadow of the object\\"
              r" ($\hat y>0.03\max_s\hat y$)")
    if correction:
        update = (r"\begin{tabular}{@{}c@{\hspace{6pt}}c@{}}\begin{tabular}{@{}c@{}}" + update
                  + r"\end{tabular} & \begin{tabular}{@{}c@{}}\includegraphics[height=1.4cm]{"
                  + correction.relative_to(TEX_DIR).as_posix()
                  + r"}\\ $M\odot\delta$ on the\\ test sample\end{tabular}\end{tabular}")
    radius, dc_radius = 0.22 / (0.2 * unit), 0.24 / (0.2 * unit)  # balls: 0.22 and 0.24 cm
    # Brace under the U-Net, and the bus from the input, which runs below the whole network
    if u_shape:
        network = r"""\draw[decorate,decoration={brace,mirror,amplitude=5pt},thick]
    ($(menc0-west |- menc3-south)+(-0.3cm,-0.35cm)$) -- ($(mdec0-east |- menc3-south)+(0,-0.35cm)$)
    node[midway,below=6pt,font=\small] {U-Net $f_\theta$ (same network as UNet2D)};
\path (in-south) coordinate (b0);
\path (b0 |- menc3-south) ++(0,-1.35cm) coordinate (b1);
\draw[bus] (b0) -- (b1) -| (sum-south);
"""
    else:
        network = r"""\draw[decorate,decoration={brace,mirror,amplitude=6pt},thick]
    ($(menc0-west |- menc0-south)+(-0.7,-1.4)$) -- ($(mdec0-east |- menc0-south)+(0,-1.4)$)
    node[midway,below=8pt,font=\small] {U-Net $f_\theta$ (same network as UNet2D)};
\path ($(in-east)+(0.9,0)$) coordinate (b0);
\draw[bus] (b0) -- ++(0,-5.0) coordinate (b1) -| (sum-south);
"""
    flowchart = r"""
% Network part: the bus from the input feeds the residual, the data consistency and the moment regression
<NETWORK>\draw[bus] (b1 -| sum-south) -| (dc-south);
\node[anchor=west,align=left,font=\small] (yhat-io) at ($(yhat-east)+(0.45cm,0)$) {\textbf{UNet2dHLCC}\\$\hat y$};

% Training loss (UNet2dHLCC only), above the output
\node[draw=black!60,dashed,rounded corners=3pt,align=center,font=\footnotesize,anchor=south east] (loss)
    at ($(yhat-io.east |- yhat-farnortheast)+(0,0.3cm)$)
    {\textbf{training only}: $\mathcal L=\mathrm{MSE}(\hat y,y^\star)+\alpha(t)\,(\lambda_0L_0+\lambda_1L_1)$\\
     HLCC penalty $L_n=\|(I-\Pi_{H_n})\,a^{\hat y}_n\|^2/V$, annealed by $\alpha(t)$};
\draw[bus,dashed] (yhat-north) -- (yhat-north |- loss.south);

% HLCC moment regression P_K. Under the output of the U-Net: moments of y_hat, then the difference with the
% first row (left to right), then the second row (right to left, top-aligned) ending with the output. The boxes
% take the width of their lines.
\node[op,anchor=north] (myhat) at ($(yhat-south |- b1)+(0,-0.6cm)$)
    {moments of $\hat y$\\ $a^{\hat y}_n(\theta)$, all the views};
\node[circle,draw=black!60,fill=\HLCCColor,inner sep=1pt,font=\small] (delta)
    at ($(myhat.south)+(0,-0.95cm)$) {$-$};
\node[op,anchor=east] (reg) at ($(delta.west)+(-0.75cm,0)$)
    {harmonic regression\\ fit $\tilde a_n\in H_n$ on $\Theta_{\mathrm{acq}}$,\\ evaluate on all views\\
     $H_n=\mathrm{span}\{\cos m\theta,\sin m\theta\}_{m\le n,\ m\equiv n\,(2)}$};
\node[op,anchor=east] (my) at ($(reg.west)+(-0.5cm,0)$)
    {moments of the\\ measured views\\
     $a^{y}_n(\theta)=\sum_s y(\theta,s)\,U_n(s/S)$\\
     $\theta\in\Theta_{\mathrm{acq}}$, $n=0,\dots,<K>$};
\node[anchor=south west,align=left,font=\footnotesize,inner sep=0pt] (title) at ($(my.north west)+(0,0.3cm)$)
    {\textbf{HLCC moment regression $P_<K>$}\\(orders $0,\dots,<K>$, no trainable parameter, inference only)};
\draw[bus] (b1) -| ($(my.west)+(-0.45cm,0)$) -- (my.west);
\draw[bus] (yhat-south) -- (myhat.north);
\draw[bus] (my) -- (reg);
\draw[bus] (reg) -- node[above,font=\small,opacity=1] {$\tilde a_n$} (delta);
\draw[bus] (myhat) -- node[right,font=\small,opacity=1] {$a^{\hat y}_n$} (delta);

\node[op,anchor=north] (corr) at ($(delta.south |- my.south)+(0,-0.65cm)$)
    {correction of the missing views\\
     $\delta(\theta,s)=\sum_n c_n(\theta)\,W(s)\,U_n(s/S)$\\
     $W(s)=\sqrt{1-(s/S)^2}$\\ $c_n$: the corrected moments are $\tilde a_n$};
\draw[bus] (delta) -- node[right,font=\small,opacity=1] {$\Delta a_n=\tilde a_n-a^{\hat y}_n$} (corr);
\node[op,anchor=north east] (upd) at ($(corr.north west)+(-0.6cm,0)$) {<UPDATE>};
\draw[bus] (corr.west) -- (corr.west -| upd.east);
"""
    flowchart = (flowchart.replace("<NETWORK>", network).replace("<UPDATE>", update)
                 .replace("<K>", str(max_order)))
    step = 0.45 / unit  # between the output blocks
    if u_shape:
        network_layers = [io_label("in", "Incomplete $y$", times(views, detectors), font=r"\footnotesize",
                                   above=True),
                          *mini_unet_u("in", views, detectors, height, unit, gap_cm=0.75)]
    else:
        network_layers = [io_label("in", "Incomplete $y$", times(views, detectors)),
                          *mini_unet_row("in", views, detectors, height)]

    return [
        image_slab("in", "0,0,0", sinogram("incomplete"), height, depth),
        *network_layers,
        ball("sum", "mdec0-east", SUM_LOGO, radius=radius, offset=(step + 0.2 * radius, 0, 0)),
        ball("dc", "sum-east", DC_LOGO, fill=r"\DCColor", radius=dc_radius, offset=(step + 0.2 * dc_radius, 0, 0)),
        slab_box("yhat", "dc-east", height, depth, offset=(step, 0, 0)),
        to_connection("mdec0", "sum"),
        to_connection("sum", "dc"),
        to_connection("dc", "yhat"),
        slab_image("yhat", sinogram("unet2dhlcc"), height, depth),
        flowchart,
        # Output under the input, the arrow arriving on the east face
        slab_box("yp4", "$(upd.west)+(-1.6cm,0)$", height, depth),
        r"""
\draw[connection] (upd.west) -- node {\midarrow} (yp4-east);
""",
        slab_image("yp4", sinogram(f"unet2dhlcc_p{max_order}"), height, depth),
        io_label("yp4", rf"UNet2dHLCC\_P{max_order}", rf"$\hat y_{{P_{max_order}}}$\\" + times(views, detectors),
                 font=r"\footnotesize"),
        rf"""
\node[draw=black!50,dashed,rounded corners,inner sep=0.25cm,
    fit=(title)(my)(reg)(myhat)(corr)(upd)(yp4-farnorthwest)(yp4-nearsoutheast)(yp4-io)] {{}};
""",
    ]


def viewgraph_figure(num_stalks: int, k: int, views: int, detectors: int, unit: float = 1.0) -> List[str]:
    """ViewGraphNet: nodes are the views, features [2F, V, D], residual blocks of detector convs + graph conv.
    Drawn at ``unit`` cm per unit; the box thicknesses, offsets and the inset are given in cm."""
    channels, hidden = 2 * num_stalks, max(1, num_stalks // 2)
    height = 32.0
    depth = round(height * detectors / views, 1)

    def u(length_cm: float) -> float:
        """A length in cm, in units of the drawing."""
        return round(length_cm / unit, 3)

    # Box thickness: 0.25 cm + 0.025 cm per doubling of the channels, room for the channel count under each box
    w = {c: round((0.25 + 0.025 * math.log2(c)) / (0.2 * unit), 2) for c in (1, 2, hidden, num_stalks, channels)}
    common = dict(height=height, depth=depth)
    radius = 0.2 / (0.2 * unit)  # residual balls: 0.2 cm

    arch = [
        box("in", "0,0,0", 2, fill=r"\IOColor", width=w[2], **common),
        io_label("in", r"$y$ + acquired flag", times(views, detectors), font=r"\footnotesize", above=True),
        banded_box("enc", "in-east", (channels,), width=w[channels], offset=(u(0.65), 0, 0), opacity=0.9, **common),
        note("enc", "Encoder", bold=True),
        to_connection("in", "enc"),
    ]

    # Residual ViewGraphBlocks: [conv 1x3, ViewNorm, GELU] [graph conv, ViewNorm, GELU] [conv 1x3] (+)
    prev = "enc"
    for idx, block in enumerate(("1", "2", "L")):
        b = f"b{block}"
        after_dots = block == "L"
        arch += [
            banded_box(f"{b}c1", f"{prev}-east", (channels,), width=w[channels],
                       offset=(u(1.15 if after_dots else 0.8 if prev == "enc" else 0.55), 0, 0), opacity=0.9,
                       **common),
            banded_box(f"{b}g", f"{b}c1-east", (channels,), fill=r"\GraphColor", bandfill=r"\GraphActColor",
                       width=w[channels], opacity=0.9, **common),
            note(f"{b}g", f"Block ${block}$", bold=True),
            box(f"{b}c2", f"{b}g-east", channels, width=w[channels], opacity=0.9, **common),
            ball(f"{b}s", f"{b}c2-east", SUM_LOGO, radius=radius, offset=(u(0.25) + 0.2 * radius, 0, 0)),
            to_connection(f"{b}c2", f"{b}s"),
            rf"""
\draw[bus] ($({b}c1-west)+(-0.45cm,0)$) -- ++(0,1.4cm) -| ({b}s-north);
""",
        ]
        if after_dots:
            arch.append(rf"""
\draw[connection] ({prev}-east) -- node[pos=0.3,fill=white,text opacity=1,fill opacity=1,inner sep=1pt]
    {{\Large$\cdots$}} ({b}c1-west);
""")
        else:
            arch.append(to_connection(prev, f"{b}c1"))
        prev = f"{b}s"

    arch += [
        rf"""
\draw[decorate,decoration={{brace,mirror,amplitude=6pt}},thick]
    ($(b1c1-west |- b1c1-south)+(-0.3cm,-1.05cm)$) -- ($(bLs-east |- b1c1-south)+(0.1cm,-1.05cm)$)
    node[midway,below=7pt,font=\small] {{$\times L$ blocks: angular reach $L\,k/2 = {k // 2}L$ views}};
""",
        # Decoder: [conv 1x3, ViewNorm, GELU] [conv 1x3, GELU] [conv 1x3]
        banded_box("dec1", "bLs-east", (num_stalks,), width=w[num_stalks], offset=(u(0.42), 0, 0), opacity=0.9,
                   **common),
        banded_box("dec2", "dec1-east", (hidden,), width=w[hidden], opacity=0.9, **common),
        note("dec2", "Decoder", bold=True),
        box("dec3", "dec2-east", 1, width=w[1], opacity=0.9, **common),
        ball("dc", "dec3-east", DC_LOGO, fill=r"\DCColor", radius=0.24 / (0.2 * unit),
             offset=(u(0.4 + 0.24), 0, 0)),
        box("out", "dc-east", 1, fill=r"\IOColor", width=w[1], offset=(u(0.5), 0, 0), **common),
        io_label("out", "Completed", times(views, detectors), font=r"\footnotesize", above=True),
        to_connection("bLs", "dec1"),
        to_connection("dec3", "dc"),
        to_connection("dc", "out"),
        r"""
\draw[bus] (in-south) -- ++(0,-1.95cm) -| (dc-south)
    node[pos=0.25,below,font=\small,opacity=1] {measured sinogram $y$: data consistency};
""",
        # Inset: message passing of one graph conv, on the view graph; (0, 0) is the middle of its bottom edge
        rf"""
\begin{{scope}}[shift={{($(b1c1-west)!0.5!(bLs-east)$)}},shift={{(0cm,2.15cm)}}]
    \draw[dashed,rounded corners,black!50] (-5.4cm,0cm) rectangle (5.4cm,2.6cm);
    \foreach \j in {{0,...,6}} {{
        \node[circle,draw=black!70,fill=\GraphColor,minimum size=0.3cm,inner sep=0pt] (v\j)
            at ({{0.8cm*(\j-3)}},1.35cm) {{}};
    }}
    \node[circle,draw=black,very thick,fill=\GraphActColor,minimum size=0.3cm,inner sep=0pt] at (v3) {{}};
    \node at (-3.05cm,1.35cm) {{$\cdots$}};  \node at (3.05cm,1.35cm) {{$\cdots$}};
    \foreach \j in {{0,1,2}} {{ \draw[-Stealth,thick,draw=\edgecolor] (v\j) to[bend left=50] (v3); }}
    \foreach \j in {{4,5,6}} {{ \draw[-Stealth,thick,draw=\edgecolor] (v\j) to[bend right=50] (v3); }}
    \node[below=0pt,font=\small,inner sep=1pt] at (v3.south) {{$i$}};
    \node[font=\small] at (0cm,2.3cm) {{graph conv on the view graph ($k/2={k // 2}$ neighbours per side)}};
    \node[font=\small] at (0cm,0.72cm) {{$m_i=\sum_{{j}}\hat A_{{ij}}\,T_{{ij}}\,x_j, \qquad x_i \leftarrow \mathrm{{Conv}}_{{1\times1}}([x_i;\,m_i])$}};
    \node[font=\small] at (0cm,0.28cm) {{GCN: $T_{{ij}}=I$ \qquad SNN: $T_{{ij}}=S\big(t_f(\theta_i-\theta_j)\big)$ (shift along the detector)}};
\end{{scope}}
\draw[dashed,black!50] (b1g-north |- 0,2.15cm) -- (b1g-north);
""",
        legend("$(in-io.west |- 0,-3.65cm)$", [
            ("band", r"\ConvColor|\ActColor", r"conv $1\times3$ along the detector + (ViewNorm +) GELU"),
            ("ball", r"\SumColor", r"residual connection"),
            ("band", r"\GraphColor|\GraphActColor", r"graph conv between views + ViewNorm + GELU"),
            ("arrow", "", r"identity path"),
            ("ball", r"\DCColor", r"data consistency (measured views kept)"),
        ], columns=2, col_width=8.9),
    ]
    return arch


def render_sinograms(sample_index: int, noise_level: float, max_order: int) -> None:
    """PNGs of the test sample shown by the U-Net figures: input, outputs and the P_K correction.
    Needs ASTRA (GPU projector) and the UNet2D / UNet2dHLCC checkpoints; the models run on the CPU."""
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from src.dataset import SinogramCompletionDataset
    from src.models.factory import get_model

    dataset = SinogramCompletionDataset(sample_index + 1, split="test", noise_level=noise_level,
                                        geometry_config=GeometryConfig())
    incomplete, full, _ = dataset[sample_index]
    device = torch.device("cpu")
    unet, _ = get_model("UNet2D", device)
    hlcc_pk, _ = get_model(f"UNet2dHLCC_P{max_order}", device)
    with torch.no_grad():
        x = incomplete.unsqueeze(0)
        outputs = {"unet2d": unet(x), "unet2dhlcc": hlcc_pk.model(x), f"unet2dhlcc_p{max_order}": hlcc_pk(x)}

    def save(name: str, sino: torch.Tensor, slab: bool = True, **style) -> None:
        # [V, D], first view (-90 deg) at the bottom. On a slab the detector axis is mirrored, because the depth
        # axis of PlotNeuralNet points towards the reader.
        image = np.flipud(sino.squeeze().cpu().numpy())
        plt.imsave(SINO_DIR / f"{name}.png", np.fliplr(image) if slab else image, **style)
        print(f"wrote {(SINO_DIR / name).relative_to(PROJECT_ROOT)}.png")

    SINO_DIR.mkdir(parents=True, exist_ok=True)
    vmax = float(full.max())
    save("incomplete", incomplete, cmap="bone", vmin=0.0, vmax=vmax)
    for name, out in outputs.items():
        save(name, out, cmap="bone", vmin=0.0, vmax=vmax)
    correction = outputs[f"unet2dhlcc_p{max_order}"] - outputs["unet2dhlcc"]
    limit = float(correction.abs().max())
    save(f"p{max_order}_correction_2d", correction, slab=False, cmap="RdBu_r", vmin=-limit, vmax=limit)


def find_latex(requested: str | None) -> str | None:
    if requested:
        return requested
    return shutil.which("tectonic") or shutil.which("pdflatex")


def compile_tex(tex_path: Path, latex: str) -> Path:
    """Compile in the .tex folder (the \\subimport path is relative to it) and move the PDF to PDF_DIR."""
    if Path(latex).name == "tectonic":
        cmd = [latex, "-X", "compile", tex_path.name]
    else:
        cmd = [latex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    result = subprocess.run(cmd, cwd=tex_path.parent, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
    for ext in (".aux", ".log"):
        tex_path.with_suffix(ext).unlink(missing_ok=True)
    pdf_path = PDF_DIR / tex_path.with_suffix(".pdf").name
    shutil.move(tex_path.with_suffix(".pdf"), pdf_path)
    return pdf_path


FIGURES = ("unet", "unet_ushape", "unet_hlcc_p4", "unet_hlcc_p4_ushape", "viewgraph")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=(*FIGURES, "all"), default="all")
    parser.add_argument("--latex", default=None, help="LaTeX compiler (default: tectonic or pdflatex from the PATH)")
    parser.add_argument("--no-compile", action="store_true", help="only write the .tex sources")
    parser.add_argument("--sinograms", action="store_true", help="re-render the sinogram images (ASTRA + checkpoints)")
    parser.add_argument("--sample", type=int, default=0, help="test sample shown by the U-Net figures")
    parser.add_argument("--noise_level", type=float, default=1e5, help="as in src.evaluate")
    args = parser.parse_args()

    max_order = 4  # the _P4 models
    if args.sinograms:
        render_sinograms(args.sample, args.noise_level, max_order)

    geometry = GeometryConfig()
    views, detectors = geometry.num_views_full, geometry.det_col_count
    unet_cfg, graph_cfg = DEFAULT_MODEL_CONFIG["UNet2D"], DEFAULT_MODEL_CONFIG["SNN"]
    # name: (body, cm per PlotNeuralNet unit). The unit of the three figures of the thesis keeps their natural width
    # under TEXT_WIDTH_CM; the row variants are drawn at the PlotNeuralNet scale.
    figures = {
        "unet": lambda: (unet_figure(unet_cfg["filters"], views, detectors), 1.0),
        "unet_ushape": lambda: (unet_figure(unet_cfg["filters"], views, detectors, u_shape=True, unit=0.25), 0.25),
        "unet_hlcc_p4": lambda: (unet_hlcc_p4_figure(views, detectors, max_order), 1.0),
        "unet_hlcc_p4_ushape": lambda: (unet_hlcc_p4_figure(views, detectors, max_order, u_shape=True, unit=0.4),
                                        0.4),
        "viewgraph": lambda: (viewgraph_figure(graph_cfg["num_stalks"], graph_cfg["k"], views, detectors, unit=0.3),
                              0.3),
    }
    selected = figures if args.model == "all" else {args.model: figures[args.model]}

    latex = None if args.no_compile else find_latex(args.latex)
    if not args.no_compile and latex is None:
        print("No LaTeX compiler found (tectonic / pdflatex): only the .tex sources are written.")

    TEX_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in selected.items():
        tex_path = TEX_DIR / f"arch_{name}.tex"
        body, unit = build()
        tex_path.write_text("".join(document(body, TEX_DIR, unit)))
        print(f"wrote {tex_path.relative_to(PROJECT_ROOT)}")
        if latex is not None:
            print(f"wrote {compile_tex(tex_path, latex).relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
