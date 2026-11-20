"""Draw the absorption placements of ``philtorch.lti.delay_state_space``.

Writes one signal-flow diagram per ``absorption_position`` next to this file:
``delay_state_space_<position>.svg``. Run it after changing a diagram, and
commit the SVGs::

    pixi exec --spec schemdraw --spec matplotlib-base python docs/figures/delay_state_space.py
"""

from pathlib import Path

import matplotlib
import schemdraw
import schemdraw.dsp as dsp

schemdraw.use("matplotlib")
schemdraw.config(fontsize=14, lw=1.25, font="serif")
# Serif maths, and SVGs that only change when the drawing does.
matplotlib.rcParams.update({"mathtext.fontset": "cm", "svg.hashsalt": "philtorch"})

STEP = 0.9  # the length of a wire between blocks
WIDTH = 1.1  # the width of a box

GAMMA = r"$\Gamma(z)$"  # the label of the filters
# The blocks on the feedback path, from the tap back to the input sum.
LOOPS = {"delay": ["$A$"], "matrix": ["$A$", GAMMA], "feedback": [GAMMA, "$A$"]}


def block(d, element, direction="right"):
    """Add a box or sum at the cursor, entering from its side, and move past it."""
    inlet, outlet = ("W", "E") if direction == "right" else ("E", "W")
    placed = d.add(element.anchor(inlet))
    d.move_from(getattr(placed, outlet))
    return placed


def box(label):
    return dsp.Box(w=WIDTH, h=0.8).label(label)


def draw(position):
    d = schemdraw.Drawing(show=False)
    d += dsp.Line().right(STEP / 2).label(r"$\mathbf{x}$", "left")
    split = d.add(dsp.Dot())
    d += dsp.Arrow().right(STEP / 2)
    block(d, box("$B$"))
    d += dsp.Arrow().right(STEP)
    add = block(d, dsp.Sum().scale(0.4))
    d += dsp.Arrow().right(STEP)
    block(d, box(r"$z^{-\mathbf{m}}$"))
    d += dsp.Line().right(STEP / 2).label(r"$\mathbf{s}$", "top")
    if position == "delay":
        d += dsp.Arrow().right(STEP / 2)
        block(d, box(GAMMA))
        d += dsp.Line().right(STEP / 2).label(r"$\hat{\mathbf{s}}$", "top")
    else:
        # The same width, so that the three diagrams line up.
        d += dsp.Line().right(WIDTH + STEP)
    tap = d.add(dsp.Dot())
    d += dsp.Arrow().right(STEP / 2)
    block(d, box("$C$"))
    d += dsp.Arrow().right(STEP)
    out = block(d, dsp.Sum().scale(0.4))
    d += dsp.Arrow().right(STEP / 2).label(r"$\mathbf{y}$", "right")

    d += dsp.Line().at(tap.center).down(1.4)
    # The first box sits under the main row's filter column and the second
    # under z^{-m}'s, so the wire between them is as long as theirs.
    for i, label in enumerate(LOOPS[position]):
        arrow = dsp.Arrow().left(STEP if i else STEP / 2)
        if position == "feedback" and label == "$A$":
            arrow.label(r"$\hat{\mathbf{s}}$", "bottom")
        d += arrow
        block(d, box(label), "left")
    d += dsp.Line().tox(add.S)
    d += dsp.Arrow().to(add.S)

    # The feedthrough, below the feedback.
    d += dsp.Line().at(split.center).down(2.8)
    d += dsp.Arrow().right(STEP / 2)
    block(d, box("$D$"))
    d += dsp.Line().tox(out.S)
    d += dsp.Arrow().to(out.S)
    return d


if __name__ == "__main__":
    here = Path(__file__).parent
    for position in LOOPS:
        draw(position).draw(show=False).fig.savefig(
            here / f"delay_state_space_{position}.svg",
            bbox_inches="tight",
            pad_inches=0.02,
            transparent=True,
            metadata={"Date": None},
        )
