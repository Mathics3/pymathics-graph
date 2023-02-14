# -*- coding: utf-8 -*-

"""
GraphBox.
"""

# uses networkx

import base64
import tempfile


from typing import Tuple

from mathics.core.element import BaseElement, BoxElementMixin


from pymathics.graph.format import png_format_graph, svg_format_graph

no_doc = True

class GraphBox(BoxElementMixin):
    def __init__(self, G, **options):
        self.G = G
        self.options = options

    def boxes_to_b64text(
        self, elements: Tuple[BaseElement] = None, **options
    ) -> Tuple[bytes, Tuple[int, int]]:
        """
        Produces a base64 png representation and a tuple with the size of the pillow image
        associated to the object.
        """
        contents, size = self.boxes_to_png(elements, **options)
        encoded = base64.b64encode(contents)
        encoded = b"data:image/png;base64," + encoded
        return encoded, size

    def boxes_to_png(self, elements=None, **options) -> Tuple[bytes, Tuple[int, int]]:
        """
        returns a tuple with the set of bytes with a png representation of the image
        and the scaled size.
        """
        return png_format_graph(self.G, **self.options), (800, 600)

    def boxes_to_svg(self, elements=None, **options):
        return svg_format_graph(self.G, **self.options), (400, 300)

    def boxes_to_tex(self, elements=None, **options) -> str:
        """
        Store the associated image as a png file and return
        a LaTeX command for including it.
        """

        data, size = self.boxes_to_png(elements, **options)
        res = 100  # pixels/cm
        width_str, height_str = (str(n / res).strip() for n in size)
        head = rf"\includegraphics[width={width_str}cm,height={height_str}cm]"

        # This produces a random name, where the png file is going to be stored.
        # LaTeX does not have a native way to store an figure embeded in
        # the source.
        fp = tempfile.NamedTemporaryFile(delete=True, suffix=".png")
        path = fp.name
        fp.close()

        with open(path, "wb") as imgfile:
            imgfile.write(data)

        return head + "{" + format(path) + "}"

    def boxes_to_text(self, elements=None, **options):
        return "-Graph-"

    def boxes_to_mathml(self, elements=None, **options):
        encoded, size = self.boxes_to_b64text(elements, **options)
        decoded = encoded.decode("utf8")
        # see https://tools.ietf.org/html/rfc2397
        return f'<mglyph src="{decoded}" width="50%" height="50%" />'
