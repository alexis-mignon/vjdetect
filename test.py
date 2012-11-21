# Author: Alexis Mignon
# 
#
#
# -*- coding: utf-8 -*-
import vjdetect
from scipy.misc import imshow

def test(filename="lena.jpg"):
    imshow(
        vjdetect.detect_and_draw(
            filename, "haarcascade_frontalface_default.xml")
        )

    
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test(sys.argv[1])
    else:
        test()
