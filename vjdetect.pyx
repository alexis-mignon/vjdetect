# -*- coding: utf-8 -*-
""" Python implementation of the Haar Cascade detector of Viola-Jones.

    This module implements only the detector itself not the process to learn
    the cascade. It *does not* rely on OpenCV but uses the same xml files
    that describe the cascades.
    
    It is basically a translation of an old version of the *jviolajones*
    project from Java to C/Cython.
    
    :Author: Alexis Mignon (c) 2012
    :E-mail: alexis.mignon@gmail.com

"""

import numpy as np
cimport numpy as np
from lxml.etree import parse
from scipy.misc import imread, imshow, toimage, fromimage
from ImageDraw import ImageDraw
import cython
import os

from libc.stdlib cimport malloc, calloc, free

cdef extern from "math.h":
    double sqrt(double)
    double fmin(double, double)
    double fabs(double)

ctypedef np.float64_t Double_t
ctypedef np.float32_t Float_t
ctypedef np.int32_t Int_t
Float = np.float32
Int = np.int32


CASCADE_DIR = os.path.dirname(__file__) + "/data/"
DEFAULT_CASCADE = "haarcascade_frontalface_default.xml"

cdef Int_t TREE_LEFT = 0
cdef Int_t TREE_RIGHT = 1

cdef extern from "src/list.h":
    ctypedef struct ListItem:
        pass
    
    ctypedef struct List "List" :
        ListItem* first
        ListItem* last
        Int_t size

    List* List_new()
    List* List_append(List* self, void* item)
    void List_delete(List* self, bint delete_items)
    
    ctypedef struct ListIterator:
        pass

    void  ListIterator_init(ListIterator* self, List* list_)
    void* ListIterator_next(ListIterator* self)
    bint  ListIterator_has_next(ListIterator* self)

cdef extern from "src/detector.h":
    ctypedef struct Rect:
        pass
    
    void Rect_init(Rect* self,
                   Int_t x1, Int_t y1,
                   Int_t x2, Int_t y2,
                   Float_t weight)
    Rect* Rect_new(Int_t x1, Int_t y1, Int_t x2, Int_t y2, Float_t weight)
    
    ctypedef struct Feature:
        pass

    Feature* Feature_new(List* rect_list,
         Float_t threshold,
         Float_t left_val , Int_t left_node , bint has_left_val,
         Float_t right_val, Int_t right_node, bint has_right_val,
         Float_t width, Float_t height)

    ctypedef struct Tree:
        pass


    Tree* Tree_new(List* feature_list)

    ctypedef struct Stage:
        pass
    
    Stage* Stage_new(List* tree_list, Float_t threshold)


    bint Stage_pass(Stage* self, 
        Int_t width, Int_t height,
        Int_t* gray_image,
        Int_t* squares,
        Int_t i, Int_t j, Float_t scale)
        
    void get_integral_canny(
                Int_t width,
                Int_t height,
                Int_t* gray_image,
                Int_t* canny)
                
    List* det_get_objects(
                List* stages,
                Float_t dwidth,
                Float_t dheight,
                Int_t width,
                Int_t height,
                Int_t* img, 
                Float_t scale_base, Float_t scale_inc,
                Float_t increment, bint do_canny_pruning)

cdef inline Rect* Rect_fromString(str text):
    cdef:
        list tab = text.split()
        Int_t x1 = int(tab[0])
        Int_t x2 = int(tab[1])
        Int_t y1 = int(tab[2])
        Int_t y2 = int(tab[3])
        Float_t w = float(tab[4])    
    return Rect_new(x1, y1, x2, y2, w)

cdef class Detector:
    cdef:
        List* stages
        Float_t width
        Float_t height

    
    def __dealloc__(self):
        List_delete(self.stages, True)
    
        
    def __cinit__(self, str filename):
        """ Constructor.
    Builds a Haar cascade detector.
    
    Parameters
    ----------
    
    filename: string
        The path to the xml file describing the Haar Cascade.
"""
        cdef:
            list trees
            list stage_elements
            list features
            List* stage_trees
            List* tree_features
            List* feature_rects
            
            Stage* st
            Tree* t
            Rect* r
            Float_t thres
            object tree, feature, stage
            list text_tab
            Feature* f
            bint has_left_val, has_right_val
            Int_t left_node, right_node
            Float_t left_val, right_val
            
        document = parse(filename)
        self.stages = List_new()
        
        root = document.getroot().getchildren()[0]
        
        text_tab = root.find("./size").text.split()
        self.width = int(text_tab[0])
        self.height = int(text_tab[1])

        stage_elements = root.find("./stages").getchildren()
        
        for stage in stage_elements:
            
            thres = float(stage.find("./stage_threshold").text)
            trees = stage.findall("./trees/_")
            stage_trees = List_new()
            for tree in trees:
                features = tree.findall("./_")
                tree_features = List_new()
                for feature in features:
                    thres2 = float(feature.find("./threshold").text)
                    left_node = -1
                    left_val = 0.0
                    has_left_val = False
                    right_node = -1
                    right_val = 0.0
                    has_right_val = False
                    
                    e = feature.find("./left_val")
                    if e is None:
                        left_node = int(feature.find("./left_node").text)
                        has_left_val = False
                    else:
                        left_val = float(feature.find("./left_val").text)
                        has_left_val = True
                    
                    e = feature.find("./right_val")
                    if e is None:
                        right_node = int(feature.find("./right_node").text)
                        has_right_val = False
                    else:
                        right_val = float(feature.find("./right_val").text)
                        has_right_val = True

                                
                    rects = feature.findall("./feature/rects/_")
                    feature_rects = List_new()
                    for rect in rects:
                        r = Rect_fromString(rect.text)
                        List_append(feature_rects, r)
                        
                    f = Feature_new(feature_rects, thres2, left_val, left_node, has_left_val,
                                right_val, right_node, has_right_val, self.width, self.height)
                    List_delete(feature_rects, False)
                    List_append(tree_features, f)
                
                t = Tree_new(tree_features)
                List_delete(tree_features, False)
                List_append(stage_trees, t)

            st = Stage_new(stage_trees, thres)
            List_delete(stage_trees, False)
            List_append(self.stages, st)
    
    @cython.boundscheck(False)
    @cython.cdivision(True)
    def get_faces(self, np.ndarray image, 
                  Float_t scale_base=1.0, Float_t scale_inc=1.25,
                  Float_t increment=0.1, Int_t min_neighbors=1,
                  bint do_canny_pruning=True):
        """ Detects an object in an image using a haar cascade.
        
    Parameters
    ----------
    image: array or string
        The image in which we want to detect objects
    
    scale_base: float, optional
        the initial ratio between the window size and the Haar classifier size
        (default 2)
    
    scale_inc: float, optional
        The scale increment of the window size, at each step (default 1.25)
    
    increment : int, optional 
        The shift of the window at each sub-step, in terms of percentage of
        the window size (default 0.1)
    
    min_neighbors: int, optional
        When aggregating responses, if the number of aggregated windows is
        heigher than *min_neighbors*, then the total final window is just
        the average of the aggregated windows. (default 1)
    
    do_canny_pruning: boolean, optional
        Do we exclude some responses based on edge intensity (default True)
"""
        cdef:
            np.ndarray[Int_t, ndim=2] img
            List *ret
            ListIterator it
            np.ndarray[Int_t, ndim=2] ret_array
            Int_t* ret_val

        img = to_gray_scale(image).astype(Int)
        ret = det_get_objects(
                self.stages,
                self.width,
                self.height,
                image.shape[1],
                image.shape[0],
                <Int_t*> img.data, 
                scale_base, scale_inc,
                increment, do_canny_pruning)
        
        if ret.size == 0:
            List_delete(ret, False)
            return np.zeros((0,4), Int)
        else:
            ret_array = np.zeros((ret.size, 4), Int)
            ListIterator_init(&it, ret)
            for i in range(ret.size):
                ret_val = <Int_t*> ListIterator_next(&it)
                ret_array[i,0] = ret_val[0]
                ret_array[i,1] = ret_val[1]
                ret_array[i,2] = ret_val[2]
                ret_array[i,3] = ret_val[3]
            List_delete(ret, True)
            return self.merge(ret_array, min_neighbors)


    @cython.cdivision(True)
    @cython.boundscheck(False)
    def merge(self, np.ndarray[Int_t, ndim=2] rects, Int_t min_neighbors):
        cdef:
            Int_t nrects = len(rects)
            list resp = []
            np.ndarray[Int_t, ndim=2] rect
            np.ndarray[Int_t, ndim=1] ret = np.zeros(nrects, Int)
            np.ndarray[Int_t, ndim=1] neighbors
            np.ndarray[Int_t, ndim=1] r
            Int_t nb_classes = 0
            Int_t i, j, n
            bint found
        
        for i in range(nrects):
            found = False
            for j in range(i):
                if self.equals(rects, j, i):
                    found = True
                    ret[i] = ret[j]

            if not found:
                ret[i] = nb_classes
                nb_classes += 1
        
        neighbors = np.zeros(nb_classes, Int)
        rect = np.zeros((nb_classes, 4), Int)
        
        for i in range(nrects):
            j = ret[i]
            neighbors[j] += 1
            rect[j] += rects[i]
        
        for i in range(nb_classes):
            n = neighbors[i]
            if n >= min_neighbors:
                r = (rect[i] * 2 + n)/(2*n)
                resp.append(r)
        return np.array(resp)

    cdef inline bint equals(self, np.ndarray rects, Int_t i, Int_t j):
        cdef:
            np.ndarray[Int_t, ndim=2] rects_ = rects
            Int_t r1x = rects_[i, 0]
            Int_t r1y = rects_[i, 1]
            Int_t r1w = rects_[i, 2]
            Int_t r1h = rects_[i, 3]
            Int_t r2x = rects_[j, 0]
            Int_t r2y = rects_[j, 1]
            Int_t r2w = rects_[j, 2]
            Int_t r2h = rects_[j, 3]
            Int_t distance = <Int_t> (r1w * 0.2)

        if  (r2x <= r1x + distance) and \
            (r2x >= r1x - distance) and \
            (r2y <= r1y + distance) and \
            (r2y >= r1y - distance) and \
            (r2w <= <Int_t> (r1w * 1.2)) and \
            (<Int_t>(r2w * 1.2) >= r1w):
                return True

        if r1x >= r2x and (r1x + r1w) <= (r2x + r2w) and \
           r1y >= r2y and (r1y + r1h) <= (r2y + r2h):
                return True

        return False

def to_gray_scale(image):
    if image.ndim == 3:
        return (image * [0.3,0.59,0.11]).sum(-1)
    elif image.ndim == 2:
        return image
    else:
        raise ValueError("Unexpected shape for an image")

def detect(image, cascade, 
           scale_base=2.0, scale_inc=1.25,
           increment=0.1, min_neighbors=1,
           bint do_canny_pruning=True):
    """ Detects an object in an image using a haar cascade.
    
    Parameters
    ----------
    image: array or string
        The image in which we want to detect objects
    
    cascade: string
        the path to the xml cascade file
    
    scale_base: float, optional
        the initial ratio between the window size and the Haar classifier size
        (default 2)
    
    scale_inc: float, optional
        The scale increment of the window size, at each step (default 1.25)
    
    increment : int, optional 
        The shift of the window at each sub-step, in terms of percentage of
        the window size (default 0.1)
    
    min_neighbors: int, optional
        When aggregating responses, if the number of aggregated windows is
        heigher than *min_neighbors*, then the total final window is just
        the average of the aggregated windows. (default 1)
    
    do_canny_pruning: boolean, optional
        Do we exclude some responses based on edge intensity (default True)
    
    Returns
    -------
    
    An 2D (Nx4) array where each line corresponds to a detection window. The
    window parmeters are (x, y, width, height), (x,y) being the coordinates
    of the top left corner.
        
"""
    cdef Detector detector = Detector(cascade)
    if isinstance(image, str):
        image = imread(image)
    return detector.get_faces(image)
    
def detect_and_draw(image, cascade,
           scale_base=2.0, scale_inc=1.25,
           increment=0.1, min_neighbors=1,
           bint do_canny_pruning=True):
    if isinstance(image, str):
        image = imread(image)
    image = to_gray_scale(image)
    faces = detect(image, cascade)
    img = toimage(image)
    draw = ImageDraw(img)
    
    for x,y,w,h in faces:
        draw.rectangle((x,y,x+w, y+h), outline=255)
        draw.rectangle((x-1,y-1,x+w+1, y+h+1), outline=255)
        draw.rectangle((x+1,y+1,x+w-1, y+h-1), outline=255)
    return fromimage(img)

