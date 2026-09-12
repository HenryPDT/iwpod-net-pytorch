
from os.path import isfile

import numpy as np


class ShapeParseError(ValueError):
    """Strict per-line parse failure (carries line context)."""


def parse_shape_line(line, lineno=0):
    """Strict-parse one annotation line -> (pts (2,n) float array, n, text).

    Unlike Shape.read (lenient, crash-prone on bad input), this validates the
    corner count and coordinate payload and raises ShapeParseError naming the
    line. Trailing fields (e.g. a vehicle label) are returned as text.
    """
    parts = [p for p in line.split(",") if p != ""]
    if not parts:
        raise ShapeParseError(f"line {lineno}: empty")
    try:
        n = int(float(parts[0]))
    except (ValueError, IndexError) as e:
        raise ShapeParseError(f"line {lineno}: bad corner count") from e
    if len(parts) < 1 + 2 * n:
        raise ShapeParseError(f"line {lineno}: expected {2 * n} coordinates")
    try:
        # First 2n values only; anything after (e.g. a label) is text.
        nums = [float(v) for v in parts[1:1 + 2 * n]]
    except ValueError as e:
        raise ShapeParseError(f"line {lineno}: non-numeric values") from e
    text = parts[1 + 2 * n] if len(parts) > 1 + 2 * n else ""
    return np.array(nums, dtype=float).reshape(2, n), n, text


class Label:

    def __init__(self,cl=-1,tl=np.array([0.,0.]),br=np.array([0.,0.]),prob=None):
        self.__tl   = tl
        self.__br   = br
        self.__cl   = cl
        self.__prob = prob

    def __str__(self):
        return 'Class: %d, top_left(x:%f,y:%f), bottom_right(x:%f,y:%f)' % (self.__cl, self.__tl[0], self.__tl[1], self.__br[0], self.__br[1])

    def copy(self):
        return Label(self.__cl,self.__tl,self.__br)

    def wh(self): return self.__br-self.__tl

    def cc(self): return self.__tl + self.wh()/2

    def tl(self): return self.__tl

    def br(self): return self.__br

    def tr(self): return np.array([self.__br[0],self.__tl[1]])

    def bl(self): return np.array([self.__tl[0],self.__br[1]])

    def cl(self): return self.__cl

    def area(self): return np.prod(self.wh())

    def prob(self): return self.__prob

    def set_class(self,cl):
        self.__cl = cl

    def set_tl(self,tl):
        self.__tl = tl

    def set_br(self,br):
        self.__br = br

    def set_wh(self,wh):
        cc = self.cc()
        self.__tl = cc - .5*wh
        self.__br = cc + .5*wh

    def set_prob(self,prob):
        self.__prob = prob


def lread(file_path,label_type=Label):

    if not isfile(file_path):
        return []

    objs = []
    with open(file_path,'r') as fd:
        for line in fd:
            v       = line.strip().split()
            cl      = int(v[0])
            ccx,ccy = float(v[1]),float(v[2])
            w,h     = float(v[3]),float(v[4])
            prob    = float(v[5]) if len(v) == 6 else None

            cc  = np.array([ccx,ccy])
            wh  = np.array([w,h])

            objs.append(label_type(cl,cc-wh/2,cc+wh/2,prob=prob))

    return objs

def lwrite(file_path,labels,write_probs=True):
    with open(file_path,'w') as fd:
        for l in labels:
            cc,wh,cl,prob = (l.cc(),l.wh(),l.cl(),l.prob())
            if prob is not None and write_probs:
                fd.write('%d %f %f %f %f %f\n' % (cl,cc[0],cc[1],wh[0],wh[1],prob))
            else:
                fd.write('%d %f %f %f %f\n' % (cl,cc[0],cc[1],wh[0],wh[1]))



class Shape():

    def __init__(self,pts=np.zeros((2,0)),max_sides=4,text=''):
        self.pts = pts
        self.max_sides = max_sides
        self.text = text

    def isValid(self):
        return self.pts.shape[1] > 2

    def write(self,fp):
        fp.write('%d,' % self.pts.shape[1])
        ptsarray = self.pts.flatten()
        fp.write(''.join([('%f,' % value) for value in ptsarray]))
        fp.write('%s,' % self.text)
        fp.write('\n')

    def read(self,line):
        data        = line.strip().split(',')
        ss          = int(data[0])
        values      = data[1:(ss*2 + 1)]
        text        = data[(ss*2 + 1)] if len(data) >= (ss*2 + 2) else ''
        self.pts    = np.array([float(value) for value in values]).reshape((2,ss))
        self.text   = text

def readShapes(path):
    shapes = []
    with open(path) as fp:
        for line in fp:
            shape = Shape()
            shape.read(line)
            shapes.append(shape)
    return shapes

def writeShapes(path,shapes):
    if len(shapes):
        with open(path,'w') as fp:
            for shape in shapes:
                if shape.isValid():
                    shape.write(fp)

