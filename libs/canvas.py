
try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

#from PyQt4.QtOpenGL import *

from shape import Shape
from lib import distance
import math
import r_box_hint as r_hint
import numpy as np

CURSOR_DEFAULT = Qt.ArrowCursor
CURSOR_POINT = Qt.PointingHandCursor
CURSOR_DRAW = Qt.CrossCursor
CURSOR_MOVE = Qt.ClosedHandCursor
CURSOR_GRAB = Qt.OpenHandCursor

# class Canvas(QGLWidget):


class Canvas(QWidget):
    zoomRequest = pyqtSignal(int)
    scrollRequest = pyqtSignal(int, int)
    newShape = pyqtSignal()
    selectionChanged = pyqtSignal(bool)
    shapeMoved = pyqtSignal()
    drawingPolygon = pyqtSignal(bool)
    autoNewShape = pyqtSignal(str)
    remLabel = pyqtSignal(Shape)

    hideRRect = pyqtSignal(bool)
    hideNRect = pyqtSignal(bool)
    status = pyqtSignal(str)

    CREATE, EDIT = list(range(2))

    epsilon = 11.0



    def __init__(self, *args, **kwargs):
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.createModeType = "box"
        self.shapes = []
        self.current = None
        self.selectedShape = None  # save the selected shape here
        self.selectedShapeCopy = None
        self.lineColor = QColor(0, 0, 255)
        self.line = Shape(line_color=self.lineColor)
        self.prevPoint = QPointF()
        self.offsets = QPointF(), QPointF()
        self.scale = 1.0
        self.pixmap = QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.hVertex = None
        self._painter = QPainter()
        self._cursor = CURSOR_DEFAULT
        # Menus:
        self.menus = (QMenu(), QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)
        self.verified = False
        # judge can draw rotate rect
        self.canDrawRotatedRect = True
        self.hideRotated = False
        self.hideNormal = False
        self.canOutOfBounding = False
        self.showCenter = False

        self.imagePath = None

        _ = r_hint.init_evaluator("pretrained_dota")
        _ = r_hint.init_evaluator("online_learning")

    def print_shape_summary(self, shape):
        obj = shape
        for a in dir(obj):
            if not a.startswith('__') and not callable(getattr(obj, a)):
                print(a, " : ", getattr(obj, a))

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if not value:  # Create
            self.unHighlight()
            self.deSelectShape()

    def setCreateModeType(self, value="box"):
        self.createModeType = value

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
        self.hVertex = self.hShape = None

    def selectedVertex(self):
        return self.hVertex is not None

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        # print("Mouse Moving")
        pos = self.transformPos(ev.pos())

        self.restoreCursor()

        # Polygon drawing.
        if self.drawing():

            self.overrideCursor(CURSOR_DRAW)
            if self.current:
                if self.createModeType == "point-3" or self.createModeType == "point-4":
                    return
                color = self.lineColor
                if self.outOfPixmap(pos):
                    # Don't allow the user to draw outside the pixmap.
                    # Project the point to the pixmap's edges.
                    pos = self.intersectionPoint(self.current[-1], pos)
                elif len(self.current) > 1 and self.closeEnough(pos, self.current[0]):
                    # Attract line to starting point and colorise to alert the
                    # user:
                    pos = self.current[0]
                    color = self.current.line_color
                    self.overrideCursor(CURSOR_POINT)
                    self.current.highlightVertex(0, Shape.NEAR_VERTEX)
                self.line[1] = pos
                self.line.line_color = color
                self.repaint()
                self.current.highlightClear()
                self.status.emit("width is %d, height is %d." % (pos.x()-self.line[0].x(), pos.y()-self.line[0].y()))
            return

        # Polygon copy moving.
        if Qt.RightButton & ev.buttons():
            # print("right button")
            # if self.selectedShapeCopy and self.prevPoint:
            #     print("select shape copy")
            #     self.overrideCursor(CURSOR_MOVE)
            #     self.boundedMoveShape(self.selectedShapeCopy, pos)
            #     self.repaint()
            # elif self.selectedShape:
            #     print("select shape")
            #     self.selectedShapeCopy = self.selectedShape.copy()
            #     self.repaint()
            if self.selectedVertex() and self.selectedShape.isRotated:
                self.boundedRotateShape(pos)
                self.shapeMoved.emit()
                self.repaint()
            self.status.emit("(%d,%d)." % (pos.x(), pos.y()))
            return

        # Polygon/Vertex moving.
        if Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                # if self.outOfPixmap(pos):
                #     print("chule ")
                #     return
                # else:
                # print("meiyou chujie")
                self.boundedMoveVertex(pos)
                self.shapeMoved.emit()
                self.repaint()
            elif self.selectedShape and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShape(self.selectedShape, pos)
                self.shapeMoved.emit()
                self.repaint()
                self.status.emit("(%d,%d)." % (pos.x(), pos.y()))
            return

        # Just hovering over the canvas, 2 posibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip("Image")
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.hVertex, self.hShape = index, shape
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                # self.setToolTip("Click & drag to move point.")
                # self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.hVertex, self.hShape = None, shape
                # self.setToolTip(
                #     "Click & drag to move shape '%s'" % shape.label)
                # self.setStatusTip(self.toolTip())
                self.overrideCursor(CURSOR_GRAB)
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            if self.hShape:
                self.hShape.highlightClear()
                self.update()
            self.hVertex, self.hShape = None, None

        self.status.emit("(%d,%d)." % (pos.x(), pos.y()))


    def mousePressEvent(self, ev):
        # print("Mouse Press event")
        pos = self.transformPos(ev.pos())
        # print('sldkfj %d %d' % (pos.x(), pos.y()))
        if ev.button() == Qt.LeftButton:
            # print("Left button press")
            self.hideBackroundShapes(True)
            if self.drawing():
                if self.createModeType != "box":
                    pass
                else:
                    self.handleDrawing(pos)
            else:
                self.selectShapePoint(pos)
                self.prevPoint = pos
                self.repaint()
        elif ev.button() == Qt.RightButton and self.editing():
            # print("right button press")
            self.selectShapePoint(pos)
            self.hideBackroundShapes(True)
            # if self.selectedShape is not None:
            #     print('point is (%d, %d)' % (pos.x(), pos.y()))
            #     self.selectedShape.rotate(10)

            self.prevPoint = pos
            self.repaint()

    def mouseReleaseEvent(self, ev):
        # print("Mouse Release event")
        self.hideBackroundShapes(False)
        if ev.button() == Qt.RightButton and not self.selectedVertex():
            menu = self.menus[bool(self.selectedShapeCopy)]
            self.restoreCursor()
            if not menu.exec_(self.mapToGlobal(ev.pos()))\
               and self.selectedShapeCopy:
                # Cancel the move by deleting the shadow copy.
                self.selectedShapeCopy = None
                self.repaint()
        elif ev.button() == Qt.LeftButton and self.selectedShape:
            self.overrideCursor(CURSOR_GRAB)
        elif ev.button() == Qt.LeftButton:
            # print("Released left button")
            pos = self.transformPos(ev.pos())
            if self.drawing():
                self.handleDrawing(pos)

    def endMove(self, copy=False):
        assert self.selectedShape and self.selectedShapeCopy
        shape = self.selectedShapeCopy
        #del shape.fill_color
        #del shape.line_color
        if copy:
            self.shapes.append(shape)
            self.selectedShape.selected = False
            self.selectedShape = shape
            self.repaint()
        else:
            self.selectedShape.points = [p for p in shape.points]
        self.selectedShapeCopy = None

    def hideBackroundShapes(self, value):
        # print("hideBackroundShapes")
        self.hideBackround = value
        if self.selectedShape:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.repaint()

    def handleDrawing(self, pos):
        print("handling drawing")
        print(self.current)
        if self.current and self.current.reachMaxPoints() is False:
            print("inside current")
            if self.createModeType == "point-3":
                # print(len(self.current.points))
                if len(self.current.points) == 1:
                    print("One Point so far")
                    self.current.addPoint(pos)

                    a = self.current.points[0]
                    b = self.current.points[1]

                    angle1 = np.rad2deg(np.arctan2(a.y() - b.y(), a.x() - b.x()))
                    angle2 = np.rad2deg(np.arctan2(b.y() - a.y(), b.x() - a.x()))
                    angle = max(angle1, angle2)
                    self.current.direction = math.radians(angle)
                    print(angle)
                    # self.line[1] = pos
                    # self.setHiding()
                    # self.drawingPolygon.emit(True)
                    self.update()
                elif len(self.current.points) == 2:
                    print("Two Points so far")
                    a = self.current.points[0]
                    b = self.current.points[1]
                    c = pos

                    l = distance(b - c)
                    mab =  (a.y() - b.y()) / (a.x() - b.x() + 0.000001)
                    mbc = -1/(mab + .000001)
                    dx = (l / math.sqrt(1 + (mbc * mbc)));
                    dy = mbc * dx;

                    cxp = b.x() + dx;
                    cyp = b.y() + dy;

                    cxn = b.x() - dx;
                    cyn = b.y() - dy;

                    cp = QPointF(cxp, cyp)
                    cn = QPointF(cxn, cyn)

                    if distance(cp - c) <= distance(cn - c):
                        c = cp
                    else:
                        c = cn

                    dx = a.x() + c.x() - b.x()
                    dy = a.y() + c.y() - b.y()
                    d = QPointF(dx, dy)

                    self.current.addPoint(c)
                    self.current.addPoint(d)

                    # self.line.addPoint(c)
                    # self.line.addPoint(d)

                    self.finalise()
            elif self.createModeType == "point-4":
                if len(self.current.points) == 1:
                    print("One Point so far")
                    self.current.addPoint(pos)

                    a = self.current.points[0]
                    b = self.current.points[1]

                    angle1 = np.rad2deg(np.arctan2(a.y() - b.y(), a.x() - b.x()))
                    angle2 = np.rad2deg(np.arctan2(b.y() - a.y(), b.x() - a.x()))
                    angle = max(angle1, angle2)
                    self.current.direction = math.radians(angle)
                    print(angle)
                    self.update()
                elif len(self.current.points) == 2:
                    self.current.addPoint(pos)
                    self.update()
                elif len(self.current.points) == 3:
                    self.current.addPoint(pos)

                    a = self.current.points[0]
                    b = self.current.points[1]
                    c = self.current.points[2]
                    d = self.current.points[3]

                    # if c and d symmetric on either side
                    # l = distance(c-d)
                    # l = l/2
                    # cntr = QPointF((a.x() + b.x())/2, (a.y() + b.y())/2)
                    #
                    # mab =  (a.y() - cntr.y()) / (a.x() - cntr.x() + 0.000001)
                    # mbc = -1/(mab + .000001)
                    # dx = (l / math.sqrt(1 + (mbc * mbc)));
                    # dy = mbc * dx;
                    #
                    # #point c
                    # cx = cntr.x() + dx;
                    # cy = cntr.y() + dy;
                    #
                    # #point d
                    # dx = cntr.x() - dx;
                    # dy = cntr.y() - dy;
                    #
                    # c = QPointF(cx, cy)
                    # d = QPointF(dx, dy)
                    #
                    #
                    # x3x = c.x() - a.x() + cntr.x()
                    # x3y = c.y() - a.y() + cntr.y()
                    #
                    # x1x = 2 * cntr.x() - x3x
                    # x1y = 2 * cntr.y() - x3y
                    #
                    # x4x = 2 * a.x() - x1x
                    # x4y = 2 * a.y() - x1y
                    # #
                    # x2x = x1x + x3x - x4x
                    # x2y = x1y + x3y - x4y
                    #

                    # determine c
                    cntr = QPointF((a.x() + b.x())/2, (a.y() + b.y())/2)
                    l = distance(c - cntr)
                    mab =  (a.y() - cntr.y()) / (a.x() - cntr.x() + 0.000001)
                    mbc = -1/(mab + .000001)
                    dx = (l / math.sqrt(1 + (mbc * mbc)));
                    dy = mbc * dx;

                    cxp = cntr.x() + dx;
                    cyp = cntr.y() + dy;

                    cxn = cntr.x() - dx;
                    cyn = cntr.y() - dy;

                    cp = QPointF(cxp, cyp)
                    cn = QPointF(cxn, cyn)

                    c = cp if distance(cp - c) <= distance(cn - c) else cn


                    # determine d
                    cntr = QPointF((a.x() + b.x())/2, (a.y() + b.y())/2)
                    l = distance(d - cntr)
                    mab =  (a.y() - cntr.y()) / (a.x() - cntr.x() + 0.000001)
                    mbc = -1/(mab + .000001)
                    dx = (l / math.sqrt(1 + (mbc * mbc)));
                    dy = mbc * dx;

                    dxp = cntr.x() + dx;
                    dyp = cntr.y() + dy;

                    dxn = cntr.x() - dx;
                    dyn = cntr.y() - dy;

                    dp = QPointF(dxp, dyp)
                    dn = QPointF(dxn, dyn)

                    d = dp if distance(dp - d) <= distance(dn - d) else dn

                    x1x = a.x() + c.x() - cntr.x()
                    x1y = a.y() + c.y() - cntr.y()

                    x2x = b.x() + c.x() - cntr.x()
                    x2y = b.y() + c.y() - cntr.y()

                    x3x = b.x() + d.x() - cntr.x()
                    x3y = b.y() + d.y() - cntr.y()

                    x4x = a.x() + d.x() - cntr.x()
                    x4y = a.y() + d.y() - cntr.y()

                    self.current.points[0] = QPointF(x1x, x1y)
                    self.current.points[1] = QPointF(x2x, x2y)
                    self.current.points[2] = QPointF(x3x, x3y)
                    self.current.points[3] = QPointF(x4x, x4y)

                    self.finalise()
            else:
                initPos = self.current[0]
                minX = initPos.x()
                minY = initPos.y()
                targetPos = self.line[1]
                maxX = targetPos.x()
                maxY = targetPos.y()
                self.current.addPoint(QPointF(maxX, minY))
                self.current.addPoint(targetPos)
                self.current.addPoint(QPointF(minX, maxY))
                self.current.addPoint(initPos)
                self.line[0] = self.current[-1]
                if self.current.isClosed():
                    self.finalise()
        elif not self.outOfPixmap(pos):
            print("creating current")
            self.current = Shape()
            self.current.addPoint(pos)
            self.line.points = [pos, pos]
            self.setHiding()
            self.drawingPolygon.emit(True)
            self.update()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        return self.drawing() and self.current and len(self.current) > 2

    def mouseDoubleClickEvent(self, ev):
        # We need at least 4 points here, since the mousePress handler
        # adds an extra one before this handler is called.
        if self.canCloseShape() and len(self.current) > 3:
            self.current.popPoint()
            self.finalise()

    def selectShape(self, shape):
        self.deSelectShape()
        shape.selected = True
        self.selectedShape = shape
        self.setHiding()
        self.selectionChanged.emit(True)
        self.update()

    def selectShapePoint(self, point):
        """Select the first shape created which contains this point."""
        self.deSelectShape()
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)

            shape.selected = True
            self.selectedShape = shape
            self.calculateOffsets(shape, point)
            self.setHiding()
            self.selectionChanged.emit(True)

            return
        for shape in reversed(self.shapes):
            if self.isVisible(shape) and shape.containsPoint(point):
                shape.selected = True
                self.selectedShape = shape
                self.calculateOffsets(shape, point)
                self.setHiding()
                self.selectionChanged.emit(True)
                return

    def calculateOffsets(self, shape, point):
        rect = shape.boundingRect()
        x1 = rect.x() - point.x()
        y1 = rect.y() - point.y()
        x2 = (rect.x() + rect.width()) - point.x()
        y2 = (rect.y() + rect.height()) - point.y()
        self.offsets = QPointF(x1, y1), QPointF(x2, y2)

    def boundedMoveVertex(self, pos):
        # print("Moving Vertex")
        index, shape = self.hVertex, self.hShape
        point = shape[index]

        if not self.canOutOfBounding and self.outOfPixmap(pos):
            return
            # pos = self.intersectionPoint(point, pos)

        # print("index is %d" % index)
        sindex = (index + 2) % 4
        # get the other 3 points after transformed
        p2,p3,p4 = self.getAdjointPoints(shape.direction, shape[sindex], pos, index)

        pcenter = (pos+p3)/2
        if self.canOutOfBounding and self.outOfPixmap(pcenter):
            return
        # if one pixal out of map , do nothing
        if not self.canOutOfBounding and (self.outOfPixmap(p2) or
            self.outOfPixmap(p3) or
            self.outOfPixmap(p4)):
                return

        # move 4 pixal one by one
        shape.moveVertexBy(index, pos - point)
        lindex = (index + 1) % 4

        rindex = (index + 3) % 4
        shape[lindex] = p2
        # shape[sindex] = p3
        shape[rindex] = p4
        shape.close()

        # calculate the height and weight, and show it
        w = math.sqrt((p4.x()-p3.x()) ** 2 + (p4.y()-p3.y()) ** 2)
        h = math.sqrt((p3.x()-p2.x()) ** 2 + (p3.y()-p2.y()) ** 2)
        self.status.emit("width is %d, height is %d." % (w,h))


    def getAdjointPoints(self, theta, p3, p1, index):
        # p3 = center
        # p3 = 2*center-p1
        a1 = math.tan(theta)
        if (a1 == 0):
            if index % 2 == 0:
                p2 = QPointF(p3.x(), p1.y())
                p4 = QPointF(p1.x(), p3.y())
            else:
                p4 = QPointF(p3.x(), p1.y())
                p2 = QPointF(p1.x(), p3.y())
        else:
            a3 = a1
            a2 = - 1/a1
            a4 = - 1/a1
            b1 = p1.y() - a1 * p1.x()
            b2 = p1.y() - a2 * p1.x()
            b3 = p3.y() - a1 * p3.x()
            b4 = p3.y() - a2 * p3.x()

            if index % 2 == 0:
                p2 = self.getCrossPoint(a1,b1,a4,b4)
                p4 = self.getCrossPoint(a2,b2,a3,b3)
            else:
                p4 = self.getCrossPoint(a1,b1,a4,b4)
                p2 = self.getCrossPoint(a2,b2,a3,b3)

        return p2,p3,p4

    def getCrossPoint(self,a1,b1,a2,b2):
        x = (b2-b1)/(a1-a2)
        y = (a1*b2 - a2*b1)/(a1-a2)
        return QPointF(x,y)

    def boundedRotateShape(self, pos):
        # print("Rotate Shape2")
        # judge if some vertex is out of pixma
        index, shape = self.hVertex, self.hShape
        point = shape[index]

        angle = self.getAngle(shape.center,pos,point)
        # for i, p in enumerate(shape.points):
        #     if self.outOfPixmap(shape.rotatePoint(p,angle)):
        #         # print("out of pixmap")
        #         return
        if not self.rotateOutOfBound(angle):
            shape.rotate(angle)
            self.prevPoint = pos

    def getAngle(self, center, p1, p2):
        dx1 = p1.x() - center.x();
        dy1 = p1.y() - center.y();

        dx2 = p2.x() - center.x();
        dy2 = p2.y() - center.y();

        c = math.sqrt(dx1*dx1 + dy1*dy1) * math.sqrt(dx2*dx2 + dy2*dy2)
        if c == 0: return 0
        y = (dx1*dx2+dy1*dy2)/c
        if y>1: return 0
        angle = math.acos(y)

        if (dx1*dy2-dx2*dy1)>0:
            return angle
        else:
            return -angle

    def boundedMoveShape(self, shape, pos):
        if shape.isRotated and self.canOutOfBounding:
            c = shape.center
            dp = pos - self.prevPoint
            dc = c + dp
            if dc.x() < 0:
                dp -= QPointF(min(0,dc.x()), 0)
            if dc.y() < 0:
                dp -= QPointF(0, min(0,dc.y()))
            if dc.x() >= self.pixmap.width():
                dp += QPointF(min(0, self.pixmap.width() - 1  - dc.x()), 0)
            if dc.y() >= self.pixmap.height():
                dp += QPointF(0, min(0, self.pixmap.height() - 1 - dc.y()))

        else:
            if self.outOfPixmap(pos):
                return False  # No need to move
            o1 = pos + self.offsets[0]
            if self.outOfPixmap(o1):
                pos -= QPointF(min(0, o1.x()), min(0, o1.y()))
            o2 = pos + self.offsets[1]
            if self.outOfPixmap(o2):
                pos += QPointF(min(0, self.pixmap.width() - 1 - o2.x()),
                               min(0, self.pixmap.height() - 1 - o2.y()))
            dp = pos - self.prevPoint
        # The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason. XXX
        #self.calculateOffsets(self.selectedShape, pos)

        if dp:
            shape.moveBy(dp)
            self.prevPoint = pos
            shape.close()
            return True
        return False


    def boundedMoveShape2(self, shape, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QPointF(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QPointF(min(0, self.pixmap.width() - o2.x()),
                           min(0, self.pixmap.height() - o2.y()))
        # The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason. XXX
        #self.calculateOffsets(self.selectedShape, pos)
        dp = pos - self.prevPoint
        if dp:
            shape.moveBy(dp)
            self.prevPoint = pos
            shape.close()
            return True
        return False

    def deSelectShape(self):
        if self.selectedShape:
            self.selectedShape.selected = False
            self.selectedShape = None
            self.setHiding(False)
            self.selectionChanged.emit(False)
            self.update()

    def deleteSelected(self):
        if self.selectedShape:
            shape = self.selectedShape
            self.shapes.remove(self.selectedShape)
            self.selectedShape = None
            self.update()
            return shape

    def copySelectedShape(self):
        if self.selectedShape:
            shape = self.selectedShape.copy()
            self.deSelectShape()
            self.shapes.append(shape)
            shape.selected = True
            self.selectedShape = shape
            self.boundedShiftShape(shape)
            return shape

    def boundedShiftShape(self, shape):
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shape[0]
        offset = QPointF(2.0, 2.0)
        self.calculateOffsets(shape, point)
        self.prevPoint = point
        if not self.boundedMoveShape(shape, point - offset):
            self.boundedMoveShape(shape, point + offset)

    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.HighQualityAntialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)
        Shape.scale = self.scale
        for shape in self.shapes:
            if (shape.selected or not self._hideBackround) and self.isVisible(shape):
                if (shape.isRotated and not self.hideRotated) or (not shape.isRotated and not self.hideNormal):
                    shape.fill = shape.selected or shape == self.hShape
                    shape.paint(p)
                elif self.showCenter:
                    shape.fill = shape.selected or shape == self.hShape
                    shape.paintNormalCenter(p)

        if self.current:
            self.current.paint(p)
            self.line.paint(p)
        if self.selectedShapeCopy:
            self.selectedShapeCopy.paint(p)

        # Paint rect
        if self.current is not None and len(self.line) == 2:
            leftTop = self.line[0]
            rightBottom = self.line[1]
            rectWidth = rightBottom.x() - leftTop.x()
            rectHeight = rightBottom.y() - leftTop.y()
            color = QColor(0, 220, 0)
            p.setPen(color)
            brush = QBrush(Qt.BDiagPattern)
            p.setBrush(brush)
            p.drawRect(leftTop.x(), leftTop.y(), rectWidth, rectHeight)

            #draw dialog line of rectangle
            p.setPen(self.lineColor)
            p.drawLine(leftTop.x(),rightBottom.y(),rightBottom.x(),leftTop.y())

        self.setAutoFillBackground(True)
        if self.verified:
            pal = self.palette()
            pal.setColor(self.backgroundRole(), QColor(184, 239, 38, 128))
            self.setPalette(pal)
        else:
            pal = self.palette()
            pal.setColor(self.backgroundRole(), QColor(232, 232, 232, 255))
            self.setPalette(pal)

        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical coordinates."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QPointF(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() < w and 0 <= p.y() < h)

    def finalise(self):
        assert self.current
        self.current.isRotated = self.canDrawRotatedRect
        self.print_shape_summary(self.current)
        # print(self.canDrawRotatedRect)
        self.current.close()
        self.shapes.append(self.current)
        self.current = None
        self.setHiding(False)
        self.newShape.emit()
        self.update()

    def closeEnough(self, p1, p2):
        #d = distance(p1 - p2)
        #m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        return distance(p1 - p2) < self.epsilon

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [(0, 0),
                  (size.width(), 0),
                  (size.width(), size.height()),
                  (0, size.height())]
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QPointF(x, y)

    def intersectingEdges(self, x1y1, x2y2, points):
        """For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen."""
        x1, y1 = x1y1
        x2, y2 = x2y2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = distance(m - QPointF(x2, y2))
                print("return=",d,i,(x,y))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        qt_version = 4 if hasattr(ev, "delta") else 5
        if qt_version == 4:
            if ev.orientation() == Qt.Vertical:
                v_delta = ev.delta()
                h_delta = 0
            else:
                h_delta = ev.delta()
                v_delta = 0
        else:
            delta = ev.angleDelta()
            h_delta = delta.x()
            v_delta = delta.y()
        # print('scrolling vdelta is %d, hdelta is %d' % (v_delta, h_delta))
        mods = ev.modifiers()
        if Qt.ControlModifier == int(mods) and v_delta:
            self.zoomRequest.emit(v_delta)
        else:
            v_delta and self.scrollRequest.emit(v_delta, Qt.Vertical)
            h_delta and self.scrollRequest.emit(h_delta, Qt.Horizontal)
        ev.accept()

    def keyPressEvent(self, ev):
        key = ev.key()

        if key == Qt.Key_Escape and self.current:
            print('ESC press')
            self.current = None
            self.drawingPolygon.emit(False)
            self.update()
        elif key == Qt.Key_Return and self.canCloseShape():
            self.finalise()
        elif key == Qt.Key_Left and self.selectedShape:
            self.moveOnePixel('Left')
        elif key == Qt.Key_Right and self.selectedShape:
            self.moveOnePixel('Right')
        elif key == Qt.Key_Up and self.selectedShape:
            self.moveOnePixel('Up')
        elif key == Qt.Key_Down and self.selectedShape:
            self.moveOnePixel('Down')
        elif key == Qt.Key_Z and self.selectedShape and self.selectedShape.isRotated and not self.rotateOutOfBound(0.1):
            self.selectedShape.rotate(0.1)
            self.shapeMoved.emit()
            self.update()
        elif key == Qt.Key_X and self.selectedShape and\
             self.selectedShape.isRotated and not self.rotateOutOfBound(0.01):
            self.selectedShape.rotate(0.01)
            self.shapeMoved.emit()
            self.update()
        elif key == Qt.Key_C and self.selectedShape and\
             self.selectedShape.isRotated and not self.rotateOutOfBound(-0.01):
            self.selectedShape.rotate(-0.01)
            self.shapeMoved.emit()
            self.update()
        elif key == Qt.Key_V and self.selectedShape and\
             self.selectedShape.isRotated and not self.rotateOutOfBound(-0.1):
            self.selectedShape.rotate(-0.1)
            self.shapeMoved.emit()
            self.update()
        elif key == Qt.Key_R:
            self.hideRotated = not self.hideRotated
            self.hideRRect.emit(self.hideRotated)
            self.update()
        elif key == Qt.Key_N:
            self.hideNormal = not self.hideNormal
            self.hideNRect.emit(self.hideNormal)
            self.update()
        elif key == Qt.Key_O:
            self.canOutOfBounding = not self.canOutOfBounding
        elif key == Qt.Key_B:
            self.showCenter = not self.showCenter
            self.update()
        elif key == Qt.Key_H and self.selectedShape:
             self.process_hint_single_object(method = "image_processing")
        elif key == Qt.Key_J:
              self.process_hint_multiple_objects(method = "image_processing")
        # elif key == Qt.Key_K and self.selectedShape:
        #      self.process_hint_single_object(method = "mask_rcnn")
        # elif key == Qt.Key_L:
        #      self.process_hint_multiple_objects(method = "mask_rcnn")
        elif key == Qt.Key_K:
             self.process_hint_based_on_detections(mode = "global")
        elif key == Qt.Key_L:
             self.process_hint_based_on_detections(mode = "box", selected=False)
        else:
            pass

    def get_outer_rect(self, points):
        points = np.array(points).astype(int)
        points[points < 0] = 0

        xmin = min(points[:, 0])
        xmax = max(points[:, 0])

        ymin = min(points[:, 1])
        ymax = max(points[:, 1])

        bbox = [xmin, ymin, xmax, ymax]
        return bbox

    def process_hint_based_on_detections(self, mode="global", method="online_learning", selected=False):
        if mode=="box":
            if selected:
                process_shapes = [self.selectedShape]
            else:
                process_shapes = self.shapes

            bboxes = []
            for shape in process_shapes:
                points = [(p.x(), p.y()) for p in shape.points]
                bboxes.append(self.get_outer_rect(points))

            hints = r_hint.detect_orientation_based_on_detections(self.imagePath, mode=mode, method = method, bboxes = np.array(bboxes))

            if selected:
                hint = hints[0]
                if hint['status']:
                    self.selectedShape.direction = hint['direction']
                    self.selectedShape.isRotated = hint['isRotated']

                    self.selectedShape.center = QPointF(hint['cx'], hint['cy'])

                    self.selectedShape.points = []
                    for i, p in enumerate(hint['points']):
                        self.selectedShape.addPoint(QPointF(p[0], p[1]))

                    self.selectedShape.close()
                    self.shapeMoved.emit()
                    self.update()
            else:
                for i in range(len(hints)):
                    if hints[i]['status']:
                        self.shapes[i].direction = hints[i]['direction']
                        self.shapes[i].isRotated = hints[i]['isRotated']

                        self.shapes[i].center = QPointF(hints[i]['cx'], hints[i]['cy'])

                        self.shapes[i].points = []
                        for j, p in enumerate(hints[i]['points']):
                            self.shapes[i].addPoint(QPointF(p[0], p[1]))

                        self.shapes[i].close()

                self.current = None
                self.selectedShape = None
                self.selectedShapeCopy = None

                self.shapeMoved.emit()
                self.update()
        elif mode=="global":
            hints = r_hint.detect_orientation_based_on_detections(self.imagePath, mode=mode, method=method)

            ln = len(self.shapes)
            while ln:
                shape = self.shapes[0]
                self.shapes.remove(self.shapes[0])
                self.remLabel.emit(shape)
                self.update()
                ln = ln - 1

            for i in range(len(hints)):
                if hints[i]['status']:
                    self.shapes.append(Shape())
                    self.shapes[i].direction = hints[i]['direction']
                    self.shapes[i].isRotated = hints[i]['isRotated']

                    self.shapes[i].center = QPointF(hints[i]['cx'], hints[i]['cy'])

                    self.shapes[i].points = []
                    for j, p in enumerate(hints[i]['points']):
                        self.shapes[i].addPoint(QPointF(p[0], p[1]))

                    self.shapes[i].close()
                    self.autoNewShape.emit(hints[i]['label'])

            self.current = None
            self.selectedShape = None
            self.selectedShapeCopy = None

            self.shapeMoved.emit()
            self.update()

    def process_hint_single_object(self, method = "image_processing"):
        self.print_shape_summary(self.selectedShape)

        points = [(p.x(), p.y()) for p in self.selectedShape.points]
        center = self.selectedShape.center
        cx = center.x()
        cy = center.y()

        hint = r_hint.generate_bbox_hint(self.imagePath, cx, cy, points, method)

        if hint['status']:
            self.selectedShape.direction = hint['direction']
            self.selectedShape.isRotated = hint['isRotated']

            self.selectedShape.center = QPointF(hint['cx'], hint['cy'])

            for i, p in enumerate(hint['points']):
                self.selectedShape.points[i] = QPointF(p[0], p[1])

            self.shapeMoved.emit()
            self.update()

    def process_hint_multiple_objects(self, method = "image_processing"):
        print("convert all")
        print(len(self.shapes))

        for i in range(len(self.shapes)):
            points = [(p.x(), p.y()) for p in self.shapes[i].points]
            center = self.shapes[i].center
            cx = center.x()
            cy = center.y()

            hint = r_hint.generate_bbox_hint(self.imagePath, cx, cy, points, method)

            if hint['status']:
                self.shapes[i].direction = hint['direction']
                self.shapes[i].isRotated = hint['isRotated']

                self.shapes[i].center = QPointF(hint['cx'], hint['cy'])

                for j, p in enumerate(hint['points']):
                    self.shapes[i].points[j] = QPointF(p[0], p[1])

        self.current = None
        self.selectedShape = None
        self.selectedShapeCopy = None

        self.shapeMoved.emit()
        self.update()


    def rotateOutOfBound(self, angle):
        if self.canOutOfBounding:
            return False
        for i, p in enumerate(self.selectedShape.points):
            if self.outOfPixmap(self.selectedShape.rotatePoint(p,angle)):
                return True
        return False

    def moveOnePixel(self, direction):
        # print(self.selectedShape.points)
        if direction == 'Left' and not self.moveOutOfBound(QPointF(-1.0, 0)):
            # print("move Left one pixel")
            self.selectedShape.points[0] += QPointF(-1.0, 0)
            self.selectedShape.points[1] += QPointF(-1.0, 0)
            self.selectedShape.points[2] += QPointF(-1.0, 0)
            self.selectedShape.points[3] += QPointF(-1.0, 0)
            self.selectedShape.center += QPointF(-1.0, 0)
        elif direction == 'Right' and not self.moveOutOfBound(QPointF(1.0, 0)):
            # print("move Right one pixel")
            self.selectedShape.points[0] += QPointF(1.0, 0)
            self.selectedShape.points[1] += QPointF(1.0, 0)
            self.selectedShape.points[2] += QPointF(1.0, 0)
            self.selectedShape.points[3] += QPointF(1.0, 0)
            self.selectedShape.center += QPointF(1.0, 0)
        elif direction == 'Up' and not self.moveOutOfBound(QPointF(0, -1.0)):
            # print("move Up one pixel")
            self.selectedShape.points[0] += QPointF(0, -1.0)
            self.selectedShape.points[1] += QPointF(0, -1.0)
            self.selectedShape.points[2] += QPointF(0, -1.0)
            self.selectedShape.points[3] += QPointF(0, -1.0)
            self.selectedShape.center += QPointF(0, -1.0)
        elif direction == 'Down' and not self.moveOutOfBound(QPointF(0, 1.0)):
            # print("move Down one pixel")
            self.selectedShape.points[0] += QPointF(0, 1.0)
            self.selectedShape.points[1] += QPointF(0, 1.0)
            self.selectedShape.points[2] += QPointF(0, 1.0)
            self.selectedShape.points[3] += QPointF(0, 1.0)
            self.selectedShape.center += QPointF(0, 1.0)
        self.shapeMoved.emit()
        self.repaint()

    def moveOutOfBound(self, step):
        points = [p1+p2 for p1, p2 in zip(self.selectedShape.points, [step]*4)]
        return True in map(self.outOfPixmap, points)

    def setLastLabel(self, text):
        assert text
        self.shapes[-1].label = text
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        self.line.points = [self.current[-1], self.current[0]]
        self.drawingPolygon.emit(True)

    def resetAllLines(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        self.line.points = [self.current[-1], self.current[0]]
        self.drawingPolygon.emit(True)
        self.current = None
        self.drawingPolygon.emit(False)
        self.update()

    def loadPixmap(self, pixmap):
        self.pixmap = pixmap
        self.shapes = []
        self.repaint()

    def loadShapes(self, shapes):
        self.shapes = list(shapes)
        self.current = None
        self.repaint()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.repaint()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.update()
