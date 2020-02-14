from Tkinter import Tk
from tkFileDialog import askopenfilename, asksaveasfile
import Tkinter
import tkMessageBox
import shapefile
import laspy
import numpy as np
import os
import time

def integer(geom):
    geometry = []
    append = geometry.append
    for point in geom:
        x, y = point[0], point[1]
        x, y = int(x*1000), int(y*1000)
        append([x, y])
    geometry = np.array(geometry) #Closed polygon
    return geometry

def inside_polygon(pt, index, minY, maxY, maxX, geom, fraction):
    condition1 = np.logical_and(pt[1] > minY, pt[1] < maxY)
    condition2 = pt[0] < maxX
    condition12 = np.logical_and(condition1, condition2)

    intersX = geom[:,0][:-1] + (pt[1] - geom[:,1][:-1])*fraction
    condition3 = pt[0] <= intersX
    
    truth = np.logical_and(condition12, condition3)
    intersections = truth[truth == True].size

    if intersections%2 == 1:
        tiling(pt, index)

def tiling(pt, index):
    found = np.where(np.logical_and(np.logical_and(pt[0] >= tiles[:,2], pt[0] < tiles[:,4]),
                                    np.logical_and(pt[1] >= tiles[:,3], pt[1] < tiles[:,5])))
    col, row = tiles[found][0][0], tiles[found][0][1]
    tile_dict[col, row].append(index)

def write():
    for i in xrange (columns):
        for j in xrange (rows):
            if tile_dict[i,j] != []:
                write_tile = tiles[i*columns + j]
                name = [j, i, write_tile[2]/1000.0,write_tile[3]/1000.0
                        , write_tile[4]/1000.0, write_tile[5]/1000.0]
                fileName = "{},{},{},{},{},{}.las".format(*name)
                outFile = laspy.file.File(path + fileName, mode = "w",
                                      header = head)
                outFile.points = inFile.points[tile_dict[i, j]]
                outFile.header.scale = [1,1,1]
                outFile.close()

if __name__ == '__main__':
    #SHP Message Box
    Tk().withdraw()
    tkMessageBox.showinfo("Information", "Choose .shp file")
    
    #SHP Selection Window
    Tk().withdraw()
    SHPfile = askopenfilename()
    shpFound = False

    while shpFound == False:
        try:
            sf = shapefile.Reader(SHPfile)
            shpFound = True
        except:
            tkMessageBox.showwarning('WARNING!', 'Choose .shp file')
            Tk().withdraw()
            SHPfile = askopenfilename()

    #Read .shp file
    records = sf.iterShapeRecords()
    record = next(records)

    #Multipolygon Geometry
    geom = record.shape.points
    geom = integer(geom)

    #Edges
    minY = np.fmin(geom[:,1][:-1], geom[:,1][1:])
    maxY = np.fmax(geom[:,1][:-1], geom[:,1][1:])
    maxX = np.fmax(geom[:,0][:-1], geom[:,0][1:])
    nom = geom[:,0][1:] - geom[:,0][:-1]
    denom = geom[:,1][1:] - geom[:,1][:-1]
    fraction = np.divide(nom, denom)
  
    #Bounding Rectangle
    x_min, x_max = np.amin(geom[:,0]), np.amax(geom[:,0])
    y_min, y_max = np.amin(geom[:,1]), np.amax(geom[:,1])

    #LAS Message Box
    Tk().withdraw()
    tkMessageBox.showinfo("Information", "Choose .las file(s)")

    #LAS Selection Window
    Tk().withdraw()
    LASfile = askopenfilename()
    lasFound = False

    while lasFound == False:
        try:
            #Read .LAS file
            begin = time.clock()
            inFile = laspy.file.File(LASfile, mode = "r")
            lasFound = True
        except:
            tkMessageBox.showwarning('WARNING!', 'Choose .las file')
            Tk().withdraw()
            LASfile = askopenfilename()

    #Create 100m tiles
    x_ext = x_max - x_min
    y_ext = y_max - y_min

    columns = (x_ext / 100000) +1
    rows = (y_ext / 100000) +1

    head = inFile.header
    path = os.getcwd() + '/Tiles/'
    tiles = []
    tile_dict = {}
    append = tiles.append
    for i in xrange(columns):
        for j in xrange(rows):
            xmin, xmax = x_min + i*10**5, x_min + (i+1)*10**5
            ymin, ymax = y_min + j*10**5, y_min + (j+1)*10**5
            tile = [i, j, xmin, ymin, xmax, ymax]
            append(tile)
            tile_dict[i, j] = []
    tiles = np.array(tiles)

    for i in xrange(500):
        print i
        #Sampling interval
        start = i*10**6
        end = (i+1)*10**6
        
        #X, Y, classes arrays
        X, Y = inFile.get_x()[start:end], inFile.get_y()[start:end]
        classes = inFile.get_raw_classification()[start:end]
        indices = np.arange(10**6) + start
        
        #If you want to filter data:Keep only vegetation, ground, buildings
        #keep = np.logical_or(np.logical_or(classes == 1, classes == 2),
        #                     classes == 6)
        #X, Y = X[keep], Y[keep]
        #indices = indices[keep]

        #Check inside bbox
        incl = np.where(np.logical_and(np.logical_and(X>x_min, X<x_max),
                                       np.logical_and(Y>y_min, Y<y_max)))
        pts = np.stack((X[incl], Y[incl]), axis=-1)
        indices = indices[incl]

        for pt, index in zip(pts, indices):
            inside_polygon(pt, index, minY, maxY, maxX, geom, fraction)

    write()
        
    end = time.clock()
    duration = end - begin
    print('Duration: {:.3f} s'.format(duration))
