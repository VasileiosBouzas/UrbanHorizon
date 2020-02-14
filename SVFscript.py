import numpy as np
import laspy as lp
import json
import math
import os
import matplotlib.pyplot as plt
import time
import sys
from shapely.geometry import Polygon

# Determine tile containing point,
# based on whether point lies within tile's extent
def find_tile():
    for tile in tilelist:
        # tile name contains coordinates of bounding box
        new_tile = tile.strip(".las").split(",")

        tile_min_x = float(new_tile[2])
        tile_min_y = float(new_tile[3])
        tile_max_x = float(new_tile[4])
        tile_max_y = float(new_tile[5])
        if tile_min_x <= x <= tile_max_x and tile_min_y <= y <= tile_max_y:
            return int(new_tile[0]), int(new_tile[1])

# Search for the tiles that are adjacent to the initially selected tile
def find_tile_grid(row, col):
    iteration_list = [-1, 0, 1]
    tile_grid = []
    
    for i in iteration_list:
        for j in iteration_list:
            tilename_start = "{},{}".format((row + i), (col + j))
            for tile in tilelist:
                if tile.startswith(tilename_start):
                    tile_grid.append(tile)
                    break
    # tile_grid: list of 9 tiles
    return tile_grid

# determine height of viewpoint by sampling the ground points
# center: location of viewpoint
def getheight(tile_grid):
    center = np.array([x, y])
    pointheight = 0
    points_number = 0
    
    for tile in tile_grid:
        # read the .las file
        file_input = lp.file.File("{}{}{}".format(path, "/Tiles/", tile), mode='r')

        # keep groundpoints satisfying ground_rules:
        # classification 2 for ground, inside las file
        # keep points within radius of 5 metres
        ground_rules = np.logical_and(
            file_input.raw_classification == 2,
            np.sqrt(np.sum((np.vstack((file_input.x, file_input.y)).transpose() - center) ** 2, axis=1)) <= 1)
        build_rules = np.logical_and(
            file_input.raw_classification == 6,
            np.sqrt(np.sum((np.vstack((file_input.x, file_input.y)).transpose() - center) ** 2, axis=1)) <= 1)
            
        ground_points = file_input.points[ground_rules]
        build_points = file_input.points[build_rules]

        # make array with heights of each point
        if ground_points.size > build_points.size:
            ground_point_heights = np.array((ground_points['point']['Z'])).transpose()
        else:
            ground_point_heights = np.array((build_points['point']['Z'])).transpose()
            
        if ground_point_heights.size > 0:
            pointheight += float(np.sum(ground_point_heights))
            points_number += ground_point_heights.size
    
    # get mean value of points' heights
    if points_number > 0:
        height = pointheight / points_number
        return height
    else:
        return 0


# function to get all points lying within range of the defined radius from the viewpoint
def getPoints(tile_grid, radius, view_height):
    # Viewpoint
    center = np.array([x, y])
    
    # Gather points
    arraysX, arraysY, arraysZ = [], [], []  # list of arrays of X,Y,Z coords
    arrayDistances = []  # Horizontal distances
    arrayClasses = []  # Classifications
    toBeAdded = []
    for tile in tile_grid:
        inFile = lp.file.File("{}{}{}".format(path, "/Tiles/", tile), mode='r')
        coords = np.vstack((inFile.x, inFile.y)).transpose()
        elevation = inFile.z
        distances = np.sqrt(np.sum((coords - center)**2, axis=1))

        keep_points = np.logical_and(np.logical_and(np.logical_or(
            inFile.raw_classification == 1,
            inFile.raw_classification == 6),
            distances < radius),
            elevation >= view_height/1000)

        # Get coordinates
        arraysX.append(inFile.x[keep_points])
        arraysY.append(inFile.y[keep_points])
        arraysZ.append(inFile.z[keep_points])
        # Get distances
        arrayDistances.append(distances[keep_points])
        # Get classifications
        arrayClasses.append(inFile.raw_classification[keep_points])
  
    # Concatenate all information
    X, Y, Z = arraysX[0], arraysY[0], arraysZ[0]
    distances = arrayDistances[0]
    classes = arrayClasses[0]
    for arrayX, arrayY in zip(arraysX[1:], arraysY[1:]):
        X = np.hstack([X, arrayX])
        Y = np.hstack([Y, arrayY])
    for arrayZ in arraysZ[1:]:
        Z = np.hstack([Z, arrayZ])
    for arDist in arrayDistances[1:]:
        distances = np.hstack([distances, arDist])
    for arClass in arrayClasses[1:]:
        classes = np.hstack([classes, arClass])

    if removeLst:
        toBeRemoved = clearpoints(removeLst, boundRemoveLst, X, Y, classes, 6)

        X=np.delete(X,toBeRemoved)
        Y=np.delete(Y,toBeRemoved)
        Z=np.delete(Z,toBeRemoved)
        distances=np.delete(distances,toBeRemoved)
        classes=np.delete(classes,toBeRemoved)

    if addLst:
        toBeCleared = clearpoints(addLst, boundAddLst, X, Y, classes, 0)

        X=np.delete(X,toBeCleared)
        Y=np.delete(Y,toBeCleared)
        Z=np.delete(Z,toBeCleared)
        distances=np.delete(distances,toBeCleared)
        classes=np.delete(classes,toBeCleared)

        for i in range(len(boundAddLst)):
            Xarray, Yarray, Zarray, Carray = create_pc(boundAddLst[i][0], boundAddLst[i][1], boundAddLst[i][2], boundAddLst[i][3], height[i], view_height)
            toBeAdded=clearpoints(addLst, boundAddLst, Xarray, Yarray, Carray, 6)

            Xarray = Xarray[toBeAdded]
            Yarray = Yarray[toBeAdded]
            Zarray = Zarray[toBeAdded]
            Carray = Carray[toBeAdded]
            coordsNew = np.vstack((Xarray, Yarray)).transpose()
            distancesNew = np.sqrt(np.sum((coordsNew - center) ** 2, axis=1))

            keepNewPoints = np.logical_and(
                distancesNew < radius,
                Zarray >= view_height / 1000)

            Xarray = Xarray[keepNewPoints]
            Yarray = Yarray[keepNewPoints]
            Zarray = Zarray[keepNewPoints]
            Carray = Carray[keepNewPoints]
            distancesNew = distancesNew[keepNewPoints]

            X = np.hstack([X, Xarray])
            Y = np.hstack([Y, Yarray])
            Z = np.hstack([Z, Zarray])
            classes = np.hstack([classes, Carray])
            distances = np.hstack([distances, distancesNew])

    return X, Y, Z, distances, classes

def clearpoints(geomLst, boundLst, X, Y, classes, theClass):
    toBeRemoved = []
    for i in range(len(boundLst)):
        x_min = boundLst[i][0]
        x_max = boundLst[i][1]
        y_min = boundLst[i][2]
        y_max = boundLst[i][3]

        if theClass != 0:
            incl = np.where(np.logical_and(np.logical_and(np.logical_and(X > x_min, X < x_max),
                                                          np.logical_and(Y > y_min, Y < y_max)),
                                           classes == theClass))
        else:
            incl = np.where(np.logical_and(np.logical_and(X > x_min, X < x_max),
                                           np.logical_and(Y > y_min, Y < y_max)))

        pts = np.stack((X[incl], Y[incl]), axis=-1)

        geom = geomLst[i]
        # Edges
        minY = np.fmin(geom[:, 1][:-1], geom[:, 1][1:])
        maxY = np.fmax(geom[:, 1][:-1], geom[:, 1][1:])
        maxX = np.fmax(geom[:, 0][:-1], geom[:, 0][1:])
        nom = geom[:, 0][1:] - geom[:, 0][:-1]
        denom = geom[:, 1][1:] - geom[:, 1][:-1]
        fraction = np.divide(nom, denom)
        curtime = time.clock()

        for i in range(len(pts)):
            pointInside = inside_polygon(pts[i], minY, maxY, maxX, geom, fraction)
            if pointInside is not None:
                toBeRemoved.append(incl[0][i])
    return toBeRemoved

# Create dome
def createDome(X, Y, Z, dists, classes, view_height):
    # Initialize dome
    # Indices = (Azimuth, Elevation)
    dome = np.zeros((180, 90), dtype=int)
    domeDists = np.zeros((180, 90), dtype=int)

    if X.size > 0:
        # Azimuths
        dX, dY = X - x, Y - y
        azimuths = np.arctan2(dY, dX) * 180 / math.pi - 90
        azimuths[azimuths < 0] += 360

        # Elevations
        dZ = Z - view_height / 1000
        elevations = np.arctan2(dZ, dists) * 180 / math.pi

        # Shade sectors
        # Array with dome indices, distances & classifications
        data = np.stack((azimuths // 2, elevations // 1, dists, classes), axis=-1)
        # Sort according to indices & classifications
        sortData = data[np.lexsort([data[:, 2], data[:, 1], data[:, 0]])]

        # Spot where azimuth & elevation values change
        azimuth_change = sortData[:, 0][:-1] != sortData[:, 0][1:]
        elevation_change = sortData[:, 1][:-1] != sortData[:, 1][1:]
        keep = np.where(np.logical_or(azimuth_change, elevation_change))
        # Take position of next element, plus add first row
        shortestDistance = sortData[
            np.insert(keep[0] + 1, 0, 0)]  # (inserts second element of change, first position, index of first point)
        # Define indices & classifications
        hor = shortestDistance[:, 0].astype(int)
        ver = shortestDistance[:, 1].astype(int)
        classif = shortestDistance[:, 3].astype(int)
        dists = shortestDistance[:, 2]

        # Update dome
        dome[hor, ver] = classif

        domeDists[hor, ver] = dists

        # Buildings as solids

        # Find building positions in dome
        # print dome[dome == 6].size
        if dome[dome == 6].size > 0:
            bhor, bver = np.where(dome == 6)
            # Create an array out of them
            builds = np.stack((bhor, bver), axis=-1)
            shape = (builds.shape[0] + 1, builds.shape[1])
            builds = np.append(builds, (bhor[0], bver[0])).reshape(shape)

            # Spot azimuth changes
            azimuth_change = builds[:, 0][:-1] != builds[:, 0][1:]
            keep = np.where(azimuth_change)
            # keep = np.insert(np.where(azimuth_change==True), 0, 0)
            # Change to building up to roof for each row
            roof_rows, roof_cols = builds[keep][:, 0], builds[keep][:, 1]
            for roof_row, roof_col in zip(roof_rows, roof_cols):
                condition = np.where(np.logical_or(domeDists[roof_row, :roof_col] > domeDists[roof_row, roof_col],
                                                   dome[roof_row, :roof_col] == 0))
                dome[roof_row, :roof_col][condition] = 6
    plot(dome)
    return dome

# Plot dome
def plot(dome):
    # Create circular grid 
    theta, radius = np.mgrid[0:(2*np.pi+2*np.pi/180):2*np.pi/180, 0:90:1]
    Z = dome.copy().astype(float)
    Z = Z[0:, ::-1]  # Reverse array rows
    # assign colors depending on class
    Z[Z == 0] = 0
    Z[Z == 1] = 0.5
    Z[Z == 6] = 1

    if Z[Z == 6].size == 0:
        Z[0,0] = 1
    axes = plt.subplot(111, projection='polar')

    cmap = plt.get_cmap('tab20c')


    axes.pcolormesh(theta, radius, Z, cmap=cmap)
    axes.set_ylim([0, 90])
    axes.tick_params(labelleft=False)
    axes.set_theta_zero_location("N")
    plt.savefig("Plots/"+'{}_point{}.png'.format(filename[8:-5],fid),bbox_inches='tight')
    plt.close()

# calculate SVF, and percentage of building/vegetation obstructions
def calculate_SVF(radius, dome):
    obstructedArea = 0
    treeObstruction = 0
    buildObstruction = 0
    for i in range(0, 180):
        for j in range(0, 90):
            if dome[i, j] != 0:
                v = 90 - (j + 1)
                R = math.cos(v * math.pi / 180) * radius
                r = math.cos((v + 1) * math.pi / 180) * radius
                # calculate area of each obstructed sector (circular sector area calculation)
                cell_area = (math.pi / 180.0) * (R ** 2 - r ** 2)
                obstructedArea += cell_area

                if dome[i, j] == 1:
                    treeObstruction += cell_area
                elif dome[i, j] == 6:
                    buildObstruction += cell_area

    circleArea = math.pi * (radius ** 2)

    # SVF: proportion of open area to total area
    SVF = (circleArea - obstructedArea) / circleArea
    treeObstructionPercentage = treeObstruction / circleArea
    buildObstructionPercentage = buildObstruction / circleArea
    return SVF, treeObstructionPercentage, buildObstructionPercentage

def integer(geom):
    geometry = []
    append = geometry.append
    for point in geom:
        x, y = point[0], point[1]
        append([x, y])
    geometry = np.array(geometry) #Closed polygon
    return geometry


def inside_polygon(pt, minY, maxY, maxX, geom, fraction):
    condition1 = np.logical_and(pt[1] > minY, pt[1] <= maxY)
    condition2 = pt[0] <= maxX
    condition = np.logical_and(condition1, condition2)

    intersX = geom[:, 0][:-1][condition] + (pt[1] - geom[:, 1][:-1][condition]) * fraction[condition]
    truth = np.logical_or(geom[:, 0][:-1][condition] == geom[:, 0][1:][condition],
                          pt[0] <= intersX)

    intersections = truth[truth == True].size

    if intersections % 2 == 1:
        return pt

def create_pc(Xmin, Xmax, Ymin, Ymax, height, view_height, density=0.5):
    dx = Xmax - Xmin
    dy = Ymax - Ymin
    # print dx, dy
    Xadd=[]
    Yadd=[]
    Zadd=[]
    Cadd=[]
    # print int(math.ceil(dx/density))
    for i in range(int(math.ceil(dx/density))):
        for j in range(int(math.ceil(dy/density))):
            Xadd.append(Xmin+i*density)
            Yadd.append(Ymin+j*density)
            Zadd.append(float(height) + view_height/1000.0)
            Cadd.append(6)
    return np.asarray(Xadd), np.asarray(Yadd), np.asarray(Zadd), np.asarray(Cadd)

def run():
    start = time.clock()
    row, col = find_tile()
    tile_grid = find_tile_grid(row, col)
    view_height = getheight(tile_grid)
    X, Y, Z, distances, classes = getPoints(tile_grid, radius, view_height)
    dome = createDome(X, Y, Z, distances, classes, view_height)

    SVF, tree_percentage, build_percentage = calculate_SVF(radius, dome)
    SVF, tree_percentage, build_percentage = round(SVF*100), round(tree_percentage*100), round(build_percentage*100)
    print ('{}%'.format(int(SVF)) + "\n" + '{}%'.format(int(tree_percentage)) + "\n" + '{}%'.format(int(build_percentage)))
    end = time.clock()
    duration = end-start

if __name__ == '__main__':
    """GLOBAL VARIABLES"""
    # path for tile directory and list of tilenames
    filename=str(sys.argv[1])
    # filename = 'filelog/'+'testdata'+'.json'
    #path for tile directory and list of tilenames
    path = os.getcwd()
    tilelist = os.listdir(path+"/Tiles")
    # define radius
    radius = 100
    bufferRadius=1.5
    """END GLOBAL VARIABLES"""

    datadict = json.loads(open(filename).read())
    nr_remove_polygons= len(datadict['polygonRemove'])
    nr_add_polygons=len(datadict['polygonAdd'])

    m = 1
    removeLst = []
    boundRemoveLst=[]
    addLst = []
    boundAddLst = []
    height = []
    while True:
        try:
            coorNum = len(datadict['polygonRemove'][m]['value'][0])
            coorLst = []
            for i in range(coorNum):
                x= float(datadict['polygonRemove'][m]['value'][0][i][0])
                y= float(datadict['polygonRemove'][m]['value'][0][i][1])
                coorLst.append((x,y))
            tempPolygon = Polygon(coorLst)

            bufferPolygon = tempPolygon.buffer(bufferRadius, resolution=2)
            geom=integer(list(zip(*bufferPolygon.exterior.coords.xy)))
            x_min, x_max = np.amin(geom[:,0]), np.amax(geom[:,0])
            y_min, y_max = np.amin(geom[:,1]), np.amax(geom[:,1])
            boundRemoveLst.append([x_min, x_max,y_min, y_max])
            removeLst.append(geom)
            # define next point
            m += 1
        except:
            break

    n = 1
    while True:
        try:
            coorNum = len(datadict['polygonAdd'][n]['value'][0])
            height.append(datadict['polygonAdd'][n]['value'][1])
            # define next point
            coorLst = []
            for i in range(coorNum):
                x = float(datadict['polygonAdd'][n]['value'][0][i][0])
                y = float(datadict['polygonAdd'][n]['value'][0][i][1])
                coorLst.append((x, y))
            
            X_min = min(coorLst, key=lambda item: item[0])[0]
            X_max = max(coorLst, key=lambda item: item[0])[0]
            Y_min = min(coorLst, key=lambda item: item[1])[1]
            Y_max = max(coorLst, key=lambda item: item[1])[1]
            
            boundAddLst.append([X_min, X_max, Y_min, Y_max])
            addLst.append(np.asarray(coorLst))

            n += 1
        except:
            break


    o=1
    while True:
        try:
            x = float(datadict['coordinates'][o]['value'][0])
            y = float(datadict['coordinates'][o]['value'][1])
            fid = int(datadict['coordinates'][o]['key'])
            # define viewpoint
            run()

            # define next point
            o += 1
        except:
            break
    
