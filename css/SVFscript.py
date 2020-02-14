import numpy as np
import laspy as lp
import json
import math
import os
import matplotlib.pyplot as plt
import sys

#define point class (with classification)
class point:

    def __init__(self, x, y, z, cl):
        self.x = x
        self.y = y
        self.z = z
        self.cl = cl

    def __str__(self):
        return str(self.x) + ", "+str(self.y) + ", "+str(self.z) + ", "+str(self.cl)

    def horizontal_dist(self, pt):
        return ((self.x - pt.x) ** 2 + (self.y - pt.y) ** 2) ** 0.5

    def horizontal_angle(self, pt):
        if self.horizontal_dist(pt) != 0:
            sin_angle = (pt.x - self.x) / self.horizontal_dist(pt)
            if self.y < pt.y:
                ang = math.asin(sin_angle)
            else:
                ang = math.pi - math.asin(sin_angle)
            if ang < 0:
                return (2 * math.pi+ang) * 180 / math.pi
            else:
                return ang * 180.0 / math.pi
        else:
            return 0

    def vertical_angle(self, pt):
        if self.horizontal_dist(pt) != 0:
            tan_angle = (pt.z - self.z) / self.horizontal_dist(pt)
            return math.atan(tan_angle) * 180.0 / math.pi
        else:
            return 90.0

def find_tile(viewpoint):
    formatted_tilelist = []
    for tile in tilelist:
        formatted_tilelist.append(tile.strip(".las").split(","))
    for tile in formatted_tilelist:
        x = viewpoint.x
        y = viewpoint.y
        tile_min_x = float(tile[2])
        tile_min_y = float(tile[3])
        tile_max_x = float(tile[4])
        tile_max_y = float(tile[5])
        if x >= tile_min_x and x <= tile_max_x and y >= tile_min_y and y <= tile_max_y:
            return int(tile[0]), int(tile[1])

def find_tile_grid(row, col):
    iteration_list = [-1, 0, 1]
    tile_grid = []
    
    for i in iteration_list:
        for j in iteration_list:
            tilename_start = "{},{}".format((row + i),(col + j))
            for tile in tilelist:
                if tile.startswith(tilename_start) == True:
                    tile_grid.append(tile)
                    break
    return tile_grid

def getheight(tile_grid, viewpoint):
    center = np.array((viewpoint.x, viewpoint.y))
    pointheight = 0
    points_no = 0
    
    for tile in tile_grid:
        f = lp.file.File("{}{}{}".format(path, "/Tiles/", tile), mode='r')

        ground_rules = np.logical_and(
            f.raw_classification == 2,
            np.sqrt(np.sum((np.vstack((f.x, f.y)).transpose() - center) ** 2, axis=1)) <= 5)

        ground_points = f.points[ground_rules]

        ground_point_heights = np.array((ground_points['point']['Z'])).transpose()
        
        if ground_point_heights.size > 0:
            pointheight += float(np.sum(ground_point_heights))
            points_no += ground_point_heights.size

    if points_no > 0 :
        height = pointheight / points_no
        return height
    else:
        return 0

def getpoints(tile_grid, viewpoint, radius):
    stupidlist = []
    arraylist = []
    
    center = np.array((viewpoint.x, viewpoint.y))
    
    for tile in tile_grid:
    
        f = lp.file.File("{}{}{}".format(path, "/Tiles/", tile), mode='r')

        # coords = np.vstack((f.x, f.y)).transpose()
        # distances = np.sqrt(np.sum((coords - center) ** 2, axis=1))

        
        # only read points that:
        # Have classification  BUILDING
        # Have classification UNCLASSIFIED
        # Have a distance from viewpoint that is lower than or equal to the radius
        # Have a heigher Z-value than the viewpoint
        unclass_build_point = np.logical_and(np.logical_and(np.logical_or(
            f.raw_classification == 1,
            f.raw_classification == 6),
            np.sqrt(np.sum((np.vstack((f.x, f.y)).transpose() - center) ** 2, axis=1)) <= radius),
            f.z > viewpoint.z)

        unclass_build_points = f.points[unclass_build_point]
        points_xyz_class = np.array((unclass_build_points['point']['X'],
                                     unclass_build_points['point']['Y'],
                                     unclass_build_points['point']['Z'],
                                     unclass_build_points['point']['raw_classification'])).transpose()
        

        arraylist.append(points_xyz_class)
        for pt in points_xyz_class:
            stupidlist.append(point(pt[0]/1000.0, pt[1]/1000.0, pt[2]/1000.0, pt[3]))

    return stupidlist

#create dome dictionary
def dome_dict(viewpoint, inpoints):
    dome={}
    for i in range(0, 180):
        for j in range(0, 90):
            dome[i, j] = [0, radius, 0]
            
    #compute non-visible dome sectors
    for p in inpoints:
        h = viewpoint.horizontal_angle(p)
        v = viewpoint.vertical_angle(p)
        if dome[h // 2, v // 1][0] == 0:
            dome[h // 2, v // 1][0] = 1
        
        if viewpoint.horizontal_dist(p) < dome[h // 2, v // 1][1]:
            dome[h // 2, v // 1][1] = viewpoint.horizontal_dist(p)
            dome[h // 2, v // 1][2] = p.cl
        
            #handle buildings as solids
            #cover all sectors beneath building as well
            if dome[h // 2, v // 1][2] == 6 and v > 0:
                for i in range(0, int(v)):
                    if dome[h // 2, i][0] == 0:
                        dome[h // 2, i][0] = 1
                    if viewpoint.horizontal_dist(p) < dome[h // 2, i][1]:
                        dome[h // 2, i][1] = viewpoint.horizontal_dist(p)
                        dome[h // 2, i][2] = p.cl     
    plot(dome)
    return dome

#added function
def plot(dome):
    theta, r = np.mgrid[0:(2*np.pi+2*np.pi/180):2*np.pi/180, 0:90:1]
    z = []
    for i in reversed(range(0, 180)):
        z_inside = []
        for j in reversed(range(0, 90)):
            if dome[i, j][0] == 1:
                if dome[i, j][2]==6:
                    z_inside.append(1)
                else:
                    z_inside.append(0.5)
            else:
                z_inside.append(0)
        z.append(z_inside)
    Z = np.array(z)
    axes = plt.subplot(111, projection='polar')
    cmap = plt.get_cmap('tab20c')
    axes.pcolormesh(theta, r, Z, cmap=cmap)
    axes.set_ylim([0,90])
    axes.tick_params(labelleft=False)
    axes.set_theta_zero_location("N")
    plt.savefig("Plots/"+'{}_point{}.png'.format(filename[8:-5],fid))
    plt.close()
#added function

# calculate SVF
def calculate_SVF(radius, dome):
    shadedArea = 0
    treeShade = 0
    buildShade = 0
    for i in range(0, 180):
        for j in range(0, 90):
            if dome[i, j][0] == 1:
                v = 90 - (j + 1)
                R = math.cos(v * math.pi / 180) * radius
                r = math.cos((v+1) * math.pi / 180) * radius
                cell_area = (math.pi / 180.0) * (R ** 2 - r ** 2)
                shadedArea += cell_area
                
                if dome[i, j][2] == 1:
                    treeShade += cell_area
                elif dome[i, j][2] == 6:
                    buildShade += cell_area              

    circleArea = math.pi * (radius ** 2)
    SVF = (circleArea - shadedArea) / circleArea
    if shadedArea != 0:
        treeShadePercentage = treeShade / shadedArea
        buildShadePercentage = buildShade / shadedArea
    else:
        treeShadePercentage = 0
        buildShadePercentage = 0
    return SVF, treeShadePercentage, buildShadePercentage

def run():
    row, col = find_tile(viewpoint)
    tile_grid = find_tile_grid(row, col)
    
    view_height = getheight(tile_grid, viewpoint)
    
    complete_viewpoint = point(viewpoint.x, viewpoint.y, view_height, viewpoint.cl)
    inpoints = getpoints(tile_grid, viewpoint, radius)

    dome = dome_dict(viewpoint, inpoints)
    SVF, tree, build = calculate_SVF(radius, dome)
    print SVF

if  __name__ == '__main__':
    """GLOBAL VARIABLES"""
    # path for tile directory and list of tilenames
    filename=str(sys.argv[1])
    #print str(filename)
    path = os.getcwd()
    tilelist = os.listdir(path+"/Tiles")

    #define radius
    radius = 100.0

    """END GLOBAL VARIABLES"""
    
    datadict = json.loads(open(filename).read())
    n = 1
    while True:
        try:
            x = float(datadict['coordinates'][n]['value'][0])
            y = float(datadict['coordinates'][n]['value'][1])
            fid = int(datadict['coordinates'][n]['key'])
            #define viewpoint
            viewpoint = point(x, y, 0.0, 0)
            run()

            #define next point
            n += 1
        except:
            break
    
