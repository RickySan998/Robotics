#!/usr/bin/env python
# --- YOUR MATRIC NUMBER HERE ---
import roslib, rospy, rospkg
from numpy import *
from geometry_msgs.msg import *
from sensor_msgs.msg import LaserScan, JointState, Imu
from nav_msgs.msg import Odometry
from std_msgs import *
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_multiply
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float64MultiArray
from pkg_turtle.msg import MsgGuider, Motion
import cv2
import numpy
import sys

# ================================= CONSTANTS ==========================================        
# let's cache the SIN and POS so we don't keep recalculating it, which is slow
DEG2RAD = [i/180.0*pi for i in xrange(360)] # DEG2RAD[3] means 3 degrees in radians
SIN = [sin(DEG2RAD[i]) for i in xrange(360)] # SIN[32] means sin(32degrees)
COS = [cos(DEG2RAD[i]) for i in xrange(360)]

# ================================== DATA STRUCTS ===========================================
# log odds initialisations
L_THRESH = 5 
L_OCC = 1
L_INF = 1

# global need path
need_path = True
path = []

# Spline Closeness measure
MIN_DIST = 0.15 #0.1 

# Equality Threshold
EQ_TH = 0.0001

def SplineFitting(start_i,start_j,end_i,end_j,start_v_i,start_v_j,nend_i = None, nend_j = None):
    # for spline, need start and end positions, as well as start and end velocity magnitudes. The direction can be extrapolated
    # by the gradient between start and end
    # if not at end, we can improve smoothness by averaging the gradient between current spline and next spline
    
    # to return the spline parameters
    #print(start_i,start_j,end_i,end_j,nend_i,nend_j)
    end_v = 0.2; n = 1 # PROJ 2: Change to 1 to NOT use splines
    
    di1 = end_i - start_i; dj1 = end_j - start_j 

    time = sqrt(di1*di1 + dj1*dj1) / end_v # t_final is adjusted based on the distance of the start - end, divided by some constant velocity
    
    # cache constants for computation
    c1 = 1/pow(time,2); c2 = 1/time; c3 = 2/pow(time,3)
    
    #print("di1,dj1: ", di1, dj1)
    
    # compute boundary velocities
    if nend_i == None and nend_j == None: # at the end, hence extrapolate using gradient of single spline
        dvert = di1 
        dhor = dj1
    else: # extrapolate using gradient of both splines
        di2 = nend_i - end_i; dj2 = nend_j - end_j
        dvert = di1*dj2 + di2*dj1
        dhor = 2*dj1*dj2
    
    # Hypotenuse
    dh = sqrt(dvert*dvert + dhor*dhor) 
    end_v_i = end_v * dvert/dh # vertical
    end_v_j = end_v * dhor/dh # horizontal
    
    # compute the constants a0,a1,a2,a3 (horizontal direction)
    a0 = start_j
    a1 = start_v_j
    a2 = -3*c1*start_j - 2 * c2 * start_v_j + 3 *c1 * end_j - c2*end_v_j
    a3 = c3 * start_j + c1 * start_v_j - c3 * end_j + c1 * end_v_j
    
    # and b0,b1,b2,b3 (vertical direction)
    b0 = start_i
    b1 = start_v_i
    b2 = -3*c1*start_i - 2 * c2 * start_v_i + 3 *c1 * end_i - c2*end_v_i
    b3 = c3 * start_i + c1 * start_v_i - c3 * end_i + c1 * end_v_i
        
    return a0,a1,a2,a3,b0,b1,b2,b3,time/n

def initOccGrid(min_pos, max_pos, cell_size, initial_value, inflation_radius):
    global di
    global dj
    di = int64(round((max_pos[0] - min_pos[0])/cell_size))
    dj = int64(round((max_pos[1] - min_pos[1])/cell_size))
    di += 1; dj += 1
    global log_odds_map
    global inflation_map
    global G_Cost
    global H_Cost
    global inflation_mask
    global IDX_ARRAY
    log_odds_map = [[initial_value for j in range(dj)] for i in range(di)]
    inflation_map = [[0 for j in range(dj)] for i in range(di)]
    IDX_ARRAY = [[0 for j in range(dj)] for i in range(di)] # array to indicate which cells are in path

    global OFFS_OCC
    global OFFS_INF
    OFFS_OCC = (di + dj) * 5
    OFFS_INF = (di + dj) * 2 
    G_Cost = [[0 for j in range(dj)] for i in range(di)]
    H_Cost = [[0 for j in range(dj)] for i in range(di)]
    # gen mask
    inflation_mask = []
    offset_idx = int(round(inflation_radius/cell_size))
    for i in range(-offset_idx,offset_idx+1):
        for j in range(-offset_idx,offset_idx+1):
            if (pow(i,2) + pow(j,2)) <= pow(inflation_radius/cell_size,2):
                inflation_mask.append((i,j))
def GeneralIntLosPP(start_pos,end_pos):
    start_idx = (int64(round((start_pos[0]))), int64(round((start_pos[1]))))
    end_idx = (int64(round((end_pos[0]))), int64(round((end_pos[1]))))
    
    indices = []
    indices.append(start_idx)
    
    diffx = end_idx[0] - start_idx[0]
    diffy = end_idx[1] - start_idx[1]
    if abs(diffx) > abs(diffy):
        diffl = diffx
        diffs = diffy
        to_long_short = lambda l,s : (l,s)
        
    else:
        diffl = diffy
        diffs = diffx
        to_long_short = lambda l,s : (s,l)
        
    ls_start_idx = list(to_long_short(start_idx[0],start_idx[1]))
    ls_end_idx = list(to_long_short(end_idx[0], end_idx[1]))
    
    # increments and error init
    sign_l = sign(diffl)
    sign_s = sign(diffs)
    e_inc = 2 * diffs
    e_dec = 2 * abs(diffl) * sign_s
    e = 0
    
    # length of a is
    a = abs(diffs) - abs(diffl)
    
    # error checker (more accurate than using absolute, because of the upper and lower bounds)
    if sign_s >= 0 :
        error_check = lambda error : error >= abs(diffl)
    else:
        error_check = lambda error : error < -abs(diffl)
        
    # start propagating to end
    while ls_start_idx != ls_end_idx:
        ls_start_idx[0] += sign_l
        e += e_inc
        if error_check(e) :
            e -= e_dec
            ls_start_idx[1] += sign_s
            
            b = sign_s * e
            
            if abs(a - b) < EQ_TH:
                pass

            elif a > b:
                indices.append(to_long_short(ls_start_idx[0],ls_start_idx[1] - sign_s))
                if(ls_start_idx[0] == ls_end_idx[0]) and (ls_start_idx[1] - sign_s) == ls_end_idx[1]:
                    break

            elif a < b:
                indices.append(to_long_short(ls_start_idx[0] - sign_l,ls_start_idx[1]))
                if(ls_start_idx[0] - sign_l == ls_end_idx[0]) and (ls_start_idx[1]) == ls_end_idx[1]:
                    break     

        indices.append(to_long_short(ls_start_idx[0],ls_start_idx[1]))
        
    return indices


def updateAtIdx(idx, occupied):
    global need_path
    global log_odds_map
    global inflation_map
    #check if within map
    if idx[0] < 0 or idx[0] >= di or idx[1] < 0 or idx[1] >= dj:
        return
    #update check if occupied previously
    was_occupied = log_odds_map[idx[0]][idx[1]] > L_THRESH
    #update current log odds
    if occupied:
        log_odds_map[idx[0]][idx[1]] += L_OCC
    else:
        log_odds_map[idx[0]][idx[1]] -= L_OCC
    #check if there is change in occupancy status
    is_occupied = log_odds_map[idx[0]][idx[1]] > L_THRESH
    if was_occupied != is_occupied:
      for rel_idx in inflation_mask:
            i = rel_idx[0] + idx[0]
            j = rel_idx[1] + idx[1]
            if i >= 0 and i < di and j >= 0 and j < dj: # cell in map
                if is_occupied:
                    inflation_map[i][j] += L_INF
                    if IDX_ARRAY[i][j] == 1:
                        need_path = True
                else:
                    inflation_map[i][j] -= L_INF

def show_map(rbt_idx, path=None, goal_idx=None):
    """ Prints the occupancy grid and robot position on it as a picture in a resizable 
        window
    Parameters:
        rbt_pos (tuple of float64): position tuple (x, y) of robot.
    """

    img_mat_copy = img_mat.copy()
    for i in range(di):
        log_odds_row = log_odds_map[i]
        inflation_row = inflation_map[i]
        for j in range(dj):
            log_odds_cell_value = log_odds_row[j]
            inflation_cell_value = inflation_row[j]
            if log_odds_cell_value > L_THRESH: # Occupied precedes over inflation; a cell that is occupied is also marked as its own inflation, but if occupied, directly set as white
                for k in range(3):
                    img_mat_copy[i, j, k] = 255 # white
            elif inflation_cell_value > 0:
                for k in range(3):
                    img_mat_copy[i, j, k] = 180 # light gray
            elif log_odds_cell_value < -L_THRESH:
                for k in range(3):
                    img_mat_copy[i, j, k] = 0 # black
                
    if path is not None:
        for k in range(len(path)):
            i = path[k][0]
            j = path[k][1]
            img_mat_copy[i, j, :] = (0, 0, 255) # red
        
        
    # color the robot position as a crosshair
    img_mat_copy[rbt_idx[0], rbt_idx[1], :] = (0, 255, 0) # green
    
    if goal_idx is not None:
        img_mat_copy[goal_idx[0], goal_idx[1], :] = (255, 0, 0) # blue

    # print to a window 'img'
    cv2.imshow('img', img_mat_copy)
    cv2.waitKey(10)

def segment(path): # only for A* path
    segs = [] # list of segments
    seg = [path[0]] # container for single segment
    start_idx_i,start_idx_j = path[0]
    segs_inf = [inflation_map[start_idx_i][start_idx_j]] # status of each segment
    past_inflation = inflation_map[start_idx_i][start_idx_j] # starting cell inflation condition
    for idx in range(1,len(path)):
        curr_idx_i,curr_idx_j = path[idx]
        curr_inflation = inflation_map[curr_idx_i][curr_idx_j]
        if curr_inflation != past_inflation:
            segs.append(seg)
            segs_inf.append(curr_inflation)
            seg = [path[idx]]
        else:
            seg.append(path[idx])
        past_inflation = curr_inflation
    segs.append(seg)
    return segs,segs_inf

def extract_turn_points(segs): # only for A* path, extract turning points regardless of inflation status
    segs_tp = []
    seg_tp = []
    for seg in segs:
        # start and end always appended hence
        seg_tp = [0]
        if len(seg) == 1:
            segs_tp.append(seg_tp)
        elif len(seg) == 2:
            seg_tp = [0,1]
            segs_tp.append(seg_tp)
        else: # length >2
            len_seg = len(seg)
            i0,j0 = seg[0]
            i1,j1 = seg[1]
            diri0 = i1 - i0 # do not pack to tuple to speed up
            dirj0 = j1 - j0
            for n in range(2,len_seg):
                inext,jnext = seg[n]
                icurr,jcurr = seg[n-1]
                diri1 = inext - icurr
                dirj1 = jnext - jcurr
                if(diri1 != diri0) or (dirj1 != dirj0):
                    seg_tp.append(n-1)
                diri0 = diri1
                dirj0 = dirj1
            seg_tp.append(len_seg - 1)
            segs_tp.append(seg_tp)
    return segs_tp

def extract_turn_pointsJPS(segs): # only for JPS path, extract turning points regardless of inflation status
    segs_tp = []
    seg_tp = []
    for seg in segs:
        # start and end always appended hence
        seg_tp = [0]
        if len(seg) == 1:
            segs_tp.append(seg_tp)
        elif len(seg) == 2:
            seg_tp = [0,1]
            segs_tp.append(seg_tp)
        else: # length >2
            len_seg = len(seg)
            i0,j0 = seg[0]
            i1,j1 = seg[1]
            diri0 = sign(i1 - i0) # do not pack to tuple to speed up
            dirj0 = sign(j1 - j0)
            for n in range(2,len_seg):
                inext,jnext = seg[n]
                icurr,jcurr = seg[n-1]
                diri1 = sign(inext - icurr)
                dirj1 = sign(jnext - jcurr)
                if(diri1 != diri0) or (dirj1 != dirj0):
                    seg_tp.append(n-1)
                diri0 = diri1
                dirj0 = dirj1
            seg_tp.append(len_seg - 1)
            segs_tp.append(seg_tp)
    return segs_tp

def GeneralLosPP(start_pos,end_pos): # General Float LOS without pos2idx
    threshold = 0.5 # set where to start drawing line
    start_idx = (int(round(start_pos[0])), int(round(start_pos[1])))
    end_idx = (int64(round(end_pos[0])), int64(round(end_pos[1])))
    
    
    start_float = (start_pos[0] , start_pos[1])
    end_float = (end_pos[0] , end_pos[1] )
    
    indices = [] # init an empty list
    indices.append(start_idx) # append the starting index into the cell

    # assign long and short axes
    diffx = end_float[0] - start_float[0]
    diffy = end_float[1] - start_float[1]
    if abs(diffx) > abs(diffy):
        diffl = diffx
        diffs = diffy
        to_long_short = lambda l,s : (l,s)
        to_x_y = lambda x,y : (x,y)

    else:
        diffl = diffy
        diffs = diffx
        to_long_short = lambda l,s : (s,l)
        to_x_y = lambda x,y : (y,x)

    # get start and final in terms of long and short axis
    ls_start_idx = list(to_long_short(start_idx[0],start_idx[1]))
    ls_end_idx = list(to_long_short(end_idx[0],end_idx[1]))
    ls_start_float = list(to_long_short(start_float[0],start_float[1]))
    ls_end_float = list(to_long_short(end_float[0],end_float[1]))

    # get signs of increments and short axis increment
    sign_l = sign(diffl)
    sign_s = sign(diffs)
    short_inc = diffs * 1.0 /diffl * sign_l

    # initiate rounding error
    error_s = ls_start_float[1] - ls_start_idx[1]
    error_l = ls_start_float[0] - ls_start_idx[0]

    # get length of a (refer to notes)
    dlong = 0.5 + error_l * sign_l
    length_a = round(abs(short_inc * dlong),8)
    
    # error checker
    if sign_s >= 0 :
        error_check = lambda error : error >= threshold
    else:
        error_check = lambda error : error < -threshold

    # perform general line, loop until end is reached
    while (ls_start_idx != ls_end_idx):
        ls_start_idx[0] += sign_l
        error_s += short_inc
        if error_check(error_s):
            error_s -= sign_s
            ls_start_idx[1] += sign_s
            # get length of b (refer to notes)
            length_b = round((1-threshold) + error_s * sign_s,8)
            
            if abs(length_a - length_b) < EQ_TH:
                pass

            elif length_a < length_b: # intermediate in long axis
                indices.append(to_x_y(ls_start_idx[0]-sign_l,ls_start_idx[1]))
                if (ls_start_idx[0] - sign_l == ls_end_idx[0]) and (ls_start_idx[1] == ls_end_idx[1]):
                    break

            elif length_a > length_b: # intermediate in short axis
                indices.append(to_x_y(ls_start_idx[0],ls_start_idx[1]-sign_s))
                if (ls_start_idx[0] == ls_end_idx[0]) and (ls_start_idx[1]-sign_s == ls_end_idx[1]):
                    break
                #indices.append(to_x_y(ls_start_idx[0]-sign_l,ls_start_idx[1]))
                #indices.append(to_x_y(ls_start_idx[0],ls_start_idx[1]-sign_s))
            
        indices.append(to_x_y(ls_start_idx[0],ls_start_idx[1]))

    return indices

def propagate_path(pts):
    start_idx = pts[0]
    path = [start_idx]
    for i in range(1,len(pts)): # propagate all cells in LOS between final turning points
        end_idx = pts[i]
        LOSindexes = GeneralLosPP(start_idx,end_idx)
        for b in range(1,len(LOSindexes)):
            path.append(LOSindexes[b])
        start_idx = end_idx
    return path

def post_process_full_Astar(path):
    if len(path) == 2:
        propagated = propagate_path(path)
        return path,propagated
    elif len(path) == 1:
        return path,path
    segs,segs_inf = segment(path)
    segs_tp = extract_turn_points(segs)
    g_los_tp = LOSviapoints(segs,segs_tp,segs_inf,forward = True)
    s_los_tp = LOSviapoints(segs,segs_tp,segs_inf,forward = False)
    pts_2dir = twodirectionLOS(segs,segs_inf,segs_tp,g_los_tp,s_los_tp)
    processed_path_all = propagate_path(pts_2dir)
    processed_int = []
    for idx in processed_path_all:
        processed_int.append((int(idx[0]),int(idx[1])))
    
    return pts_2dir, processed_int # returns turning points in float and cell indexes in int
def post_process_full_JPS(JPSpath):
    if len(JPSpath) == 2:
        propagated = propagate_path(JPSpath)
        return JPSpath,propagated
    elif len(JPSpath) == 1:
        return JPSpath,JPSpath
    JPSfull = propagate_path(JPSpath)
    JPSfull = [JPSfull]
    JPSpure = extract_turn_pointsJPS(JPSpath)
    segs_inf_jps = [0]
    JPSlosg = LOSviapoints(JPSfull,JPSpure,segs_inf_jps,True)
    JPSloss = LOSviapoints(JPSfull,JPSpure,segs_inf_jps,False)
    JPStwodir = twodirectionLOS(JPSfull,segs_inf_jps,JPSpure,JPSlosg,JPSloss)
    JPSfullpath = propagate_path(JPStwodir)
    full_path = []
    for idx in JPSfullpath:
        full_path.append((int(idx[0]),int(idx[1])))
    return JPStwodir,full_path # returns turning points in float and cell indexes in int

def twodirectionLOS(segs,segs_inf,segs_tp,g_segs_tp,s_segs_tp):
    pts = [] # output wanted is de-segmented, i.e. 1 list of MAP COORDINATES
    
    for n in range(len(segs)):
        if segs_inf[n] : #if inflated, we preserve all the TURNING points
            seg_tp = segs_tp[n]
            seg = segs[n]
            for i in seg_tp:
                r,c = seg[i]
                point = (float(r),float(c))
                pts.append(point)
        else:
            g_seg_tp = g_segs_tp[n]
            s_seg_tp = s_segs_tp[n]
            seg_tp = segs_tp[n]
            seg = segs[n]
            g_seg_len = len(g_seg_tp)
            
            q = g_seg_len - 1 # cache and decrement to be faster
            for p in range(g_seg_len):
                g = g_seg_tp[p]
                s = s_seg_tp[q]
                
                if g == s: # if both same kind, we preserve to point
                    i,j = seg[g]
                    point = (float(i),float(j))
                    pts.append(point)
                    q -= 1
                else:
                    g_elbow = g in seg_tp
                    s_elbow = s in seg_tp
                    # if both different, will definitely be between 2 of other kind
                    # we keep the one that is NOT elbow
                    
                    if g_elbow and not s_elbow:
                        i,j = seg[s]
                        point = (float(i),float(j))
                        pts.append(point)
                    elif not g_elbow and s_elbow:
                        i,j = seg[g]
                        point = (float(i),float(j))
                        pts.append(point)
                    elif not g_elbow and not s_elbow: # both are not elbow, then we discard both and keep intersection between lines passing thru g and prevg, s and prevs
                        i0,j0 = seg[g_seg_tp[p]]
                        i1,j1 = seg[g_seg_tp[p-1]]
                        
                        
                        i2,j2 = seg[s_seg_tp[q]]
                        i3,j3 = seg[s_seg_tp[q-1]]
                        
                        # cache constant for faster speed
                        a0 = i1 - i0
                        b0 = j1 - j0
                        a1 = i3 - i2
                        b1 = j3 - j2
                        k0 = i0*b0 - a0*j0
                        k1 = i2*b1 - a1*j2
                        d = float(b1*a0 - a1*b0)
                        
                        # point of intersection (i,j) given by: refer to notes
                        idx = ((a0*k1 - a1*k0)/d , (-b1*k0 + b0*k1)/d)
                        pts.append(idx)
                        
                    q -= 1
    return pts

def LOSviapoints(segs,segs_tp,segs_inflation,forward=True): # the 2 A_star parameters is not required for the main code if we can access cell inflation status globally
    # create definitions; these differ depending on the direction of the path
    if forward:
        get_first = lambda length: 0
        get_third = lambda length: 2
        get_last = lambda length: length - 1
        next_p = lambda p, hasLOS: p + 2 if hasLOS else p+1
        prev_q = lambda q : q - 1
        at_end = lambda p,length : p==length
        aft_end = lambda p,length : p > length
    else:
        get_first = lambda length: length-1
        get_third = lambda length: length-3
        get_last = lambda length: 0
        next_p = lambda p, hasLOS: p - 2 if hasLOS else p-1
        prev_q = lambda q : q + 1
        at_end = lambda p,length : p== -1
        aft_end = lambda p,length : p < -1
    
    LOS_segs_pts = []
    
    for n in range(len(segs)): # iterate thru each segment
        # p and q are the positions of elements in seg_tp and seg respectively, and seg_tp elements are positions of seg elements
        
        if segs_inflation[n]: # if inflation, skip the segment entirely
            LOS_segs_pts.append(False)
        else:
            # inits for each segment, note start is always included
            seg_tp = segs_tp[n] # get the segment and turning point positions in that segment
            seg = segs[n]
            seg_tp_len = len(seg_tp)
            p = get_first(seg_tp_len)
            start_intidx = seg_tp[p]
            LOS_seg_pts = [start_intidx]
            
            
            # check length of seg_tp, if 1, then simply finish segment, if 2 then add the end and finish segment
            if seg_tp_len == 1:
                LOS_segs_pts.append(LOS_seg_pts)
            elif seg_tp_len == 2:
                p = get_last(seg_tp_len)
                LOS_seg_pts.append(seg_tp[p])
                LOS_segs_pts.append(LOS_seg_pts)
            else:
                # p is always next 2 tp, q is always the current target. p is position of seg_tp elements, q is position of seg
                # start and end_tupidx is the map index of the end, to be used in LOS
                # hence end_tupidx is always the map index of the target, could be 1 or 2 next tp
                start_tupidx = seg[start_intidx]
                p = get_third(seg_tp_len)
                q = seg_tp[p]
                end_tupidx = seg[q]
                l = 0
                hasLOS = True # only True if we have LOS to the next 2 tp
                LOSindexes = GeneralIntLosPP(start_tupidx, end_tupidx)
                
                while True:
                    l += 1
                    idx = LOSindexes[l]
                    
                    if idx == end_tupidx: # target reached, could be reaching next 2 or 1 tp, but is always q
                        LOS_seg_pts.append(q) # q must always be the element in seg_tp
                        p = next_p(p,hasLOS)
                        
                        # if p exceeds last index by 1, that means we miss the end, if p exceeds last index by 2, means we had LOS to end, and end ard appended
                        # in both cases, we reached the end of the segment, hence break the while loop
                        if at_end(p,seg_tp_len):
                            p = get_last(seg_tp_len)
                            LOS_seg_pts.append(seg_tp[p])
                            LOS_segs_pts.append(LOS_seg_pts)
                            break
                        elif aft_end(p,seg_tp_len):
                            LOS_segs_pts.append(LOS_seg_pts)
                            break
                        else: # means there are still turning points in the segment
                            # initiate for the next tp pair
                            start_tupidx = end_tupidx
                            q = seg_tp[p]
                            end_tupidx = seg[q]
                            l = 0
                            hasLOS = True
                            LOSindexes = GeneralIntLosPP(start_tupidx ,end_tupidx)
                            continue # i.e. skip the inflation check, note that for non-inflated segments, start and end of that segment cannot be inflated
                    
                    # if not reached end_tupidx, check whether inflation
                    cell_inf_status = inflation_map[idx[0]][idx[1]]# replace this with accessing cell_inflation_status globally 
                    if cell_inf_status:
                        hasLOS = False
                        q = prev_q(q)
                        end_tupidx = seg[q]
                        LOSindexes = GeneralIntLosPP(start_tupidx,end_tupidx)
                        l = 0
                        
    return LOS_segs_pts

# LOS codes
def LOS(start_pos, end_pos):
    # sets up the LOS object to prepare return a list of indices on the map starting from start_pos (world coordinates) to end_pos (world)
    # start_pos is the robot position.
    # end_pos is the maximum range of the LIDAR, or an obstacle.
    # every index returned in the indices will be the index of a FREE cell
    # you can return the indices, or update the cells in here
    # General Line Algorithm
    threshold = 0.5 # set where to start drawing line
    start_idx = (int64(round((start_pos[0] - min_pos[0]) / cell_size)), int64(round((start_pos[1] - min_pos[1]) / cell_size)))
    end_idx = (int64(round((end_pos[0] - min_pos[0]) / cell_size)), int64(round((end_pos[1] - min_pos[1]) / cell_size)))
    start_float = ((start_pos[0] - min_pos[0]) / cell_size, (start_pos[1] - min_pos[1]) / cell_size)
    end_float = ((end_pos[0] - min_pos[0]) / cell_size, (end_pos[1] - min_pos[1]) / cell_size)
    
    indices = [] # init an empty list
    indices.append(start_idx) # append the starting index into the cell

    # assign long and short axes
    diffx = end_float[0] - start_float[0]
    diffy = end_float[1] - start_float[1]
    if abs(diffx) > abs(diffy):
        diffl = diffx
        diffs = diffy
        to_long_short = lambda l,s : (l,s)
        to_x_y = lambda x,y : (x,y)

    else:
        diffl = diffy
        diffs = diffx
        to_long_short = lambda l,s : (s,l)
        to_x_y = lambda x,y : (y,x)

    # get start and final in terms of long and short axis
    ls_start_idx = list(to_long_short(start_idx[0],start_idx[1]))
    ls_end_idx = list(to_long_short(end_idx[0],end_idx[1]))
    ls_start_float = list(to_long_short(start_float[0],start_float[1]))
    ls_end_float = list(to_long_short(end_float[0],end_float[1]))

    # get signs of increments and short axis increment
    sign_l = sign(diffl)
    sign_s = sign(diffs)
    short_inc = diffs * 1.0 /diffl * sign_l

    # initiate rounding error
    error_s = ls_start_float[1] - ls_start_idx[1]
    error_l = ls_start_float[0] - ls_start_idx[0]

    # get length of a (refer to notes)
    dlong = 0.5 + error_l * sign_l
    length_a = round(abs(short_inc * dlong),8)
    
    # error checker
    if sign_s >= 0 :
        error_check = lambda error : error >= threshold
    else:
        error_check = lambda error : error < -threshold

    # perform general line, loop until end is reached
    while (ls_start_idx != ls_end_idx):
        ls_start_idx[0] += sign_l
        error_s += short_inc
        if error_check(error_s):
            error_s -= sign_s
            ls_start_idx[1] += sign_s
            # get length of b (refer to notes)
            length_b = round((1-threshold) + error_s * sign_s,8)

            if abs(length_a - length_b) < EQ_TH:
                pass
            
            elif length_a < length_b: # intermediate in long axis
                indices.append(to_x_y(ls_start_idx[0]-sign_l,ls_start_idx[1]))
                if (ls_start_idx[0] - sign_l == ls_end_idx[0]) and (ls_start_idx[1] == ls_end_idx[1]):
                    break

            elif length_a > length_b: # intermediate in short axis
                indices.append(to_x_y(ls_start_idx[0],ls_start_idx[1]-sign_s))
                if (ls_start_idx[0] == ls_end_idx[0]) and (ls_start_idx[1]-sign_s == ls_end_idx[1]):
                    break
                # indices.append(to_x_y(ls_start_idx[0]-sign_l,ls_start_idx[1]))
                # indices.append(to_x_y(ls_start_idx[0],ls_start_idx[1]-sign_s))
            
        indices.append(to_x_y(ls_start_idx[0],ls_start_idx[1]))

    return indices

# A Star codes
def getNeighborAstar(starti,startj):
    
    indexes = []
    for i in range(-1,2):
        for j in range(-1,2):
            if i==0 and j == 0:
                continue
            elif starti + i < 0 or starti + i >= di or startj + j < 0 or startj + j >= dj:
                continue
            else:
                indexes.append((starti+i,startj+j))
    return indexes

def sortappendStar(OpenList, new_element_i,new_element_j,g_cost_arr,h_cost_arr):
    
    if not OpenList:
        OpenList.append((new_element_i,new_element_j))
        return
    
    else:
        el_g_cost = g_cost_arr[new_element_i][new_element_j]
        el_h_cost = h_cost_arr[new_element_i][new_element_j]
        el_f_cost_o,el_f_cost_c = (el_g_cost[0] + el_h_cost[0], el_g_cost[1] + el_h_cost[1])
        length = len(OpenList); i=0
        while i < length:
            curr_el_i,curr_el_j = OpenList[i]
            curr_el_gcost = g_cost_arr[curr_el_i][curr_el_j]
            curr_el_hcost = h_cost_arr[curr_el_i][curr_el_j]
            curr_el_fcost_o,curr_el_fcost_c = (curr_el_gcost[0] + curr_el_hcost[0], curr_el_gcost[1] + curr_el_hcost[1])
            if (el_f_cost_o * sqrt(2) + el_f_cost_c < curr_el_fcost_o * sqrt(2) + curr_el_fcost_c):
                break
            elif (el_f_cost_o * sqrt(2) + el_f_cost_c == curr_el_fcost_o * sqrt(2) + curr_el_fcost_c) and (el_h_cost[0] * sqrt(2) + el_h_cost[1] < curr_el_hcost[0] * sqrt(2) + curr_el_hcost[1]):
                break
            i += 1
        OpenList.insert(i,(new_element_i,new_element_j))

def AstarModified(start_idx_i,start_idx_j,end_idx_i,end_idx_j): # allowed to go inside inflation zones, and even occupied zones, but occupied and inflation has high offset in g-cost
    global di,dj
    A_infl = inflation_map
    A_obs = log_odds_map
    foundgoal = False
    path = []
    OpenList = []
    # initialize g and h cost arrays (since A* only needs 1 function in searching, no need to create global g and h cost arrays)
    g_cost_star = [[0 for j in range(dj)] for i in range(di)]
    h_cost_star = [[0 for j in range(dj)] for i in range(di)]
    visitedlist = [[0 for j in range(dj)] for i in range(di)]
    parent_arr = [[0 for j in range(dj)] for i in range(di)]
    
    # initiate global g and h cost arrays, change to global one in main code if used global
    for i in range(di):
        for j in range(dj):
            g_cost_star[i][j] = (inf,inf) # costs in terms of ordinal,cardinal

            # compute h-cost to goal
            diff_i = abs(end_idx_i - i)
            diff_j = abs(end_idx_j - j)
            h_ord = min(diff_i,diff_j)
            h_card = max(diff_i,diff_j) - h_ord
            h_cost_star[i][j] = (h_ord,h_card)
    
    g_cost_star[start_idx_i][start_idx_j] = (0,0)
    
    OpenList.append((start_idx_i,start_idx_j))
    
    while not not OpenList and not foundgoal: # iterate until goal is found or OpenList empty
        curr_i,curr_j = OpenList.pop(0)
        if visitedlist[curr_i][curr_j] == 0:
            visitedlist[curr_i][curr_j] = 1
            
            indexes = getNeighborAstar(curr_i,curr_j)
            
            for idx in indexes:
                i = idx[0]; j = idx[1]
                if i == end_idx_i and j == end_idx_j: # found goal during 8-N expansion about curr_i,curr_j
                    i = end_idx_i; j = end_idx_j
                    parent_arr[i][j] = (curr_i,curr_j)
                    path.append((i,j))
                    while i!= start_idx_i or j != start_idx_j:
                        # print(parent_arr[i][j])
                        i,j = parent_arr[i][j]
                        path.append((i,j))
                    return path
                
                diffi = idx[0] - curr_i
                diffj = idx[1] - curr_j
                g_cost_par = g_cost_star[curr_i][curr_j]
                g_cost_before = g_cost_star[i][j]
                
                
                # update g-cost first, if cheaper, then add to open list and append parent
                if abs(diffi) == 1 and abs(diffj) == 1:# ordinal neighbor
                    if A_obs[i][j] > L_THRESH: # occupied
                        g_cost_curr = (g_cost_par[0] + OFFS_OCC, g_cost_par[1])
                    elif A_infl[i][j] > 0: # inflation, but not occupied
                        g_cost_curr = (g_cost_par[0] + OFFS_INF, g_cost_par[1])
                    else: # free
                        g_cost_curr = (g_cost_par[0] + 1, g_cost_par[1])
                else: # cardinal neighbor
                    if A_obs[i][j] > L_THRESH: # occupied
                        g_cost_curr = (g_cost_par[0], g_cost_par[1] + OFFS_OCC)
                    elif A_infl[i][j] > 0: # inflation, but not occupied
                        g_cost_curr = (g_cost_par[0], g_cost_par[1] + OFFS_INF)
                    else: # free
                        g_cost_curr = (g_cost_par[0], g_cost_par[1] + 1)
                
                # if cheaper, append to open list and modify parent
                if g_cost_before[0] * sqrt(2) + g_cost_before[1] > g_cost_curr[0] * sqrt(2) + g_cost_curr[1] :
                    g_cost_star[i][j] = g_cost_curr
                    parent_arr[i][j] = (curr_i,curr_j)
                    # print("appended ",i,j)
                    sortappendStar(OpenList,i,j,g_cost_star,h_cost_star)

# JPS codes
def sortappendJPS(OpenListJPS, new_element_i,new_element_j):
    g_cost_arr = G_Cost
    h_cost_arr = H_Cost
    
    if not OpenListJPS:
        OpenListJPS.append((new_element_i,new_element_j))
        return
    
    else:
        el_g_cost = g_cost_arr[new_element_i][new_element_j]
        el_h_cost = h_cost_arr[new_element_i][new_element_j]
        el_f_cost_o,el_f_cost_c = (el_g_cost[0] + el_h_cost[0], el_g_cost[1] + el_h_cost[1])
        length = len(OpenListJPS); i=0
        while i < length:
            curr_el_i,curr_el_j = OpenListJPS[i]
            curr_el_gcost = g_cost_arr[curr_el_i][curr_el_j]
            curr_el_hcost = h_cost_arr[curr_el_i][curr_el_j]
            curr_el_fcost_o,curr_el_fcost_c = (curr_el_gcost[0] + curr_el_hcost[0], curr_el_gcost[1] + curr_el_hcost[1])
            if (el_f_cost_o * sqrt(2) + el_f_cost_c < curr_el_fcost_o * sqrt(2) + curr_el_fcost_c):
                break
            elif (el_f_cost_o * sqrt(2) + el_f_cost_c == curr_el_fcost_o * sqrt(2) + curr_el_fcost_c) and (el_h_cost[0] * sqrt(2) + el_h_cost[1] < curr_el_hcost[0] * sqrt(2) + curr_el_hcost[1]):
                break
            i += 1
        OpenListJPS.insert(i,(new_element_i,new_element_j))

def searchHorizontalJPS(OpenListJPS, parentArr, istart, jstart, goalJPS, dir_right): #(istart,jstart) = index of root cell of expansion
    # returns OpenList with appended forced neighbor if any, and sorts it at the end based on f and then h cost
    # parentArr is locally from diagonal, OpenList also local
    # also returns status of whether or not forced neighbor is found, and if goal is found
   
    g_cost_arr = G_Cost
    
    JPSmap = inflation_map # replace this with inflation as well as obstacle status (inflation is sufficient because obstacles are automatically inflation)
    increment = 1 if dir_right == 1 else -1
    foundNeighbor = False
    curr_j = jstart
    
    # instead of checking ahead before incrementing, we increment first, and then check
    while True:
        curr_j += increment
        #print("searching horizontally ",istart,curr_j)
    # check off-map first
        if curr_j >= dj or curr_j < 0: # Horizontal Expansion off map
            return False,False # last 2 booleans: foundNeighbor and isGoal
    
    # if infront is blocked, simply return 
        if JPSmap[istart][curr_j] > 0: # inflation count greater than 1, hence blocked
            return False,False
        
       # if goal, then update parent of goal, append goal to OpenList (no need to sort anymore), and then return
        if (istart,curr_j) == goalJPS:            
            #print((istart,curr_j),(istart,jstart - increment), "horizontal")
            parentArr[istart][curr_j] = (istart,jstart - increment)
            OpenListJPS.append((istart,curr_j))
            return False,True
            
        # if all 3 checks are not true, then either still moving (uninteresting points) or found forced neighbors
        if istart-1 >= 0 and JPSmap[istart-1][curr_j] == 0 and JPSmap[istart-1][curr_j-increment] > 0 : # forced neighbor above
            foundNeighbor = True # terminate when FN are FOUND, even though its not cheaper
            # compute g-cost of forced neighbors, if lower then update parent and add to OpenListJPS
            g_cost_root = g_cost_arr[istart][jstart] #g_cost_par = g_cost of root + how much we move. This is g_cost of parent before forced neighbor
            g_cost_par = (g_cost_root[0], g_cost_root[1] + abs(curr_j - increment - jstart)) # ordinal remains same, cardinal is + how much we move
            g_cost_fn = (g_cost_par[0] + 1,g_cost_par[1]) # fn wrt to parent, always increment ordinal, but cardinal same
            g_cost_fn_before = g_cost_arr[istart-1][curr_j]
            
            # compare the g-costs
            # can change to optimise for integer
            if (g_cost_fn_before[0] * sqrt(2) + g_cost_fn_before[1]) > (g_cost_fn[0] * sqrt(2) + g_cost_fn[1]): # strictly cheaper
                g_cost_arr[istart-1][curr_j] = g_cost_fn # update g_cost of FN
                parentArr[istart-1][curr_j] = (istart,curr_j-increment) # update parent of FN
                parentArr[istart][curr_j - increment] = (istart,jstart - increment) # update parent of parent of FN
                sortappendJPS(OpenListJPS, istart-1,curr_j) # append location of forced neighbor
        
        # Check other direction
        if istart+1 <di and JPSmap[istart+1][curr_j] == 0 and JPSmap[istart+1][curr_j-increment] > 0: # forced neighbor below
            foundNeighbor = True # terminate when FN are FOUND, even though its not cheaper
            # compute g-cost of forced neighbors, if lower then update parent and add to OpenListJPS
            g_cost_root = g_cost_arr[istart][jstart] #g_cost_par = g_cost of root + how much we move. This is g_cost of parent before forced neighbor
            g_cost_par = (g_cost_root[0], g_cost_root[1] + abs(curr_j - increment - jstart)) # ordinal remains same, cardinal is + how much we move
            g_cost_fn = (g_cost_par[0] + 1,g_cost_par[1]) # fn wrt to parent, always increment ordinal, but cardinal same
            g_cost_fn_before = g_cost_arr[istart+1][curr_j]
            
            #print("root ",g_cost_root, "g_cost_fn", g_cost_fn, "before", g_cost_arr[istart+1][curr_j], "par", g_cost_par)
            # compare the g-costs
            # can change to optimise for integer
            if (g_cost_fn_before[0] * sqrt(2) + g_cost_fn_before[1]) > (g_cost_fn[0] * sqrt(2) + g_cost_fn[1]): # strictly cheaper
                g_cost_arr[istart+1][curr_j] = g_cost_fn # update g_cost of FN
                parentArr[istart+1][curr_j] = (istart,curr_j-increment) # update parent of FN
                parentArr[istart][curr_j - increment] = (istart,jstart - increment) # update parent of parent of FN
                sortappendJPS(OpenListJPS, istart+1,curr_j) # append location of forced neighbor
                
            
        # if ANY forced Neighbor found, then foundNeighbor would be true. Hence, we add istart,curr_j (the vertex ahead) if its cheaper, and terminate
        if foundNeighbor:
            g_cost_front = (g_cost_par[0],g_cost_par[1] + 1) # g_cost of front is just +1 cardinal
            g_cost_front_before = g_cost_arr[istart][curr_j]
            if (g_cost_front_before[0] * sqrt(2) + g_cost_front_before[1]) > (g_cost_front[0] * sqrt(2) + g_cost_front[1]): # strictly cheaper
                g_cost_arr[istart][curr_j] = g_cost_front # update g_cost of Front
                parentArr[istart][curr_j] = (istart,jstart-increment) # update parent of Front, directly to 1 - root
                sortappendJPS(OpenListJPS, istart,curr_j) # append location of Front
                
            return True,False # same level as foundNeighbor, because must exit even though not cheaper

def searchVerticalJPS(OpenListJPS, parentArr, istart, jstart, goalJPS, dir_bot): #(istart,jstart) = index of root cell of expansion
    # returns OpenList with appended forced neighbor if any, and sorts it at the end based on f and then h cost
    # parentArr is locally from diagonal, OpenList also local
    # also returns status of whether or not forced neighbor is found, and if goal is found

    g_cost_arr = G_Cost
    
    JPSmap = inflation_map # replace this with inflation as well as obstacle status (inflation is sufficient because obstacles are automatically inflation)
    increment = 1 if dir_bot == 1 else -1
    foundNeighbor = False
    curr_i = istart
    
    # instead of checking ahead before incrementing, we increment first, and then check
    while True:
        curr_i += increment
        #print("searching vertically ",curr_i,jstart)
    # check off-map first
        if curr_i >= di or curr_i < 0: # Vertical Expansion off map
            return False,False # last 2 booleans: foundNeighbor and isGoal
    
    # if infront is blocked, simply return 
        if JPSmap[curr_i][jstart] > 0: # inflation count greater than 1, hence blocked
            return False,False
        
       # if goal, then update parent of goal, append goal to OpenList (no need to sort anymore), and then return
        if (curr_i,jstart) == goalJPS:
            #print((curr_i,jstart),(istart-increment,jstart), "vertical")
            parentArr[curr_i][jstart] = (istart-increment,jstart)
            OpenListJPS.append((curr_i,jstart))
            return False,True
            
        # if all 3 checks are not true, then either still moving (uninteresting points) or found forced neighbors
        if jstart+1 < dj and JPSmap[curr_i][jstart+1] == 0 and JPSmap[curr_i - increment][jstart+1] > 0 : # forced neighbor right
            foundNeighbor = True # terminate when FN are FOUND, even though its not cheaper
            # compute g-cost of forced neighbors, if lower then update parent and add to OpenListJPS
            g_cost_root = g_cost_arr[istart][jstart] #g_cost_par = g_cost of root + how much we move. This is g_cost of parent before forced neighbor
            g_cost_par = (g_cost_root[0], g_cost_root[1] + abs(curr_i - increment - istart)) # ordinal remains same, cardinal is + how much we move
            g_cost_fn = (g_cost_par[0] + 1,g_cost_par[1]) # fn wrt to parent, always increment ordinal, but cardinal same
            g_cost_fn_before = g_cost_arr[curr_i][jstart+1]
            
            # compare the g-costs
            # can change to optimise for integer
            if (g_cost_fn_before[0] * sqrt(2) + g_cost_fn_before[1]) > (g_cost_fn[0] * sqrt(2) + g_cost_fn[1]): # strictly cheaper
                g_cost_arr[curr_i][jstart+1] = g_cost_fn # update g_cost of FN
                parentArr[curr_i][jstart+1] = (curr_i - increment,jstart) # update parent of FN
                parentArr[curr_i-increment][jstart] = (istart-increment,jstart) # update parent of parent of FN
                sortappendJPS(OpenListJPS, curr_i,jstart+1) # append location of forced neighbor
                
        
        # Check other direction
        if jstart-1 >=0 and JPSmap[curr_i][jstart-1] == 0 and JPSmap[curr_i - increment][jstart-1] > 0 : # forced neighbor left
            foundNeighbor = True # terminate when FN are FOUND, even though its not cheaper
            # compute g-cost of forced neighbors, if lower then update parent and add to OpenListJPS
            g_cost_root = g_cost_arr[istart][jstart] #g_cost_par = g_cost of root + how much we move. This is g_cost of parent before forced neighbor
            g_cost_par = (g_cost_root[0], g_cost_root[1] + abs(curr_i - increment - istart)) # ordinal remains same, cardinal is + how much we move
            g_cost_fn = (g_cost_par[0] + 1,g_cost_par[1]) # fn wrt to parent, always increment ordinal, but cardinal same
            g_cost_fn_before = g_cost_arr[curr_i][jstart-1]
            
            # compare the g-costs
            # can change to optimise for integer
            if (g_cost_fn_before[0] * sqrt(2) + g_cost_fn_before[1]) > (g_cost_fn[0] * sqrt(2) + g_cost_fn[1]): # strictly cheaper
                g_cost_arr[curr_i][jstart-1] = g_cost_fn # update g_cost of FN
                parentArr[curr_i][jstart-1] = (curr_i - increment,jstart) # update parent of FN
                parentArr[curr_i-increment][jstart] = (istart-increment,jstart) # update parent of parent of FN
                sortappendJPS(OpenListJPS, curr_i,jstart-1) # append location of forced neighbor
                
            
        # if ANY forced Neighbor found, then foundNeighbor would be true. Hence, we add istart,curr_j (the vertex ahead) if its cheaper, and terminate
        if foundNeighbor:
            g_cost_front = (g_cost_par[0],g_cost_par[1] + 1) # g_cost of front is just +1 cardinal
            g_cost_front_before = g_cost_arr[curr_i][jstart]
            if (g_cost_front_before[0] * sqrt(2) + g_cost_front_before[1]) > (g_cost_front[0] * sqrt(2) + g_cost_front[1]): # strictly cheaper
                g_cost_arr[curr_i][jstart] = g_cost_front # update g_cost of Front
                parentArr[curr_i][jstart] = (istart-increment,jstart) # update parent of Front, directly to 1 before root
                sortappendJPS(OpenListJPS, curr_i,jstart) # append location of Front
                
            return True,False # same line as foundNeighbor, because regardless of added or not must return

def searchDiagonal(OpenListJPS,parentArr,istart,jstart,goalJPS, dir_right, dir_bot):
    g_cost_arr = G_Cost
    
    found_n_h = False # by default, False, needed in case one of the directions is blocked from first iteration
    found_n_v = False
    
    # only need to return whether or not goal is found
    
    JPSmap = inflation_map # replace this with inflation as well as obstacle status (inflation is sufficient because obstacles are automatically inflation)
    
    j_inc = 1 if dir_right == 1 else -1
    i_inc = 1 if dir_bot == 1 else -1
    
    
    curr_i_d = istart
    curr_j_d = jstart # current diagonal
    
    # store the first cell we start, as the root
    g_cost_root = g_cost_arr[istart][jstart] # this is to compute the 1-side FN costs
    stop_diagonal = False
    goal_reached = False
    
    while True: # this loop is for the WHOLE ordinal expansion, at the end of each iteration, increment curr_i_d and curr_j_d
        #print("searching diagonally ",curr_i_d,curr_j_d)
        # going horizontal first
        curr_j = curr_j_d
        # for each cardinal, check if front is blocked or out of map, then check its 1-side FN before calling searchHorizontal
        # if front is blocked or out of map, then don't bother with this cardinal direction, go to the other cardinal direction
        curr_j += j_inc
        if curr_j >= dj or curr_j < 0: # out of map
            pass
        elif JPSmap[curr_i_d][curr_j] > 0: # blocked
            pass
        elif (curr_i_d,curr_j) == goalJPS: # if goal is directly next, then just return
            #print((curr_i_d,curr_j_d),(istart-i_inc,jstart- j_inc), "diagonal next horizontal")
            parentArr[curr_i_d][curr_j] = (curr_i_d,curr_j_d)
            parentArr[curr_i_d][curr_j_d] = (istart-i_inc,jstart- j_inc)
            OpenListJPS.append((curr_i_d,curr_j))
            return True
        else: # free front, check 1-side FN
            if JPSmap[curr_i_d-i_inc][curr_j] == 0 and JPSmap[curr_i_d-i_inc][curr_j-j_inc] > 0: # no nid to check istart-i_inc in map, because start of diagonal expansion cannot be at corners or edges
                # we only start expanding diagonally if we move diagonally FROM another cell, which means there is at least length 1 in both directions
                # FN 1-side horizontal
                g_cost_par = (g_cost_root[0] + abs(curr_j_d - jstart), g_cost_root[1]) # only + ordinals from current diagonal to starting diagonal, and no. of ordinals = diff in cardinals
                g_cost_fn = (g_cost_par[0] + 1, g_cost_par[1])
                g_cost_fn_before = g_cost_arr[curr_i_d-i_inc][curr_j]
                if (g_cost_fn_before[0] * sqrt(2) + g_cost_fn_before[1]) > (g_cost_fn[0] * sqrt(2) + g_cost_fn[1]):
                    g_cost_arr[curr_i_d-i_inc][curr_j] = g_cost_fn
                    parentArr[curr_i_d-i_inc][curr_j] = (curr_i_d,curr_j_d)
                    parentArr[curr_i_d][curr_j_d] = (istart-i_inc,jstart- j_inc) # update parent of parent
                    sortappendJPS(OpenListJPS,curr_i_d - i_inc, curr_j)
                    stop_diagonal = True
            # then regardless of whether or not we found a forced neighbor, expand this side, starting from curr_j (1 ahead curr diagonal)
            # but first, must compute the starting g_cost of the cardinal (because not done in searchHorizontal; it assume the start g_cost ard computed)
            # up to this point, only have g_cost_root, hence must extrapolate current diag from root
            g_cost_diag = (g_cost_root[0] + abs(curr_j_d - jstart), g_cost_root[1])
            g_cost_arr[curr_i_d][curr_j] = (g_cost_diag[0], g_cost_diag[1] + 1) # +1 cardinal from current diagonal
            found_n_h, goal_reached = searchHorizontalJPS(OpenListJPS,parentArr,curr_i_d,curr_j,goalJPS,dir_right)
            
            # rule: if forced N found, still nid to continue other direction. If goal_reached, immediately return (parent ard appended in searchHorizontal)
            if goal_reached:
                parentArr[curr_i_d][curr_j_d] = (istart-i_inc, jstart-j_inc)
                return True
            
        # going vertical, if goal found, then would have returned before
        curr_i = curr_i_d
        # for each cardinal, check if front is blocked or out of map, then check its 1-side FN before calling searchVertical
        # if front is blocked or out of map, then don't bother with this cardinal direction, go to the other cardinal direction
        curr_i += i_inc
        if curr_i >= di or curr_i < 0: # out of map
            pass
        elif JPSmap[curr_i][curr_j_d] > 0: # blocked
            pass
        elif (curr_i,curr_j_d) == goalJPS: # if goal is directly next, then just return
            #print((curr_i,curr_j_d),(curr_i_d,curr_j_d), "diagonal next vetical")
            parentArr[curr_i][curr_j_d] = (curr_i_d,curr_j_d)
            parentArr[curr_i_d][curr_j_d] = (istart-i_inc,jstart- j_inc)
            OpenListJPS.append((curr_i,curr_j_d))
            return True
        else: # free front, check 1-side FN
            if JPSmap[curr_i][curr_j_d - j_inc] == 0 and JPSmap[curr_i-i_inc][curr_j_d-j_inc] > 0: # no nid to check istart-i_inc in map, because start of diagonal expansion cannot be at corners or edges
                # we only start expanding diagonally if we move diagonally FROM another cell, which means there is at least length 1 in both directions
                # FN 1-side horizontal
                g_cost_par = (g_cost_root[0] + abs(curr_j_d - jstart), g_cost_root[1]) # only + ordinals from current diagonal to starting diagonal, and no. of ordinals = diff in cardinals
                g_cost_fn = (g_cost_par[0] + 1, g_cost_par[1])
                g_cost_fn_before = g_cost_arr[curr_i][curr_j_d-j_inc]
                if (g_cost_fn_before[0] * sqrt(2) + g_cost_fn_before[1]) > (g_cost_fn[0] * sqrt(2) + g_cost_fn[1]):
                    g_cost_arr[curr_i][curr_j_d - j_inc] = g_cost_fn
                    parentArr[curr_i][curr_j_d - j_inc] = (curr_i_d,curr_j_d)
                    parentArr[curr_i_d][curr_j_d] = (istart-i_inc,jstart-j_inc)
                    sortappendJPS(OpenListJPS,curr_i, curr_j_d - j_inc)
                    stop_diagonal = True
            # then regardless of whether or not we found a forced neighbor, expand this side, starting from curr_j (1 ahead curr diagonal)
            # but first, must compute the starting g_cost of the cardinal (because not done in searchHorizontal; it assume the start g_cost ard computed)
            # up to this point, only have g_cost_root, hence must extrapolate current diag from root
            g_cost_diag = (g_cost_root[0] + abs(curr_j_d - jstart), g_cost_root[1])
            g_cost_arr[curr_i][curr_j_d] = (g_cost_diag[0], g_cost_diag[1] + 1) # +1 cardinal from current diagonal
            found_n_v, goal_reached = searchVerticalJPS(OpenListJPS,parentArr,curr_i,curr_j_d,goalJPS,dir_bot)
            
            # rule: if forced N found, still nid to continue other direction. If goal_reached, immediately return (parent ard appended in searchVertical)
            # goal reached cardinally, hence current diaggonal is special. Must update its parent
            if goal_reached: # error previously, if goal found directly return. The current diagonal parent is not updated; parent of goal is ard updated to be current diagonal (i.e. 1 before the start of cardinal expansion, but 1 before start is still diagonal, i.e. no parent yet)
                parentArr[curr_i_d][curr_j_d] = (istart-i_inc, jstart-j_inc)
                return True
        
        # end of both cardinals, increment diagonal and check for diagonally blocked, or if diagonal is out of map or goal or blocked
        # if neither, then see whether any 1-side neighbors (check stop diagonal) or cardinal FN (check found_n_v and found_n_h). If any found, then append
        # diagonal and return, else carry on to next loop
        curr_i_d += i_inc
        curr_j_d += j_inc
        
        if curr_j_d >= dj or curr_j_d < 0 or curr_i_d >= di or curr_i_d < 0: # out of map
            return False
        elif JPSmap[curr_i_d][curr_j_d] > 0: # in-front blocked
            return False
        elif JPSmap[curr_i_d - i_inc][curr_j_d] > 0 and JPSmap[curr_i_d][curr_j_d - j_inc] > 0: # diagonally blocked
            return False
        elif JPSmap[curr_i_d][curr_j_d] == goalJPS: # goal reached diagonally, hence directly append to root
            parentArr[curr_i_d][curr_j_d] = (istart-i_inc,jstart-j_inc)
            OpenListJPS.append((curr_i_d,curr_j_d))
            return True
        elif stop_diagonal or found_n_v or found_n_h: # some forced neighbor found, hence append infront diagonal if cheaper
            g_cost_diag = (g_cost_root[0] + abs(curr_j_d - jstart), g_cost_root[1]) # only diagonal from root
            g_cost_diag_before = g_cost_arr[curr_i_d][curr_j_d]
            if (g_cost_diag_before[0] * sqrt(2) + g_cost_diag_before[1]) > (g_cost_diag[0] * sqrt(2) + g_cost_diag[1]):
                g_cost_arr[curr_i_d][curr_j_d] = g_cost_diag
                parentArr[curr_i_d][curr_j_d] = (istart-i_inc, jstart-j_inc)
                parentArr[curr_i_d-i_inc][curr_j_d-j_inc] = (istart-i_inc,jstart-j_inc) # error previously was if found_n_v or found_n_h the current diagonal parent isn't updated, was only updated during 1-side FN check
                sortappendJPS(OpenListJPS,curr_i_d, curr_j_d)
                return False

def JPS(start_idx_i,start_idx_j,goal_idx_i,goal_idx_j):
    g_cost_arr = G_Cost
    h_cost_arr = H_Cost
    
    # Define OpenList to use in iteration
    OpenListJPS = []
    path = []
    
    # initialization
    goal_found = False
    goalJPS = (goal_idx_i,goal_idx_j)
    visitedlist = [[0 for j in range(dj)] for i in range(di)]
    parent_arr = [[0 for j in range(dj)] for i in range(di)]
    
    # initiate global g and h cost arrays, change to global one in main code if used global
    for i in range(di):
        for j in range(dj):
            g_cost_arr[i][j] = (inf,inf) # costs in terms of ordinal,cardinal

            # compute h-cost to goal
            diff_i = abs(goal_idx_i - i)
            diff_j = abs(goal_idx_j - j)
            h_ord = min(diff_i,diff_j)
            h_card = max(diff_i,diff_j) - h_ord
            h_cost_arr[i][j] = (h_ord,h_card)
            
    # Initialize start g-cost, no need to append start to OpenList because it is always picked first
    g_cost_arr[start_idx_i][start_idx_j] = (0,0)
    visitedlist[start_idx_i][start_idx_j] = 1
    
    # To start, expand the 8-neighbors of the starting position, and compute their respective g-cost
    for idx in getFreeNeighborsJPS(start_idx_i,start_idx_j):
        i = idx[0];j = idx[1]
        diffi = i - start_idx_i
        diffj = j - start_idx_j
        parent_arr[i][j] = (start_idx_i,start_idx_j)
        if abs(diffi) == 1 and abs(diffj) == 1: # ordinal neighbors
            g_cost_arr[i][j] = (1,0)
            sortappendJPS(OpenListJPS,i,j) # so that OpenListJPS remains sorted
        else:
            g_cost_arr[i][j] = (0,1) # cardinal neighbors
            sortappendJPS(OpenListJPS,i,j)
    #print(OpenListJPS)

    # pick start of JPS expansion from these original neighbors, base on f-cost then h-cost (since OpenList ard sorted, simply pick first element)
    # iterate until open_list is empty or if goal is reached
    
    while not not OpenListJPS and not goal_found:
    
        curr_i, curr_j = OpenListJPS.pop(0)
        
        if visitedlist[curr_i][curr_j] == 0:
            visitedlist[curr_i][curr_j] = 1 # mark as visisted
            
            curr_par = parent_arr[curr_i][curr_j]
            dir_right = sign(curr_j - curr_par[1])
            dir_bot = sign(curr_i - curr_par[0])
            
            
            if abs(dir_right) == 1 and abs(dir_bot) == 1:
                goal_found = searchDiagonal(OpenListJPS,parent_arr,curr_i,curr_j,goalJPS,dir_right,dir_bot)
            elif abs(dir_right) == 1:
                fn,goal_found = searchHorizontalJPS(OpenListJPS,parent_arr,curr_i,curr_j,goalJPS,dir_right)
            elif abs(dir_bot) == 1:
                fn,goal_found = searchVerticalJPS(OpenListJPS,parent_arr,curr_i,curr_j,goalJPS,dir_bot)
    
    if goal_found:
        idx_i,idx_j = (goal_idx_i,goal_idx_j)
        #print("start and goal\n", idx_i,idx_j,start_idx_i, start_idx_j)
        path.append((idx_i,idx_j))
        while idx_i != start_idx_i or idx_j != start_idx_j:
            #print(parent_arr[idx_i][idx_j])
            if parent_arr[idx_i][idx_j] == 0:
                return False
            idx_i,idx_j = parent_arr[idx_i][idx_j]
            path.append((idx_i,idx_j))
     
    if not path:
        return False
    else:                 
        return path

def getFreeNeighborsJPS(start_i,start_j):
    # simple 8-neighbors, return the list of indexes
    indexes = []
    
    JPSmap = inflation_map
    
    for i in range(-1,2):
        for j in range(-1,2):
            if i==0 and j==0:
                continue
            elif start_i + i < 0 or start_i + i >= di or start_j + j < 0 or start_j + j >=dj: # out of map
                continue
            else:
                if JPSmap[start_i + i][start_j + j] == 0:
                    indexes.append((start_i + i, start_j + j))
                
    return indexes

# Post processing
def GeneralIntLos(min_pos, cell_size, start_pos,end_pos): # for world position inputs
    start_idx = (int64(round((start_pos[0] - min_pos[0])/cell_size)), int64(round((start_pos[1] - min_pos[1])/cell_size)))
    end_idx = (int64(round((end_pos[0] - min_pos[0])/cell_size)), int64(round((end_pos[1] - min_pos[1])/cell_size)))
    
    indices = []
    indices.append(start_idx)
    
    diffx = end_idx[0] - start_idx[0]
    diffy = end_idx[1] - start_idx[1]
    if abs(diffx) > abs(diffy):
        diffl = diffx
        diffs = diffy
        to_long_short = lambda l,s : (l,s)
        
    else:
        diffl = diffy
        diffs = diffx
        to_long_short = lambda l,s : (s,l)
        
    ls_start_idx = list(to_long_short(start_idx[0],start_idx[1]))
    ls_end_idx = list(to_long_short(end_idx[0], end_idx[1]))
    
    # increments and error init
    sign_l = sign(diffl)
    sign_s = sign(diffs)
    e_inc = 2 * diffs
    e_dec = 2 * abs(diffl) * sign_s
    e = 0
    
    # length of a is
    a = abs(diffs) - abs(diffl)
    
    # error checker (more accurate than using absolute, because of the upper and lower bounds)
    if sign_s >= 0 :
        error_check = lambda error : error >= abs(diffl)
    else:
        error_check = lambda error : error < -abs(diffl)
        
    # start propagating to end
    while ls_start_idx != ls_end_idx:
        ls_start_idx[0] += sign_l
        e += e_inc
        if error_check(e) :
            e -= e_dec
            ls_start_idx[1] += sign_s
            
            b = sign_s * e
            
            
            if a > b:
                indices.append(to_long_short(ls_start_idx[0],ls_start_idx[1] - sign_s))
                if(ls_start_idx[0] == ls_end_idx[0]) and (ls_start_idx[1] - sign_s) == ls_end_idx[1]:
                    break
            elif a < b:
                indices.append(to_long_short(ls_start_idx[0] - sign_l,ls_start_idx[1]))
                if(ls_start_idx[0] - sign_l == ls_end_idx[0]) and (ls_start_idx[1]) == ls_end_idx[1]:
                    break                
            else:
                #indices.append(to_long_short(ls_start_idx[0] - sign_l,ls_start_idx[1]))
                #indices.append(to_long_short(ls_start_idx[0],ls_start_idx[1] - sign_s))
                pass
        indices.append(to_long_short(ls_start_idx[0],ls_start_idx[1]))
        
    return indices

def GeneralLosPP(start_pos,end_pos): # for post processing, int optimised, takes in map indexes
    threshold = 0.5 # set where to start drawing line
    start_idx = (int(round(start_pos[0])), int(round(start_pos[1])))
    end_idx = (int64(round(end_pos[0])), int64(round(end_pos[1])))
    
    
    start_float = (start_pos[0] , start_pos[1])
    end_float = (end_pos[0] , end_pos[1] )
    
    indices = [] # init an empty list
    indices.append(start_idx) # append the starting index into the cell

    # assign long and short axes
    diffx = end_float[0] - start_float[0]
    diffy = end_float[1] - start_float[1]
    if abs(diffx) > abs(diffy):
        diffl = diffx
        diffs = diffy
        to_long_short = lambda l,s : (l,s)
        to_x_y = lambda x,y : (x,y)

    else:
        diffl = diffy
        diffs = diffx
        to_long_short = lambda l,s : (s,l)
        to_x_y = lambda x,y : (y,x)

    # get start and final in terms of long and short axis
    ls_start_idx = list(to_long_short(start_idx[0],start_idx[1]))
    ls_end_idx = list(to_long_short(end_idx[0],end_idx[1]))
    ls_start_float = list(to_long_short(start_float[0],start_float[1]))
    ls_end_float = list(to_long_short(end_float[0],end_float[1]))

    # get signs of increments and short axis increment
    sign_l = sign(diffl)
    sign_s = sign(diffs)
    short_inc = diffs * 1.0 /diffl * sign_l

    # initiate rounding error
    error_s = ls_start_float[1] - ls_start_idx[1]
    error_l = ls_start_float[0] - ls_start_idx[0]

    # get length of a (refer to notes)
    dlong = 0.5 + error_l * sign_l
    length_a = round(abs(short_inc * dlong),8)
    
    # error checker
    if sign_s >= 0 :
        error_check = lambda error : error >= threshold
    else:
        error_check = lambda error : error < -threshold

    # perform general line, loop until end is reached
    while (ls_start_idx != ls_end_idx):
        ls_start_idx[0] += sign_l
        error_s += short_inc
        if error_check(error_s):
            error_s -= sign_s
            ls_start_idx[1] += sign_s
            # get length of b (refer to notes)
            length_b = round((1-threshold) + error_s * sign_s,8)
            
            
            if length_a < length_b: # intermediate in long axis
                indices.append(to_x_y(ls_start_idx[0]-sign_l,ls_start_idx[1]))
                if (ls_start_idx[0] - sign_l == ls_end_idx[0]) and (ls_start_idx[1] == ls_end_idx[1]):
                    break

            elif length_a > length_b: # intermediate in short axis
                indices.append(to_x_y(ls_start_idx[0],ls_start_idx[1]-sign_s))
                if (ls_start_idx[0] == ls_end_idx[0]) and (ls_start_idx[1]-sign_s == ls_end_idx[1]):
                    break
            else:
                pass
                #indices.append(to_x_y(ls_start_idx[0]-sign_l,ls_start_idx[1]))
                #indices.append(to_x_y(ls_start_idx[0],ls_start_idx[1]-sign_s))
            
        indices.append(to_x_y(ls_start_idx[0],ls_start_idx[1]))

    return indices


# =============================== SUBSCRIBERS =========================================      
def subscribe_scan(msg):
    # stores a 360 long tuple of LIDAR Range data into global variable rbt_scan. 
    # 0 deg facing forward. anticlockwise from top.
    global rbt_scan, write_scan, read_scan
    write_scan = True # acquire lock
    if read_scan: 
        write_scan = False # release lock
        return
    rbt_scan = msg.ranges
    write_scan = False # release lock
    
def get_scan():
    # returns scan data after acquiring a lock on the scan data to make sure it is not overwritten by the subscribe_scan handler while using it.
    global write_scan, read_scan
    read_scan = True # lock
    while write_scan:
        pass
    scan = rbt_scan # create a copy of the tuple
    read_scan = False
    return scan

def subscribe_motion(msg):
    global msg_motion
    msg_motion = msg
    
# ================================== PUBLISHERS ========================================

# =================================== TO DO ============================================
# Define the LIDAR maximum range
MAX_RNG = 3.5 #??? # search in topic /scan

# Define the inverse sensor model for mapping a range reading to world coordinates
def inverse_sensor_model(rng, deg, pose):
    rbt_degree = pose[2]*180/pi

    xk = pose[0] + rng*COS[deg + rbt_degree]
    yk = pose[1] + rng*SIN[deg + rbt_degree]
    return (xk, yk)

# ================================ BEGIN ===========================================
def main(GOALS, CELL_SIZE, MIN_POS, MAX_POS):
    # ---------------------------------- INITS ----------------------------------------------
    # init node
    rospy.init_node('main')
    # Set the labels below to refer to the global namespace (i.e., global variables)
    # global is required for writing to global variables. For reading, it is not necessary
    global rbt_scan, rbt_true, read_scan, write_scan, rbt_wheels, rbt_control
    global path,need_path,IDX_ARRAY,vt
    global msg_motion
    
    # Initialise global vars with NaN values 
    # nan and inf are imported from numpy. If you use "import numpy as np", then nan is np.nan, and inf is np.inf.
    rbt_scan = [nan]*360 # a list of 360 nans
    #rbt_true = [nan]*3
    read_scan = False
    write_scan = False
    msg_motion = None

    # Subscribers, # PROJ 2: adapt to project 2 requirement
    rospy.Subscriber('/turtle/scan', LaserScan, subscribe_scan, queue_size=1)
    rospy.Subscriber('/turtle/motion', Motion, subscribe_motion, queue_size=1)

    publisher_main = rospy.Publisher('/turtle/guider', MsgGuider, latch=True, queue_size=1)
    msg_guider = MsgGuider()
    msg_guider.stop = False

    # Publish dummy message, and then wait until turtle_movefunc.py to start
    publisher_main.publish(msg_guider)
    # Wait for Subscribers to receive data.
    while (isnan(rbt_scan[0])  or msg_motion is None) and not rospy.is_shutdown():
        pass
    
    # Data structures
    global cell_size 
    cell_size = CELL_SIZE
    global initial_value
    initial_value = 0
    global min_pos
    min_pos = MIN_POS
    max_pos = MAX_POS
    initOccGrid(min_pos, max_pos, cell_size, 0, 0.2)             
    # get the first goal pos
    goals = GOALS
    goal_pos = goals[0]
    # number of goals (i.e. areas)
    g_len = len(goals)
    # set the goal number as zero
    g = 0

    # show_map initialisation
    global img_mat
    img_mat = full((di,dj,3), uint8(127))
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', di*5, dj*5)
    
    # ---------------------------------- BEGIN ----------------------------------------------
    t = rospy.get_time()
    while (not rospy.is_shutdown()): # required to Keyboard interrupt nicely
#        print('---')
#        print('True pos: ({:.3f} {:.3f} {:.3f})'.format(rbt_true[0], rbt_true[1], rbt_true[2]))
#        continue
        
        if (rospy.get_time() > t): # every 50 ms
            
            # get scan
            scan = get_scan() #already gets ranges, because of subscribe_scan, rbt_scan global variable is assigned to be ranges
            
            # calculate the robot position using the motion model
            rbt_pos = (msg_motion.x,msg_motion.y,msg_motion.o)
            vt = msg_motion.v
            
            # for each degree in the scan
            for i in xrange(360):
                # if you use log-odds Binary Bayes
                if scan[i] != inf: # range reading is < max range ==> occupied
                    end_pos = (rbt_pos[0] + scan[i]*COS[(i+int(rbt_pos[2]*180/pi))%360],rbt_pos[1] + scan[i]*SIN[(i+int(rbt_pos[2]*180/pi))%360])
                    end_pos_idx = (int64(round((end_pos[0] - min_pos[0]) / cell_size)), int64(round((end_pos[1] - min_pos[1]) / cell_size)))
                    updateAtIdx(end_pos_idx, True)
                    # # set the obstacle cell as occupied
                else: # range reading is inf ==> no obstacle found
                    end_pos = (rbt_pos[0] + MAX_RNG*COS[(i+int(rbt_pos[2]*180/pi))%360],rbt_pos[1] + MAX_RNG*SIN[(i+int(rbt_pos[2]*180/pi))%360])
                    end_pos_idx = (int64(round((end_pos[0] - min_pos[0]) / cell_size)), int64(round((end_pos[1] - min_pos[1]) / cell_size)))
                    updateAtIdx(end_pos_idx, False)
                    # # set the last cell as free
                # # set all cells between current cell and last cell as free
                for idx in LOS(rbt_pos, end_pos):
                    ipos = int(idx[0])
                    jpos = int(idx[1])
                    if(idx[0] != end_pos_idx[0]) or (idx[1] != end_pos_idx[1]):
                        updateAtIdx((ipos, jpos), False)

            # plan
            rbt_idx = (int64(round((rbt_pos[0] - min_pos[0]) / cell_size)), int64(round((rbt_pos[1] - min_pos[1]) / cell_size)))
            goal_idx = (int64(round((goal_pos[0] - min_pos[0]) / cell_size)), int64(round((goal_pos[1] - min_pos[1]) / cell_size)))
            
            if need_path :#or not path: # if path is empty or if path is needed
                #print("Inflation in path, replanning")
                for idx in path: # set previous idx of path in IDX_ARR to be 0
                    i = idx[0]; j = idx[1]
                    IDX_ARRAY[i][j] = 0

                # Then replan a new path and set these idx to be 1
                path = JPS(rbt_idx[0], rbt_idx[1], goal_idx[0], goal_idx[1])
                #print("JPS path",path)
                if path == False:
                    path = AstarModified(rbt_idx[0], rbt_idx[1], goal_idx[0], goal_idx[1])
                    turning_points,path = post_process_full_Astar(path)
                else:
                    turning_points,path = post_process_full_JPS(path)

                #print("replanning",path,goal_pos,goal_idx,rbt_idx,rbt_pos)
                for idx in path:
                    i = idx[0];j = idx[1]
                    #print(idx)
                    IDX_ARRAY[i][j] = 1
                #print("post process after replan",turning_points,path)



            # spline and trajectory generation
            # orientation is rbt_pos[2]
            # only generate spline if need_path = true, or when we reach first turning point. If new spline generated due to need_path,  then reset the turning point counter
            if need_path: # means we just replanned, reset turning point counter to 1 after start and current waypoint to 1
                need_path = False # means we finished doing everything related to re-planning
                curr_waypoint_idx = 1 # which waypoint of a spline we are at now
                curr_turnpoint_idx = len(turning_points) - 2 # which turnpoint we are at now
                spline_start_idx = rbt_idx # reset start to current robot idx, this is the map index to fit a spline
                curr_turnpoint = turning_points[curr_turnpoint_idx] # in index
                curr_turnpoint_x = curr_turnpoint[0] * cell_size + min_pos[0]
                curr_turnpoint_y = curr_turnpoint[1] * cell_size + min_pos[1] # to measure distance to current TURN POINT in world coordinate
                curr_v_i = vt * cos(rbt_pos[2]); curr_v_j = vt*sin(rbt_pos[2]) # since using angle, must consider actual direction
                
                if len(turning_points) > 2: # means can interpolate gradient
                    next_turnpoint = turning_points[curr_turnpoint_idx-1]
                    a0,a1,a2,a3,b0,b1,b2,b3,spt = SplineFitting(spline_start_idx[0],spline_start_idx[1],curr_turnpoint[0],curr_turnpoint[1],curr_v_i,curr_v_j,next_turnpoint[0],next_turnpoint[1]) # spt is the spline time INCREMENT
                    
                else: # can't interpolate gradient since only start and end
                    a0,a1,a2,a3,b0,b1,b2,b3,spt = SplineFitting(spline_start_idx[0],spline_start_idx[1],curr_turnpoint[0],curr_turnpoint[1],curr_v_i,curr_v_j)
                # then calculate the index as well as convert to world position of the next waypoint
                l_t = spt*curr_waypoint_idx
                i_tar = b0 + b1 * l_t + b2 * (l_t*l_t) + b3 * (l_t*l_t*l_t)
                j_tar = a0 + a1 * l_t + a2 * (l_t*l_t) + a3 * (l_t*l_t*l_t)
                curr_xpoint = i_tar * cell_size + min_pos[0]
                curr_ypoint = j_tar * cell_size + min_pos[1] # the actual world position of the WAY POINT 
            elif (rbt_pos[0] - curr_turnpoint_x)*(rbt_pos[0] - curr_turnpoint_x) +  (rbt_pos[1] - curr_turnpoint_y)*(rbt_pos[1] - curr_turnpoint_y) <= MIN_DIST*MIN_DIST:# near curr_turnpoint, increment curr_turnpoint and generate the NEXT Spline, reset waypoint idx
            # must also check if near goal
                print(" turnpoint reached, generating new spline")
                #print(turning_points,turning_points[curr_turnpoint_idx],curr_turnpoint_x,curr_turnpoint_y)
                if curr_turnpoint[0] == goal_idx[0] and curr_turnpoint[1] == goal_idx[1]:
                    need_path = True
                    g += 1
                    if g >= g_len:
                        msg_guider.stop = True
                        publisher_main.publish(msg_guider)
                        # wait for sometime for move node to pick up message
                        t += 0.3
                        while rospy.get_time() < t:
                            pass
                        break
                    goal_pos = goals[g]                     
                else: # re-fit spline with next turning point
                    curr_waypoint_idx = 1
                    spline_start_idx = curr_turnpoint # put the start idx to the current turnpoint map idx first, then move the turnpoint
                    curr_turnpoint_idx -= 1
                    curr_turnpoint = turning_points[curr_turnpoint_idx]
                    curr_turnpoint_x = curr_turnpoint[0] * cell_size + min_pos[0]
                    curr_turnpoint_y = curr_turnpoint[1] * cell_size + min_pos[1]
                    curr_v_i = vt * cos(rbt_pos[2]); curr_v_j = vt*sin(rbt_pos[2])
                    #print("current turning points",turning_points, turning_points[curr_turnpoint_idx])
                    if len(turning_points) > 2 and curr_turnpoint_idx + 1 < len(turning_points): # means can interpolate gradient
                        next_turnpoint = turning_points[curr_turnpoint_idx-1]
                        a0,a1,a2,a3,b0,b1,b2,b3,spt = SplineFitting(spline_start_idx[0],spline_start_idx[1],curr_turnpoint[0],curr_turnpoint[1],curr_v_i,curr_v_j,next_turnpoint[0],next_turnpoint[1]) # spt is the spline time INCREMENT
                        
                    else: # can't interpolate gradient since only start and end
                        a0,a1,a2,a3,b0,b1,b2,b3,spt = SplineFitting(spline_start_idx[0],spline_start_idx[1],curr_turnpoint[0],curr_turnpoint[1],curr_v_i,curr_v_j)

                    # generate next waypoint
                    l_t = spt*curr_waypoint_idx
                    i_tar = b0 + b1 * l_t + b2 * (l_t*l_t) + b3 * (l_t*l_t*l_t)
                    j_tar = a0 + a1 * l_t + a2 * (l_t*l_t) + a3 * (l_t*l_t*l_t)
                    curr_xpoint = i_tar * cell_size + min_pos[0]
                    curr_ypoint = j_tar * cell_size + min_pos[1]
            elif (rbt_pos[0] - curr_xpoint)*(rbt_pos[0] - curr_xpoint) +  (rbt_pos[1] - curr_ypoint)*(rbt_pos[1] - curr_ypoint) <= MIN_DIST*MIN_DIST:#if we near curr_waypoint
                print("waypoint reached, generating new waypoint")
                curr_waypoint_idx += 1
                print(curr_waypoint_idx)
                l_t = spt * curr_waypoint_idx
                i_tar = b0 + b1 * l_t + b2 * (l_t*l_t) + b3 * (l_t*l_t*l_t)
                j_tar = a0 + a1 * l_t + a2 * (l_t*l_t) + a3 * (l_t*l_t*l_t)
                curr_xpoint = i_tar * cell_size + min_pos[0]
                curr_ypoint = j_tar * cell_size + min_pos[1]
            #else do nothing, which means just publish the curr_xpoint and curr_ypoint. Published waypoint to follow is already in GAZEBO WORLD coordinates
            msg_guider.target.a = curr_xpoint; msg_guider.target.b = curr_ypoint;
            publisher_main.publish(msg_guider)      
            # show the map as a picture
            show_map((rbt_idx[0], rbt_idx[1]), path, (goal_idx[0], goal_idx[1]))
            
            # increment the time counter
            et = rospy.get_time() - t
            print(et <= 0.2, et)
            t += 0.2
    print('[INFO] MAIN stopped')       
        
if __name__ == '__main__':      
    try: 
        # parse goals
        goals = sys.argv[1]
        goals = goals.split('|')
        for i in xrange(len(goals)):
            tmp = goals[i].split(',')
            tmp[0] = float(tmp[0])
            tmp[1] = float(tmp[1])
            goals[i] = tmp
        
        # parse cell_size
        cell_size = float(sys.argv[2])
        
        # parse min_pos
        min_pos = sys.argv[3]
        min_pos = min_pos.split(',')
        min_pos = (float(min_pos[0]), float(min_pos[1]))
        
        # parse max_pos
        max_pos = sys.argv[4]
        max_pos = max_pos.split(',')
        max_pos = (float(max_pos[0]), float(max_pos[1]))
        main(goals, cell_size, min_pos, max_pos)
    except rospy.ROSInterruptException:
        pass