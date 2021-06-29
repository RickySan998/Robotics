#!/usr/bin/env python

import roslib, rospy, rospkg
from hector_uav_msgs.srv import EnableMotors
from math import sqrt, cos, sin, pi, atan2, asin
import sys

from pkg_hector.msg import MotionH, MsgGuiderH
from sensor_msgs.msg import Range
# ================================= PARAMETERS ========================================== 
_ALTITUDE = 5.5
_NEAR = 0.2
_SPIN_RATE = pi/5.
_DT = 0.05
# ================================= CONSTANTS ==========================================       
_PI = pi
# =============================== SUBSCRIBERS =========================================  
def subscribe_motion(msg):
    global motion
    motion = msg
       
def subscribe_turtle_motion(msg):
    global turtle_motion
    turtle_motion = msg
    
def subscribe_turtle_guider(msg):
    global turtle_guider
    turtle_guider = msg

def subscribe_sonar(msg):
    global sonar
    sonar = msg
# ================================ BEGIN ===========================================
def guider(sx=2., sy=2., gx=2., gy=2.):
    # ---------------------------------- INITS ----------------------------------------------
    # --- init node ---
    rospy.init_node('hector_motion')
    
    # --- cache global vars / constants ---
    global motion, turtle_motion, turtle_guider, sonar
    motion = None
    turtle_motion = None #Motion(5., 5., 0., 0., 0.)
    turtle_guider = None # False
    
    ALTITUDE = _ALTITUDE
    NEAR = _NEAR
    NEAR_SQ = NEAR**2
    PI = pi
    TWOPI = 2*pi
    DT = _DT
    INC_O = _SPIN_RATE * DT
    LAND_TARGET = ALTITUDE
    MAX_DESCEND = 0.5
    
    # --- Subscribers ---
    rospy.Subscriber('/hector/sonar_height', Range, subscribe_sonar, queue_size=1)
    rospy.Subscriber('/hector/motion', MotionH, subscribe_motion, queue_size=1)
    rospy.Subscriber('/turtle/motion', Motion, subscribe_turtle_motion, queue_size=1)
    rospy.Subscriber('/turtle/guider', MsgGuider, subscribe_turtle_guider, queue_size=1)
    
    while (sonar is None or motion is None or turtle_motion is None or turtle_guider is None or rospy.get_time() == 0) and not rospy.is_shutdown(): 
        pass
    if rospy.is_shutdown():
        return
    print('*GUIDER* Done waiting for topics')
    
    # --- Publishers ---
    pub_guider = rospy.Publisher('/hector/guider', MsgGuiderH, latch=True, queue_size=1)
    msg_guider = MsgGuiderH()
    pub_guider.publish(msg_guider)
    
    # --- state enums ---
    TAKEOFF = 0
    TURTLE = 1
    GOAL = 2
    BASE = 3
    LAND = 4
    
    # ---------------------------------- LOOP ----------------------------------------------
    o = 0.
    state = TAKEOFF
    msg_guider.state = TAKEOFF # send to hector_move to know which set of constants to use for PID
    stop = False // false until turtle and hector have completed
    t = rospy.get_time()
    while not rospy.is_shutdown() and not msg_guider.stop:
        if rospy.get_time() > t:
            # --- Get pose --- //of hector
            rx = motion.x
            ry = motion.y
            rz = motion.z
            ro = motion.o 
            
            if state == TAKEOFF: # takeoff
                tx = sx
                ty = sy
                tz = ALTITUDE
                to = 0  # !-- flying up no need to rotate
                if (rz - ALTITUDE) * (rz - ALTITUDE) < NEAR_SQ:
                    state = TURTLE
            elif state == TURTLE:
                turX = turtle_motion.x
                turY = turtle_motion.y
                turningPoint = turtle_guider.target
                distanceSQTtoH = (rx - turX) * (rx - turX) + (ry - turY) * (ry - turY) 
                distanceSQTtoTurn = (turningPoint[0] - turX) * (turningPoint[0] - turX) + (turningPoint[1] - turY) * (turningPoint[1] - turY)
                distanceRatioSQ = distanceSQTtoH / distanceSQTtoTurn 
                if distanceRatioSQ >= 100: # !-- assuming speed of hector is 10x speed of turtle, distance ratio should be 10 -> distanceSQ = 100
                    tx = turningPoint[0] # !-- if overshoot, just set turning point as common target, might miss but can hit next one
                    ty = turningPoint[1]
                else: # !-- if distance ratio < 10, means target within path of turtle to turning point
                    tx = turX + sqrt(distanceRatioSQ) / 10 * (turningPoint[0] - turX)
                    ty = turY + sqrt(distanceRatioSQ) / 10 * (turningPoint[1] - turY)
                distanceApartSq = (tx - rx) * (tx - rx) + (ty - ry) * (ty - ry)
                if distanceApartSq <= NEAR_SQ:
                    if turtle_guider.stop == False: # !-- turtle still has yet to reach end
                        state = GOAL
                    elif turtle_guider.stop == True and ((rx - gx) * (rx - gx) + (ry - gy) * (ry - gy)) > NEAR_SQ: # !-- turtle has terminated but hector far from goal
                        state = GOAL
                    elif turtle_guider.stop == True and ((rx - gx) * (rx - gx) + (ry - gy) * (ry - gy)) <= NEAR_SQ: # !-- turtle has terminated and hector near to goal
                        state = BASE    
            elif state == GOAL:
                tx = gx
                ty = gy
                # to = (ro + INC_O + PI) % TWOPI - PI
                distanceApartSq = (tx - rx) * (tx - rx) + (ty - ry) * (ty - ry)
                if distanceApartSq <= NEAR_SQ:
                    if turtle_guider.stop == False: # !-- turtle still has yet to reach end
                        state = TURTLE
                    else: # !-- turtle has reached end
                        state = BASE
            elif state == BASE:
                # following BASE is always LAND
                tx = sx
                ty = sy
                # to = (ro + INC_O + PI) % TWOPI - PI
                distanceApartSq = (tx - rx) * (tx - rx) + (ty - ry) * (ty - ry)
                if distanceApartSq <= NEAR_SQ:
                    state = LAND 
            elif state == LAND:
                # might want to subscribe to sonar to tell if you landed
                tx = sx
                ty = sy
                if LAND_TARGET > 0:
                    LAND_TARGET -= 5*DT*MAX_DESCEND
                else:
                    LAND_TARGET = 0
                tz = LAND_TARGET
                if sonar.range < sonar.max_range: # !-- some buffer before activating sonar checks
                    if sonar.range <= NEAR: 
                        stop = True
            msg_guider.x = tx
            msg_guider.y = ty
            msg_guider.z = tz
            msg_guider.o = to
            msg_guider.state = state
            msg_guider.stop = stop
            pub_guider.publish(msg_guider)
            
            # --- Timing ---
            et = rospy.get_time() - t
            t += DT
            if et > DT:
                print('*GUIDER* {} OVERSHOOT'.format(int(et*1000)))
    
    print('=== *GUIDER* Terminated ===')
    
if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            goals = sys.argv[3]
            goals = goals.split('|')
            goals = goals[-1]
            gx = float(goals[0]); gy = float(goals[1])
            guider(float(sys.argv[1]), float(sys.argv[2]), gx, gy)
        else:
            guider()
    except rospy.ROSInterruptException:
        pass
