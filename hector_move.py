#!/usr/bin/env python

import roslib, rospy, rospkg
from hector_uav_msgs.srv import EnableMotors
from geometry_msgs.msg import Twist
from math import sqrt, cos, sin, pi, atan2
import sys

from pkg_hector.msg import MotionH, MsgGuiderH
# ================================= PARAMETERS ========================================== 
# Best to have different set of parameters for different states
# K: to turtle
# T: takeoff and landing
# B: to base
_KP_X = 0.6;    _TP_X = 0.003;    _BP_X = 0.5;
_KI_X = 0.;     _TI_X = 0.01;     _BI_X = 0.2;
_KD_X = 0.03;   _TD_X = 0.0015;   _BD_X = 0.02;
_KP_Y = _KP_X;  _TP_Y = _TP_X;    _BP_Y = _BP_X;
_KI_Y = _KI_X;  _TI_Y = _TI_X;    _BI_Y = _BI_X;
_KD_Y = _KD_X;  _TD_Y = _TD_X;    _BD_Y = _BD_X;
_KP_Z = 0.3;    _TP_Z = 0.4;      _BP_Z = 0.3;
_KI_Z = 0.1;    _TI_Z = 0.15;     _BI_Z = 0.1;
_KD_Z = 0.015;  _TD_Z = 0.028;    _BD_Z = 0.015;
_MAX_ASCEND = 1.
_MAX_DESCEND = -0.5

# ================================= CONSTANTS ==========================================       
_PI = pi
# =============================== SUBSCRIBERS =========================================  
def subscribe_motion(msg):
    global motion
    motion = msg
    
def subscribe_guider(msg):
    global msg_guider
    msg_guider = msg
       
# ================================ BEGIN ===========================================
def move():
    # ---------------------------------- INITS ----------------------------------------------
    # --- init node ---
    rospy.init_node('hector_move')
    
    # --- cache global vars / constants ---

    err_x = 0
    err_y = 0
    err_z = 0
    sumErr_x = 0
    sumErr_y = 0
    sumErr_z = 0

    global msg_guider, motion
    msg_guider = None
    motion = None
    KP_X = _KP_X; KI_X = _KI_X; KD_X = _KD_X
    KP_Y = _KP_Y; KI_Y = _KI_Y; KD_Y = _KD_Y
    KP_Z = _KP_Z; KI_Z = _KI_Z; KD_Z = _KD_Z
    TP_X = _TP_X; TI_X = _TI_X; TD_X = _TD_X
    TP_Y = _TP_Y; TI_Y = _TI_Y; TD_Y = _TD_Y
    TP_Z = _TP_Z; TI_Z = _TI_Z; TD_Z = _TD_Z
    BP_X = _BP_X; BI_X = _BI_X; BD_X = _BD_X
    BP_Y = _BP_Y; BI_Y = _BI_Y; BD_Y = _BD_Y
    BP_Z = _BP_Z; BI_Z = _BI_Z; BD_Z = _BD_Z
    MAX_HORZ_V = _MAX_HORZ_V
    MAX_ASCEND = _MAX_ASCEND
    MAX_DESCEND = _MAX_DESCEND
    PI = _PI
    TWOPI = 2*PI
    
    # --- Service: Enable Motors ---
    enable_motors = rospy.ServiceProxy('/hector/enable_motors', EnableMotors)
    # Shutdown handler
    def shutdown_handler():
        # Messages cannot be published in topics
        # Disable motors   
        enable_motors(False)
        print('* MOVE * Motors Disabled')
    rospy.on_shutdown(shutdown_handler)
    
    # --- Subscribers ---
    rospy.Subscriber('/hector/guider', MsgGuiderH, subscribe_guider, queue_size=1)
    rospy.Subscriber('/hector/motion', MotionH, subscribe_motion, queue_size=1)
    while (motion is None or msg_guider is None or rospy.get_time() == 0) and not rospy.is_shutdown():
        pass
    if rospy.is_shutdown():
        return
    print('* MOVE * Done waiting for topics')
    
    # --- Publishers ---
    pub_cmd = rospy.Publisher('/hector/cmd_vel', Twist, latch=True, queue_size=1)
    cmd_vel = Twist()
    cmd_lin = cmd_vel.linear
    cmd_ang = cmd_vel.angular
    
    # --- Enable motors ---
    enable_motors(True)
    print('* MOVE * Enabled motors')
    
    # --- state enums ---
    TAKEOFF = 0
    TURTLE = 1
    GOAL = 2
    BASE = 3
    LAND = 4
        
    # ---------------------------------- LOOP ----------------------------------------------
    t = rospy.get_time();
    prev_state = -1
    while not rospy.is_shutdown() and not msg_guider.stop:
        if rospy.get_time() > t:
            # --- Get pose ---
            rx = motion.x
            ry = motion.y
            rz = motion.z
            
            # --- Get target ---
            tx = msg_guider.x
            ty = msg_guider.y
            tz = msg_guider.z
            
            # --- state dependent constants ---
            if prev_state != msg_guider.state:
                prev_state = msg_guider.state
                if prev_state == TAKEOFF or prev_state == LAND:
                    kp_x = TP_X; ki_x = TI_X; kd_x = TD_X
                    kp_y = TP_Y; ki_y = TI_Y; kd_y = TD_Y
                    kp_z = TP_Z; ki_z = TI_Z; kd_z = TD_Z
                elif prev_state == BASE or prev_state == GOAL:
                    kp_x = BP_X; ki_x = BI_X; kd_x = BD_X
                    kp_y = BP_Y; ki_y = BI_Y; kd_y = BD_Y
                    kp_z = BP_Z; ki_z = BI_Z; kd_z = BD_Z
                else: # State = Turtle
                    kp_x = KP_X; ki_x = KI_X; kd_x = KD_X
                    kp_y = KP_Y; ki_y = KI_Y; kd_y = KD_Y
                    kp_z = KP_Z; ki_z = KI_Z; kd_z = KD_Z
                    
            # !-- Express errors in world frame ---
            prevErr_x = err_x
            prevErr_y = err_y
            prevErr_z = err_z
            err_x = tx - rx
            err_y = ty - ry
            err_z = tz - tz
            sumErr_x += err_x
            sumErr_y += err_y
            sumErr_z += err_z

            # !--- PID using errors from world frame---//from updated position, calculate velocity
            x = kp_x*err_x + ki_x*sumErr_x + kd_x*(err_x-prevErr_x)
            y = kp_x*err_y + ki_y*sumErr_y + kd_y*(err_y-prevErr_y)
            z = kp_x*err_z + ki_z*sumErr_z + kd_z*(err_z-prevErr_z)
            if state == TAKEOFF or state == LAND:
                o = 0
            else:
                o = PI / 5 

            # !--- Limits ---
            # !-- horizontal max speed constraint
            if sqrt(x*x+y*y) > 2: #//horizontal speed
                x = 2*cos(ro)
                y = 2*sin(ro)
            # !-- ascend / descend constraint
            if z > 1:
                z = 1
            if z < -0.5:
                z = -0.5
            
            # -- Teleop (convert to robot frame before teleop) ---
            cmd_lin.x = x*cos(ro) + y*sin(ro)
            cmd_lin.y = - x*sin(ro) + y*cos(ro)
            cmd_lin.z = z
            cmd_ang.z = o
            pub_cmd.publish(cmd_vel)
            
            # --- Timing ---
            et = rospy.get_time() - t
            t += 0.05
            if et > 0.05:
                print('* MOVE * {} OVERSHOOT'.format(int(et*1000)))
    
    print('=== * MOVE * Terminated ===')
    
if __name__ == '__main__':
    try:
        move()
    except rospy.ROSInterruptException:
        pass
