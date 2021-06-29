#!/usr/bin/env python

import roslib, rospy, rospkg
from numpy import *
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_multiply
from tf2_msgs.msg import TFMessage
from pkg_turtle.msg import MsgGuider, Motion
import cv2
import numpy
import sys

# cache PI and 2PI
TWOPI = 2*pi
PI = pi
TIME_INC = 0.1
vt = 0
wt = 0

# declare PID constants
Kp_t = 0.6 #0.6
Kp_r = 0.5 #0.6
Ki_t = 0 #0.001
Ki_r = 0 #0.001
Kd_t = 0.5 #0.1 #0.5
Kd_r = 0.03 #0.5

# OdometryMM initialisations
WR = 0.066/2.0 # m, robot's wheel RADIUS, not DIAMETER
wl = 0 # both wheel displacement angles in radians
wr = 0
L = 0.16
t_odometry = 0

def subscribe_wheels(msg):
    global rbt_wheels
    rbt_wheels = (msg.position[1], msg.position[0])

def subscribe_guider(msg):
    global msg_guider
    msg_guider = msg

def subscribe_imu(msg):
    global rbt_imu_o, rbt_imu_w, rbt_imu_a
    t = msg.orientation
    rbt_imu_o = euler_from_quaternion([\
        t.x,\
        t.y,\
        t.z,\
        t.w\
        ])[2]
    rbt_imu_w = msg.angular_velocity.z
    rbt_imu_a = msg.linear_acceleration.x


def motion_control(START_POSE)
    rospy.init_node('move')
    global xt,yt,ot,xp,yp,rbt_true, rbt_wheels, rbt_imu_a,msg_guider, vt, wt, x,y,o,wl,wr,t_odometry

    x = START_POSE[0]
    y = START_POSE[1]
    o = START_POSE[2] # angle in radians
    #print(x, y, o)

    msg_guider = None
    rbt_wheels = [nan] * 2
    rbt_imu_a = None

    xt = 0; yt = 0; ot = 0; xp = 0; yp = 0
    u_vel = 0
    u_ang = 0

    rospy.Subscriber('/turtle/guider', MsgGuider, subscribe_guider, queue_size=1)
    rospy.Subscriber('/turtle/joint_states', JointState, subscribe_wheels, queue_size=1)
    rospy.Subscriber('/turtle/imu', Imu, subscribe_imu, queue_size=1)
    pub_u = rospy.Publisher('/turtle/cmd_vel', Twist, latch=True, queue_size=1)
    publisher_move = rospy.Publisher('/turtle/motion', Motion, latch=True, queue_size=1)
    u = Twist()


    # Odometry Message
    msg_motion = Motion()
    msg_motion.x = x
    msg_motion.y = y
    msg_motion.o = o
    msg_motion.v = 0
    msg_motion.w = 0

    # Initialize limits
    max_err_t = 3
    prev_u_t = 0
    prev_u_r = 0
    max_u_t = 0.22
    max_u_r = 2.84 
    max_du_t = 0.1
    max_du_r = 0.2

    # Initialize the errors for linear
    err_vel = sqrt((yp-yt)*(yp-yt) + (xp-xt)*(xp-xt)) 
    sum_err_vel = err_vel # For I
    prev_err_vel = err_vel # later err_vel is recomputed in the loop, and at the end of the loop, assigned to prev

    # for angular
    err_ang = arctan2(yp-yt,xp-xt) - ot
    if err_ang >= PI:
        err_ang -= TWOPI
    elif err_ang < -PI:
        err_ang += TWOPI

    sum_err_ang = err_ang
    prev_err_ang = err_ang

    # Wait until every topic subscribed has a message
    while (isnan(rbt_wheels[0]) or rbt_imu_a is None or rospy.get_time() == 0 or msg_guider is None) and not rospy.is_shutdown():
        pass
    # publish dummy message to also start turtle_mainfunc
    publisher_move.publish(msg_motion)

    wl = rbt_wheels[0]; wr = rbt_wheels[1]
    t_odometry = rospy.get_time()
    t = rospy.get_time()
    while not rospy.is_shutdown() :
        if rospy.get_time() > t:
            if msg_guider.stop is True:
                break
            calculatedResults = calculate_OdometryMM(rbt_wheels)
            xt = calculatedResults[0]
            yt = calculatedResults[1]
            ot = calculatedResults[2]
            xp = msg_guider.target.a
            yp = msg_guider.target.b


            msg_motion.x = xt; msg_motion.y = yt; msg_motion.o = ot; msg_motion.v = vt; msg_motion.w = wt
            publisher_move.publish(msg_motion)
            # linear command
            err_vel = sqrt((yp-yt)*(yp-yt) + (xp-xt)*(xp-xt))

            # angular error
            err_ang = arctan2(yp-yt,xp-xt) - ot
            
            P_t = Kp_t * err_vel
            I_t = Ki_t * sum_err_vel
            sum_err_vel += err_vel
            D_t = Kd_t * (err_vel - prev_err_vel)
            prev_err_vel = err_vel

            # angular command
            if err_ang >= PI:
                err_ang -= TWOPI
            elif err_ang < -PI:
                err_ang += TWOPI
            P_r = Kp_r * err_ang
            I_r = Ki_r * sum_err_ang
            sum_err_ang += err_ang
            D_r = Kd_r *(err_ang - prev_err_ang)
            prev_err_ang = err_ang

            # clamp u_ang error
            prev_u_r = u_ang
            u_ang = P_r + I_r + D_r

            # clamping u_trans error
            prev_u_t = u_vel
            u_vel = P_t + I_t + D_t
            
            u_vel = u_vel + 0.05 * sign(u_vel)
            c = cos(err_ang)
            u_vel = u_vel * c * c* c

            # clamp output
            if abs(u_ang) > 0.45:
                u_ang = 0.45 * sign(u_ang)
            if abs(u_vel) > 0.22:
                u_vel = 0.22 * sign(u_vel)

            # clamp output change
            du_t = u_vel - prev_u_t; du_r = u_ang - prev_u_r
            if abs(du_t) > max_du_t:
                u_vel = prev_u_t + max_du_t*sign(du_t)
            if abs(du_r) > max_du_r:
                u_ang = prev_u_r + max_du_r*sign(du_r)

            u.linear.x = u_vel
            u.angular.z = u_ang
            pub_u.publish(u)
            t += TIME_INC
    t += 0.3
    u.linear.x = 0
    u.angular.z = 0
    pub_u.publish(u)

    while not rospy.is_shutdown() and rospy.get_time() < t:
        pass


# Odometry Motion Model codes
def calculate_OdometryMM(wheels):
    # calculates the robot's new pose based on wheel encoder angles
    # INPUT: wheels: (left_wheel_angle, right_wheel_angle)
    # OUTPUT: a new pose (x, y, theta)
    global x
    global y 
    global o  # angle in radians
    global wl  # both wheel displacement angles in radians
    global wr 
    global vt # keep track of the current fused velocity
    global wt
    global t_odometry
    global rbt_imu_a,rbt_imu_w,rbt_imu_o
    
    
    # previous wheel angles stored in self.wl and self.wr, respectively. Remember to overwrite them
    # previous pose stored in self.x, self.y, self.o, respectively. Remember to overwrite them
    # previous time stored in self.t. Remember to overwrite it
    # axle track stored in self.L. Should not be overwritten.
    # wheel radius, NOT DIAMETER, stored in self.WR. Should not be overwritten.
    dt = rospy.get_time() - t_odometry # current time minus previous time
    # IMU Portion, always compute IMU velocity based on fused one, which is the more correct one

    v_imu = vt + rbt_imu_a * dt

    # Odometry Portion
    dwl = wheels[0] - wl #thetaL
    dwr = wheels[1] - wr #thetaR
    vt_o = WR * (dwl + dwr) / (2 * dt) #tangential velociy
    wt_o = WR * (dwr - dwl) / (L * dt)
    
    # Infuse IMU And Odometry
    #print("v_imu and v_odom ",v_imu, vt_o)
    vt = v_imu * 0.3 + vt_o * 0.7

    #print("w_imu and w_odom",rbt_imu_w,wt_o)
    wt = rbt_imu_w * 0.3 + wt_o * 0.7

    dpsi = wt*dt # change in angle about COR from odometry

    # Update x,y, and bearing based on Odometry Model
    o = 0.2 * (o+dpsi) + 0.8 * rbt_imu_o

    if abs(wt) < 3e-2:
    #   MM for move straight
        x = x + vt*dt*cos(o)
        y = y + vt*dt*sin(o)
        #o = o
    else:
        # MM for curve turns
        rt = vt/wt
        x = x + rt*sin(o + dpsi) - rt*sin(o)
        y = y - rt*cos(o + dpsi) + rt*cos(o)
    if o >= PI:
        o -= TWOPI
    elif o < -PI:
        o += TWOPI

    # Fuse o afterwards
    wl = wheels[0]
    wr = wheels[1]
    t_odometry += dt # update the current time. There's a reason why resampling the time is discouraged
    return (x, y, o)

if __name__ == '__main__':
    try:
    # parse start_pose
        start_pose = sys.argv[1]
        #print("start_pose", start_pose)
        start_pose = start_pose.split(',')
        start_pose = (float(start_pose[0]), float(start_pose[1]), 0.)
        motion_control(start_pose)
    except rospy.ROSInterruptException:
        pass
