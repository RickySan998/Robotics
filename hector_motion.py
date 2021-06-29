#!/usr/bin/env python

import roslib, rospy, rospkg
from tf.transformations import euler_from_quaternion
from std_srvs.srv import Empty # calibrate imu
from math import sqrt, cos, sin, pi, atan2
from numpy import array, transpose, matmul, subtract, add
from numpy.linalg import inv
import sys
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import Vector3Stamped
from hector_uav_msgs.msg import Altimeter

from pkg_hector.msg import MotionH, MsgGuiderH
# ================================= PARAMETERS ========================================== 
# !-- Find the noises from topic, or tune them yourself --
_IMU_NX = 0.1225 # noise of IMU X component
_IMU_NY = 0.1225 # noise of IMU Y component
_IMU_NZ = 0.09 # noise of IMU Z component
_IMU_NO = 0.0025 # noise of IMU O component
_GPS_NX = 0.04 # noise of GPS X component
_GPS_NY = 0.04 # noise of GPS Y component
_GPS_NZ = 0.04 # noise of GPS Z component 
_BAR_NZ = 0.15 # noise of Barometer Z component
_CMP_NO = 0.0001 # noise of Compass O component
# ================================= CONSTANTS ==========================================        
_RAD_EQUATOR = 6378137. # m
_RAD_POLAR = 6356752.3 # m
_G = 9.80665 # m/s/s
_DEG2RAD = pi/180.
_PI = pi
N = 10 # number of messages to use for initialization
# =============================== SUBSCRIBERS =========================================  
_rbt_true = [False, False, False, False]
# x, y, z, o
def subscribe_true(msg):
    # !-- subscribe to ground truth ---
    rbt_true = _rbt_true
    # !-- subscribe to ground truth ---
    
_rbt_gps = [False, False, False, False]
gpsmsgcount = 0
gpsinit = [0,0,0] # lat,longi,alt
gpsinitdone = False
# lat, long, alt, seq
# "seq" is the message count. May be required for asynchronous EKF
# measurement for EKF
def subscribe_gps(msg):
    # !-- subscribe to GPS ---
    global rbt_gps, gpsmsgcount
    gpsmsgcount += 1
    rbt_gps = _rbt_gps
    rbt_gps[0] = msg.latitude
    rbt_gps[1] = msg.longitude
    rbt_gps[2] = msg.altitude
    rbt_gps[3] = msg.header.seq # or no need to init msg count, just use msg.header.seq
    if gpsmsgcount <= N:
        gpsinit[0] += rbt_gps[0]
        gpsinit[1] += rbt_gps[1]
        gpsinit[2] += rbt_gps[2]

    # !-- subscribe to GPS ---

_rbt_imu = [False, False, False, False]
imumsgcount = 0
# ax, ay, az, w
# input into EKF
def subscribe_imu(msg):
    # !-- subscribe to IMU ---
    global rbt_imu, imumsgcount
    imumsgcount += 1
    #rbt_imu = _rbt_imu
    rbt_imu[0] = msg.linear_acceleration.x
    rbt_imu[1] = msg.linear_acceleration.y
    rbt_imu[2] = msg.linear_acceleration.z
    rbt_imu[3] = msg.angular_velocity.z
    # !-- subscribe to IMU ---
    
_rbt_compass = [False, False, False]
compassmsgcount = 0
# Fx, Fy, seq. --or-- o, seq
# Fx and Fy are vector values. seq is the message count.
# measurement for EKF
def subscribe_compass(msg):
    # !-- subscribe to compass ---
    global rbt_compass, compassmsgcount
    compassmsgcount += 1
    #rbt_compass = _rbt_compass
    rbt_compass[0] = msg.vector.x
    rbt_compass[1] = msg.vector.y
    rbt_compass[2] = msg.header.seq # or no need to init msg count, just use msg.header.seq
    # !-- subscribe to compass ---

_rbt_bar = [False, False]
baromsgcount = 0
baro_init = 0
baro_init_done = False
# z, seq.
# measurement for EKF
def subscribe_barometer(msg):
    # !-- subscribe to altimeter ---
    global baromsgcount, rbt_bar
    baromsgcount += 1
    #rbt_bar = _rbt_bar
    rbt_bar[0] = msg.altitude
    rbt_bar[1] = msg.header.seq
    if baromsgcount <= N:
        baro_init += rbt_bar[0]
    # !-- subscribe to altimeter ---

def subscribe_guider(msg):
    global msg_guider
    msg_guider = msg
    
# ================================ BEGIN ===========================================
def motion(rx0=2.0, ry0=2.0, rz0 =0.172, ro0=0.0):
    # ---------------------------------- INITS ----------------------------------------------
    # --- init node ---
    rospy.init_node('hector_motion')
    global rbt_gps, rbt_imu, rbt_compass,rbt_bar, baro_offset,mag_offset,imu_wo_offset,x0,y0,z0,o0
    global baro_init,baro_init_done,gpsinit,gpsinitdone,N
    # --- cache global vars / constants ---
    rbt_ecef_init = [0,0,0] # initialize rbf ECEF
    baro_offset = 0 # offset to the starting height
    mag_offset = 0 # heading offset to the starting heading which is o0 (assumed to be known)
    imu_wo_offset = 0 # offset to the angular velocity around z-axis which is assumed to be 0
    x0 = rx0; y0 = ry0; z0 = rz0; o0 = ro0; # to be consistent with project document
    rbt_true = _rbt_true
    rbt_gps = _rbt_gps
    rbt_imu = _rbt_imu
    rbt_compass = _rbt_compass
    rbt_bar = _rbt_bar
    RAD_EQUATOR = _RAD_EQUATOR
    RAD_POLAR = _RAD_POLAR
    DEG2RAD = _DEG2RAD
    IMU_NX = _IMU_NX
    IMU_NY = _IMU_NY
    IMU_NZ = _IMU_NZ
    IMU_NO = _IMU_NO
    GPS_NX = _GPS_NX
    GPS_NY = _GPS_NY
    GPS_NZ = _GPS_NZ
    BAR_NZ = _BAR_NZ
    CMP_NO = _CMP_NO
    G = _G
    PI = _PI
    TWOPI = 2.*PI
    global msg_guider
    msg_guider = None
    
    # --- Service: Calibrate ---
    # !-- Calibrate your IMU
    imu_service = '' # !--
    calibrate_imu = rospy.ServiceProxy(imu_service, Empty)
    calibrate_imu()
    print('*MOTION* Imu calibrated')
    # !-- Calibrate your IMU

    # --- Subscribers --- # to fill in
    # !-- ground truth --
    # !-- GPS --
    rospy.Subscriber('/hector/fix',NavSatFix,subscribe_gps,queue_size=1)
    # !-- IMU --
    rospy.Subscriber('/hector/raw_imu',Imu,subscribe_imu,queue_size=1)
    # !-- Compass --
    rospy.Subscriber('/hector/magnetic',Vector3Stamped,subscribe_compass,queue_size=1)
    # !-- Barometer --
    rospy.Subscriber('/hector/altimeter',Altimeter,subscribe_barometer,queue_size=1)

    rospy.Subscriber('/hector/guider', MsgGuiderH, subscribe_guider, queue_size=1)

    # --- Compute Initializations for Barometer and GPS --- #
    # These first N readings are obtained when the move and guider node is still waiting, because we haven't published the first message from motion
    # which would start the other nodes. Therefore, up to now, the drone would still be in its starting position. Hence, the first N readings collected
    # are all from the drone's starting position. Averaging is done to alleviate the effect of variance on initialization.
    while gpsmsgcount <=N or baromsgcount <= N:
        pass
    gpsinit[0] = gpsinit[0]/N
    gpsinit[1] = gpsinit[1]/N
    gpsinit[2] = gpsinit[2]/N
    baro_init = baro_init/N
    
    # --- Publishers ---
    pub_motion = rospy.Publisher('/hector/motion', MotionH, latch=True, queue_size=1)
    msg_motion = MotionH()
    pub_motion.publish(msg_motion)
    
    
    # !-- Make sure the while loop does not jam your program while waiting for topics
    while (not rbt_imu[-1] or not rbt_gps[-1] or not rbt_compass[-1] or\
        rospy.get_time() == 0 or not rbt_bar[-1] or msg_guider is None) and not rospy.is_shutdown():
        pass
    if rospy.is_shutdown():
        return
    print('*MOTION* Done waiting for topics')
    
       
    # --- build sensor functions ---
    # !-- design nested functions or lambdas that convert messages to useful world coordinates
    seq_gps = rbt_gps[-1]
    seq_compass = rbt_compass[-1]
    seq_baro = rbt_bar[-1]

    # GPS Conversion from Geodetic to Gazebo World Coordinate
    def GPStoGazWorld():
        # get the GPS Geodetic Measurement from the topic /hector/fix
        # which uses message type sensor_msgs/NavSatFix as global variable rbt_gps
        global rbt_gps, rbt_ecef_init, gpsmsgcount, gpsinit, gpsinitdone # rbt_gps = [lat,long,alt,count]
        # declare variable to store the results
        gpspos = [0,0,0]
        # cache the geodetic values
        lat = rbt_gps[0] # psi
        longi = rbt_gps[1] # lambda
        alt = rbt_gps[2] # h
        a_square = RAD_EQUATOR*RAD_EQUATOR; b_square = RAD_POLAR*RAD_POLAR
        N_psi = a_square/sqrt(a_square*pow(cos(lat),2) + b_square*pow(sin(lat),2))

        # Geodetic to ECEF conversion
        gpspos[0] = (N_psi + alt) * cos(lat) * cos(longi)
        gpspos[1] = (N_psi + alt) * cos(lat) * sin(longi)
        gpspos[2] = ((b_square* N_psi/a_square) + alt) * sin(lat)

        if not gpsinitdone:
        # Compute ECEF starting position ONCE
            lat_s = gpsinit[0] # psi
            longi_s = gpsinit[1] # lambda
            alt_s = gpsinit[2] # h
            N_psi_s = a_square/sqrt(a_square*pow(cos(lat_s),2) + b_square*pow(sin(lat_s),2))
            rbt_ecef_init[0] = (N_psi_s + alt_s) * cos(lat_s) * cos(longi_s)
            rbt_ecef_init[1] = (N_psi_s + alt_s) * cos(lat_s) * sin(longi_s)
            rbt_ecef_init[2] = ((b_square* N_psi_s/a_square) + alt_s) * sin(lat_s)
            gpsinitdone = True

        # ECEF to Local NED conversion
        Ren = array([[-sin(lat)*cos(longi), -sin(longi), -cos(lat) * cos(longi)],
                        [-sin(lat)*sin(longi), cos(longi), -cos(lat) * sin(longi)],
                        [cos(lat), 0, -sin(lat)]]) # define the rotation matrix
        gpspos = array([gpspos]); rbt_ecef_init = array(rbt_ecef_init)

        gpspos = matmul((gpspos-rbt_ecef_init),Ren) # transposed equation from the lecture notes, so that results is a 1x3 vector

        # Local NED to Map Coordinates
        map_init = array([[x0,y0,z0]])
        Rmn = array([[1,0,0], [0,-1,0], [0,0,-1]]) # define rotation matrix
        gpspos = transpose(transpose(map_init) + matmul(Rmn,transpose(gpspos)))

        return gpspos # 1x3 vector

    def MagtoHeading():
        # get the compass vector direction wrt to robot body frame. This gives where the needle is pointing in terms of body coordinate frame
        # subscribe to the topic /hector/magnetic, which uses a message type geometry_msgs/Vector3Stamped, and only take the vector attribute of
        # the message. Store them in a global variable called rbt_compass
        global rbt_compass, mag_offset, compassmsgcount
        magV = sqrt(rbt_compass[0]*rbt_compass[0] + rbt_compass[1] * rbt_compass[1])
        Vn = [rbt_compass[0]/magV, rbt_compass[1]/magV] # get a unit vector direction
        heading = atan2(-Vn[1],Vn[0]) # must put negative on y, as seen from the hint in project document. Use atan2 to get the correct quadrant angle
        # if compassmsgcount == 1: # if it is the first measurement, calculate offset from the known starting heading o0 which is known
        #     mag_offset = heading - o0

        return heading

    def BartoAlt():
        # get the barometer altitude measurement by subscribing to the topic /hector/altimeter, which uses the message type hector_uav_msgs/Altimeter.
        # Take only the altitude attribute of the message and store it in a global variable called rbt_bar

        global rbt_bar, baro_offset, baromsgcount, baro_init, baro_init_done,z0

        # Compute barometer offset
        if not baro_init_done:
            baro_offset = baro_init - z0
            baro_init_done = True

        return rbt_bar[0] - baro_offset

    def IMUtoWorld(heading): # takes in the heading in radians, to perform rotations on body x and y accelerations
        # get the IMU body linear acceleration in x,y, and z direction, as well as the angular velocity about z-axis by subscribing to the topic 
        # /hector/raw_imu, which uses the message type sensor_msgs/Imu. Take only the x,y,z of the linear_acceleration attribute, as well as the 
        # z of the angular_acceleration attribute, store it in a global variable called rbt_imu
        global rbt_imu, imu_wo_offset, imumsgcount
        ax = cos(heading) * rbt_imu[0] - sin(heading) * rbt_imu[1]
        ay = sin(heading) * rbt_imu[0] + cos(heading) * rbt_imu[1]
        az = rbt_imu[2]
        wz = rbt_imu[3]

        # if imumsgcount == 1: # if first reading, measure offset of angular velocity with respect to start, assumed to be 0. If small enough, can delete
        #     imu_wo_offset = 0 - wz

        return ax,ay,az,wz

    # --- EKF inits ---
    # !-- for the numpy functions to work, need to initialise with LIST OF LIST

    # for x position and vel
    X = array([[rx0], [0.]]) # column 2x1 vector
    Px = array([[0., 0.], [0., 0.]]) # 2x2 matrix, 0 variance because we are certain at the start
    Qx = array([[IMU_NX]]) # 1x1 matrix
    VRVx = array([[GPS_NX]]) # cache redundant calculations
    

    # for y position and vel 
    Y = array([[ry0], [0.]]) # column 2x1 vector
    Py = array([[0., 0.], [0., 0.]]) # 2x2 matrix, 0 variance because we are certain at the start
    Qy = array([[IMU_NY]]) # 1x1 matrix
    VRVy = array([[GPS_NY]]) # cache redundant calculations

    # for z position and vel
    Z = array([[rz0], [0.]]) # column 2x1 vector
    Pz = array([[0., 0.], [0., 0.]]) # 2x2 matrix, 0 variance because we are certain at the start
    Qz = array([[IMU_NZ]]) # 1x1 matrix
    VRVzgps = array([[GPS_NZ]]) # cache redundant calculations
    VRVzbaro = array([[BAR_NZ]]) # for z-position, measurement using both barometer and gps

    # for heading position and vel
    PSI = array([[ro0], [0.]]) # column 2x1 vector
    Ppsi = array([[0., 0.], [0., 0.]]) # 2x2 matrix, 0 variance because we are certain at the start
    Qpsi = array([[IMU_NO]]) # 1x1 matrix
    VRVp = array([[CMP_NO]]) # cache redundant calculations

    
    # state
    state_str = ['TAKEOFF', 'TURTLE', 'GOAL', 'BASE', 'LAND']
    
    # ---------------------------------- LOOP ----------------------------------------------
    iter_t = rospy.get_time();
    t = iter_t
    while not rospy.is_shutdown() and not msg_guider.stop:
        if rospy.get_time() > iter_t:
            # --- fetch globals ---
            # [1] TRUE
            # !-- get the true pose information --
            # !-- get the true pose information --
            
            # [2] GPS
            # fetch seq information
            measure_gps = seq_gps != rbt_gps[-1]
            if measure_gps:
                # !-- get the gps information --
                gps_pos = GPStoGazWorld() # 1x3 vector of gps positions (x,y,z)
                gps_pos_x = gps_pos[0][0]
                gps_pos_y = gps_pos[0][1]
                gps_pos_z = gps_pos[0][2]
                seq_gps = rbt_gps[-1]
                # !-- get the gps information --
            
            # [3] COMPASS
            measure_compass = seq_compass != rbt_compass[-1]
            if measure_compass:
                # !-- get the compass information --
                comp_heading = MagtoHeading()
                seq_compass = rbt_compass[-1]
                # !-- get the compass information --
            
            # [4] BAR
            measure_bar = seq_bar != rbt_bar[-1]
            if measure_bar:
                # !-- get the bar information --
                bar_z = BartoAlt()
                seq_bar = rbt_bar[-1]
                # !-- get the bar information --
            
            # [5] IMU (our input)
            # !-- get the imu information --
            [ax,ay,az,wz] = IMUtoWorld(PSI[0][0]) # PSI[0][0] gives the current estimated heading, needed for IMU acceleration rotation
            az = az - G # must compensate by G to get body's true acceleration
            # !-- get the imu information --
            
            # [6] TIME
            dt = rospy.get_time() - t
            
            # ==== EKF PREDICTION =====
            # !-- EKF Prediction --
            # remember to convert the IMU information to world frame,
            # it is literally a rotation about z

            # Define F and W for x,y,z
            Fp = array([[1,dt],[0,1]]); Wp = array([[0.5*dt*dt],[dt]])

            # F and W for heading
            Fpsi = array([[1,0.5*dt],[0,0]]); Wpsi = array([[0.5*dt],[1]])

            # for x state
            # prediction mean
            X = matmul(Fp,X) + Wp*ax
            # prediction covariance
            Px = matmul(matmul(Fp,Px),transpose(Fp)) + matmul(matmul(Wp,Qx),transpose(Wp))

            # for y state
            # prediction mean
            Y = matmul(Fp,Y) + Wp*ay
            # prediction covariance
            Py = matmul(matmul(Fp,Py),transpose(Fp)) + matmul(matmul(Wp,Qy),transpose(Wp))

            # for z state
            # prediction mean
            Z = matmul(Fp,Z) + Wp*az
            # prediction covariance
            Pz = matmul(matmul(Fp,Pz),transpose(Fp)) + matmul(matmul(Wp,Qz),transpose(Wp))

            # for heading state
            # prediction mean
            PSI = matmul(Fpsi,PSI) + Wpsi*wz
            # prediction covariance
            Ppsi = matmul(matmul(Fpsi,Ppsi),transpose(Fpsi)) + matmul(matmul(Wpsi,Qpsi),transpose(Wpsi))
            # !-- EKF Prediction --
            

            # ==== EKF CORRECTION ====

            # Define H, since all measuremets give the position part of the state, H is common for all sensors
            H = array([[1., 0.]]) # 1x2 vector

            if measure_compass:
                # !-- CORRECT o --
                # For heading ONLY
                IC = inv(matmul(matmul(H,Ppsi),transpose(H)) + VRVp)
                K = matmul(matmul(Ppsi,transpose(H)),IC)

                # correction on mean
                diff = array([comp_heading]) - matmul(H,PSI)
                PSI = PSI + matmul(K,diff)

                # correction on variance
                Ppsi = Ppsi - matmul(matmul(K,H),Ppsi)
                # !-- CORRECT o --

                
            if measure_gps:
                # !-- CORRECT x, y, z --
                # for x
                # compute Kalman Gain
                IC = inv(matmul(matmul(H,Px),transpose(H)) + VRVx)
                K = matmul(matmul(Px,transpose(H)),IC)

                # correction on mean
                diff = array([gps_pos_x]) - matmul(H,X)
                X = X + matmul(K,diff)

                # correction on variance
                Px = Px - matmul(matmul(K,H),Px)

                # for y
                # compute Kalman Gain
                IC = inv(matmul(matmul(H,Py),transpose(H)) + VRVy)
                K = matmul(matmul(Py,transpose(H)),IC)

                # correction on mean
                diff = array([gps_pos_y]) - matmul(H,Y)
                Y = Y + matmul(K,diff)

                # correction on variance
                Py = Py - matmul(matmul(K,H),Py)

                # for z
                # compute Kalman Gain
                IC = inv(matmul(matmul(H,Pz),transpose(H)) + VRVzgps)
                K = matmul(matmul(Pz,transpose(H)),IC)

                # correction on mean
                diff = array([gps_pos_z]) - matmul(H,Z)
                Z = Z + matmul(K,diff)

                # correction on variance
                Pz = Pz - matmul(matmul(K,H),Pz)
                # !-- CORRECT x, y, z --

                
            if measure_bar: # optional
                # !-- CORRECT z --
                # for z ONLY
                IC = inv(matmul(matmul(H,Pz),transpose(H)) + VRVzbaro)
                K = matmul(matmul(Pz,transpose(H)),IC)

                # correction on mean
                diff = array([bar_z]) - matmul(H,Z)
                Z = Z + matmul(K,diff)

                # correction on variance
                Pz = Pz - matmul(matmul(K,H),Pz)

                # !-- CORRECT z --
                
            # === Publish motion ===

            # Ensure heading published is within -pi to pi
            head_angle = PSI[0][0]
            while head_angle > PI or head_angle < -PI:
            	if head_angle > PI:
            		head_angle -= TWOPI
            	else:
            		head_angle += TWOPI

            # !-- Publish pose --
            msg_motion.x = X[0][0]
            msg_motion.y = Y[0][0] # !-- y
            msg_motion.z = Z[0][0] # !-- z
            msg_motion.o = head_angle # !-- o
            pub_motion.publish(msg_motion)
            # !-- Publish pose --
                
            # ==== DEBUG ====
            print('*MOTION* --- ')
            print('  STATE : {}'.format(state_str[msg_guider.state]))
            print('  ROTSD ({:6f})'.format(0.)) # !-- sqrt the variance to get std dev.
            print('    ROT ({:6.3f}) TRU({:6.3f}) compass({:6.3f}) '.format(0., 0., 0.))
            print('  POSSD ({:7f}, {:7f}, {:6f})'.format(0., 0., 0.))
            print('    POS ({:7.3f}, {:7.3f}, {:6.3f})'.format(0., 0., 0.))
            print('    TRU ({:7.3f}, {:7.3f}, {:6.3f})'.format(0., 0., 0.))
            print('    GPS ({:7.3f}, {:7.3f}, {:6.3f})'.format(0., 0., 0.))
            print('   BARO (  ---  ,   ---  , {:6.3f})'.format(0.))
            
            
            t += dt
            
            # --- Timing ---
            et = rospy.get_time() - iter_t
            iter_t += 0.05
            if et > 0.05:
                print('*MOTION* {} OVERSHOOT'.format(int(et*1000)))
#    
            
    print('=== *MOTION* Terminated ===')
    
if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            motion(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), 0.0)
        else:
            motion()
    except rospy.ROSInterruptException:
        pass
