from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, 
	wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        
	# TODO: Implement
 	self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
	
	kp = 60
	ki = 0.1
	kd = 0.
	mn = 0. # Minimum throlttle value
	mx = 0.2 # Maximum throttle value
	self.throttle_controller = PID(0.3, 0.1, 0., mn, mx)
	self.steer_controller = PID(5.0, 0.05, 1.0, -max_steer_angle, max_steer_angle)#####

	tau = 0.5 # 1/(2pi * tau) = cutoff frequency
	ts = 0.02 # Sample time
	self.vel_lpf = LowPassFilter(tau, ts)
	
	self.vehicle_mass = vehicle_mass
	self.fuel_capacity = fuel_capacity
	self.brake_deadband = brake_deadband
	self.decel_limit = decel_limit
	self.accel_limit = accel_limit
	self.wheel_radius = wheel_radius

	self.last_time = rospy.get_time()
	pass

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

	if not dbw_enabled:
	    self.throttle_controller.reset()
	    self.steer_controller.reset()
	    return 0., 0., 0.

	current_vel = self.vel_lpf.filt(current_vel)

	#rospy.logwarn("Angular vel: {0}".format(angular_vel))
	#rospy.logwarn("Target velocity: {0}".format(linear_vel))
	#rospy.logwarn("Target angular vel: {0}".format(angular_vel))
	#rospy.logwarn("Current velocity: {0}".format(current_vel))
	#rospy.logwarn("Filtered velocity: {0}".format(self.vel_lpf.get()))

	


	vel_error = linear_vel - current_vel
	self.last_vel = current_vel

	current_time = rospy.get_time()
	sample_time = current_time - self.last_time
	self.last_time = current_time
	

	steering = self.steer_controller.step(angular_vel, sample_time)#####
	steering += self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
	#steering = self.vel_lpf.filt(steering)

	throttle = self.throttle_controller.step(vel_error, sample_time)
	brake = 0

	if linear_vel == 0. and current_vel < 0.1:
	    throttle = 0.
	    brake = 700 #Nm to hold the car in place if we are stopped at a light. Acceleration ~ 1m/s^2

	elif throttle < 0.1 and vel_error < 0:
	    throttle = 0.
	    decel = max(vel_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius # Torque Nm

        return throttle, brake, steering