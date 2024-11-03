#!/usr/bin/env python

from __future__ import print_function

from grasp_service_node.srv import pose_dof, pose_dofResponse
from geometry_msgs.msg import Pose
import rospy


def callback(request):
	result = pose_dofResponse()
	#right now copying the values from client and printing it back to server to see if communication is working properly
	result.gripper_pose.position.x = request.obj_pose.position.x
	result.gripper_pose.position.y = request.obj_pose.position.y
	result.gripper_pose.position.z = request.obj_pose.position.z
	
	result.gripper_pose.orientation.x = request.obj_pose.orientation.x
	result.gripper_pose.orientation.y = request.obj_pose.orientation.y
	result.gripper_pose.orientation.z = request.obj_pose.orientation.z
	result.gripper_pose.orientation.w = request.obj_pose.orientation.w
	result.dofs = [0.3959,0,-0.05236,0.395913,0,-0.05236,0.3959,0,-0.05236,-0.1561,0.1561]
	
	return result
	
def grasp_node():
	rospy.init_node('grasp_server')
	rospy.Service('grasp_server', pose_dof, callback)
	print('Grasp_server_running')
	rospy.spin()
	
if __name__ == '__main__':
	#name of the function changed because message and function had same name
	grasp_node()
	