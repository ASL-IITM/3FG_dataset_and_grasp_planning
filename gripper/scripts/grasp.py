#!/usr/bin/python
import csv
import rospy
from graspit_commander import GraspitCommander
import numpy as np
from geometry_msgs.msg import Pose
import time
import tf
from World_scene import load_world
import os

epsilon_quality = []
file_names = []
#file_names = ['A0','A1','A2','A3','A4','A5','A6','B0','B1','B2','B3','B4','B5','B6',\
#			  'C0','C1','C2','C3','C4','C5','C6','D0','D1','D2','D3','D4','D5','D6',\
#			  'E0','E1','E2','E3','E4','E5','E6','F0','F1','F2','F3','F4','F5','F6',\
#			  'G0','G1','G2','G3','G4','G5','G6']

missing_file_names =['J23_2', 'S18_2', 'R16_1', 'R11_1', 'B21_1', 'U07_0', 'N16_1', 'P18_1', 'K07_1', 'O04_0', 'F23_3', 'E05_2', 'O06_1', 'J21_0', 'X12_1', 'O14_3', 'D11_0', 'B12_3', 'D02_3', 'P08_3', 'S10_1', 'P13_2', 'S10_0', 'E02_2', 'V05_1', 'A10_0', 'O06_2', 'D23_0', 'C08_2', 'Q08_3', 'J10_1', 'O15_2', 'L07_0', 'B12_1', 'E16_3', 'X08_1', 'R17_0', 'C06_1', 'N07_2', 'Q03_3', 'S03_2', 'N09_0', 'V24_1', 'L17_1', 'X09_0', 'B23_3', 'M18_1', 'P23_2', 'N13_0', 'M21_0', 'E20_1', 'J17_3', 'K06_2', 'P15_1', 'U02_1', 'I09_1', 'S11_3', 'E13_3', 'K24_2', 'P15_2', 'U02_3', 'H21_1', 'B12_2', 'A23_0', 'S24_2', 'N00_0', 'O18_3', 'L22_2', 'R07_2', 'R20_3', 'G01_2', 'C07_2', 'P24_3', 'S05_0', 'B07_3', 'N25_3', 'M14_3', 'B25_1', 'L20_2', 'O18_1', 'P13_1', 'P09_3', 'D24_1', 'S21_2', 'N25_2', 'J04_1', 'K02_0', 'Q02_0', 'G23_2', 'G18_3', 'H24_3', 'S05_2']
#directory = '/home/asl-7/Downloads/egad_train_set_0.15_stl'
directory = '/home/asl-7/Downloads/egad_eval_set_0.12/processed_meshes'
files = os.listdir(directory)
all_files = [file.replace('.stl', '') for file in files if file.endswith('.stl')]
for file in all_files:
	file_names.append(file)


def grasp():
	# Plan grasps
	for file_name in file_names:
		#function to load the scene for world
		print('Doing grasp planning for '+file_name+'.stl ....')
		load_world(file_name)
		planned_grasps = GraspitCommander.planGrasps(max_steps=70000)#many times atleast these many iterations are required to get a good solution
		rospy.sleep(0.1)
		csv_file = '/home/asl-7/graspit_ros_ws/src/Grasp_Planning/docs/trial/'+file_name+'.csv'
		
		#writing all 10 grasp in a csv file with all parameters
		for grasp in planned_grasps.grasps:
			quaternion = (grasp.pose.orientation.x,grasp.pose.orientation.y,grasp.pose.orientation.z,grasp.pose.orientation.w)
			roll,pitch,yaw = tf.transformations.euler_from_quaternion(quaternion)
			with open(csv_file, mode='a') as f:
				header = ['X','Y','Z','roll', 'pitch', 'yaw','x_q','y_q','z_q','w_q','lateral_joint_1','lateral_joint_2','d','epsilon_quality','volume_quality']
				thewriter = csv.DictWriter(f, fieldnames=header)
				if f.tell() == 0:
					thewriter.writeheader()
				thewriter.writerow({'X':grasp.pose.position.x,'Y':grasp.pose.position.y,'Z':grasp.pose.position.z,'roll':roll, 'pitch':pitch, 'yaw':yaw, \
						'x_q':grasp.pose.orientation.x,'y_q':grasp.pose.orientation.y,'z_q':grasp.pose.orientation.z,'w_q':grasp.pose.orientation.w,'d':12, \
							'lateral_joint_1':grasp.dofs[3],'lateral_joint_2':grasp.dofs[7],'epsilon_quality':grasp.epsilon_quality, 'volume_quality':grasp.volume_quality})

		print('Grasp planning for '+file_name+'.stl completed and saved in path:/home/asl-7/graspit_ros_ws/src/Grasp_Planning/docs/egad_eval_0.12/'+file_name+'.csv')

def grasp_2():
	# Plan grasps for missing files
	for file_name in missing_file_names:
		#function to load the scene for world
		print('Doing grasp planning for '+file_name+'.stl ....')
		load_world(file_name)
		planned_grasps = GraspitCommander.planGrasps(max_steps=70000)#many times atleast these many iterations are required to get a good solution
		rospy.sleep(0.1)
		csv_file = '/home/asl-7/graspit_ros_ws/src/Grasp_Planning/docs/egad_0.15/'+file_name+'.csv'
		
		#writing all 10 grasp in a csv file with all parameters
		for grasp in planned_grasps.grasps:
			quaternion = (grasp.pose.orientation.x,grasp.pose.orientation.y,grasp.pose.orientation.z,grasp.pose.orientation.w)
			roll,pitch,yaw = tf.transformations.euler_from_quaternion(quaternion)
			with open(csv_file, mode='a') as f:
				header = ['X','Y','Z','roll', 'pitch', 'yaw','x_q','y_q','z_q','w_q','lateral_joint_1','lateral_joint_2','d','epsilon_quality','volume_quality']
				thewriter = csv.DictWriter(f, fieldnames=header)
				if f.tell() == 0:
					thewriter.writeheader()
				thewriter.writerow({'X':grasp.pose.position.x,'Y':grasp.pose.position.y,'Z':grasp.pose.position.z,'roll':roll, 'pitch':pitch, 'yaw':yaw, \
						'x_q':grasp.pose.orientation.x,'y_q':grasp.pose.orientation.y,'z_q':grasp.pose.orientation.z,'w_q':grasp.pose.orientation.w,'d':12, \
							'lateral_joint_1':grasp.dofs[3],'lateral_joint_2':grasp.dofs[7],'epsilon_quality':grasp.epsilon_quality, 'volume_quality':grasp.volume_quality})

		print('Grasp planning for '+file_name+'.stl completed and saved in path:/home/asl-7/graspit_ros_ws/src/Grasp_Planning/docs/egad_0.15/'+file_name+'.csv')


if __name__ == "__main__":	
	try:
		grasp()
	except rospy.ROSInterruptException:
		pass
