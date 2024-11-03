#! /usr/bin/env python

from graspit_commander import GraspitCommander
from geometry_msgs.msg import Pose
from math import pi


#CHANGE THE ORIGIN OF THE OBJECT TO SOMETHING FEASIBLE WITH REAL ROBOT
gripper = Pose()
object_pose = Pose()
obstacle_pose = Pose()
obstacle_pose_1 = Pose()
obstacle_pose_2 = Pose()
obstacle_pose_3 = Pose()

object_pose.position.x = -0.05
object_pose.position.y = 0
object_pose.position.z = 0.025
object_pose.orientation.x = 0
object_pose.orientation.y = 0
object_pose.orientation.z = 0
object_pose.orientation.w = 1

obstacle_pose.position.x = 0
obstacle_pose.position.y = 0
obstacle_pose.position.z = 0
obstacle_pose.orientation.x = 0
obstacle_pose.orientation.y = 0
obstacle_pose.orientation.z = 0
obstacle_pose.orientation.w = 1

obstacle_pose_1.position.x = 0.18
obstacle_pose_1.position.y = 0.18
obstacle_pose_1.position.z = 0.125
obstacle_pose_1.orientation.x = 0
obstacle_pose_1.orientation.y = 0
obstacle_pose_1.orientation.z = 0
obstacle_pose_1.orientation.w = 1

obstacle_pose_2.position.x = 0
obstacle_pose_2.position.y = 0.25
obstacle_pose_2.position.z = 0.125
obstacle_pose_2.orientation.x = 0
obstacle_pose_2.orientation.y = 0
obstacle_pose_2.orientation.z = 0
obstacle_pose_2.orientation.w = 1

obstacle_pose_3.position.x = -0.18
obstacle_pose_3.position.y = 0.18
obstacle_pose_3.position.z = 0.125
obstacle_pose_3.orientation.x = 0
obstacle_pose_3.orientation.y = 0
obstacle_pose_3.orientation.z = 0
obstacle_pose_3.orientation.w = 1

gripper.position.x = 0
gripper.position.y = 0
gripper.position.z = 1

gripper.orientation.x = 0
gripper.orientation.y = 0
gripper.orientation.z = 0
gripper.orientation.w = 1

def load_world(file_name):
    GraspitCommander.clearWorld()

    GraspitCommander.importRobot("RobotIQ",gripper)
    #GraspitCommander.importGraspableBody("/home/asl-7/Downloads/egad_train_set_0.15_stl/" + file_name + ".stl", object_pose)
    GraspitCommander.importGraspableBody('/home/asl-7/Downloads/egad_eval_set_0.12/processed_meshes/' + file_name + ".stl", object_pose)
    GraspitCommander.importObstacle("/home/asl-7/graspit/models/obstacles/table.xml",obstacle_pose)
    GraspitCommander.importObstacle("/home/asl-7/graspit/models/obstacles/stand_cubic.xml",obstacle_pose_1)
    GraspitCommander.importObstacle("/home/asl-7/graspit/models/obstacles/stand_cubic.xml",obstacle_pose_2)
    GraspitCommander.importObstacle("/home/asl-7/graspit/models/obstacles/stand_cubic.xml",obstacle_pose_3)
    return 

'''
Axis information
1. The Y axis is aligned with the normal of the palm.
2. The X axis passes through the line bisecting the thumb finger/middle finger
3. The Z axis is orthogonal to X and Y axis.

Finger DOF information
dof[0]: thumb finger base joint
dof[1]: thumb finger middle joint
dof[2]: thumb finger tip joint
dof[3]: finger 2 lateral joint
dof[4]: finger 2 base joint
dof[5]: finger 2 middle joint
dof[6]: finger 2 fingertip joint
dof[7]: finger 3 lateral joint
dof[8]: finger 3 base joint
dof[9]: finger 3 middle joint
dof[10]: finger 3 fingertip joint
'''