#!/usr/bin/env python

import rospy
from tf.transformations import quaternion_matrix, quaternion_from_matrix
import numpy as np
from geometry_msgs.msg import Pose

def object_world_transformation(gripper_pose, object_pose):
    pos = []

    # Obtain the 4x4 homogeneous rotation matrix without the translation information.
    gripper_rotation_matrix = quaternion_matrix([gripper_pose.orientation.x,gripper_pose.orientation.y,gripper_pose.orientation.z,gripper_pose.orientation.w])
    object_rotation_matrix = quaternion_matrix([object_pose.orientation.x, object_pose.orientation.y, object_pose.orientation.z, object_pose.orientation.w])

    # Update the translational terms of both the homogeneous matrices
    gripper_rotation_matrix[0,3] = gripper_pose.position.x
    object_rotation_matrix[0,3] = object_pose.position.x
    gripper_rotation_matrix[1,3] = gripper_pose.position.y
    object_rotation_matrix[1,3] = object_pose.position.y
    gripper_rotation_matrix[2,3] = gripper_pose.position.z
    object_rotation_matrix[2,3] = object_pose.position.z

    # Matrix Multiplying to get the final homogeneous transform
    final_transformation = np.matmul(gripper_rotation_matrix, object_rotation_matrix)
    # Finding the quaternions from the final matrix
    quat = quaternion_from_matrix(final_transformation)

    for i in range(3):
        pos.append(final_transformation[i,3])
    
    # Adding all the calculated elements into the final_gripper_pose variable
    final_gripper_pose = Pose()
    final_gripper_pose.position.x = pos[0]
    final_gripper_pose.position.y = pos[1]
    final_gripper_pose.position.z = pos[2]
    final_gripper_pose.orientation.x = quat[0]
    final_gripper_pose.orientation.y = quat[1]
    final_gripper_pose.orientation.z = quat[2]
    final_gripper_pose.orientation.w = quat[3]

    rospy.loginfo(final_gripper_pose)

    return final_gripper_pose

if __name__ == "__main__":
    gripper = Pose()
    object = Pose()

    gripper.position.x = 0
    gripper.position.y = 0.5
    gripper.position.z = 0.5
    gripper.orientation.x = 3.14
    gripper.orientation.y = 0
    gripper.orientation.z = 3.14
    gripper.orientation.w = 1

    object.position.x = 0.5
    object.position.y = 0
    object.position.z = 0
    object.orientation.x = 1.57
    object.orientation.y = 3.14
    object.orientation.z = 0
    object.orientation.w = 1

    world_gripper_pose = object_world_transformation(gripper, object)
    print(world_gripper_pose)
    