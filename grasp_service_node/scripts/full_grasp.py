#! /usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import Pose
from grasp_service_node.srv import pose_dof, pose_dofResponse
from graspit_commander import GraspitCommander

epsi = []

def callback(request):

    if request.counter == 0:
        global planned_grasps
        # Import all the files in the world of graspit and perform the grasps
        GraspitCommander.clearWorld()
        #GraspitCommander.loadWorld("grasp")
        GraspitCommander.loadWorld("grasp_dataset")
        # Spawn the object at the correct object_pose
        GraspitCommander.importGraspableBody("/home/asl-7/graspit/models/objects/cylinder.xml", request.obj_pose)
        rospy.sleep(1)
        # Plan grasps
        planned_grasps = GraspitCommander.planGrasps()
         
        for grasp in planned_grasps.grasps:
            epsi.append(grasp.epsilon_quality)

        # Determining the best grasp by finding the index of the grasp with highest epsilon quality
        arg_max = np.argmax(np.array(epsi))
        best_pose = planned_grasps.grasps[arg_max].pose

        rospy.loginfo(best_pose)
        dofs = planned_grasps.grasps[arg_max].dofs

        # Manipulating the value to not include this same grasp result in the subsequent request iteration
        epsi[arg_max] = -10.0
        response = pose_dofResponse()
        response.gripper_pose = best_pose
        response.dofs = dofs
        return response
    
    # The counter variable value is 0 at the start. After the grasp is planned, the counter variable is changed. In the next iteration
    # All information regarding the grasp is already available, just finding the best grasp pose and passing it to the execution system.
    # TODO: Instead of counter, think of how to send all the data together(not the priority right now)
    elif request.counter > 0:
        # Determining the best grasp by finding the index of the grasp with highest epsilon quality
        arg_max = np.argmax(np.array(epsi))
        best_pose = planned_grasps.grasps[arg_max].pose
        rospy.loginfo(best_pose)
        dofs = planned_grasps.grasps[arg_max].dofs
        
        # Manipulating the value to not include this same grasp result in the subsequent request iteration
        epsi[arg_max] = -10.0
        response = pose_dofResponse()
        response.gripper_pose = best_pose
        response.dofs = dofs
        return response
       
def main():
    rospy.init_node("gripper_pose_server")
    # Starting the server
    s = rospy.Service('grasp_server', pose_dof, callback)
    print("server_running")
    rospy.spin()
    
if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    