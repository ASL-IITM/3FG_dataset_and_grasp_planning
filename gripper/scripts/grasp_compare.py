import rospy
from graspit_commander import GraspitCommander
from geometry_msgs.msg import Pose
from World_scene import load_world
import os

def graspit_grasp(file_name):
    planned_grasps = GraspitCommander.planGrasps(max_steps=70000)
    best_pose = planned_grasps.grasps[0].pose
    GraspitCommander.setRobotPose(best_pose)

def cnn_grasp():
    pass

def main():
    rospy.init_node('graspit_commander', anonymous=True)
    directory = '/home/asl-7/Downloads/egad_eval_set_0.12/processed_meshes'
    file_names = [file[:-4] for file in os.listdir(directory) if file.endswith('.stl')]

    for file_name in file_names:
        load_world(file_name)
        graspit_grasp(file_name)
        input()

if __name__ == "__main__":
    main()
