#include "ros/ros.h"
#include "geometry_msgs/Pose.h"
#include "actionlib/client/simple_action_client.h"
#include "std_msgs/String.h"
#include "graspit_interface/PlanGraspsAction.h"
#include "graspit_interface/ClearWorld.h"
#include "graspit_interface/ImportGraspableBody.h"
#include "graspit_interface/ImportRobot.h"
#include "graspit_interface/Planner.h"
#include "graspit_interface/PlanGraspsGoal.h"
#include "graspit_interface/SearchSpace.h"
#include "graspit_interface/SearchContact.h"

// float temp = 0;
typedef actionlib::SimpleActionClient<graspit_interface::PlanGraspsAction> Client;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "grasper_client");
    
    ros::NodeHandle n;
    ros::ServiceClient clear_world = n.serviceClient<graspit_interface::ClearWorld>("/graspit/clearWorld");
    ros::ServiceClient import_graspable_body = n.serviceClient<graspit_interface::ImportGraspableBody>("/graspit/importGraspableBody");
    ros::ServiceClient import_robot = n.serviceClient<graspit_interface::ImportRobot>("/graspit/importRobot");    
    
    Client client("/graspit/planGrasps", true);
    client.waitForServer();

    graspit_interface::ImportGraspableBody srv;
    graspit_interface::ImportRobot robot;
    graspit_interface::ClearWorld clear;
    graspit_interface::Planner plan;
    graspit_interface::SearchSpace ss;
    graspit_interface::SearchContact sc;

    graspit_interface::PlanGraspsGoal planner;
    

    geometry_msgs::Pose object_pose;
    geometry_msgs::Pose robot_pose;


    object_pose.orientation.w = 1;
    object_pose.orientation.x = 0;
    object_pose.orientation.y = 0;
    object_pose.orientation.z = 0;
    object_pose.position.x = 0;
    object_pose.position.y = 0;
    object_pose.position.z = 0;

    robot_pose.orientation.w = 1;
    robot_pose.orientation.x = 0;
    robot_pose.orientation.y = 0;
    robot_pose.orientation.z = 0;
    robot_pose.position.x = 0.3;
    robot_pose.position.y = 0;
    robot_pose.position.z = 0;



    srv.request.filename = "cylinder";
    srv.request.pose = object_pose;


    robot.request.filename = "RobotIQ";
    robot.request.pose = robot_pose;

    clear_world.call(clear);
    import_graspable_body.call(srv);
    import_robot.call(robot);


    // object_id.call(id);
    planner.graspable_body_id = 0;
    planner.max_steps = 50000;
    planner.feedback_num_steps = -1;
    planner.planner = plan;
    planner.search_energy = "GUIDED_POTENTIAL_QUALITY_ENERGY";
    planner.search_space = ss;
    planner.search_contact = sc;

    client.sendGoal(planner);
    client.waitForResult(ros::Duration(0));

    graspit_interface::PlanGraspsResultConstPtr result = client.getResult(); 

    std::cout<<(result->grasps[0].pose.orientation.w);
    return 0;    
}   
