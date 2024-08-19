#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from apriltag_ros.msg import AprilTagDetectionArray

# quaternion --> rotation matrix
def correct_quaternion_to_rotation_matrix(q):
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx**2 - 2*qy**2]
    ])

def apriltag(x, y, z):
    p = PoseStamped()
    p.pose.position.x = x
    p.pose.position.y = y
    p.pose.position.z = z
    return p

apriltags = [apriltag(1, 1, 0.05),
             apriltag(1, -1, 0.05),
             apriltag(-1, 1, 0.05),
             apriltag(-1, -1, 0.05),
             apriltag(0, 0, 0.05),]


def tag_detections_callback(msg):
    for detection in msg.detections:
        # 獲取AprilTag相對於相機的位置（rotation matrix）＆方向（quaternion）
        tag_id = detection.id[0]
        position = detection.pose.pose.pose.position
        orientation = detection.pose.pose.pose.orientation
        
        rel_pos_b = np.array([position.x, position.y, position.z])
        q = [orientation.x, orientation.y, orientation.z, orientation.w]
        R = correct_quaternion_to_rotation_matrix(q)

        # 將相對位置從body frame轉換到世界座標系
        rel_pos = np.dot(R, rel_pos_b)

        # 檢查tag_id是否在有效範圍內
        if tag_id < len(apriltags):
            tag_pos = np.array([apriltags[tag_id].pose.position.x,
                                apriltags[tag_id].pose.position.y,
                                apriltags[tag_id].pose.position.z])
            
            # 將相對位置 + AprilTag絕對座標 ＝ 無人機測量位置（世界座標系）
            measurement = tag_pos - rel_pos
            rospy.loginfo("AprilTag detection: ID = %d\n measurement = %s", tag_id, measurement)

        else:
            rospy.logwarn("Detected AprilTag with ID %d is out of range.", tag_id)


def main():
    rospy.init_node('apriltag_measure_uav')

    rospy.Subscriber('/camera_down/tag_detections', AprilTagDetectionArray, tag_detections_callback)

    rate = rospy.Rate(50)

    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass