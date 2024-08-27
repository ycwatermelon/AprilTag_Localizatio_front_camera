#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Vector3, TwistStamped
from apriltag_ros.msg import AprilTagDetectionArray
from sensor_msgs.msg import Imu
from kalman_filter import kalman_filter

# quaternion --> rotation matrix
def correct_quaternion_to_rotation_matrix(q):
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx**2 - 2*qy**2]
    ])
def front_camera_rotation_matrix():
    # 假設前置攝像頭沿著 X 軸正方向安裝，鏡頭朝前
    # 這個矩陣將把攝像頭坐標系轉換為 UAV 體坐標系
    return np.array([
        [0, 0, 1],  # 攝像頭的 Z 軸對應 UAV 的 X 軸
        [-1, 0, 0], # 攝像頭的 X 軸對應 UAV 的 -Y 軸
        [0, -1, 0]  # 攝像頭的 Y 軸對應 UAV 的 -Z 軸
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
             apriltag(0, 0, 0.05),
             apriltag(5, 0, 3),
             apriltag(5, 1, 4),
             apriltag(5, -1, 4),
             apriltag(5, 1, 2),
             apriltag(5, -1, 2),]
             
        
# 初始化卡爾曼濾波器
dt = 1.0 / 50  # 假設更新頻率為50Hz
nx = 9  # 狀態變量數量 (x, y, z, vx, vy, vz, ax, ay, az)
initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
initial_P = [10**2] * 3 + [5**2] * 3 + [2**2] * 3  # 增加初始不確定性
Q_scale = [0.5**2] * 3 + [0.3**2] * 3 + [0.2**2] * 3  # 適度增加過程噪聲
R_scale = [8**2, 8**2, 8**2] + [4**2] * 3 + [2**2] * 3  # 調整測量噪聲

kf_front = None
kf_down = None

# 用於存儲最近的估計結果
last_estimate_front = np.zeros((9, 1))
last_estimate_down = np.zeros((9, 1))

current_velocity = np.zeros(3)
current_acceleration = np.zeros(3)

# 下鏡頭處理函數
def process_tag_detection_down(detection):
    tag_id = detection.id[0]
    position = detection.pose.pose.pose.position
    orientation = detection.pose.pose.pose.orientation
    
    rel_pos_b = np.array([position.x, position.y, position.z])
    q = [orientation.x, orientation.y, orientation.z, orientation.w]
    R = correct_quaternion_to_rotation_matrix(q)

    rel_pos = np.dot(R, rel_pos_b)

    if tag_id < len(apriltags):
        tag_pos = np.array([apriltags[tag_id].pose.position.x,
                            apriltags[tag_id].pose.position.y,
                            apriltags[tag_id].pose.position.z])
        
        measurement = tag_pos - rel_pos
        
        rospy.loginfo("AprilTag detection (Down): ID = %d\n measurement = %s", tag_id, measurement)

        return tag_id, measurement
    else:
        rospy.logwarn("Detected AprilTag with ID %d is out of range.", tag_id)
        return None, None

# 前鏡頭處理函數
def process_tag_detection_front(detection):
    tag_id = detection.id[0]
    position = detection.pose.pose.pose.position
    orientation = detection.pose.pose.pose.orientation
    
    rel_pos_camera = np.array([position.x, position.y, position.z])
    q = [orientation.x, orientation.y, orientation.z, orientation.w]
    R_tag_to_camera = correct_quaternion_to_rotation_matrix(q)

    # 將相對位置從攝像頭坐標系轉換到無人機本體坐標系
    R_camera_to_body = front_camera_rotation_matrix()
    rel_pos_body = R_camera_to_body @ rel_pos_camera

    if tag_id < len(apriltags):
        tag_pos = np.array([apriltags[tag_id].pose.position.x,
                            apriltags[tag_id].pose.position.y,
                            apriltags[tag_id].pose.position.z])
        
        # 計算無人機在世界坐標系中的位置
        measurement = tag_pos - rel_pos_body
        
        rospy.loginfo("AprilTag detection (Front): ID = %d\n measurement = %s", tag_id, measurement)

        return tag_id, measurement
    else:
        rospy.logwarn("Detected AprilTag with ID %d is out of range.", tag_id)
        return None, None

def tag_detections_callback_measurement_down(msg, pub):
    for detection in msg.detections:
        tag_id, measurement = process_tag_detection_down(detection)
        if measurement is not None:
            measurement_msg = Vector3()
            measurement_msg.x = measurement[0]
            measurement_msg.y = measurement[1]
            measurement_msg.z = measurement[2]
            pub.publish(measurement_msg)

def tag_detections_callback_measurement_front(msg, pub):
    for detection in msg.detections:
        tag_id, measurement = process_tag_detection_front(detection)
        if measurement is not None:
            measurement_msg = Vector3()
            measurement_msg.x = measurement[0]
            measurement_msg.y = measurement[1]
            measurement_msg.z = measurement[2]
            pub.publish(measurement_msg)

def tag_detections_callback_kalman_down(msg, pub):
    global kf_down, current_velocity, current_acceleration, last_estimate_down
    for detection in msg.detections:
        tag_id, measurement = process_tag_detection_down(detection)
        if measurement is not None:
            full_measurement = np.concatenate([measurement, current_velocity, current_acceleration])
            
            kf_down.predict()
            estimated_state, _ = kf_down.update(full_measurement.reshape(9, 1))

            last_estimate_down = estimated_state

            kalman_msg = Vector3()
            kalman_msg.x = estimated_state[0, 0]
            kalman_msg.y = estimated_state[1, 0]
            kalman_msg.z = estimated_state[2, 0]
            pub.publish(kalman_msg)

            rospy.loginfo("AprilTag detection (Down): ID = %d\n measurement = %s\n kalman estimate = %s", 
                          tag_id, measurement, estimated_state[:3, 0])

def tag_detections_callback_kalman_front(msg, pub):
    global kf_front, current_velocity, current_acceleration, last_estimate_front
    for detection in msg.detections:
        tag_id, measurement = process_tag_detection_front(detection)
        if measurement is not None:
            full_measurement = np.concatenate([measurement, current_velocity, current_acceleration])
            
            kf_front.predict()
            estimated_state, _ = kf_front.update(full_measurement.reshape(9, 1))

            last_estimate_front = estimated_state

            kalman_msg = Vector3()
            kalman_msg.x = estimated_state[0, 0]
            kalman_msg.y = estimated_state[1, 0]
            kalman_msg.z = estimated_state[2, 0]
            pub.publish(kalman_msg)

            rospy.loginfo("AprilTag detection (Front): ID = %d\n measurement = %s\n kalman estimate = %s", 
                          tag_id, measurement, estimated_state[:3, 0])

def fuse_estimates():
    global last_estimate_front, last_estimate_down
    # 簡單的加權平均，可以根據需要調整權重
    weight_front = 0.5
    weight_down = 0.5
    fused_estimate = weight_front * last_estimate_front + weight_down * last_estimate_down
    return fused_estimate

def publish_fused_estimate(pub):
    fused_estimate = fuse_estimates()
    fused_msg = Vector3()
    fused_msg.x = fused_estimate[0, 0]
    fused_msg.y = fused_estimate[1, 0]
    fused_msg.z = fused_estimate[2, 0]
    pub.publish(fused_msg)

def local_position_callback(data, pub):
    local_pos_msg = Vector3()
    local_pos_msg.x = data.pose.position.x
    local_pos_msg.y = data.pose.position.y
    local_pos_msg.z = data.pose.position.z
    pub.publish(local_pos_msg)

def velocity_callback(msg):
    global current_velocity
    current_velocity = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])

def imu_callback(msg):
    global current_acceleration
    gravity = np.array([0, 0, 9.81])
    measured_acc = np.array([msg.linear_acceleration.x, 
                             msg.linear_acceleration.y, 
                             msg.linear_acceleration.z])
    current_acceleration = measured_acc - gravity

def main():
    global kf_front, kf_down
    rospy.init_node('apriltag_measure_uav')
    
    # 為前置和下置攝像頭分別初始化卡爾曼濾波器
    kf_front = kalman_filter(dt, nx, initial_state, initial_P, Q_scale, R_scale)
    kf_down = kalman_filter(dt, nx, initial_state, initial_P, Q_scale, R_scale)

    # 分別為前置和下置攝像頭創建發布者
    measurement_pub_front = rospy.Publisher('/uav/measurement/front', Vector3, queue_size=10)
    measurement_pub_down = rospy.Publisher('/uav/measurement/down', Vector3, queue_size=10)
    kalman_pub_front = rospy.Publisher('/uav/kalman/front', Vector3, queue_size=10)
    kalman_pub_down = rospy.Publisher('/uav/kalman/down', Vector3, queue_size=10)
    
    # 創建融合估計的發布者
    fused_estimate_pub = rospy.Publisher('/uav/kalman/fused', Vector3, queue_size=10)

    local_position_pub = rospy.Publisher('/uav/local_position', Vector3, queue_size=10)
    
    rospy.Subscriber('/mavros/local_position/pose', PoseStamped, local_position_callback, local_position_pub)
    
    # 下置攝像頭訂閱器
    rospy.Subscriber('/camera_down/tag_detections', AprilTagDetectionArray, 
                     lambda msg: tag_detections_callback_measurement_down(msg, measurement_pub_down))
    rospy.Subscriber('/camera_down/tag_detections', AprilTagDetectionArray, 
                     lambda msg: tag_detections_callback_kalman_down(msg, kalman_pub_down))
    
    # 前置攝像頭訂閱器
    rospy.Subscriber('/camera_front/tag_detections', AprilTagDetectionArray, 
                     lambda msg: tag_detections_callback_measurement_front(msg, measurement_pub_front))
    rospy.Subscriber('/camera_front/tag_detections', AprilTagDetectionArray, 
                     lambda msg: tag_detections_callback_kalman_front(msg, kalman_pub_front))
    
    rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, velocity_callback)
    rospy.Subscriber('/mavros/imu/data', Imu, imu_callback)

    rate = rospy.Rate(50)

    while not rospy.is_shutdown():
        # 在每次循環中發布融合估計
        publish_fused_estimate(fused_estimate_pub)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
