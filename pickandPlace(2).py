import Command
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
# coppiliasim 연결 
client = RemoteAPIClient()
sim = client.require('sim')
# 실습환경 on
sim.loadScene('/home/ehdtod001009/Downloads/CoppeliaSim_Edu_V4_9_0_rev6_Ubuntu22_04/실습파일/simple_pick_and_place(2).ttt')
# 로봇객체 받아오기
robotHandle = sim.getObject('/UR10')
# 조인트 값 6개 (조인트값을 받아올 로봇 / 타입자지정 )
joints = sim.getObjectsInTree(robotHandle, sim.object_joint_type)[:6]
#그리퍼, 손끝 , 베이스 , 타켓 , 물체 객체화
gripper_handle = sim.getObject('/UR10/RG2')
tip_handle = sim.getObject('/UR10/UR10_tip')
target_handle = sim.getObject('/UR10/UR10_target')
base_handle = sim.getObject('/UR10')
productHandle = sim.getObject('/Product')
#속도,가속도 설정
vel_x = 0.5
accel_x = 0.5
jerk_x = 0.5

vel_j = 60
accel_j = 40
jerk_j = 80



home_pose_joint = [0 * np.pi/180, 0 * np.pi/180, 90 * np.pi/180, 0 * np.pi/180, -90.0 * np.pi/180, 0 * np.pi/180]
place_direction_pose_joint = [-90 * np.pi/180, 0 * np.pi/180, 90 * np.pi/180, 0 * np.pi/180, -90.0 * np.pi/180, 0 * np.pi/180]
#속도, 가속도 리스트 재구성 
maxvel_j = [vel_j, vel_j, vel_j, vel_j, vel_j, vel_j]
maxacc_j = [accel_j, accel_j, accel_j, accel_j, accel_j, accel_j]
maxjerk_j = [jerk_j, jerk_j, jerk_j, jerk_j, jerk_j, jerk_j]

maxvel_x = [vel_x, vel_x, vel_x, 1]
maxacc_x = [accel_x, accel_x, accel_x, 1]
maxjerk_x = [jerk_x, jerk_x, jerk_x, 1]

appOffset = 0.1
pick_pos = [0.03536, 0.650, 0.715]
pick_app_pos = pick_pos
pick_app_pos[2] = pick_app_pos[2] + appOffset

place_pos = [0.7, 0.0, 0.615]
place_x_offset = 0.06
place_y_offset = 0.06

place_pos_list = []
count_x = 3
count_y = 3

product_pose = sim.getObjectPose(productHandle)

for i in range(count_x):
    for j in range(count_y):
        offset_x = i * place_x_offset
        offset_y = j * place_y_offset
        new_pos = place_pos + np.array([offset_x, offset_y, 0])
        place_pos_list.append(new_pos)
print('place_pos_list= ',place_pos_list)
sim.startSimulation()

for place_pos_iter in place_pos_list:
   
    place_pos = place_pos_iter
    place_app_pos = place_pos
    place_app_pos[2] = place_app_pos[2] + appOffset
    Command.move_joint(sim, joints, tip_handle, target_handle, maxvel_j, maxacc_j, maxjerk_j, home_pose_joint)
    Command.move_linear(sim, tip_handle, target_handle, maxvel_x, maxacc_x, maxjerk_x, pick_app_pos)
    Command.move_linear(sim, tip_handle, target_handle, maxvel_x, maxacc_x, maxjerk_x, pick_pos)
    Command.open_gripper(sim,'RG2_open')
    Command.move_linear(sim, tip_handle, target_handle, maxvel_x, maxacc_x, maxjerk_x, pick_app_pos)
    Command.move_joint(sim, joints, tip_handle, target_handle, maxvel_j, maxacc_j, maxjerk_j, home_pose_joint)
    Command.move_joint(sim, joints, tip_handle, target_handle, maxvel_j, maxacc_j, maxjerk_j, place_direction_pose_joint)
    Command.move_linear(sim, tip_handle, target_handle, maxvel_x, maxacc_x, maxjerk_x, place_app_pos)
    Command.move_linear(sim, tip_handle, target_handle, maxvel_x, maxacc_x, maxjerk_x, place_pos)
    Command.close_gripper(sim,'RG2_open')
    Command.move_linear(sim, tip_handle, target_handle, maxvel_x, maxacc_x, maxjerk_x, place_app_pos)
    Command.move_joint(sim, joints, tip_handle, target_handle, maxvel_j, maxacc_j, maxjerk_j, place_direction_pose_joint)
    Command.move_joint(sim, joints, tip_handle, target_handle, maxvel_j, maxacc_j, maxjerk_j, home_pose_joint)
sim.stopSimulation()