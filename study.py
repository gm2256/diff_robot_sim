from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import math
import time
import setting
import cv2
from ultralytics import YOLO

client = RemoteAPIClient()

sim = client.require('sim')
sim.loadScene('/home/ehdtod001009/Downloads/CoppeliaSim_Edu_V4_9_0_rev6_Ubuntu22_04/scenes/practice.ttt')  

robotHandle = sim.getObject('/UR5')
tip_handle = sim.getObject('/UR5/tip')
target_handle = sim.getObject('/target')
base_handle = sim.getObject('/UR5/link')
gripper_handle = sim.getObject('/UR5/RG2')
vision_handle = sim.getObject('/visionSensor')
conveyor_handle = sim.getObject('/conveyorSystem') 
sensorHandle = sim.getObject('/proximitySensor')

#박스 크기 저장 [X,Y,Z] 
box1 = [0.07 , 0.07 ,0.07]
box2 = [0.15 , 0.075 ,0.075]
box3 = [0.10 , 0.05 ,0.10]

# 조인트 값 6개, joint 6
joints = sim.getObjectsInTree(robotHandle, sim.object_joint_type)[:6]

k = np.pi/180
# move_joint 함수에 쓸 변수 , variable using move_joint function
#grap_product_joints = [0 * k, -18* k, 0 * k,-13 * k ,0 * k ,-150 * k , 0 * k]
home_joints = [0 * k, 0 * k, -90 * k,0 * k , 90 * k , 0 * k]
app_product_joints = [0 * k, -10 * k, -70* k, -8 * k ,90* k , 0 * k]
move_product_joints1 = [-40 * k, -16* k, -46 * k, -24 * k ,90* k ,-1.378 * k]
move_product_joints2 = [-60 * k, -16* k, -46 * k, -24 * k ,90* k ,-1.378 * k]
move_product_joints3 = [-90 * k, -16* k, -46 * k, -24 * k ,90* k ,-1.378 * k]

object_detected = False
color = "없음"
# 속도,가속도 설정 ,v acc setting
vel_x = 0.5
accel_x = 0.5
jerk_x = 0.5

vel_j = 60
accel_j = 40
jerk_j = 80

# reconfiguration
maxvel_j = [vel_j, vel_j, vel_j, vel_j, vel_j, vel_j]
maxacc_j = [accel_j, accel_j, accel_j, accel_j, accel_j, accel_j]
maxjerk_j = [jerk_j, jerk_j, jerk_j, jerk_j, jerk_j, jerk_j]

maxvel_x = [vel_x, vel_x, vel_x, 1]
maxacc_x = [accel_x, accel_x, accel_x, 1]
maxjerk_x = [jerk_x, jerk_x, jerk_x, 1]


# 빈 통 위치  

left_pos1 = [0.025, -0.300, 0.400]
left_app_pos1 = left_pos1.copy()
left_app_pos1[2] = left_app_pos1[2] + 0.2

left_pos2 = [-0.300, -0.300, 0.400]
left_app_pos2 = left_pos2.copy()
left_app_pos2[2] = left_app_pos2[2] + 0.2

left_pos3 = [-0.600, -0.300, 0.400]
left_app_pos3 = left_pos3.copy()
left_app_pos3[2] = left_app_pos3[2] + 0.2

move_pos1 = [0.025, -0.300, 0.200]

# 색상 범위 정의 (HSV 기준)
color_ranges = {
    "RED": [(np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([160, 100, 100]), np.array([180, 255, 255]))],
    "GREEN": [(np.array([35, 100, 100]), np.array([85, 255, 255]))],
    "BLUE": [(np.array([100, 150, 0]), np.array([140, 255, 255]))]
}


def act(type,color,x):
    # 카메라로 물체위치 보정
    # 물체 위치 -- 정사각형 기준  
    position = [0.125, 0.275, 0.21]    
    
    if type == "정사각형" :
        
       if x > 140 :
        position[0] = position[0] + 0.03       
       elif x < 95:
        position[0] = position[0] - 0.03       
       else:     
        position[0] = 0.125    
                    
    elif type == "직사각형":

        if x > 140 :
            position[0] = position[0] + 0.03       
        elif x < 95:
            position[0] = position[0] - 0.03       
        else:     
            position[0] = 0.125    

        position[1] = position[1] + 0.05
                  
    else :
        print("해당하는 박스가 없음")                
    
    pick_pos = position.copy()
    pick_pos[2] = pick_pos[2] + 0.2

    pick_up = position.copy()
    pick_up[2] = pick_up[2] + 0.4

    #grap_stuff
    setting.move_joint(sim, joints, tip_handle, target_handle, maxvel_j, maxacc_j, maxjerk_j,home_joints)
    setting.move_joint(sim, joints, tip_handle, target_handle, maxvel_j, maxacc_j, maxjerk_j,app_product_joints)
    setting.open_gripper(sim,'RG2_open')
    setting.move_linear(sim, tip_handle, target_handle, maxvel_x, maxacc_x, maxjerk_x, pick_pos)
    setting.close_gripper(sim,'RG2_open')
                            
    if color == "RED":
        #move
        setting.move_linear(sim, tip_handle, target_handle, maxvel_x, maxacc_x, maxjerk_x, pick_up)
        setting.move_joint(sim, joints, tip_handle, target_handle, maxvel_j, maxacc_j, maxjerk_j,move_product_joints1)
        setting.move_linear(sim, tip_handle, target_handle, maxvel_x, maxacc_x, maxjerk_x, left_app_pos1)
        setting.move_linear(sim, tip_handle, target_handle, maxvel_x, maxacc_x, maxjerk_x, left_pos1)
        setting.open_gripper(sim,'RG2_open')
        #return    
        setting.move_linear(sim, tip_handle, target_handle, maxvel_x, maxacc_x, maxjerk_x, left_app_pos1)
        setting.move_joint(sim, joints, tip_handle, target_handle, maxvel_j, maxacc_j, maxjerk_j,move_product_joints1)
        setting.move_joint(sim, joints, tip_handle, target_handle, maxvel_j, maxacc_j, maxjerk_j,home_joints)    
        setting.close_gripper(sim,'RG2_open')

    elif color == "GREEN":
        #move
        setting.move_linear(sim, tip_handle, target_handle, maxvel_x, maxacc_x, maxjerk_x, pick_up)
        setting.move_joint(sim, joints, tip_handle, target_handle, maxvel_j, maxacc_j, maxjerk_j,move_product_joints2)
        setting.move_linear(sim, tip_handle, target_handle, maxvel_x, maxacc_x, maxjerk_x, left_app_pos2)
        setting.move_linear(sim, tip_handle, target_handle, maxvel_x, maxacc_x, maxjerk_x, left_pos2)
        setting.open_gripper(sim,'RG2_open')
        #return    
        setting.move_linear(sim, tip_handle, target_handle, maxvel_x, maxacc_x, maxjerk_x, left_app_pos2)
        setting.move_joint(sim, joints, tip_handle, target_handle, maxvel_j, maxacc_j, maxjerk_j,move_product_joints2)
        setting.move_joint(sim, joints, tip_handle, target_handle, maxvel_j, maxacc_j, maxjerk_j,home_joints)    
        setting.close_gripper(sim,'RG2_open')

    elif color == "BLUE":
        #move
        setting.move_linear(sim, tip_handle, target_handle, maxvel_x, maxacc_x, maxjerk_x, pick_up)
        setting.move_joint(sim, joints, tip_handle, target_handle, maxvel_j, maxacc_j, maxjerk_j,move_product_joints3)
        setting.move_linear(sim, tip_handle, target_handle, maxvel_x, maxacc_x, maxjerk_x, left_app_pos3)
        setting.move_linear(sim, tip_handle, target_handle, maxvel_x, maxacc_x, maxjerk_x, left_pos3)
        setting.open_gripper(sim,'RG2_open')    
        #return    
        setting.move_linear(sim, tip_handle, target_handle, maxvel_x, maxacc_x, maxjerk_x, left_app_pos3)
        setting.move_joint(sim, joints, tip_handle, target_handle, maxvel_j, maxacc_j, maxjerk_j,move_product_joints3)
        setting.move_joint(sim, joints, tip_handle, target_handle, maxvel_j, maxacc_j, maxjerk_j,home_joints)    
        setting.close_gripper(sim,'RG2_open')
        
    
    else :
        print("not find bin included color") 

sim.startSimulation()
end_sim = 0

while True:
     
    result, distance, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = sim.readProximitySensor(sensorHandle)
    #end_sim
   
    if result:
        #센서감지후 정지
        setting.stop_container(sim,conveyor_handle)
        #초기화
        color = "없음"        
        i = 5
        # 캠키기        
        while i > 0 :
    
            # 이미지 프레임 가져오기 (RGB)
            img_bytes, resolution = sim.getVisionSensorImg(vision_handle, 0)
            width, height = resolution

            # 바이트 → numpy 이미지
            img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(height, width, 3)
            img = cv2.flip(img, 0)  # Y축 반전

        
            # 프레임을 HSV로 변환
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            for color_name, hsv_ranges in color_ranges.items():
                mask = None
                for lower, upper in hsv_ranges:
                    temp_mask = cv2.inRange(hsv, lower, upper)
                    mask = temp_mask if mask is None else cv2.bitwise_or(mask, temp_mask)

                # 노이즈 제거
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)

                # 외곽선 검출
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
                        
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # 너무 작은 것 제외
                        x, y, w, h = cv2.boundingRect(contour)
                        cx, cy = x + w // 2, y + h // 2
                                                
                        # 사각형과 텍스트 표시
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
                        cv2.circle(img, (cx, cy), 5, (255,255,255), -1)
                        cv2.putText(img, f"{color_name} ({cx},{cy})", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                        #저장                         
                        if w < h:
                            type = "직사각형"
                        else:
                            type = "정사각형"
                        
                        center_x = cx
                        color = color_name

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # 결과 프레임 출력
            cv2.imshow("Color Detection", img_bgr)
            i = i - 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  
        # 종료
        cv2.destroyAllWindows()

        # 동작                
        act(type,color,center_x)

    else:
        print("감지되는 것이 없습니다.")
        setting.activate_container(sim,conveyor_handle)
        end_sim = end_sim + 1
        time.sleep(0.2)
        if end_sim == 500 :
            sim.stopSimulation()

