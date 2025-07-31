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
model = YOLO("runs/detect/custom_yolov8/weights/best.pt")  

robotHandle = sim.getObject('/redundantRobot')
gripper_right_joint = sim.getObject('/redundantRobot/right_joint')
gripper_left_joint = sim.getObject('/redundantRobot/left_joint')
tip_handle = sim.getObject('/redundantRobot/redundantRob_tip')
target_handle1 = sim.getObject('/redundantRobot/redundantRob_target1')
base_handle = sim.getObject('/redundantRobot/redundantRob_link1')
productHandle = sim.getObject('/Product')
vision_handle = sim.getObject('/visionSensor')
conveyor_handle = sim.getObject('/conveyorSystem') 

# 조인트 값 7개, joint 7
joints = sim.getObjectsInTree(robotHandle, sim.object_joint_type)[:7]

k = np.pi/180

# move_joint 함수에 쓸 변수 , variable using move_joint function
#grap_product_joints = [0 * k, -18* k, 0 * k,-13 * k ,0 * k ,-150 * k , 0 * k]
app_product_joints = [0* k, -18 * k, 0 * k, 0 * k ,0 * k ,-180 * k , 0 * k]
grap_product_joints = [0 * k, -18* k, 0 * k,-13 * k ,0 * k ,-140 * k , 0 * k]
home_joints = [0 * k, 0 * k, 0 * k,0 * k ,0 * k ,0 * k , 0 * k]
move_product_joints = [-70 * k, -30* k, 0 * k, 0 * k ,0 * k ,-150 * k , 0 * k]

# move_linear
appOffset = 0.1
pick_pos = [0.12328, 0.2781, 0.20384]
pick_app_pos = pick_pos
pick_app_pos[2] = pick_app_pos[2] + appOffset

left_pos = [0.100, -0.325, 0.400]
left_app_pos = left_pos
left_app_pos[2] = left_app_pos[2] - appOffset

# 속도,가속도 설정 ,v acc setting
vel_x = 0.5
accel_x = 0.5
jerk_x = 0.5

vel_j = 60
accel_j = 40
jerk_j = 80

#재구성 , reconfiguration
maxvel_j = [vel_j, vel_j, vel_j, vel_j, vel_j, vel_j ,vel_j]
maxacc_j = [accel_j, accel_j, accel_j, accel_j, accel_j, accel_j,accel_j]
maxjerk_j = [jerk_j, jerk_j, jerk_j, jerk_j, jerk_j, jerk_j,jerk_j]

maxvel_x = [vel_x, vel_x, vel_x, 1]
maxacc_x = [accel_x, accel_x, accel_x, 1]
maxjerk_x = [jerk_x, jerk_x, jerk_x, 1]

# area 1 2 3 4 
area0_y = [221,200,0] 
area1_y = [200,186,1] 
area2_y = [186,175,2]
area3_y = [175,161,3]

x0 = 0.125
y0 = 0.025
d =  0.052

def real_coord(n,pixel_ymax,area_ymax):
    if n == 0:
        real_y = d * 3
    else:
        real_y = d * 3 + (pixel_ymax/area_ymax)*(1/2)*d + y0 + (n-1)*(1/2)*d
    
    return real_y

def find_location(pixel_y1,pixel_y2):
    
    print("크기 정하기",pixel_y1, pixel_y2)
    if pixel_y1 > pixel_y2:
        y= pixel_y1
    else:
        y= pixel_y2   
    
    if area0_y[0]> y and area0_y[1]< y :
        n = area0_y[2]
        pixel_ymax = abs(pixel_y2 - pixel_y1)
        area1_ymax = area0_y[0] - area0_y[1]
        real_y = real_coord(n,pixel_ymax,area1_ymax)     
        return real_y
    elif area1_y[0]> y and area1_y[1]< y :
        n = area1_y[2]
        pixel_ymax = abs(pixel_y2 - pixel_y1)
        area1_ymax = area1_y[0] - area1_y[1]
        real_y = real_coord(n,pixel_ymax,area1_ymax)     
        return real_y    
    elif area2_y[0]>y and area2_y[1]<y :
        n = area2_y[2]
        pixel_ymax = abs(pixel_y2 - pixel_y1)           
        area2_ymax = area2_y[0] - area2_y[1]
        real_y=real_coord(n,pixel_ymax,area2_ymax)
        return real_y
    elif area3_y[0]>y and area3_y[1]<y :
        n = area3_y[2]
        pixel_ymax = abs(pixel_y2 - pixel_y1)
        area3_ymax = area3_y[0] - area3_y[1]
        real_y = real_coord(n,pixel_ymax,area3_ymax)
        return real_y
    else:
        print("위치 추적실패")
        sim.stopSimulation()




sim.startSimulation()

# 그리퍼 열기 , open_gripper

setting.open_gripper(sim,gripper_right_joint,gripper_left_joint)
time.sleep(1)
try:
    while True:
       # 이미지 프레임 가져오기 (RGB) , take image frame
        img_bytes, resolution = sim.getVisionSensorImg(vision_handle, 0)
        width, height = resolution
        
        # 바이트 → numpy 이미지 , change byte to numpy image
        img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(height, width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 0)  # Y축 반전
        
        results = model(img, stream=True)
        #바운딩 박스 그리기 , write bounding box
        for result in results:
           boxes = result.boxes
           if len(boxes) == 0:
               x1, x2, y1, y2 = 0, 0, 0, 0  # 탐지된 박스 없음
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
            conf = box.conf[0].item()               # 신뢰도
            cls = int(box.cls[0])                   # 클래스 인덱스
            label = model.names[cls]                # 클래스 이름

            if conf >= 0.7:
               cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
               cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # OpenCV로 화면에 출력
        cv2.imshow('Vision Sensor Stream', img)
        
        box_value = abs(x1-x2)*abs(y1-y2)
        
        if box_value > 300 and box_value < 2000 :
            real_y = find_location(y1,y2)
            print("y는 ",real_y)
            setting.stop_container(sim,conveyor_handle)
            print("영상종료")
            break
        
        # ESC 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # 약간의 지연 (FPS 조정용)
        time.sleep(0.03)

finally:
    cv2.destroyAllWindows()
    