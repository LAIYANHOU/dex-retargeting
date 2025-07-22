import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import tyro
from loguru import logger


from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector



import numpy as np

from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu


#######################################################
"""This can control and query the LEAP Hand

I recommend you only query when necessary and below 90 samples a second.  Each of position, velociy and current costs one sample, so you can sample all three at 30 hz or one at 90hz.

#Allegro hand conventions:
#0.0 is the all the way out beginning pose, and it goes positive as the fingers close more and more
#http://wiki.wonikrobotics.com/AllegroHandWiki/index.php/Joint_Zeros_and_Directions_Setup_Guide I belive the black and white figure (not blue motors) is the zero position, and the + is the correct way around.  LEAP Hand in my videos start at zero position and that looks like that figure.

#LEAP hand conventions:
#180 is flat out for the index, middle, ring, fingers, and positive is closing more and more.

"""
########################################################
class LeapNode:
    def __init__(self):
        ####Some parameters
        # self.ema_amount = float(rospy.get_param('/leaphand_node/ema', '1.0')) #take only current
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 550
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
           
        #You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        try:
            self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 4000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, 'COM13', 4000000)
                self.dxl_client.connect()
        #Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain stiffness     
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2) # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2) # Igain
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain damping
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2) # Dgain damping for side to side should be a bit less
        #Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    #Receive LEAP pose and directly control the robot
    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #allegro compatibility
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #Sim compatibility, first read the sim value in range [-1,1] and then convert to leap
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #read position
    def read_pos(self):
        return self.dxl_client.read_pos()
    #read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()
    #read current
    def read_cur(self):
        return self.dxl_client.read_cur()



def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str, leaphand :LeapNode):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)


    # Different robot loader may have different orders for joints
    # sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    # retargeting_joint_names = retargeting.joint_names
    # retargeting_to_sapien = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names]).astype(int)

    while True:
        start_t = time.time()
        try:
            rgb = queue.get(timeout=50)
        except Empty:
            logger.error(f"Fail to fetch image from camera in 50 secs. Please check your web camera device.")
            return

        _, joint_pos, _, _ = detector.detect(rgb)
        if joint_pos is None:
            logger.warning(f"{hand_type} hand is not detected.")
        else:
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            qpos = retargeting.retarget(ref_value)
            # print("qpos: " + ", ".join(f"{pos:.4f}" for pos in qpos))

            qpos_cmd = np.zeros(16)
            # current_pos = leaphand.read_pos()
            # diff = (qpos[0]- current_pos[0] + 3.14) 
            # if  abs(diff)> 0.001:
            #     diff = np.sign(diff) * 0.001
                 
            # qpos_cmd[0] =  current_pos[0] +  diff 

            # qpos_cmd[0] = qpos[0]
            # qpos_cmd[1] = qpos[1]
            # qpos_cmd[2] = qpos[2]
            # qpos_cmd[3] = qpos[3]

            # qpos_cmd[4] = qpos[8] # thumb - middle
            # qpos_cmd[5] = qpos[9]
            # qpos_cmd[6] = qpos[10]
            # qpos_cmd[7] = qpos[11]

            # qpos_cmd[8] = qpos[12] # none
            # qpos_cmd[9] = qpos[13]
            # qpos_cmd[10] = qpos[14]
            # qpos_cmd[11] = qpos[15]

            # qpos_cmd[12] = qpos[4] # thumb - middle 
            # qpos_cmd[13] = qpos[5]
            # qpos_cmd[14] = qpos[6]
            # qpos_cmd[15] = qpos[7]

            # ['1', '0', '2', '3', '12', '13', '14', '15', '5', '4', '6', '7', '9', '8', '10', '11']

            qpos_cmd[0] = qpos[1]
            qpos_cmd[1] = qpos[0]
            qpos_cmd[2] = qpos[2]
            qpos_cmd[3] = qpos[3]

            qpos_cmd[4] = qpos[9] # thumb - middle
            qpos_cmd[5] = qpos[8]
            qpos_cmd[6] = qpos[10]
            qpos_cmd[7] = qpos[11]

            qpos_cmd[8] = qpos[13] # none
            qpos_cmd[9] = qpos[12]
            qpos_cmd[10] = qpos[14]
            qpos_cmd[11] = qpos[15]

            qpos_cmd[12] = qpos[4] # thumb - middle 
            qpos_cmd[13] = qpos[5]
            qpos_cmd[14] = qpos[6]
            qpos_cmd[15] = qpos[7]

            # qpos_cmd[8] = qpos[8]        

            # qpos_cmd = qpos
            print(f"{qpos_cmd[1]:.4f}")
            # print("qpos_cmd: " + ", ".join(f"{pos:.4f}" for pos in qpos_cmd))
            end_t = time.time()
            # print(f"time: {end_t - start_t:.4f} s")

            leaphand.set_allegro(qpos_cmd)


            # print("Position: " + str(leaphand.read_pos()))
            # time.sleep(0.02)
            # a = input("test")



def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    if camera_path is None:
        print("test")
        #cap = cv2.VideoCapture("/dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_405_Intel_R__RealSense_TM__Depth_Camera_405-video-index0")
        # cap = cv2.VideoCapture("/dev/v4l/by-id/usb-Gemeric_USB_Camera_0001-video-index0")      
        #if not cap.isOpened:
        for i in range(20):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_path = i
                print(f"[INFO] Camera opened successfully: {camera_path}")
                break
            else:
                print(f"[INFO] Failed to open camera: {i}")
    else:
        print("test2")
        cap = cv2.VideoCapture(camera_path)

    if not cap.isOpened():
        print(f"[ERROR] Failed to open camera: {camera_path}")
        return
    else:
        print(f"[INFO] Camera opened successfully: {camera_path}")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        queue.put(image)
        time.sleep(1 / 60.0)
        cv2.imshow("demo", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main(
    robot_name: RobotName, retargeting_type: RetargetingType, hand_type: HandType, camera_path: Optional[str] = None
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
    """
    # config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    # robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    
    # Update these paths to your local paths
    #config_path = Path("../../dex_retargeting/configs/teleop/leap_hand_right_dexpilot.yml")
    #robot_dir = Path("../../assets/robots/hands")
    config_path = Path(__file__).resolve().parents[2] / "dex_retargeting/configs/teleop/leap_hand_right_dexpilot.yml"
    robot_dir = Path(__file__).resolve().parents[2] / "assets/robots/hands"

    queue = multiprocessing.Queue(maxsize=1)
    producer_process = multiprocessing.Process(target=produce_frame, args=(queue, camera_path))
    # print("test3")

    leap_hand = LeapNode()
    # print("test4")
    consumer_process = multiprocessing.Process(target=start_retargeting, args=(queue, str(robot_dir), str(config_path), leap_hand))

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(5)

    print("done")


if __name__ == "__main__":
    tyro.cli(main)
